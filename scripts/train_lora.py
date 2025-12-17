"""
LoRA Training Script for Stable Diffusion 1.5
Optimized for H100 GPU using diffusers + PEFT

This is an alternative to Kohya's sd-scripts for simpler setup
"""

import os
import math
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm.auto import tqdm
import numpy as np

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)


class LoRADataset(Dataset):
    """LoRA 학습용 데이터셋"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: CLIPTokenizer,
        size: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        
        # 이미지-캡션 쌍 수집
        self.image_paths = []
        self.captions = []
        
        for img_path in sorted(self.data_dir.glob("*.png")):
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                self.image_paths.append(img_path)
                with open(caption_path, 'r', encoding='utf-8') as f:
                    self.captions.append(f.read().strip())
        
        # JPEG도 포함
        for img_path in sorted(self.data_dir.glob("*.jpg")):
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                self.image_paths.append(img_path)
                with open(caption_path, 'r', encoding='utf-8') as f:
                    self.captions.append(f.read().strip())
        
        print(f"Loaded {len(self.image_paths)} image-caption pairs from {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        # 이미지 로드 및 전처리
        image = Image.open(img_path).convert("RGB")
        
        # 리사이징
        image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # 랜덤 플립
        if self.random_flip and np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 텐서 변환 및 정규화 [-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) * 2.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # 캡션 토큰화
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
        }


def train_lora(
    pretrained_model: str,
    data_dir: str,
    output_dir: str,
    style_name: str,
    # Training params
    num_epochs: int = 20,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    lr_warmup_steps: int = 100,
    # LoRA params
    lora_rank: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    # Other params
    mixed_precision: str = "bf16",
    save_every_n_epochs: int = 5,
    seed: int = 42,
    resume_from: Optional[str] = None,
):
    """LoRA 학습 메인 함수"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Accelerator 설정
    project_config = ProjectConfiguration(
        project_dir=str(output_path),
        logging_dir=str(output_path / "logs"),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_config=project_config,
        log_with="tensorboard",
    )
    
    # 시드 설정
    if seed is not None:
        set_seed(seed)
    
    # 로깅 시작
    accelerator.init_trackers(
        project_name=f"lora_{style_name}",
        config={
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
        }
    )
    
    # 모델 로드
    print(f"\n{'='*60}")
    print(f"Loading Stable Diffusion model: {pretrained_model}")
    print(f"{'='*60}")
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        pretrained_model,
        subfolder="vae",
        torch_dtype=torch.float32,  # VAE는 fp32 유지
    )
    vae.requires_grad_(False)
    
    # Text Encoder
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    
    # Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model,
        subfolder="tokenizer",
    )
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model,
        subfolder="scheduler",
    )
    
    # LoRA 설정 적용
    print(f"\nApplying LoRA (rank={lora_rank}, alpha={lora_alpha})...")
    
    # UNet에 LoRA 적용
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
    )
    unet = get_peft_model(unet, unet_lora_config)
    
    # Text Encoder에도 LoRA 적용 (선택적)
    text_encoder_lora_config = LoraConfig(
        r=lora_rank // 2,  # Text encoder는 더 작은 rank
        lora_alpha=lora_alpha // 2,
        lora_dropout=lora_dropout,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
    
    # 학습 가능한 파라미터 출력
    unet_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    te_trainable = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    print(f"UNet trainable parameters: {unet_trainable:,}")
    print(f"Text Encoder trainable parameters: {te_trainable:,}")
    print(f"Total trainable parameters: {unet_trainable + te_trainable:,}")
    
    # 데이터셋 및 DataLoader
    print(f"\nLoading dataset from: {data_dir}")
    dataset = LoRADataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        size=512,
        random_flip=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(text_encoder.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Learning rate scheduler
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Accelerator 준비
    unet, text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, dataloader, lr_scheduler
    )
    
    # VAE를 GPU로
    vae.to(accelerator.device)
    
    # 학습 시작
    print(f"\n{'='*60}")
    print(f"Starting LoRA training for {style_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(dataloader)}")
    print(f"Total steps: {num_training_steps}")
    print(f"{'='*60}\n")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        unet.train()
        text_encoder.train()
        
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # Latent 인코딩
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=torch.float32)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=torch.float16)
                
                # 노이즈 추가
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Text embedding
                encoder_hidden_states = text_encoder(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )[0]
                
                # 노이즈 예측
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
                # Loss 계산
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(text_encoder.parameters()),
                        1.0
                    )
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 로깅
            epoch_loss += loss.item()
            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            accelerator.log({
                "train_loss": loss.item(),
                "learning_rate": current_lr,
            }, step=global_step)
            
            global_step += 1
        
        # Epoch 종료
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
        
        accelerator.log({"epoch_loss": avg_epoch_loss}, step=global_step)
        
        # 체크포인트 저장
        if (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_dir = output_path / f"checkpoint-epoch-{epoch + 1}"
            save_lora_weights(
                accelerator,
                unet,
                text_encoder,
                checkpoint_dir,
                style_name,
            )
            print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Best model 저장
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_dir = output_path / "best"
            save_lora_weights(
                accelerator,
                unet,
                text_encoder,
                best_dir,
                style_name,
            )
    
    # 최종 모델 저장
    final_dir = output_path / "final"
    save_lora_weights(
        accelerator,
        unet,
        text_encoder,
        final_dir,
        style_name,
    )
    
    accelerator.end_training()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model saved to: {final_dir}")
    print(f"{'='*60}")
    
    # 학습 정보 저장
    training_info = {
        "style_name": style_name,
        "pretrained_model": pretrained_model,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_loss": best_loss,
        "total_steps": global_step,
    }
    
    with open(output_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    return output_path


def save_lora_weights(accelerator, unet, text_encoder, output_dir, style_name):
    """LoRA 가중치 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # UNet LoRA 저장
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(output_dir / "unet_lora")
    
    # Text Encoder LoRA 저장
    unwrapped_te = accelerator.unwrap_model(text_encoder)
    unwrapped_te.save_pretrained(output_dir / "text_encoder_lora")
    
    print(f"Saved LoRA weights to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Training for Stable Diffusion")
    
    # Required arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--style_name", type=str, required=True, help="Style name (e.g., anime, watercolor)")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Other arguments
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_lora(
        pretrained_model=args.pretrained_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        style_name=args.style_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mixed_precision=args.mixed_precision,
        save_every_n_epochs=args.save_every_n_epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
