"""
Multi-LoRA Inference Pipeline
여러 LoRA를 조합하여 이미지 생성
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)
from peft import PeftModel

import warnings
warnings.filterwarnings('ignore')


@dataclass
class LoRAConfig:
    """LoRA 설정"""
    name: str
    path: str
    trigger_word: str
    weight: float = 1.0


class MultiLoRAGenerator:
    """다중 LoRA 이미지 생성기"""
    
    # 기본 트리거 워드
    DEFAULT_TRIGGERS = {
        'anime': 'anistyle',
        'watercolor': 'wcstyle',
        'cartoon': 'cartoonstyle',
        'pixel': 'pixstyle',
    }
    
    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.base_model = base_model
        
        print(f"Loading base model: {base_model}")
        
        # 기본 파이프라인 로드
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        
        # DPM++ 스케줄러 (더 빠르고 품질 좋음)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        
        # xFormers 활성화 (메모리 효율)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled")
        except:
            print("xFormers not available, using default attention")
        
        # 로드된 LoRA 관리
        self.loaded_loras: Dict[str, LoRAConfig] = {}
        self.original_unet_state = None
        self.original_te_state = None
        
        print("Base model loaded successfully")
    
    def load_lora(
        self,
        lora_path: str,
        lora_name: str,
        trigger_word: Optional[str] = None,
        weight: float = 1.0,
    ):
        """LoRA 로드"""
        lora_path = Path(lora_path)
        
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")
        
        # 트리거 워드 설정
        if trigger_word is None:
            trigger_word = self.DEFAULT_TRIGGERS.get(lora_name, f"{lora_name}style")
        
        # UNet LoRA 로드
        unet_lora_path = lora_path / "unet_lora"
        if unet_lora_path.exists():
            self.pipe.unet = PeftModel.from_pretrained(
                self.pipe.unet,
                unet_lora_path,
                adapter_name=lora_name,
            )
            print(f"Loaded UNet LoRA from {unet_lora_path}")
        
        # Text Encoder LoRA 로드
        te_lora_path = lora_path / "text_encoder_lora"
        if te_lora_path.exists():
            self.pipe.text_encoder = PeftModel.from_pretrained(
                self.pipe.text_encoder,
                te_lora_path,
                adapter_name=lora_name,
            )
            print(f"Loaded Text Encoder LoRA from {te_lora_path}")
        
        # LoRA 정보 저장
        self.loaded_loras[lora_name] = LoRAConfig(
            name=lora_name,
            path=str(lora_path),
            trigger_word=trigger_word,
            weight=weight,
        )
        
        print(f"LoRA '{lora_name}' loaded (trigger: {trigger_word}, weight: {weight})")
    
    def load_lora_safetensors(
        self,
        safetensors_path: str,
        lora_name: str,
        trigger_word: Optional[str] = None,
        weight: float = 1.0,
    ):
        """Safetensors 형식의 LoRA 로드 (Kohya 형식)"""
        
        if trigger_word is None:
            trigger_word = self.DEFAULT_TRIGGERS.get(lora_name, f"{lora_name}style")
        
        self.pipe.load_lora_weights(
            safetensors_path,
            adapter_name=lora_name,
        )
        
        self.loaded_loras[lora_name] = LoRAConfig(
            name=lora_name,
            path=safetensors_path,
            trigger_word=trigger_word,
            weight=weight,
        )
        
        print(f"LoRA '{lora_name}' loaded from safetensors (trigger: {trigger_word})")
    
    def set_lora_weights(self, lora_weights: Dict[str, float]):
        """LoRA 가중치 설정"""
        adapter_names = list(lora_weights.keys())
        adapter_weights = [lora_weights[name] for name in adapter_names]
        
        # 활성화할 어댑터 설정
        if hasattr(self.pipe.unet, 'set_adapters'):
            self.pipe.unet.set_adapters(adapter_names, adapter_weights)
        
        if hasattr(self.pipe.text_encoder, 'set_adapters'):
            self.pipe.text_encoder.set_adapters(adapter_names, adapter_weights)
        
        print(f"Set adapters: {dict(zip(adapter_names, adapter_weights))}")
    
    def disable_all_loras(self):
        """모든 LoRA 비활성화"""
        if hasattr(self.pipe.unet, 'disable_adapters'):
            self.pipe.unet.disable_adapters()
        if hasattr(self.pipe.text_encoder, 'disable_adapters'):
            self.pipe.text_encoder.disable_adapters()
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        lora_configs: Optional[List[Tuple[str, float]]] = None,
        num_images: int = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """이미지 생성"""
        
        # LoRA 설정
        if lora_configs:
            lora_weights = {name: weight for name, weight in lora_configs}
            self.set_lora_weights(lora_weights)
            
            # 트리거 워드 추가
            trigger_words = []
            for name, _ in lora_configs:
                if name in self.loaded_loras:
                    trigger_words.append(self.loaded_loras[name].trigger_word)
            
            if trigger_words:
                prompt = f"{', '.join(trigger_words)}, {prompt}"
        
        # Generator 설정
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 이미지 생성
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images
        
        return images
    
    def generate_comparison(
        self,
        prompt: str,
        lora_configs_list: List[List[Tuple[str, float]]],
        include_base: bool = True,
        **kwargs,
    ) -> Dict[str, List[Image.Image]]:
        """여러 LoRA 설정 비교 생성"""
        
        results = {}
        
        # Base model (LoRA 없이)
        if include_base:
            self.disable_all_loras()
            results["base"] = self.generate(prompt, lora_configs=None, **kwargs)
        
        # 각 LoRA 설정으로 생성
        for i, lora_configs in enumerate(lora_configs_list):
            config_name = "_".join([f"{name}_{weight}" for name, weight in lora_configs])
            results[config_name] = self.generate(prompt, lora_configs=lora_configs, **kwargs)
        
        return results
    
    def batch_generate(
        self,
        prompts: List[str],
        output_dir: str,
        lora_configs: Optional[List[Tuple[str, float]]] = None,
        **kwargs,
    ):
        """배치 이미지 생성"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
            images = self.generate(prompt, lora_configs=lora_configs, **kwargs)
            
            for j, img in enumerate(images):
                filename = f"{i:04d}_{j:02d}.png"
                img.save(output_path / filename)
        
        print(f"Generated {len(prompts)} prompts to {output_path}")


def create_comparison_grid(
    images_dict: Dict[str, List[Image.Image]],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Image.Image:
    """비교 그리드 이미지 생성"""
    import matplotlib.pyplot as plt
    
    n_configs = len(images_dict)
    n_images = max(len(imgs) for imgs in images_dict.values())
    
    fig, axes = plt.subplots(n_configs, n_images, figsize=(4 * n_images, 4 * n_configs))
    
    if n_configs == 1:
        axes = [axes]
    if n_images == 1:
        axes = [[ax] for ax in axes]
    
    for row, (config_name, images) in enumerate(images_dict.items()):
        for col, img in enumerate(images):
            if col < n_images:
                axes[row][col].imshow(img)
                axes[row][col].axis('off')
                if col == 0:
                    axes[row][col].set_title(config_name, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison grid saved to {save_path}")
    
    # Figure를 Image로 변환
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return Image.fromarray(img_array)


def main():
    """예제 사용법"""
    
    # 생성기 초기화
    generator = MultiLoRAGenerator(
        base_model="runwayml/stable-diffusion-v1-5",
        device="cuda",
        torch_dtype=torch.float16,
    )
    
    # LoRA 로드 (실제 경로로 수정 필요)
    PROJECT_ROOT = Path("/home/claude/lora_project")
    
    lora_paths = {
        "anime": PROJECT_ROOT / "output" / "anime_lora" / "final",
        "watercolor": PROJECT_ROOT / "output" / "watercolor_lora" / "final",
        "cartoon": PROJECT_ROOT / "output" / "cartoon_lora" / "final",
    }
    
    # LoRA 로드
    for name, path in lora_paths.items():
        if path.exists():
            generator.load_lora(str(path), name)
        else:
            print(f"Warning: LoRA not found at {path}")
    
    # === 테스트 1: 단일 LoRA ===
    print("\n--- Single LoRA Test ---")
    test_prompt = "a portrait of a young woman in a garden, beautiful lighting"
    
    # Anime LoRA
    anime_images = generator.generate(
        prompt=test_prompt,
        lora_configs=[("anime", 1.0)],
        num_images=2,
        seed=42,
    )
    for i, img in enumerate(anime_images):
        img.save(PROJECT_ROOT / f"evaluation/anime_single_{i}.png")
    
    # === 테스트 2: Multi-LoRA ===
    print("\n--- Multi-LoRA Test ---")
    
    # Anime + Watercolor 조합
    combined_images = generator.generate(
        prompt=test_prompt,
        lora_configs=[("anime", 0.6), ("watercolor", 0.4)],
        num_images=2,
        seed=42,
    )
    for i, img in enumerate(combined_images):
        img.save(PROJECT_ROOT / f"evaluation/anime_watercolor_{i}.png")
    
    # === 테스트 3: 비교 그리드 ===
    print("\n--- Comparison Grid ---")
    
    comparison = generator.generate_comparison(
        prompt=test_prompt,
        lora_configs_list=[
            [("anime", 1.0)],
            [("watercolor", 1.0)],
            [("anime", 0.7), ("watercolor", 0.3)],
            [("anime", 0.5), ("watercolor", 0.5)],
        ],
        include_base=True,
        num_images=1,
        seed=42,
    )
    
    grid = create_comparison_grid(
        comparison,
        save_path=str(PROJECT_ROOT / "evaluation/comparison_grid.png"),
    )
    
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    main()
