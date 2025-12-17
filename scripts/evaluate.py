"""
평가 파이프라인
- CLIP Score
- FID
- Style Accuracy
- Visual Comparison
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import time

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from transformers import CLIPProcessor, CLIPModel
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EvaluationResult:
    """평가 결과"""
    model_name: str
    clip_score_mean: float
    clip_score_std: float
    fid_score: Optional[float] = None
    style_accuracy: Optional[float] = None
    num_images: int = 0
    generation_time: float = 0.0


class CLIPEvaluator:
    """CLIP 기반 평가"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"CLIP loaded on {self.device}")
    
    def compute_clip_score(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> Tuple[float, List[float]]:
        """CLIP Score 계산"""
        
        scores = []
        
        for image, prompt in zip(images, prompts):
            inputs = self.processor(
                text=[prompt],
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Cosine similarity between image and text embeddings
                logits_per_image = outputs.logits_per_image
                score = logits_per_image.item() / 100.0  # 정규화
                scores.append(score)
        
        return np.mean(scores), scores
    
    def compute_style_similarity(
        self,
        generated_images: List[Image.Image],
        reference_images: List[Image.Image],
    ) -> float:
        """스타일 유사도 계산 (이미지-이미지)"""
        
        # 생성 이미지 임베딩
        gen_embeddings = []
        for img in generated_images:
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
                gen_embeddings.append(emb.cpu().numpy())
        gen_embeddings = np.vstack(gen_embeddings)
        
        # 레퍼런스 이미지 임베딩
        ref_embeddings = []
        for img in reference_images:
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
                ref_embeddings.append(emb.cpu().numpy())
        ref_embeddings = np.vstack(ref_embeddings)
        
        # 평균 임베딩 간 코사인 유사도
        gen_mean = gen_embeddings.mean(axis=0, keepdims=True)
        ref_mean = ref_embeddings.mean(axis=0, keepdims=True)
        
        similarity = np.dot(gen_mean, ref_mean.T) / (
            np.linalg.norm(gen_mean) * np.linalg.norm(ref_mean)
        )
        
        return float(similarity[0, 0])


class FIDCalculator:
    """FID (Fréchet Inception Distance) 계산"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Inception v3 로드
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()  # 마지막 FC 레이어 제거
        self.model.to(self.device)
        self.model.eval()
        
        # 전처리
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print("Inception v3 loaded for FID calculation")
    
    def get_features(self, images: List[Image.Image]) -> np.ndarray:
        """이미지에서 feature 추출"""
        features = []
        
        for img in tqdm(images, desc="Extracting features", leave=False):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.model(tensor)
                features.append(feat.cpu().numpy())
        
        return np.vstack(features)
    
    def calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """평균과 공분산 계산"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def compute_fid(
        self,
        real_images: List[Image.Image],
        generated_images: List[Image.Image],
    ) -> float:
        """FID 계산"""
        
        print("Computing FID...")
        
        # Feature 추출
        real_features = self.get_features(real_images)
        gen_features = self.get_features(generated_images)
        
        # 통계량 계산
        mu1, sigma1 = self.calculate_statistics(real_features)
        mu2, sigma2 = self.calculate_statistics(gen_features)
        
        # FID 계산
        diff = mu1 - mu2
        
        # 공분산 행렬의 제곱근
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)


class Evaluator:
    """통합 평가 클래스"""
    
    def __init__(self):
        self.clip_evaluator = CLIPEvaluator()
        self.fid_calculator = None  # 필요시 로드
    
    def evaluate_lora(
        self,
        generator,
        lora_name: str,
        lora_config: List[Tuple[str, float]],
        test_prompts: List[str],
        reference_images_dir: Optional[str] = None,
        num_images_per_prompt: int = 4,
        seed: int = 42,
    ) -> EvaluationResult:
        """LoRA 평가"""
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {lora_name}")
        print(f"LoRA config: {lora_config}")
        print(f"Test prompts: {len(test_prompts)}")
        print(f"{'='*60}")
        
        all_images = []
        all_prompts = []
        
        # 이미지 생성
        start_time = time.time()
        
        for prompt in tqdm(test_prompts, desc="Generating"):
            images = generator.generate(
                prompt=prompt,
                lora_configs=lora_config,
                num_images=num_images_per_prompt,
                seed=seed,
            )
            all_images.extend(images)
            all_prompts.extend([prompt] * len(images))
        
        generation_time = time.time() - start_time
        
        # CLIP Score
        clip_mean, clip_scores = self.clip_evaluator.compute_clip_score(
            all_images, all_prompts
        )
        clip_std = np.std(clip_scores)
        
        print(f"CLIP Score: {clip_mean:.4f} ± {clip_std:.4f}")
        
        # Style Accuracy (레퍼런스 이미지가 있는 경우)
        style_accuracy = None
        if reference_images_dir and Path(reference_images_dir).exists():
            ref_images = []
            for img_path in Path(reference_images_dir).glob("*.png"):
                ref_images.append(Image.open(img_path))
            for img_path in Path(reference_images_dir).glob("*.jpg"):
                ref_images.append(Image.open(img_path))
            
            if ref_images:
                style_accuracy = self.clip_evaluator.compute_style_similarity(
                    all_images[:20],  # 처음 20장만 사용
                    ref_images[:20],
                )
                print(f"Style Similarity: {style_accuracy:.4f}")
        
        # FID (선택적)
        fid_score = None
        
        return EvaluationResult(
            model_name=lora_name,
            clip_score_mean=clip_mean,
            clip_score_std=clip_std,
            fid_score=fid_score,
            style_accuracy=style_accuracy,
            num_images=len(all_images),
            generation_time=generation_time,
        )
    
    def compute_fid_score(
        self,
        real_images_dir: str,
        generated_images_dir: str,
    ) -> float:
        """FID Score 계산"""
        
        if self.fid_calculator is None:
            self.fid_calculator = FIDCalculator()
        
        # 이미지 로드
        real_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in Path(real_images_dir).glob(ext):
                real_images.append(Image.open(img_path))
        
        gen_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in Path(generated_images_dir).glob(ext):
                gen_images.append(Image.open(img_path))
        
        if len(real_images) < 10 or len(gen_images) < 10:
            print("Warning: Not enough images for FID calculation (need at least 10)")
            return -1
        
        return self.fid_calculator.compute_fid(real_images, gen_images)
    
    def run_ablation_study(
        self,
        generator,
        test_prompts: List[str],
        ablation_configs: Dict[str, List[Tuple[str, float]]],
        seed: int = 42,
    ) -> pd.DataFrame:
        """Ablation Study 실행"""
        
        results = []
        
        for config_name, lora_config in ablation_configs.items():
            result = self.evaluate_lora(
                generator=generator,
                lora_name=config_name,
                lora_config=lora_config,
                test_prompts=test_prompts,
                seed=seed,
            )
            results.append(asdict(result))
        
        df = pd.DataFrame(results)
        return df


def create_evaluation_report(
    results: List[EvaluationResult],
    output_dir: str,
):
    """평가 리포트 생성"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # DataFrame 생성
    df = pd.DataFrame([asdict(r) for r in results])
    
    # CSV 저장
    csv_path = output_path / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # CLIP Score 비교
    ax1 = axes[0]
    x = range(len(results))
    ax1.bar(x, df['clip_score_mean'], yerr=df['clip_score_std'], capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax1.set_ylabel('CLIP Score')
    ax1.set_title('CLIP Score Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    # Style Accuracy (있는 경우)
    ax2 = axes[1]
    if df['style_accuracy'].notna().any():
        ax2.bar(x, df['style_accuracy'].fillna(0))
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['model_name'], rotation=45, ha='right')
        ax2.set_ylabel('Style Similarity')
        ax2.set_title('Style Similarity to Reference')
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No reference images provided', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Style Similarity (N/A)')
    
    plt.tight_layout()
    
    fig_path = output_path / "evaluation_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to {fig_path}")
    
    # JSON 저장
    json_path = output_path / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    return df


def main():
    """평가 실행"""
    
    PROJECT_ROOT = Path("/home/claude/lora_project")
    
    # 테스트 프롬프트
    test_prompts = [
        "a portrait of a young woman in a garden",
        "a landscape with mountains and a river",
        "a cat sitting on a windowsill",
        "a futuristic city at night",
        "a cozy coffee shop interior",
        "a dragon flying over a castle",
        "a peaceful forest scene with sunlight",
        "a robot playing chess",
        "a beautiful sunset over the ocean",
        "a magical library with floating books",
    ]
    
    # Evaluator 초기화
    evaluator = Evaluator()
    
    # 생성기 로드 (inference.py에서 import)
    from inference import MultiLoRAGenerator
    
    generator = MultiLoRAGenerator(
        base_model="runwayml/stable-diffusion-v1-5",
    )
    
    # LoRA 로드
    lora_configs = {
        "anime": PROJECT_ROOT / "output" / "anime_lora" / "final",
        "watercolor": PROJECT_ROOT / "output" / "watercolor_lora" / "final",
        "cartoon": PROJECT_ROOT / "output" / "cartoon_lora" / "final",
    }
    
    for name, path in lora_configs.items():
        if path.exists():
            generator.load_lora(str(path), name)
    
    # 평가 설정
    evaluation_configs = {
        "base": [],  # LoRA 없이
        "anime_1.0": [("anime", 1.0)],
        "watercolor_1.0": [("watercolor", 1.0)],
        "cartoon_1.0": [("cartoon", 1.0)],
        "anime_0.7_watercolor_0.3": [("anime", 0.7), ("watercolor", 0.3)],
        "anime_0.5_watercolor_0.5": [("anime", 0.5), ("watercolor", 0.5)],
    }
    
    # 평가 실행
    results = []
    for config_name, lora_config in evaluation_configs.items():
        result = evaluator.evaluate_lora(
            generator=generator,
            lora_name=config_name,
            lora_config=lora_config if lora_config else None,
            test_prompts=test_prompts,
            num_images_per_prompt=2,
            seed=42,
        )
        results.append(result)
    
    # 리포트 생성
    df = create_evaluation_report(
        results,
        output_dir=str(PROJECT_ROOT / "evaluation"),
    )
    
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(df.to_string(index=False))
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
