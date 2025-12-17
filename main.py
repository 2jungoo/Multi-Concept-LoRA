"""
Multi-LoRA Fine-tuning Pipeline
메인 실행 스크립트

Usage:
    python main.py preprocess  # 데이터 전처리
    python main.py train       # 모든 LoRA 학습
    python main.py inference   # 이미지 생성
    python main.py evaluate    # 평가 실행
    python main.py all         # 전체 파이프라인 실행
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def print_header(title: str):
    """헤더 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_info(msg: str):
    """정보 메시지 출력"""
    print(f"[INFO] {msg}")


def print_success(msg: str):
    """성공 메시지 출력"""
    print(f"[✓] {msg}")


def print_error(msg: str):
    """에러 메시지 출력"""
    print(f"[✗] {msg}")


def check_environment():
    """환경 체크"""
    print_header("Environment Check")
    
    # Python 버전
    print_info(f"Python: {sys.version}")
    
    # PyTorch 버전 및 CUDA
    print_info(f"PyTorch: {torch.__version__}")
    print_info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print_info(f"CUDA version: {torch.version.cuda}")
        print_info(f"GPU: {torch.cuda.get_device_name(0)}")
        print_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print_error("CUDA not available! Training will be very slow.")
        return False
    
    # 디렉토리 확인
    required_dirs = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "configs",
    ]
    
    for dir_path in required_dirs:
        if dir_path.exists():
            print_success(f"Directory exists: {dir_path}")
        else:
            print_error(f"Directory missing: {dir_path}")
    
    return True


def run_preprocess(args):
    """데이터 전처리 실행"""
    print_header("Data Preprocessing")
    
    from preprocess_data import DataPreprocessor
    
    # 설정
    raw_data_paths = {
        'anime': args.anime_dir,
        'watercolor': args.watercolor_dir,
        'cartoon': args.cartoon_dir,
        'pixelart': getattr(args, 'pixelart_dir', None),
    }
    
    processed_paths = {
        'anime': PROJECT_ROOT / 'data' / 'anime',
        'watercolor': PROJECT_ROOT / 'data' / 'watercolor',
        'cartoon': PROJECT_ROOT / 'data' / 'cartoon',
        'pixelart': PROJECT_ROOT / 'data' / 'pixelart',
    }
    
    # 전처리기 초기화
    preprocessor = DataPreprocessor(
        output_size=512,
        min_quality_score=50,
    )
    
    # 각 스타일 처리
    for style, raw_path in raw_data_paths.items():
        if raw_path and Path(raw_path).exists():
            print_info(f"Processing {style}...")
            preprocessor.process_dataset(
                input_folder=raw_path,
                output_folder=str(processed_paths[style]),
                style_name=style,
                max_images=100,
            )
        else:
            print_error(f"Skipping {style}: path not found ({raw_path})")
    
    print_success("Preprocessing complete!")


def run_train(args):
    """LoRA 학습 실행"""
    print_header("LoRA Training")
    
    from train_lora import train_lora
    
    styles = ['anime', 'watercolor', 'cartoon', 'pixelart']
    
    for style in styles:
        data_dir = PROJECT_ROOT / 'data' / style
        output_dir = PROJECT_ROOT / 'output' / f'{style}_lora'
        
        if not data_dir.exists():
            print_error(f"Data not found for {style}: {data_dir}")
            continue
        
        print_info(f"Training {style} LoRA...")
        
        try:
            train_lora(
                pretrained_model="runwayml/stable-diffusion-v1-5",
                data_dir=str(data_dir),
                output_dir=str(output_dir),
                style_name=style,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.1,
                mixed_precision="bf16",
                save_every_n_epochs=5,
                seed=42,
            )
            print_success(f"{style} LoRA training complete!")
        except Exception as e:
            print_error(f"{style} training failed: {e}")
    
    print_success("All training complete!")


def run_inference(args):
    """추론 실행"""
    print_header("Inference")
    
    from inference import MultiLoRAGenerator, create_comparison_grid
    
    # 생성기 초기화
    generator = MultiLoRAGenerator(
        base_model="runwayml/stable-diffusion-v1-5",
    )
    
    # LoRA 로드 (4가지 스타일)
    lora_paths = {
        "anime": PROJECT_ROOT / "output" / "anime_lora" / "final",
        "watercolor": PROJECT_ROOT / "output" / "watercolor_lora" / "final",
        "cartoon": PROJECT_ROOT / "output" / "cartoon_lora" / "final",
        "pixelart": PROJECT_ROOT / "output" / "pixelart_lora" / "final",
    }
    
    for name, path in lora_paths.items():
        if path.exists():
            generator.load_lora(str(path), name)
            print_success(f"Loaded {name} LoRA")
        else:
            print_error(f"LoRA not found: {path}")
    
    # 테스트 프롬프트
    test_prompts = args.prompts if args.prompts else [
        "a portrait of a young woman in a garden, beautiful lighting",
        "a landscape with mountains at sunset",
        "a magical forest with glowing mushrooms",
    ]
    
    # 출력 디렉토리
    output_dir = PROJECT_ROOT / "evaluation" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 생성
    for i, prompt in enumerate(test_prompts):
        print_info(f"Generating: {prompt[:50]}...")
        
        # Base model
        base_imgs = generator.generate(prompt, num_images=1, seed=42)
        base_imgs[0].save(output_dir / f"prompt{i}_base.png")
        
        # Single LoRAs
        for name in lora_paths.keys():
            if name in generator.loaded_loras:
                imgs = generator.generate(
                    prompt, 
                    lora_configs=[(name, 1.0)], 
                    num_images=1, 
                    seed=42
                )
                imgs[0].save(output_dir / f"prompt{i}_{name}.png")
        
        # Multi-LoRA combinations
        multi_lora_configs = [
            ("watercolor_pixelart", [("watercolor", 0.6), ("pixelart", 0.4)]),
            ("cartoon_watercolor", [("cartoon", 0.6), ("watercolor", 0.4)]),
            ("watercolor_anime", [("watercolor", 0.6), ("anime", 0.4)]),
        ]
        
        for combo_name, config in multi_lora_configs:
            # 필요한 LoRA가 모두 로드되었는지 확인
            required_loras = [c[0] for c in config]
            if all(lora in generator.loaded_loras for lora in required_loras):
                combined = generator.generate(
                    prompt,
                    lora_configs=config,
                    num_images=1,
                    seed=42,
                )
                combined[0].save(output_dir / f"prompt{i}_{combo_name}.png")
    
    print_success(f"Generated images saved to {output_dir}")


def run_evaluate(args):
    """평가 실행 - 단일 LoRA + Multi-LoRA 조합 모두 평가"""
    print_header("Evaluation - Single & Multi-LoRA")
    
    from evaluate import Evaluator, create_evaluation_report, EvaluationResult
    from inference import MultiLoRAGenerator
    from PIL import Image
    import numpy as np
    
    # 생성기 및 평가자 초기화
    generator = MultiLoRAGenerator(base_model="runwayml/stable-diffusion-v1-5")
    evaluator = Evaluator()
    
    # LoRA 로드 (4가지 스타일 모두)
    lora_paths = {
        "anime": PROJECT_ROOT / "output" / "anime_lora" / "final",
        "watercolor": PROJECT_ROOT / "output" / "watercolor_lora" / "final",
        "cartoon": PROJECT_ROOT / "output" / "cartoon_lora" / "final",
        "pixelart": PROJECT_ROOT / "output" / "pixelart_lora" / "final",
    }
    
    loaded_loras = []
    for name, path in lora_paths.items():
        if path.exists():
            generator.load_lora(str(path), name)
            loaded_loras.append(name)
            print_success(f"Loaded {name} LoRA from {path}")
        else:
            print_error(f"LoRA not found: {path}")
    
    print_info(f"Loaded LoRAs: {loaded_loras}")
    
    # 레퍼런스 이미지 디렉토리 (스타일 유사도 계산용)
    reference_dirs = {
        "anime": PROJECT_ROOT / "data" / "anime_processed",
        "watercolor": PROJECT_ROOT / "data" / "watercolor",
        "cartoon": PROJECT_ROOT / "data" / "cartoon",
        "pixelart": PROJECT_ROOT / "data" / "pixelart_processed",
    }
    
    # 테스트 프롬프트 (기존 0,1,2번 + 추가)
    test_prompts = [
        "a portrait of a young woman in a garden",      # prompt 0
        "a landscape with mountains and sunset",        # prompt 1
        "a cat sitting by a window",                    # prompt 2
        "a magical forest scene with glowing lights",   # prompt 3
        "a futuristic cityscape at night",              # prompt 4
    ]
    
    # ============================================
    # 평가 설정: 단일 LoRA + Multi-LoRA 조합
    # ============================================
    evaluation_configs = {
        # Base model (LoRA 없음)
        "base": None,
        
        # 단일 LoRA (4가지)
        "anime": [("anime", 1.0)],
        "watercolor": [("watercolor", 1.0)],
        "cartoon": [("cartoon", 1.0)],
        "pixelart": [("pixelart", 1.0)],
        
        # Multi-LoRA 조합 (3가지) - 다양한 가중치 조합
        "watercolor_pixelart_0.6_0.4": [("watercolor", 0.6), ("pixelart", 0.4)],
        "watercolor_pixelart_0.5_0.5": [("watercolor", 0.5), ("pixelart", 0.5)],
        
        "cartoon_watercolor_0.6_0.4": [("cartoon", 0.6), ("watercolor", 0.4)],
        "cartoon_watercolor_0.5_0.5": [("cartoon", 0.5), ("watercolor", 0.5)],
        
        "watercolor_anime_0.6_0.4": [("watercolor", 0.6), ("anime", 0.4)],
        "watercolor_anime_0.5_0.5": [("watercolor", 0.5), ("anime", 0.5)],
    }
    
    # 출력 디렉토리 설정
    output_base = PROJECT_ROOT / "evaluation"
    all_comparison_dir = output_base / "all_comparison"
    multi_lora_dir = output_base / "multi_lora_comparison"
    
    all_comparison_dir.mkdir(parents=True, exist_ok=True)
    multi_lora_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # 이미지 생성 및 저장
    # ============================================
    print_header("Generating Images for All Configurations")
    
    generated_images = {}  # {config_name: {prompt_idx: [images]}}
    
    for config_name, lora_config in evaluation_configs.items():
        print_info(f"Generating images for: {config_name}")
        generated_images[config_name] = {}
        
        # 필요한 LoRA가 로드되었는지 확인
        if lora_config is not None:
            required_loras = [c[0] for c in lora_config]
            if not all(lora in loaded_loras for lora in required_loras):
                print_error(f"  Skipping {config_name}: required LoRAs not loaded")
                continue
        
        for prompt_idx, prompt in enumerate(test_prompts):
            try:
                images = generator.generate(
                    prompt=prompt,
                    lora_configs=lora_config,
                    num_images=2,  # 각 프롬프트당 2장
                    seed=42,
                )
                generated_images[config_name][prompt_idx] = images
                
                # 이미지 저장
                for img_idx, img in enumerate(images):
                    # all_comparison에 저장
                    save_path = all_comparison_dir / f"prompt{prompt_idx}_{config_name}_{img_idx}.png"
                    img.save(save_path)
                    
                    # Multi-LoRA인 경우 별도 폴더에도 저장
                    if lora_config is not None and len(lora_config) > 1:
                        multi_save_path = multi_lora_dir / f"prompt{prompt_idx}_{config_name}_{img_idx}.png"
                        img.save(multi_save_path)
                
            except Exception as e:
                print_error(f"  Error generating {config_name} for prompt {prompt_idx}: {e}")
    
    print_success(f"Images saved to {all_comparison_dir}")
    print_success(f"Multi-LoRA images saved to {multi_lora_dir}")
    
    # ============================================
    # 평가 실행
    # ============================================
    print_header("Running Evaluation")
    
    results = []
    
    for config_name, lora_config in evaluation_configs.items():
        if config_name not in generated_images or not generated_images[config_name]:
            continue
        
        print_info(f"Evaluating: {config_name}")
        
        # 해당 config의 모든 생성 이미지 수집
        all_images = []
        all_prompts = []
        
        for prompt_idx, images in generated_images[config_name].items():
            all_images.extend(images)
            all_prompts.extend([test_prompts[prompt_idx]] * len(images))
        
        if not all_images:
            continue
        
        # CLIP Score 계산
        clip_mean, clip_scores = evaluator.clip_evaluator.compute_clip_score(
            all_images, all_prompts
        )
        clip_std = np.std(clip_scores)
        
        # Style Similarity 계산 (단일 LoRA인 경우)
        style_accuracy = None
        if lora_config is not None and len(lora_config) == 1:
            style_name = lora_config[0][0]
            ref_dir = reference_dirs.get(style_name)
            
            if ref_dir and ref_dir.exists():
                ref_images = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in ref_dir.glob(ext):
                        try:
                            ref_images.append(Image.open(img_path).convert('RGB'))
                            if len(ref_images) >= 20:
                                break
                        except:
                            continue
                    if len(ref_images) >= 20:
                        break
                
                if ref_images:
                    style_accuracy = evaluator.clip_evaluator.compute_style_similarity(
                        all_images[:min(20, len(all_images))],
                        ref_images[:20],
                    )
        
        # Multi-LoRA인 경우 각 스타일에 대한 유사도 계산
        multi_style_scores = {}
        if lora_config is not None and len(lora_config) > 1:
            for lora_name, weight in lora_config:
                ref_dir = reference_dirs.get(lora_name)
                if ref_dir and ref_dir.exists():
                    ref_images = []
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        for img_path in ref_dir.glob(ext):
                            try:
                                ref_images.append(Image.open(img_path).convert('RGB'))
                                if len(ref_images) >= 20:
                                    break
                            except:
                                continue
                        if len(ref_images) >= 20:
                            break
                    
                    if ref_images:
                        score = evaluator.clip_evaluator.compute_style_similarity(
                            all_images[:min(20, len(all_images))],
                            ref_images[:20],
                        )
                        multi_style_scores[lora_name] = score
        
        result = EvaluationResult(
            model_name=config_name,
            clip_score_mean=clip_mean,
            clip_score_std=clip_std,
            fid_score=None,
            style_accuracy=style_accuracy,
            num_images=len(all_images),
            generation_time=0.0,
        )
        results.append(result)
        
        # 결과 출력
        print(f"  CLIP Score: {clip_mean:.4f} ± {clip_std:.4f}")
        if style_accuracy is not None:
            print(f"  Style Accuracy: {style_accuracy:.4f}")
        if multi_style_scores:
            for style, score in multi_style_scores.items():
                print(f"  {style} Similarity: {score:.4f}")
    
    # ============================================
    # 리포트 생성
    # ============================================
    print_header("Generating Reports")
    
    df = create_evaluation_report(
        results,
        output_dir=str(output_base),
    )
    
    # Multi-LoRA 전용 리포트
    multi_lora_results = [r for r in results if '_' in r.model_name and r.model_name != "base"]
    if multi_lora_results:
        create_evaluation_report(
            multi_lora_results,
            output_dir=str(multi_lora_dir),
        )
    
    # ============================================
    # 비교 그리드 이미지 생성
    # ============================================
    print_header("Creating Comparison Grids")
    
    create_comparison_grids(
        generated_images=generated_images,
        test_prompts=test_prompts,
        output_dir=output_base,
    )
    
    # ============================================
    # 최종 결과 출력
    # ============================================
    print("\n" + "=" * 80)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 분석 요약
    print("\n" + "-" * 80)
    print("  ANALYSIS")
    print("-" * 80)
    
    # 단일 LoRA 최고 점수
    single_lora_results = [r for r in results if r.model_name in ["anime", "watercolor", "cartoon", "pixelart"]]
    if single_lora_results:
        best_single = max(single_lora_results, key=lambda x: x.clip_score_mean)
        print(f"Best Single LoRA: {best_single.model_name} (CLIP: {best_single.clip_score_mean:.4f})")
    
    # Multi-LoRA 최고 점수
    multi_results = [r for r in results if '_' in r.model_name and '0.' in r.model_name]
    if multi_results:
        best_multi = max(multi_results, key=lambda x: x.clip_score_mean)
        print(f"Best Multi-LoRA: {best_multi.model_name} (CLIP: {best_multi.clip_score_mean:.4f})")
    
    print_success("Evaluation complete!")
    print_info(f"Results saved to: {output_base}")


def create_comparison_grids(generated_images, test_prompts, output_dir):
    """비교 그리드 이미지 생성"""
    import matplotlib.pyplot as plt
    from PIL import Image
    
    output_dir = Path(output_dir)
    grids_dir = output_dir / "comparison_grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    
    # 프롬프트별 비교 그리드
    for prompt_idx, prompt in enumerate(test_prompts):
        configs_with_images = []
        
        for config_name, prompt_images in generated_images.items():
            if prompt_idx in prompt_images and prompt_images[prompt_idx]:
                configs_with_images.append((config_name, prompt_images[prompt_idx][0]))
        
        if len(configs_with_images) < 2:
            continue
        
        # 그리드 생성
        n_cols = min(4, len(configs_with_images))
        n_rows = (len(configs_with_images) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (config_name, img) in enumerate(configs_with_images):
            if idx < len(axes):
                axes[idx].imshow(img)
                axes[idx].set_title(config_name, fontsize=10)
                axes[idx].axis('off')
        
        # 빈 subplot 숨기기
        for idx in range(len(configs_with_images), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f"Prompt {prompt_idx}: {prompt[:50]}...", fontsize=12)
        plt.tight_layout()
        
        grid_path = grids_dir / f"comparison_prompt{prompt_idx}.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_info(f"Saved comparison grid: {grid_path}")
    
    # Multi-LoRA 전용 비교 그리드
    multi_configs = [c for c in generated_images.keys() if '_' in c and '0.' in c]
    if multi_configs:
        for prompt_idx in range(len(test_prompts)):
            multi_images = []
            
            for config_name in multi_configs:
                if prompt_idx in generated_images.get(config_name, {}):
                    imgs = generated_images[config_name][prompt_idx]
                    if imgs:
                        multi_images.append((config_name, imgs[0]))
            
            if len(multi_images) < 2:
                continue
            
            n_cols = min(3, len(multi_images))
            n_rows = (len(multi_images) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for idx, (config_name, img) in enumerate(multi_images):
                if idx < len(axes):
                    axes[idx].imshow(img)
                    axes[idx].set_title(config_name.replace('_', '\n'), fontsize=9)
                    axes[idx].axis('off')
            
            for idx in range(len(multi_images), len(axes) if hasattr(axes, '__len__') else 1):
                if hasattr(axes, '__getitem__'):
                    axes[idx].axis('off')
            
            plt.suptitle(f"Multi-LoRA Comparison - Prompt {prompt_idx}", fontsize=12)
            plt.tight_layout()
            
            grid_path = grids_dir / f"multi_lora_prompt{prompt_idx}.png"
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close()
            print_info(f"Saved Multi-LoRA grid: {grid_path}")


def run_all(args):
    """전체 파이프라인 실행"""
    print_header("Full Pipeline Execution")
    
    start_time = time.time()
    
    # 1. 환경 체크
    if not check_environment():
        print_error("Environment check failed!")
        return
    
    # 2. 전처리
    if args.anime_dir or args.watercolor_dir or args.cartoon_dir:
        run_preprocess(args)
    else:
        print_info("Skipping preprocessing (no raw data paths provided)")
    
    # 3. 학습
    run_train(args)
    
    # 4. 추론
    run_inference(args)
    
    # 5. 평가
    run_evaluate(args)
    
    # 완료
    elapsed = time.time() - start_time
    print_header("Pipeline Complete!")
    print_info(f"Total time: {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-LoRA Fine-tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py preprocess --anime_dir ./raw/anime --watercolor_dir ./raw/watercolor
  python main.py train --epochs 20 --batch_size 2
  python main.py inference --prompts "a portrait" "a landscape"
  python main.py evaluate
  python main.py all --anime_dir ./raw/anime --epochs 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # preprocess
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument('--anime_dir', type=str, help='Raw anime images directory')
    preprocess_parser.add_argument('--watercolor_dir', type=str, help='Raw watercolor images directory')
    preprocess_parser.add_argument('--cartoon_dir', type=str, help='Raw cartoon images directory')
    preprocess_parser.add_argument('--pixelart_dir', type=str, help='Raw pixelart images directory')
    
    # train
    train_parser = subparsers.add_parser('train', help='Train LoRAs')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--lora_rank', type=int, default=32, help='LoRA rank')
    train_parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    
    # inference
    inference_parser = subparsers.add_parser('inference', help='Generate images')
    inference_parser.add_argument('--prompts', nargs='+', help='Test prompts')
    
    # evaluate
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    
    # all
    all_parser = subparsers.add_parser('all', help='Run full pipeline')
    all_parser.add_argument('--anime_dir', type=str, help='Raw anime images directory')
    all_parser.add_argument('--watercolor_dir', type=str, help='Raw watercolor images directory')
    all_parser.add_argument('--cartoon_dir', type=str, help='Raw cartoon images directory')
    all_parser.add_argument('--pixelart_dir', type=str, help='Raw pixelart images directory')
    all_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    all_parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    all_parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    all_parser.add_argument('--lora_rank', type=int, default=32, help='LoRA rank')
    all_parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    all_parser.add_argument('--prompts', nargs='+', help='Test prompts')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        run_preprocess(args)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'inference':
        run_inference(args)
    elif args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'all':
        run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
