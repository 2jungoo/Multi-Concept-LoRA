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
    }
    
    processed_paths = {
        'anime': PROJECT_ROOT / 'data' / 'anime',
        'watercolor': PROJECT_ROOT / 'data' / 'watercolor',
        'cartoon': PROJECT_ROOT / 'data' / 'cartoon',
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
    
    styles = ['anime', 'watercolor', 'cartoon']
    
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
    
    # LoRA 로드
    lora_paths = {
        "anime": PROJECT_ROOT / "output" / "anime_lora" / "final",
        "watercolor": PROJECT_ROOT / "output" / "watercolor_lora" / "final",
        "cartoon": PROJECT_ROOT / "output" / "cartoon_lora" / "final",
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
            imgs = generator.generate(
                prompt, 
                lora_configs=[(name, 1.0)], 
                num_images=1, 
                seed=42
            )
            imgs[0].save(output_dir / f"prompt{i}_{name}.png")
        
        # Multi-LoRA combinations
        if "anime" in generator.loaded_loras and "watercolor" in generator.loaded_loras:
            combined = generator.generate(
                prompt,
                lora_configs=[("anime", 0.6), ("watercolor", 0.4)],
                num_images=1,
                seed=42,
            )
            combined[0].save(output_dir / f"prompt{i}_anime_watercolor.png")
    
    print_success(f"Generated images saved to {output_dir}")


def run_evaluate(args):
    """평가 실행"""
    print_header("Evaluation")
    
    from evaluate import Evaluator, create_evaluation_report, EvaluationResult
    from inference import MultiLoRAGenerator
    
    # 생성기 및 평가자 초기화
    generator = MultiLoRAGenerator(base_model="runwayml/stable-diffusion-v1-5")
    evaluator = Evaluator()
    
    # LoRA 로드
    lora_paths = {
        "anime": PROJECT_ROOT / "output" / "anime_lora" / "final",
        "watercolor": PROJECT_ROOT / "output" / "watercolor_lora" / "final",
        "cartoon": PROJECT_ROOT / "output" / "cartoon_lora" / "final",
    }
    
    for name, path in lora_paths.items():
        if path.exists():
            generator.load_lora(str(path), name)
    
    # 테스트 프롬프트
    test_prompts = [
        "a portrait of a young woman",
        "a landscape with mountains",
        "a cat sitting by a window",
        "a magical forest scene",
        "a futuristic cityscape",
    ]
    
    # 평가 설정
    configs = {
        "base": None,
        "anime": [("anime", 1.0)],
        "watercolor": [("watercolor", 1.0)],
        "cartoon": [("cartoon", 1.0)],
        "anime+watercolor": [("anime", 0.6), ("watercolor", 0.4)],
    }
    
    # 평가 실행
    results = []
    for name, config in configs.items():
        result = evaluator.evaluate_lora(
            generator=generator,
            lora_name=name,
            lora_config=config,
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
    
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(df.to_string(index=False))
    
    print_success("Evaluation complete!")


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
