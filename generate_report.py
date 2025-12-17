"""
평가 결과 리포트 재생성 스크립트
기존 평가 결과를 바탕으로 CSV, JSON, 차트를 다시 생성합니다.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def create_evaluation_report(results: List[EvaluationResult], output_dir: str):
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # CLIP Score 비교
    ax1 = axes[0]
    x = range(len(results))
    bars1 = ax1.bar(x, df['clip_score_mean'], yerr=df['clip_score_std'], capsize=5, color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['model_name'], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('CLIP Score')
    ax1.set_title('CLIP Score Comparison')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 0.40)  # Y축 범위 고정
    
    # Style Accuracy (있는 경우)
    ax2 = axes[1]
    style_values = df['style_accuracy'].fillna(0).values
    colors = ['steelblue' if v > 0 else 'lightgray' for v in style_values]
    bars2 = ax2.bar(x, style_values, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['model_name'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Style Similarity')
    ax2.set_title('Style Similarity to Reference')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1.0)  # Y축 범위 고정
    
    plt.tight_layout()
    
    fig_path = output_path / "evaluation_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to {fig_path}")
    
    # JSON 저장
    json_path = output_path / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"JSON saved to {json_path}")
    
    return df


def main():
    # 프로젝트 루트 설정 - 필요시 수정하세요
    PROJECT_ROOT = Path("/home/work/INC_share/JG/CV")
    
    # 출력 디렉토리
    output_dir = PROJECT_ROOT / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # 평가 결과 데이터 (실제 측정 결과)
    # ============================================
    results = [
        EvaluationResult(
            model_name="base",
            clip_score_mean=0.309720,
            clip_score_std=0.031094,
            style_accuracy=None,
            num_images=10,
        ),
        EvaluationResult(
            model_name="anime",
            clip_score_mean=0.320001,
            clip_score_std=0.022725,
            style_accuracy=0.674256,
            num_images=10,
        ),
        EvaluationResult(
            model_name="watercolor",
            clip_score_mean=0.318658,
            clip_score_std=0.028335,
            style_accuracy=0.767652,
            num_images=10,
        ),
        EvaluationResult(
            model_name="cartoon",
            clip_score_mean=0.312666,
            clip_score_std=0.016790,
            style_accuracy=0.786117,
            num_images=10,
        ),
        EvaluationResult(
            model_name="pixelart",
            clip_score_mean=0.307846,
            clip_score_std=0.027875,
            style_accuracy=0.798448,
            num_images=10,
        ),
        EvaluationResult(
            model_name="watercolor_pixelart_0.6_0.4",
            clip_score_mean=0.319550,
            clip_score_std=0.021180,
            style_accuracy=0.761400,  # (0.7323 + 0.7905) / 2 평균
            num_images=10,
        ),
        EvaluationResult(
            model_name="watercolor_pixelart_0.5_0.5",
            clip_score_mean=0.319550,
            clip_score_std=0.021180,
            style_accuracy=0.761400,
            num_images=10,
        ),
        EvaluationResult(
            model_name="cartoon_watercolor_0.6_0.4",
            clip_score_mean=0.312960,
            clip_score_std=0.016557,
            style_accuracy=0.785800,  # (0.7834 + 0.7882) / 2 평균
            num_images=10,
        ),
        EvaluationResult(
            model_name="cartoon_watercolor_0.5_0.5",
            clip_score_mean=0.312960,
            clip_score_std=0.016557,
            style_accuracy=0.785800,
            num_images=10,
        ),
        EvaluationResult(
            model_name="watercolor_anime_0.6_0.4",
            clip_score_mean=0.309614,
            clip_score_std=0.027164,
            style_accuracy=0.722650,  # (0.7786 + 0.6667) / 2 평균
            num_images=10,
        ),
        EvaluationResult(
            model_name="watercolor_anime_0.5_0.5",
            clip_score_mean=0.306641,
            clip_score_std=0.026344,
            style_accuracy=0.732050,  # (0.7930 + 0.6711) / 2 평균
            num_images=10,
        ),
    ]
    
    print("=" * 60)
    print("  Generating Reports")
    print("=" * 60)
    
    # 전체 리포트 생성
    df = create_evaluation_report(results, str(output_dir))
    
    # Multi-LoRA 전용 리포트
    multi_lora_results = [r for r in results if '_' in r.model_name and r.model_name != "base"]
    if multi_lora_results:
        multi_lora_dir = output_dir / "multi_lora_comparison"
        multi_lora_dir.mkdir(parents=True, exist_ok=True)
        create_evaluation_report(multi_lora_results, str(multi_lora_dir))
    
    # 결과 출력
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
        print(f"Best Single LoRA (CLIP): {best_single.model_name} ({best_single.clip_score_mean:.4f})")
        
        best_style = max(single_lora_results, key=lambda x: x.style_accuracy or 0)
        print(f"Best Single LoRA (Style): {best_style.model_name} ({best_style.style_accuracy:.4f})")
    
    # Multi-LoRA 최고 점수
    multi_results = [r for r in results if '_' in r.model_name and '0.' in r.model_name]
    if multi_results:
        best_multi = max(multi_results, key=lambda x: x.clip_score_mean)
        print(f"Best Multi-LoRA (CLIP): {best_multi.model_name} ({best_multi.clip_score_mean:.4f})")
    
    print("\n✅ Reports generated successfully!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()