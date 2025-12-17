# 🎨 Multi-LoRA Style Transfer for Stable Diffusion

Stable Diffusion 1.5에 여러 개의 독립적인 LoRA를 학습하여 텍스트 프롬프트로 다양한 스타일을 제어할 수 있는 이미지 생성 시스템

## 🎯 프로젝트 목표

- 4가지 스타일 LoRA 학습 (Watercolor, Cartoon, Anime, Pixelart)
- Multi-LoRA 조합을 통한 새로운 스타일 생성
- 데이터셋 해상도에 따른 학습 품질 분석 (Ablation Study)

## 📊 실험 결과

### 스타일별 생성 결과

| Base Model | Watercolor | Cartoon | Anime | Pixelart |
|:----------:|:----------:|:-------:|:-----:|:--------:|
| ![base](results/base.png) | ![wc](results/watercolor.png) | ![ct](results/cartoon.png) | ![an](results/anime.png) | ![px](results/pixelart.png) |

### 데이터셋 해상도 vs 학습 품질 (Ablation Study)

| Dataset | 원본 해상도 | 이미지 수 | Best Loss | 결과 품질 |
|---------|------------|----------|-----------|----------|
| Cartoon | ~2000×2000 | 61 | 0.1194 | ✅ 우수 |
| Watercolor | 416×416 | 100 | 0.0871 | ✅ 양호 |
| Anime | 61~94×61~94 | 100 | 0.0871 | ⚠️ 보통 |
| **Pixelart** | **16×16** | 100 | 0.0221 | ❌ 실패 |

> **결론**: LoRA 학습에는 최소 256×256 이상의 해상도 권장. 극저해상도(16×16) 이미지는 업스케일해도 정보 손실로 의미 있는 스타일 학습 불가

### CLIP Score 비교

| Model | CLIP Score | 비고 |
|-------|------------|------|
| Base (SD 1.5) | 0.310 | 기준 |
| Watercolor | 0.299 | 스타일 적용됨 |
| Cartoon | 0.286 | 스타일 적용됨 |
| Multi-LoRA | 0.292 | 조합 효과 |

## 🛠️ 기술 스택

- **Base Model**: Stable Diffusion v1.5
- **Fine-tuning**: LoRA (Low-Rank Adaptation) with PEFT
- **Framework**: PyTorch 2.2.0, Diffusers 0.27.0
- **Hardware**: NVIDIA H100 80GB
- **Evaluation**: CLIP Score, FID

## 📁 프로젝트 구조
```
├── main.py                 # CLI 메인 스크립트
├── requirements.txt        # 의존성 패키지
├── scripts/
│   ├── preprocess_data.py  # 데이터 전처리 + BLIP 캡셔닝
│   ├── train_lora.py       # LoRA 학습
│   ├── inference.py        # Multi-LoRA 추론
│   └── evaluate.py         # CLIP/FID 평가
├── configs/
│   └── lora_base_config.toml
├── data/                   # 전처리된 데이터셋 (Git 제외)
├── output/                 # 학습된 LoRA 가중치 (Git 제외)
└── evaluation/             # 평가 결과
```

## 🚀 사용법

### 1. 환경 설정
```bash
# Conda 환경 생성
conda create -n lora_env python=3.10 -y
conda activate lora_env

# PyTorch 설치 (CUDA 12.1)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 전처리
```bash
python main.py preprocess \
    --watercolor_dir /path/to/watercolor \
    --cartoon_dir /path/to/cartoon
```

### 3. LoRA 학습
```bash
python main.py train --epochs 20 --batch_size 1 --learning_rate 1e-4
```

### 4. 이미지 생성
```bash
python main.py inference
```

### 5. 평가
```bash
python main.py evaluate
```

## 💡 주요 발견사항

### 1. 데이터 해상도의 중요성
- 고해상도(400px+): 스타일 특징을 잘 학습
- 저해상도(100px-): 업스케일해도 정보 손실
- 극저해상도(16px): 학습 자체가 무의미

### 2. Multi-LoRA 조합
- 서로 다른 스타일의 LoRA를 가중치 조합하여 새로운 스타일 생성 가능
- 예: `watercolor(0.6) + cartoon(0.4)` = 수채화 느낌의 만화풍

### 3. 학습 안정성
- `mixed_precision="no"` (fp32) 사용 시 NaN loss 방지
- Learning rate 1e-4가 안정적

## 📚 참고 자료

- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [PEFT LoRA Documentation](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

**AI응용학과 2191192 이준구** | 시각지능학습[A] 기말 프로젝트
