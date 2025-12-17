# 🎨 Multi-Concept LoRA Fine-tuning for Text-to-Image Style Transfer
Stable Diffusion 1.5 기반 다중 스타일 이미지 생성 시스템
---
## 🎯 Project Goal

### 문제점
- **Problem 1**: SD 모델의 스타일 제어 한계 ("anime style" 프롬프트만으로 일관성 X)
- **Problem 2**: Full Fine-tuning 비효율 (4-7GB 모델 재학습)

### 해결 방안
- 4가지 독립 LoRA 학습 (각 3-10MB)
- **Multi-LoRA 조합**으로 새로운 스타일 창출
- 가중치 조절로 무한한 스타일 변형 가능

---

## 📊 Dataset

| 스타일 | 원본 해상도 | 이미지 수 | 출처 | 트리거 워드 |
|--------|------------|----------|------|------------|
| Anime | ~60×60 | 100장 | Kaggle | `anistyle` |
| Watercolor | 416×416 | 100장 | Roboflow | `wcstyle` |
| Cartoon | ~2000×2000 | 60장 | Kaggle | `ctstyle` |
| Pixelart | 16×16 | 100장 | Kaggle | `pixstyle` |

### 전처리 파이프라인
1. **Quality Filter** → Laplacian variance > 100 (블러 이미지 제거)
2. **Resize & Center Crop** → 512×512
3. **BLIP Auto Captioning** → 자동 캡션 생성
4. **Trigger Word 삽입** → `"anistyle, a portrait of..."`

---

## 🛠️ Training

### 학습 설정

| Parameter | Value |
|-----------|-------|
| Base Model | Stable Diffusion v1.5 |
| LoRA Rank | 32 |
| LoRA Alpha | 16 |
| Learning Rate | 1e-4 |
| Epochs | 20 |
| Batch Size | 2 |
| Optimizer | AdamW |
| Mixed Precision | bf16 |
| Scheduler | cosine |

### Target Modules
- **UNet Attention**: `to_q`, `to_k`, `to_v`, `to_out`
- **Text Encoder**: `q_proj`, `k_proj`, `v_proj`, `out_proj`

### 학습 결과
```
output/
├── anime_lora/final/      (~8MB)
├── watercolor_lora/final/ (~8MB)
├── cartoon_lora/final/    (~8MB)
└── pixelart_lora/final/   (~8MB)
```

---

## 🔀 Inference: Multi-LoRA 조합

### Multi-LoRA 수식
```
h = Wx + α₁·B₁A₁x + α₂·B₂A₂x
         └─LoRA 1─┘   └─LoRA 2─┘

αᵢ: 각 어댑터 가중치
```

### 실험한 조합 (6가지)

| Multi-LoRA 조합 | α₁ | α₂ |
|----------------|----|----|
| watercolor + pixelart | 0.6 | 0.4 |
| watercolor + pixelart | 0.5 | 0.5 |
| cartoon + watercolor | 0.6 | 0.4 |
| cartoon + watercolor | 0.5 | 0.5 |
| watercolor + anime | 0.6 | 0.4 |
| watercolor + anime | 0.5 | 0.5 |

---

## 🧪 Experiments

### 생성 설정 (총 11가지 모델)

| 구분 | 모델 수 | 설명 |
|------|--------|------|
| Base | 1 | SD v1.5 원본 (기준선) |
| Single LoRA | 4 | anime, watercolor, cartoon, pixelart |
| Multi-LoRA | 6 | 3가지 조합 × 2가지 가중치 |

### 테스트 프롬프트 (5개)
0. "a portrait of a young woman in a garden"
1. "a landscape with mountains and sunset"
2. "a cat sitting by a window"
3. "a magical forest scene with glowing lights"
4. "a futuristic cityscape at night"

### 생성 이미지 수
**11개 모델 × 5개 프롬프트 × 2장씩 = 110장**

---

## 📈 Evaluation Results

### CLIP Score & Style Similarity

| Model | CLIP Score | Style Similarity |
|-------|------------|------------------|
| base | 0.3097 | - |
| anime | **0.3200** | 0.6743 |
| watercolor | 0.3187 | 0.7677 |
| cartoon | 0.3127 | 0.7861 |
| pixelart | 0.3078 | **0.7984** |
| watercolor_pixelart_0.6_0.4 | 0.3196 | 0.7614 |
| watercolor_pixelart_0.5_0.5 | 0.3196 | 0.7614 |
| cartoon_watercolor_0.6_0.4 | 0.3130 | 0.7858 |
| cartoon_watercolor_0.5_0.5 | 0.3130 | 0.7858 |
| watercolor_anime_0.6_0.4 | 0.3096 | 0.7227 |
| watercolor_anime_0.5_0.5 | 0.3066 | 0.7321 |

### 주요 발견
- **Best CLIP Score**: anime (0.3200)
- **Best Style Similarity**: pixelart (0.7984) - 픽셀아트의 명확한 시각적 특성 때문
- **Best Multi-LoRA**: watercolor+pixelart (0.3196)
- Multi-LoRA 조합들: 0.72~0.79 수준을 유지하여 두 스타일이 성공적으로 혼합

---

## 🚀 Quick Start

### 1. 환경 설정
```bash
conda create -n lora_env python=3.10 -y
conda activate lora_env

# PyTorch (CUDA 12.1)
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

---

## 📁 Project Structure
```
├── main.py                 # CLI 메인 스크립트
├── requirements.txt        # 의존성 패키지
├── scripts/
│   ├── preprocess_data.py  # 데이터 전처리 + BLIP 캡셔닝
│   ├── train_lora.py       # LoRA 학습
│   ├── inference.py        # Multi-LoRA 추론
│   └── evaluate.py         # CLIP/Style 평가
├── configs/
│   └── lora_base_config.toml
├── data/                   # 전처리된 데이터셋
├── output/                 # 학습된 LoRA 가중치
└── evaluation/             # 평가 결과 및 생성 이미지
```

---

## ⚠️ Problems & Solutions

| 문제 | 증상 | 해결 |
|------|------|------|
| 가상환경 충돌 | PyTorch 버전 불일치 | 가상환경 재생성, 호환 버전 통일 |
| LoRA 로딩 오류 | adapter_config.json 경로 인식 실패 | unet_lora/, text_encoder_lora/ 폴더 구조 맞춤 |
| 스타일 미적용 | LoRA 적용해도 변화 없음 | 트리거 워드 캡션 앞에 삽입, weight 1.0 설정 |
| Multi-LoRA 충돌 | 두 LoRA 동시 적용 시 이상한 결과 | 가중치 합 1.0 이하로 조절 (0.6 + 0.4) |
| NaN Loss | 학습 중 loss가 nan | mixed_precision="no" 사용 |

---

## 🎯 Conclusion

### 주요 성과
- ✅ 4가지 스타일 LoRA 학습 성공 (각 20 epochs)
- ✅ Multi-LoRA 조합으로 새로운 스타일 생성 검증
- ✅ 정량적 평가로 효과 입증 (CLIP Score, Style Accuracy)
- ✅ Multi-LoRA가 두 스타일을 균형있게 혼합 (유사도 0.7 이상)

### 활용 방안
1. **아트 디렉션 도구** - 게임/애니메이션 프로토타이핑
2. **개인화 이미지 생성** - 커스텀 스타일 조합
3. **콘텐츠 크리에이터** - 일관된 브랜드 스타일 유지

### 향후 연구
- 더 많은 스타일 조합 실험
- 가중치 최적화 자동화
- 3개 이상 LoRA 동시 조합

---

## 📚 References

- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Diffusers Library](https://github.com/huggingface/diffusers)
