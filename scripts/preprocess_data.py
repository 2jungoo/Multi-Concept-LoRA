"""
데이터 전처리 파이프라인
- 이미지 품질 검사
- 리사이징 및 크롭
- BLIP 자동 캡셔닝
- 트리거 워드 삽입
"""

import os
import shutil
import json
from pathlib import Path
from PIL import Image, ImageFilter
import cv2
import numpy as np
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, output_size=512, min_quality_score=100):
        self.output_size = output_size
        self.min_quality_score = min_quality_score
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # BLIP 모델 로드
        print("Loading BLIP model for auto-captioning...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print(f"BLIP model loaded on {self.device}")
        
        # 트리거 워드 설정
        self.trigger_words = {
            'anime': 'anistyle',
            'watercolor': 'wcstyle', 
            'cartoon': 'cartoonstyle',
            'pixel': 'pixstyle',
            'face': 'sksperson'
        }
    
    def calculate_sharpness(self, image_path: str) -> float:
        """Laplacian variance로 이미지 선명도 계산"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        return laplacian_var
    
    def check_image_quality(self, image_path: str) -> dict:
        """이미지 품질 검사"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # 선명도 계산
            sharpness = self.calculate_sharpness(image_path)
            
            # 해상도 체크 (60x60은 너무 작음)
            min_dimension = min(width, height)
            is_too_small = min_dimension < 256  # 256 미만은 경고
            is_very_small = min_dimension < 128  # 128 미만은 사용 불가
            
            return {
                'path': image_path,
                'width': width,
                'height': height,
                'sharpness': sharpness,
                'is_sharp_enough': sharpness >= self.min_quality_score,
                'is_too_small': is_too_small,
                'is_very_small': is_very_small,
                'is_valid': not is_very_small,
                'mode': img.mode
            }
        except Exception as e:
            return {
                'path': image_path,
                'error': str(e),
                'is_valid': False
            }
    
    def resize_and_crop(self, img: Image.Image, target_size: int = 512) -> Image.Image:
        """중앙 크롭 후 리사이징"""
        # RGB로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        
        # 정사각형 크롭
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
        
        # 리사이징 (작은 이미지는 LANCZOS 업스케일)
        img = img.resize((target_size, target_size), Image.LANCZOS)
        
        return img
    
    def generate_caption(self, image: Image.Image, style_hint: str = "") -> str:
        """BLIP을 사용한 자동 캡션 생성"""
        # 조건부 캡션 생성 (스타일 힌트 제공)
        if style_hint:
            prompt = f"a {style_hint}"
        else:
            prompt = "a"
        
        inputs = self.blip_processor(image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(
                **inputs,
                max_length=75,
                num_beams=3,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def process_dataset(
        self,
        input_folder: str,
        output_folder: str,
        style_name: str,
        max_images: int = 100,
        use_manual_captions: bool = False,
        manual_caption_template: str = None
    ):
        """데이터셋 전처리 메인 함수"""
        
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        trigger_word = self.trigger_words.get(style_name, style_name)
        
        # 지원하는 이미지 확장자
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        
        # 이미지 파일 수집
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in valid_extensions
        ]
        
        print(f"\n{'='*60}")
        print(f"Processing {style_name} dataset")
        print(f"Found {len(image_files)} images in {input_folder}")
        print(f"Trigger word: {trigger_word}")
        print(f"{'='*60}")
        
        # 품질 검사
        print("\n[1/3] Checking image quality...")
        quality_results = []
        for img_file in tqdm(image_files, desc="Quality check"):
            result = self.check_image_quality(str(img_file))
            quality_results.append(result)
        
        # 품질 통계
        valid_images = [r for r in quality_results if r.get('is_valid', False)]
        small_images = [r for r in quality_results if r.get('is_too_small', False)]
        very_small_images = [r for r in quality_results if r.get('is_very_small', False)]
        
        print(f"\n📊 Quality Report:")
        print(f"   - Total images: {len(image_files)}")
        print(f"   - Valid for training: {len(valid_images)}")
        print(f"   - Small (< 256px, will upscale): {len(small_images)}")
        print(f"   - Too small (< 128px, skipped): {len(very_small_images)}")
        
        if very_small_images:
            print(f"\n⚠️  WARNING: {len(very_small_images)} images are too small!")
            print("   Consider replacing these with higher resolution images.")
            for r in very_small_images[:5]:  # 처음 5개만 표시
                print(f"   - {Path(r['path']).name}: {r.get('width', 'N/A')}x{r.get('height', 'N/A')}")
        
        # 처리할 이미지 선택
        images_to_process = [r for r in valid_images][:max_images]
        
        # 처리 및 캡셔닝
        print(f"\n[2/3] Processing and captioning {len(images_to_process)} images...")
        processed_count = 0
        caption_data = []
        
        for idx, quality_info in enumerate(tqdm(images_to_process, desc="Processing")):
            try:
                img_path = Path(quality_info['path'])
                img = Image.open(img_path)
                
                # 리사이징
                processed_img = self.resize_and_crop(img, self.output_size)
                
                # 새 파일명
                new_filename = f"{style_name}_{idx:04d}.png"
                output_img_path = output_path / new_filename
                processed_img.save(output_img_path, 'PNG', quality=95)
                
                # 캡션 생성
                if use_manual_captions and manual_caption_template:
                    caption = manual_caption_template
                else:
                    style_hints = {
                        'anime': 'anime style illustration',
                        'watercolor': 'watercolor painting',
                        'cartoon': 'cartoon style drawing',
                        'pixel': 'pixel art'
                    }
                    base_caption = self.generate_caption(
                        processed_img, 
                        style_hints.get(style_name, '')
                    )
                    caption = f"{trigger_word}, {base_caption}"
                
                # 캡션 파일 저장
                caption_file_path = output_path / f"{style_name}_{idx:04d}.txt"
                with open(caption_file_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                
                caption_data.append({
                    'image': new_filename,
                    'caption': caption,
                    'original_path': str(img_path),
                    'original_size': f"{quality_info['width']}x{quality_info['height']}"
                })
                
                processed_count += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {img_path}: {e}")
                continue
        
        # 메타데이터 저장
        metadata = {
            'style': style_name,
            'trigger_word': trigger_word,
            'total_processed': processed_count,
            'output_resolution': self.output_size,
            'captions': caption_data
        }
        
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Dataset processing complete!")
        print(f"   - Processed images: {processed_count}")
        print(f"   - Output folder: {output_path}")
        print(f"   - Metadata saved: {metadata_path}")
        
        return metadata


def main():
    """메인 실행 함수"""
    
    # 프로젝트 경로 설정 (실제 환경에 맞게 수정 필요)
    PROJECT_ROOT = Path("/home/claude/lora_project")
    
    # 원본 데이터셋 경로 (실제 경로로 수정 필요)
    RAW_DATA_PATHS = {
        'anime': '/path/to/your/anime/images',       # ⚠️ 수정 필요
        'watercolor': '/path/to/your/watercolor/images',  # ⚠️ 수정 필요
        'cartoon': '/path/to/your/cartoon/images'    # ⚠️ 수정 필요
    }
    
    # 처리된 데이터셋 저장 경로
    PROCESSED_DATA_PATHS = {
        'anime': PROJECT_ROOT / 'data' / 'anime',
        'watercolor': PROJECT_ROOT / 'data' / 'watercolor',
        'cartoon': PROJECT_ROOT / 'data' / 'cartoon'
    }
    
    # 전처리기 초기화
    preprocessor = DataPreprocessor(output_size=512, min_quality_score=50)
    
    # 각 스타일별 처리
    for style_name, raw_path in RAW_DATA_PATHS.items():
        if not Path(raw_path).exists():
            print(f"\n⚠️  Skipping {style_name}: Path does not exist: {raw_path}")
            continue
        
        preprocessor.process_dataset(
            input_folder=raw_path,
            output_folder=str(PROCESSED_DATA_PATHS[style_name]),
            style_name=style_name,
            max_images=100  # 각 스타일당 최대 100장
        )
    
    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)


if __name__ == "__main__":
    main()
