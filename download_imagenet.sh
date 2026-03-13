#!/bin/bash
# ImageNet-1K 다운로드 스크립트 (Kaggle API 사용)

# 사용법:
# 1. Kaggle 계정 생성: https://www.kaggle.com
# 2. API 토큰 생성: https://www.kaggle.com/settings -> API -> Create New Token
#    -> kaggle.json 파일이 다운로드됨
# 3. kaggle.json을 서버로 업로드하고 ~/.kaggle/ 폴더에 배치
# 4. 이 스크립트 실행

set -e

echo "=== ImageNet-1K 다운로드 스크립트 ==="

# Step 1: Kaggle API 설치
echo "Step 1: Kaggle API 설치 중..."
pip install kaggle

# Step 2: Kaggle 설정 확인
echo "Step 2: Kaggle 설정 확인 중..."
mkdir -p ~/.kaggle
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: ~/.kaggle/kaggle.json 파일이 없습니다!"
    echo "다음 단계를 따라주세요:"
    echo "1. https://www.kaggle.com/settings 에서 'Create New API Token' 클릭"
    echo "2. 다운로드된 kaggle.json 파일을 ~/.kaggle/ 폴더로 복사"
    echo "3. chmod 600 ~/.kaggle/kaggle.json 실행"
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json

# Step 3: ImageNet 데이터셋 다운로드
echo "Step 3: ImageNet 데이터셋 다운로드 중..."
cd /workspace/etri_iitp/JS/EfficientViT/data
mkdir -p imagenet

# ImageNet Object Localization Challenge 데이터셋 다운로드
# 참고: 전체 크기 약 150GB
kaggle competitions download -c imagenet-object-localization-challenge

# Step 4: 압축 해제
echo "Step 4: 압축 해제 중..."
unzip -q imagenet-object-localization-challenge.zip -d imagenet/
cd imagenet
unzip -q ILSVRC/Data/CLS-LOC/train.tar -d train/
unzip -q ILSVRC/Data/CLS-LOC/val.tar -d val/

# Step 5: 폴더 구조 정리
echo "Step 5: ImageNet 폴더 구조 정리 중..."
# Training 데이터는 이미 클래스별 폴더로 구성되어 있음
# Validation 데이터는 재구성 필요

python3 << 'EOF'
import os
import shutil
from pathlib import Path

val_dir = Path('/workspace/etri_iitp/JS/EfficientViT/data/imagenet/val')
val_annotations = Path('/workspace/etri_iitp/JS/EfficientViT/data/imagenet/ILSVRC/Annotations/CLS-LOC/val')

# val 폴더 내에 클래스별 폴더 생성
if val_annotations.exists():
    print("Validation 데이터 재구성 중...")
    # TODO: val 이미지를 클래스별 폴더로 재구성
    # 이 부분은 ILSVRC devkit의 ground truth 파일이 필요합니다
else:
    print("WARNING: Validation annotations를 찾을 수 없습니다")
EOF

echo "=== 다운로드 완료! ==="
echo "데이터셋 위치: /workspace/etri_iitp/JS/EfficientViT/data/imagenet"
