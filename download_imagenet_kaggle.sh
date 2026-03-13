#!/bin/bash
# ImageNet 다운로드 - Kaggle 자동화 스크립트

set -e

echo "==================================================="
echo "  ImageNet-1K 다운로드 (Kaggle)"
echo "==================================================="
echo ""

# Kaggle API 설치 확인
if ! command -v kaggle &> /dev/null; then
    echo "Step 1: Kaggle API 설치 중..."
    pip install -q kaggle
else
    echo "Step 1: Kaggle API 이미 설치됨"
fi

# kaggle.json 확인
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "ERROR: ~/.kaggle/kaggle.json 파일이 없습니다!"
    echo ""
    echo "다음 단계를 따라주세요:"
    echo "1. https://www.kaggle.com/settings 접속"
    echo "2. 'API' 섹션에서 'Create New Token' 클릭"
    echo "3. 다운로드된 kaggle.json 파일 내용 복사"
    echo "4. 서버에서 다음 명령어 실행:"
    echo ""
    echo "   mkdir -p ~/.kaggle"
    echo "   nano ~/.kaggle/kaggle.json"
    echo "   (내용 붙여넣기)"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# 권한 설정
chmod 600 ~/.kaggle/kaggle.json
echo "Step 2: Kaggle 인증 확인됨"

# Competition 참가 확인
echo ""
echo "Step 3: Competition 참가 확인 중..."
echo "https://www.kaggle.com/c/imagenet-object-localization-challenge"
echo "위 페이지에서 'Late Submission' 클릭했는지 확인하세요."
echo ""
read -p "Competition에 참가했습니까? (y/n): " participated

if [ "$participated" != "y" ]; then
    echo ""
    echo "먼저 Competition에 참가해주세요:"
    echo "https://www.kaggle.com/c/imagenet-object-localization-challenge"
    echo ""
    exit 1
fi

# 다운로드 디렉토리 생성
DOWNLOAD_DIR="/workspace/etri_iitp/JS/EfficientViT/data/imagenet"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo ""
echo "Step 4: ImageNet 다운로드 시작..."
echo "다운로드 경로: $DOWNLOAD_DIR"
echo "예상 크기: 약 155GB"
echo "예상 시간: 네트워크 속도에 따라 수 시간 소요"
echo ""

# 다운로드
kaggle competitions download -c imagenet-object-localization-challenge

echo ""
echo "Step 5: 압축 해제 중... (시간 소요)"
unzip -q imagenet-object-localization-challenge.zip

echo ""
echo "Step 6: Train 데이터 처리 중..."
cd ILSVRC/Data/CLS-LOC
mkdir -p train
cd train

# Train tar 압축 해제
if [ -f ../ILSVRC2012_img_train.tar ]; then
    tar -xf ../ILSVRC2012_img_train.tar

    # 각 클래스별 tar 압축 해제
    total_files=$(ls -1 *.tar 2>/dev/null | wc -l)
    current=0

    for f in *.tar; do
        current=$((current + 1))
        echo -ne "Processing class $current/$total_files: $f\r"

        class_name=$(basename "$f" .tar)
        mkdir -p "$class_name"
        tar -xf "$f" -C "$class_name"
        rm "$f"
    done
    echo ""
fi

echo ""
echo "Step 7: Validation 데이터 처리 중..."
cd ..
mkdir -p val
cd val

if [ -f ../ILSVRC2012_img_val.tar ]; then
    tar -xf ../ILSVRC2012_img_val.tar
fi

echo ""
echo "==================================================="
echo "  다운로드 완료!"
echo "==================================================="
echo ""
echo "데이터 경로: $DOWNLOAD_DIR/ILSVRC/Data/CLS-LOC"
echo ""
echo "Train 클래스 수: $(ls -d $DOWNLOAD_DIR/ILSVRC/Data/CLS-LOC/train/*/ 2>/dev/null | wc -l)"
echo "Val 이미지 수: $(ls -1 $DOWNLOAD_DIR/ILSVRC/Data/CLS-LOC/val/*.JPEG 2>/dev/null | wc -l)"
echo ""
echo "다음 단계: Validation 데이터를 클래스별 폴더로 재구성"
echo "  valprep.sh 스크립트 실행 또는 수동 재구성 필요"
echo ""
