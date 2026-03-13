#!/bin/bash
# ImageNet-1K 다운로드 (Hugging Face - 가장 간단한 방법)

set -e

echo "=== ImageNet-1K 다운로드 (Hugging Face) ==="
echo ""

# Step 1: 필요한 패키지 설치
echo "Step 1: 필요한 패키지 설치 중..."
pip install -q datasets huggingface_hub pillow tqdm

# Step 2: Hugging Face 로그인
echo ""
echo "Step 2: Hugging Face 로그인"
echo "------------------------------------------------"
echo "1. https://huggingface.co 에서 계정 생성"
echo "2. https://huggingface.co/settings/tokens 에서 Access Token 생성"
echo "3. 아래 명령어 실행 후 토큰 입력:"
echo ""
echo "   huggingface-cli login"
echo ""
echo "4. https://huggingface.co/datasets/imagenet-1k 접속"
echo "   'Access repository' 버튼 클릭하여 접근 권한 요청"
echo "------------------------------------------------"
echo ""

read -p "위 단계를 완료했습니까? (y/n): " response
if [ "$response" != "y" ]; then
    echo "먼저 위 단계를 완료한 후 다시 실행해주세요."
    exit 1
fi

# Step 3: Python 스크립트로 다운로드
echo ""
echo "Step 3: ImageNet-1K 다운로드 시작..."
python3 download_imagenet_hf.py

echo ""
echo "=== 완료 ==="
