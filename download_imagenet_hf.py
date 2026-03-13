#!/usr/bin/env python3
"""
Hugging Face Datasets API를 사용하여 ImageNet-1K 다운로드
"""
import os
from pathlib import Path

def download_imagenet():
    """
    Hugging Face에서 ImageNet-1K 다운로드 및 로컬 저장
    """
    print("=== ImageNet-1K 다운로드 (Hugging Face) ===\n")

    # Step 1: datasets 라이브러리 설치 확인
    try:
        from datasets import load_dataset
    except ImportError:
        print("Step 1: datasets 라이브러리 설치 중...")
        os.system("pip install datasets huggingface_hub")
        from datasets import load_dataset

    # Step 2: Hugging Face 로그인 (필요한 경우)
    print("\nStep 2: Hugging Face 인증 확인 중...")
    print("ImageNet 다운로드를 위해 Hugging Face 계정이 필요합니다.")
    print("https://huggingface.co 에서 계정 생성 후 Access Token 발급")
    print("\n다음 명령어로 로그인:")
    print("  huggingface-cli login")
    print("\n이미 로그인했다면 계속 진행됩니다...\n")

    # Step 3: ImageNet-1K 다운로드
    print("Step 3: ImageNet-1K 다운로드 중...")
    print("참고: 전체 데이터셋 크기는 약 150GB입니다.")
    print("다운로드 위치: ~/.cache/huggingface/datasets/imagenet-1k/\n")

    try:
        # ImageNet-1K 다운로드 (train + validation)
        dataset = load_dataset(
            "imagenet-1k",
            cache_dir="/workspace/etri_iitp/JS/EfficientViT/data/.cache",
            trust_remote_code=True
        )

        print(f"\n✓ 다운로드 완료!")
        print(f"  Train: {len(dataset['train'])} 이미지")
        print(f"  Validation: {len(dataset['validation'])} 이미지")

        # Step 4: PyTorch ImageFolder 형식으로 저장 (선택사항)
        save_path = Path("/workspace/etri_iitp/JS/EfficientViT/data/imagenet")
        save_choice = input("\nPyTorch ImageFolder 형식으로 저장하시겠습니까? (y/n): ")

        if save_choice.lower() == 'y':
            print("\nImageFolder 형식으로 저장 중...")
            save_as_imagefolder(dataset, save_path)
        else:
            print("\nHugging Face 포맷으로만 유지합니다.")
            print("훈련 시 datasets.Dataset을 직접 사용하세요.")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\n문제 해결 방법:")
        print("1. Hugging Face 로그인: huggingface-cli login")
        print("2. ImageNet 접근 권한 요청:")
        print("   https://huggingface.co/datasets/imagenet-1k")
        print("   페이지에서 'Access repository' 클릭")
        return

def save_as_imagefolder(dataset, save_path):
    """
    Hugging Face Dataset을 PyTorch ImageFolder 형식으로 저장
    """
    from PIL import Image
    from tqdm import tqdm

    for split in ['train', 'validation']:
        print(f"\n{split} 저장 중...")
        split_path = save_path / split
        split_path.mkdir(parents=True, exist_ok=True)

        for idx, example in enumerate(tqdm(dataset[split])):
            # 클래스 폴더 생성
            label = example['label']
            class_path = split_path / str(label)
            class_path.mkdir(exist_ok=True)

            # 이미지 저장
            img = example['image']
            img.save(class_path / f"{idx:08d}.JPEG")

    print(f"\n✓ 저장 완료: {save_path}")

if __name__ == "__main__":
    download_imagenet()
