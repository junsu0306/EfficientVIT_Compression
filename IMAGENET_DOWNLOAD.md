# ImageNet-1K 다운로드 가이드 (Hugging Face)

## 가장 간단한 방법 (추천)

### 1. 사전 준비 (한 번만)

```bash
# Hugging Face CLI 설치
pip install huggingface_hub datasets

# Hugging Face 로그인
huggingface-cli login
# → Access Token 입력 (https://huggingface.co/settings/tokens 에서 생성)

# ImageNet-1K 접근 권한 요청
# → https://huggingface.co/datasets/imagenet-1k 접속
# → "Access repository" 버튼 클릭 (승인까지 몇 분 소요)
```

### 2. 다운로드 실행

#### 옵션 A: Hugging Face 포맷으로 사용 (디스크 절약, 추천)

```bash
cd /workspace/etri_iitp/JS/EfficientViT

python3 << 'EOF'
from datasets import load_dataset

# ImageNet-1K 다운로드 (자동으로 ~/.cache/huggingface/datasets/ 에 저장)
dataset = load_dataset(
    "imagenet-1k",
    cache_dir="./data/.cache",
    trust_remote_code=True
)

print(f"Train: {len(dataset['train'])} images")
print(f"Validation: {len(dataset['validation'])} images")
print("다운로드 완료!")
EOF
```

#### 옵션 B: PyTorch ImageFolder 형식으로 저장 (호환성 최고)

```bash
cd /workspace/etri_iitp/JS/EfficientViT
python3 download_imagenet_hf.py
# → ImageFolder 형식 저장 여부 물어보면 'y' 입력
```

### 3. 학습 시 데이터 경로 설정

#### Hugging Face 포맷 사용 시:
datasets.py를 수정하여 Hugging Face Dataset 직접 사용

#### ImageFolder 형식 사용 시:
```bash
--data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet
```

---

## 대안: 미니 데이터셋으로 빠른 테스트

전체 ImageNet-1K (150GB)는 시간이 오래 걸리므로, 먼저 작은 데이터셋으로 테스트:

### ImageNet-100 (Hugging Face)

```bash
python3 << 'EOF'
from datasets import load_dataset

# ImageNet의 100개 클래스만 포함 (훨씬 빠름, 약 13GB)
dataset = load_dataset("Maysee/tiny-imagenet", cache_dir="./data/.cache")
print(f"Train: {len(dataset['train'])} images")
EOF
```

### Tiny-ImageNet (200 클래스, 64×64, 약 237MB)

```bash
cd /workspace/etri_iitp/JS/EfficientViT/data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

---

## 다운로드 속도 개선 팁

1. **멀티스레드 다운로드**: Hugging Face datasets는 자동으로 병렬 다운로드
2. **캐시 활용**: `cache_dir` 설정으로 원하는 위치에 저장
3. **부분 다운로드**: train만 먼저 다운로드

```python
# Train만 다운로드
dataset = load_dataset("imagenet-1k", split="train", cache_dir="./data/.cache")
```

---

## 문제 해결

### "Access to this repository is restricted"
→ https://huggingface.co/datasets/imagenet-1k 에서 접근 권한 요청

### "Authentication token not found"
→ `huggingface-cli login` 실행

### 디스크 용량 부족
→ Tiny-ImageNet 또는 ImageNet-100 사용

### 다운로드 속도 느림
→ 서버 네트워크 환경에 따라 다름, `wget` 또는 `aria2c` 사용 고려
