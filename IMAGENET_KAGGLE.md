# ImageNet 다운로드 - Kaggle 방법 (상세 가이드)

## 사전 준비 (한 번만)

### 1. Kaggle 계정 생성 및 API 토큰 발급

1. **Kaggle 계정 생성**
   - https://www.kaggle.com 접속
   - 구글/이메일로 회원가입

2. **Competition 참가 (필수)**
   - https://www.kaggle.com/c/imagenet-object-localization-challenge 접속
   - "Join Competition" 또는 "Late Submission" 버튼 클릭
   - 규칙 동의 체크 → 참가 완료

3. **API 토큰 생성**
   - https://www.kaggle.com/settings 접속
   - 아래로 스크롤하여 "API" 섹션 찾기
   - "Create New Token" 클릭
   - `kaggle.json` 파일 자동 다운로드됨

### 2. 서버에 kaggle.json 업로드

#### 방법 A: scp 명령어 (로컬 → 서버)

로컬 컴퓨터 터미널에서:
```bash
# kaggle.json이 다운로드 폴더에 있다고 가정
scp ~/Downloads/kaggle.json root@서버주소:/root/.kaggle/kaggle.json
```

#### 방법 B: 직접 파일 내용 복사 (더 간단)

1. **로컬 컴퓨터에서 kaggle.json 내용 복사**
   ```bash
   # Windows
   notepad Downloads\kaggle.json

   # Mac/Linux
   cat ~/Downloads/kaggle.json
   ```
   내용 예시:
   ```json
   {"username":"your_username","key":"1234567890abcdef1234567890abcdef"}
   ```

2. **서버에서 파일 생성**
   ```bash
   # 서버 터미널에서
   mkdir -p ~/.kaggle
   nano ~/.kaggle/kaggle.json
   # 또는
   vi ~/.kaggle/kaggle.json
   ```

3. **복사한 내용 붙여넣기**
   - nano: `Ctrl+Shift+V` → `Ctrl+X` → `Y` → `Enter`
   - vi: `i` → 붙여넣기 → `ESC` → `:wq`

4. **권한 설정 (중요!)**
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## ImageNet 다운로드 (서버에서 실행)

### 1. Kaggle API 설치

```bash
cd /workspace/etri_iitp/JS/EfficientViT
pip install kaggle
```

### 2. 설치 확인

```bash
kaggle --version
# 출력: Kaggle API 1.x.x
```

### 3. 인증 확인

```bash
kaggle competitions list
# 오류 없이 competition 리스트 출력되면 성공
```

만약 에러 발생 시:
```bash
# 401 Unauthorized → kaggle.json 확인
cat ~/.kaggle/kaggle.json

# 403 Forbidden → Competition 참가 안 함
# https://www.kaggle.com/c/imagenet-object-localization-challenge 재방문
```

### 4. ImageNet 다운로드

```bash
cd /workspace/etri_iitp/JS/EfficientViT/data
mkdir -p imagenet
cd imagenet

# ImageNet Competition 데이터 다운로드 (약 155GB)
kaggle competitions download -c imagenet-object-localization-challenge

# 다운로드 진행 상황 표시됨
# Downloading imagenet-object-localization-challenge.zip to /workspace/...
# 100%|████████████████████████████████████| 155G/155G [XX:XX<00:00, XXMiB/s]
```

### 5. 압축 해제

```bash
# ZIP 압축 해제 (시간 소요)
unzip -q imagenet-object-localization-challenge.zip

# 폴더 구조:
# imagenet/
# ├── ILSVRC/
# │   ├── Data/
# │   │   └── CLS-LOC/
# │   │       ├── train/  (TAR 파일들)
# │   │       ├── val/    (TAR 파일)
# │   │       └── test/
# │   └── Annotations/
# └── LOC_synset_mapping.txt
```

### 6. Train 데이터 압축 해제

```bash
cd ILSVRC/Data/CLS-LOC

# Train 데이터 압축 해제 (1000개 클래스 폴더 생성)
mkdir -p train
cd train
tar -xf ../ILSVRC2012_img_train.tar

# 각 클래스별 tar 파일 압축 해제
for f in *.tar; do
    class_name=$(basename "$f" .tar)
    mkdir -p "$class_name"
    tar -xf "$f" -C "$class_name"
    rm "$f"
done

cd ..
```

### 7. Validation 데이터 압축 해제

```bash
# Validation 데이터 압축 해제
mkdir -p val
cd val
tar -xf ../ILSVRC2012_img_val.tar
cd ..
```

### 8. Validation 데이터 재구성 (클래스별 폴더)

Validation 이미지들을 클래스별 폴더로 재구성:

```bash
cd /workspace/etri_iitp/JS/EfficientViT/data/imagenet

# valprep.sh 스크립트 다운로드
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x valprep.sh

# Validation 재구성 실행
cd ILSVRC/Data/CLS-LOC/val
bash /workspace/etri_iitp/JS/EfficientViT/data/imagenet/valprep.sh
```

또는 Python으로 직접 재구성:

```bash
cd /workspace/etri_iitp/JS/EfficientViT
python3 << 'EOF'
import os
import shutil
from pathlib import Path

val_dir = Path('data/imagenet/ILSVRC/Data/CLS-LOC/val')
annotations_file = Path('data/imagenet/ILSVRC/Annotations/CLS-LOC/val')

# Read ground truth labels
with open('data/imagenet/LOC_val_solution.csv', 'r') as f:
    lines = f.readlines()[1:]  # Skip header

# Create class folders and move images
for line in lines:
    parts = line.strip().split(',')
    img_name = parts[0]
    class_id = parts[1].split()[0]  # First prediction

    # Create class folder
    class_dir = val_dir / class_id
    class_dir.mkdir(exist_ok=True)

    # Move image
    src = val_dir / f'{img_name}.JPEG'
    dst = class_dir / f'{img_name}.JPEG'
    if src.exists():
        shutil.move(str(src), str(dst))

print("Validation data reorganized!")
EOF
```

### 9. 최종 폴더 구조 확인

```bash
cd /workspace/etri_iitp/JS/EfficientViT/data/imagenet/ILSVRC/Data/CLS-LOC

# Train: 1000개 클래스 폴더
ls train | wc -l
# 출력: 1000

# Val: 1000개 클래스 폴더
ls val | wc -l
# 출력: 1000

# 각 클래스별 이미지 수 확인
ls train/n01440764 | wc -l
# 출력: 1300 (클래스마다 다름)
```

---

## PyTorch ImageFolder로 로드 테스트

```bash
cd /workspace/etri_iitp/JS/EfficientViT

python3 << 'EOF'
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ImageFolder 형식으로 로드
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    root='data/imagenet/ILSVRC/Data/CLS-LOC/train',
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root='data/imagenet/ILSVRC/Data/CLS-LOC/val',
    transform=transform
)

print(f'Train: {len(train_dataset)} images')
print(f'Val: {len(val_dataset)} images')
print(f'Classes: {len(train_dataset.classes)}')
print('ImageNet loaded successfully!')
EOF
```

예상 출력:
```
Train: 1281167 images
Val: 50000 images
Classes: 1000
ImageNet loaded successfully!
```

---

## Phase B 실행 (ImageNet-1K)

```bash
cd /workspace/etri_iitp/JS/EfficientViT

python -m classification.main \
    --model EfficientViT_M4 \
    --finetune pretrained_efficientvit_m4.pth \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet/ILSVRC/Data/CLS-LOC \
    --data-set IMNET \
    --pruning \
    --lambda-ffn 0.005 --lambda-qk 0.01 --lambda-v 0.001 \
    --mu 1.0 --m-max-mb 8.45 \
    --output_dir ./output_pgm_imagenet_pretrained \
    --epochs 100 \
    --batch-size 64 \
    --device cuda:0
```

---

## 문제 해결

### 1. "401 Unauthorized" 에러
```bash
# kaggle.json 확인
cat ~/.kaggle/kaggle.json

# 권한 확인
ls -la ~/.kaggle/kaggle.json
# 출력: -rw------- 1 root root ... (600 권한이어야 함)

# 권한 재설정
chmod 600 ~/.kaggle/kaggle.json
```

### 2. "403 Forbidden" 에러
- Competition에 참가하지 않았음
- https://www.kaggle.com/c/imagenet-object-localization-challenge 접속
- "Late Submission" 클릭 → 규칙 동의

### 3. 디스크 공간 부족
```bash
# 현재 디스크 사용량 확인
df -h /workspace

# 불필요한 파일 정리
rm -rf ~/.cache/pip
rm -rf /tmp/*

# ZIP 파일 삭제 (압축 해제 후)
rm imagenet-object-localization-challenge.zip
```

### 4. 다운로드 중단 시 재개
```bash
# Kaggle API는 자동으로 이어받기 지원
kaggle competitions download -c imagenet-object-localization-challenge
# 이미 다운로드된 부분은 건너뜀
```

### 5. 속도 개선
```bash
# 다운로드 속도가 느리면 병렬 다운로드
kaggle competitions download -c imagenet-object-localization-challenge -f ILSVRC2012_img_train.tar
kaggle competitions download -c imagenet-object-localization-challenge -f ILSVRC2012_img_val.tar

# 또는 aria2로 병렬 다운로드 (Kaggle URL 직접 사용 불가능, 우회 필요)
```

---

## 요약: 빠른 실행 스크립트

```bash
#!/bin/bash
# ImageNet 다운로드 및 설정 전체 자동화

set -e

echo "=== Kaggle API 설정 ==="
pip install kaggle
chmod 600 ~/.kaggle/kaggle.json

echo "=== ImageNet 다운로드 ==="
cd /workspace/etri_iitp/JS/EfficientViT/data
mkdir -p imagenet
cd imagenet
kaggle competitions download -c imagenet-object-localization-challenge

echo "=== 압축 해제 ==="
unzip -q imagenet-object-localization-challenge.zip

echo "=== Train 데이터 처리 ==="
cd ILSVRC/Data/CLS-LOC
mkdir -p train
cd train
tar -xf ../ILSVRC2012_img_train.tar

for f in *.tar; do
    class_name=$(basename "$f" .tar)
    mkdir -p "$class_name"
    tar -xf "$f" -C "$class_name"
    rm "$f"
done

echo "=== Validation 데이터 처리 ==="
cd ../
mkdir -p val
cd val
tar -xf ../ILSVRC2012_img_val.tar

echo "=== 완료! ==="
echo "데이터 경로: /workspace/etri_iitp/JS/EfficientViT/data/imagenet/ILSVRC/Data/CLS-LOC"
```

이 스크립트를 `download_imagenet_kaggle.sh`로 저장 후:
```bash
chmod +x download_imagenet_kaggle.sh
./download_imagenet_kaggle.sh
```
