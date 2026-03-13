# Kaggle로 ImageNet 다운로드 - 빠른 시작 가이드

## 체크리스트

- [ ] 1. Kaggle 계정 생성
- [ ] 2. Competition 참가
- [ ] 3. API 토큰 발급
- [ ] 4. 서버에 kaggle.json 설정
- [ ] 5. ImageNet 다운로드

---

## 1. Kaggle 계정 생성

https://www.kaggle.com 접속 → 구글 계정으로 회원가입

---

## 2. Competition 참가 (필수!)

https://www.kaggle.com/c/imagenet-object-localization-challenge

→ **"Late Submission"** 또는 **"Join Competition"** 버튼 클릭
→ 규칙 동의 체크

**이 단계를 건너뛰면 403 Forbidden 에러 발생!**

---

## 3. API 토큰 발급

https://www.kaggle.com/settings

→ 아래로 스크롤 → **"API"** 섹션
→ **"Create New Token"** 클릭
→ `kaggle.json` 파일 자동 다운로드

---

## 4. 서버에 kaggle.json 설정

### 방법 A: 파일 내용 복사 (가장 간단)

**로컬 컴퓨터에서:**
```bash
# Windows
notepad Downloads\kaggle.json

# Mac/Linux
cat ~/Downloads/kaggle.json
```

파일 내용 복사 (예시):
```json
{"username":"your_username","key":"abc123..."}
```

**서버에서:**
```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# 복사한 내용 붙여넣기 (Ctrl+Shift+V)
# Ctrl+X → Y → Enter로 저장

chmod 600 ~/.kaggle/kaggle.json
```

### 방법 B: scp로 직접 전송

**로컬 컴퓨터에서:**
```bash
scp ~/Downloads/kaggle.json root@서버주소:/root/.kaggle/kaggle.json
```

**서버에서:**
```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## 5. ImageNet 다운로드

### 옵션 A: 자동 스크립트 사용 (권장)

```bash
cd /workspace/etri_iitp/JS/EfficientViT
chmod +x download_imagenet_kaggle.sh
./download_imagenet_kaggle.sh
```

### 옵션 B: 수동 다운로드

```bash
# Kaggle API 설치
pip install kaggle

# 인증 확인
kaggle competitions list

# ImageNet 다운로드 (약 155GB)
cd /workspace/etri_iitp/JS/EfficientViT/data
mkdir -p imagenet
cd imagenet
kaggle competitions download -c imagenet-object-localization-challenge

# 압축 해제
unzip imagenet-object-localization-challenge.zip

# Train 데이터 압축 해제
cd ILSVRC/Data/CLS-LOC
mkdir -p train
cd train
tar -xf ../ILSVRC2012_img_train.tar

# 각 클래스별 tar 해제
for f in *.tar; do
    class_name=$(basename "$f" .tar)
    mkdir -p "$class_name"
    tar -xf "$f" -C "$class_name"
    rm "$f"
done

# Validation 데이터 압축 해제
cd ..
mkdir -p val
cd val
tar -xf ../ILSVRC2012_img_val.tar
```

---

## 다운로드 완료 후 확인

```bash
cd /workspace/etri_iitp/JS/EfficientViT

# 폴더 구조 확인
ls data/imagenet/ILSVRC/Data/CLS-LOC/train | wc -l
# 출력: 1000 (클래스 수)

# PyTorch로 로드 테스트
python3 -c "
import torchvision.datasets as datasets
train = datasets.ImageFolder('data/imagenet/ILSVRC/Data/CLS-LOC/train')
print(f'Train images: {len(train)}')
print(f'Classes: {len(train.classes)}')
"
# 출력: Train images: 1281167, Classes: 1000
```

---

## Phase B 실행

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
    --output_dir ./output_pgm_imagenet \
    --epochs 100 \
    --batch-size 64 \
    --device cuda:0
```

---

## 문제 해결

### "401 Unauthorized"
```bash
# kaggle.json 확인
cat ~/.kaggle/kaggle.json

# 권한 재설정
chmod 600 ~/.kaggle/kaggle.json
```

### "403 Forbidden"
→ Competition 참가 안 함
→ https://www.kaggle.com/c/imagenet-object-localization-challenge 재방문
→ "Late Submission" 클릭

### "Could not find kaggle.json"
```bash
# 경로 확인
ls -la ~/.kaggle/kaggle.json

# 없으면 다시 생성
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
```

### 디스크 공간 부족
```bash
# 현재 용량 확인
df -h /workspace

# 캐시 정리
rm -rf ~/.cache/pip
rm -rf /tmp/*
```

---

## 예상 소요 시간

- **다운로드**: 2-6시간 (네트워크 속도에 따라)
- **압축 해제**: 1-2시간
- **전체**: 3-8시간

---

## 다음 단계

1. ✅ ImageNet 다운로드 완료
2. ⏭️ Pretrained 모델 로드 테스트
3. ⏭️ Phase B PGM Pruning 실행
4. ⏭️ λ Sweep 실험
5. ⏭️ 최종 평가 (76% 압축 목표)
