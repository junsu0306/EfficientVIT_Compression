# EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention

> **논문:** EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention
> **학회:** CVPR 2023
> **저자:** Xinyu Liu, Houwen Peng, Ningxin Zheng, Yuqing Yang, Han Hu, Yixuan Yuan
> **소속:** Microsoft Research Asia & Chinese University of Hong Kong

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [핵심 아이디어 (논문)](#3-핵심-아이디어-논문)
4. [모델 아키텍처 상세](#4-모델-아키텍처-상세)
5. [모델 변형 (M0~M5)](#5-모델-변형-m0m5)
6. [환경 설정](#6-환경-설정)
7. [학습 방법](#7-학습-방법)
8. [평가 방법](#8-평가-방법)
9. [속도 벤치마크](#9-속도-벤치마크)
10. [Downstream Task (Object Detection)](#10-downstream-task-object-detection)
11. [지식 증류 (Knowledge Distillation)](#11-지식-증류-knowledge-distillation)
12. [추론 최적화](#12-추론-최적화)
13. [코드 구성 요소 상세](#13-코드-구성-요소-상세)

---

## 1. 프로젝트 개요

EfficientViT는 **메모리 효율적인 Vision Transformer** 아키텍처로, 모바일 및 엣지 디바이스에서도 실시간으로 동작할 수 있는 고속 비전 모델을 목표로 합니다.

### 주요 성능 지표 (ImageNet-1K)

| 모델 | 파라미터 | Top-1 Acc | 처리속도 (GPU) |
|------|---------|-----------|---------------|
| M0   | 2.3M    | 63.2%     | 27,644 img/s  |
| M1   | 3.0M    | 68.4%     | 20,093 img/s  |
| M2   | 4.2M    | 70.8%     | 18,218 img/s  |
| M3   | 6.9M    | 73.4%     | 16,644 img/s  |
| M4   | 8.8M    | 74.3%     | 15,914 img/s  |
| M5   | 12.4M   | 77.1%     | 10,621 img/s  |

---

## 2. 프로젝트 구조

```
EfficientVIT_Compression/
├── DOCUMENTATION.md               # 본 문서
├── README.md                      # 원본 프로젝트 README
├── LICENSE
│
├── classification/                # ImageNet 이미지 분류 태스크
│   ├── main.py                    # 메인 학습/평가 스크립트
│   ├── engine.py                  # 학습/평가 루프 구현
│   ├── losses.py                  # Knowledge Distillation 손실함수
│   ├── utils.py                   # 분산학습 유틸리티, MetricLogger 등
│   ├── speed_test.py              # 처리속도 벤치마크
│   ├── requirements.txt           # 의존성 패키지 목록
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── efficientvit.py        # 핵심 모델 아키텍처 정의
│   │   └── build.py               # 모델 빌더 (M0~M5 6가지 변형)
│   │
│   └── data/
│       ├── __init__.py
│       ├── datasets.py            # 데이터셋 로더 (ImageNet, CIFAR 등)
│       ├── samplers.py            # 분산학습용 RASampler
│       └── threeaugment.py        # ThreeAugment 데이터 증강
│
└── downstream/                    # Object Detection & Instance Segmentation
    ├── train.py                   # MMDetection 학습 스크립트
    ├── test.py                    # MMDetection 평가 스크립트
    ├── dist_train.sh              # 분산 학습 쉘 스크립트
    ├── dist_test.sh               # 분산 평가 쉘 스크립트
    ├── efficientvit.py            # Detection Backbone 구현
    ├── efficientvit_fpn.py        # Feature Pyramid Network
    │
    ├── configs/
    │   ├── _base_/                # 기본 설정 (datasets, models, schedules)
    │   ├── mask_rcnn_efficientvit_m4_fpn_1x_coco.py   # Mask R-CNN 설정
    │   └── retinanet_efficientvit_m4_fpn_1x_coco.py  # RetinaNet 설정
    │
    ├── mmcv_custom/               # 커스텀 MMCv 유틸리티
    └── mmdet_custom/              # 커스텀 MMDetection 유틸리티
```

---

## 3. 핵심 아이디어 (논문)

### 3.1 문제 정의

기존 Vision Transformer들은 두 가지 주요 병목이 존재합니다:

1. **메모리 접근 비용**: Multi-Head Self-Attention(MHSA)에서 Q, K, V 계산 시 메모리 읽기/쓰기가 빈번하게 발생
2. **연산 중복성**: 서로 다른 Attention Head들이 유사한 패턴을 학습하는 경향 → 비효율적

### 3.2 핵심 혁신: Cascaded Group Attention (CGA)

```
기존 MHSA:
  입력 X → 각 Head에 동일한 X 복사 → Q,K,V 계산 → 독립적 Attention

Cascaded Group Attention (CGA):
  입력 X → 채널 방향으로 Head 수만큼 분할 (Chunk)
  Head 0: chunk_0 → Q,K,V → Attention → feat_0
  Head 1: chunk_1 + feat_0 → Q,K,V → Attention → feat_1  ← 이전 출력 누적
  Head 2: chunk_2 + feat_1 → Q,K,V → Attention → feat_2
  ...
  최종: concat(feat_0, feat_1, ...) → Linear Projection
```

**효과:**
- 각 Head가 다른 채널 부분집합을 처리 → 채널 중복성 제거
- 이전 Head 출력을 다음 Head 입력에 추가 → 점진적으로 더 넓은 수용 영역
- 입력 특징의 다양성 증가 → Attention 다양성 향상

### 3.3 Sandwich Layout (EfficientViT Block)

```
입력 x
  ↓
[DW-Conv 3×3] + Residual     ← 지역 특징 추출 (선행)
  ↓
[FFN (1×1 Conv × 2)] + Residual
  ↓
[LocalWindowAttention (CGA)] + Residual    ← 전역 의존성
  ↓
[DW-Conv 3×3] + Residual     ← 지역 특징 정제 (후행)
  ↓
[FFN (1×1 Conv × 2)] + Residual
  ↓
출력
```

**기존 Transformer:** FFN → Attention → FFN
**EfficientViT:** DW-Conv → FFN → **Attention** → DW-Conv → FFN

DW-Conv를 Attention 앞뒤에 배치함으로써 지역 정보와 전역 정보를 효과적으로 결합합니다.

### 3.4 Local Window Attention

전체 이미지에 Attention을 적용하면 O(N²) 복잡도가 발생합니다. EfficientViT는 **7×7 윈도우** 내에서만 Attention을 수행하여 복잡도를 크게 줄입니다.

```python
# LocalWindowAttention.forward() 핵심 동작
# 이미지를 window_resolution×window_resolution 블록으로 분할
# 각 윈도우 내에서 독립적으로 CGA 수행
# 윈도우를 다시 합쳐서 원래 크기 복원
```

### 3.5 Relative Position Encoding (Attention Bias)

학습 가능한 상대적 위치 인코딩을 Attention Score에 직접 더합니다:

```python
# attention_biases: (num_heads, num_unique_offsets)
# attention_bias_idxs: (N, N) - 각 픽셀 쌍의 상대 거리 인덱스
attn = (q.T @ k) * scale + attention_biases[:, attention_bias_idxs]
```

- 추론 시 `self.ab`로 미리 계산하여 캐싱 → 중복 계산 방지

---

## 4. 모델 아키텍처 상세

### 4.1 전체 구조 (`classification/model/efficientvit.py`)

```
입력: (B, 3, 224, 224)
         ↓
  [Patch Embedding]
  4개의 Conv2d_BN (stride=2씩) → 총 16× 다운샘플링
         ↓ (B, embed_dim[0], 14, 14)
  [Stage 1: blocks1]
  EfficientViTBlock × depth[0]
         ↓
  [Transition 1→2: Subsample]
  DW-Conv → FFN → PatchMerging → DW-Conv → FFN
         ↓ (B, embed_dim[1], 7, 7)
  [Stage 2: blocks2]
  EfficientViTBlock × depth[1]
         ↓
  [Transition 2→3: Subsample]
  DW-Conv → FFN → PatchMerging → DW-Conv → FFN
         ↓ (B, embed_dim[2], 4, 4)
  [Stage 3: blocks3]
  EfficientViTBlock × depth[2]
         ↓
  [Global Average Pooling] → (B, embed_dim[2])
         ↓
  [분류 헤드: BN_Linear]
         ↓
  출력: (B, num_classes)
```

### 4.2 핵심 모듈 설명

#### `Conv2d_BN` - 융합 가능한 Conv+BN
```
classification/model/efficientvit.py:13
```
- Conv2d(bias=False) + BatchNorm2d 결합
- `fuse()` 메서드: BN 파라미터를 Conv 가중치에 수학적으로 흡수 → 추론 속도 향상
- 수식: `w_fused = w_conv × (γ / √(σ² + ε))`, `b_fused = β - μ × γ / √(σ² + ε)`

#### `BN_Linear` - 분류 헤드
```
classification/model/efficientvit.py:37
```
- BatchNorm1d + Linear 결합 (분류 헤드용)
- `fuse()` 메서드 지원

#### `Residual` - Stochastic Depth 지원 잔차 연결
```
classification/model/efficientvit.py:78
```
```python
# 학습 시: 확률적으로 서브모듈 출력을 0으로 설정 (Stochastic Depth)
if self.training and self.drop > 0:
    return x + m(x) * random_mask.ge_(drop_prob) / (1 - drop_prob)
# 추론 시: 일반 잔차 연결
else:
    return x + m(x)
```

#### `PatchMerging` - 해상도 축소 블록
```
classification/model/efficientvit.py:63
```
```
입력 (B, dim, H, W)
  → Conv1×1 (dim → 4×dim, 채널 확장)
  → ReLU
  → DW-Conv 3×3 stride=2 (공간 축소)
  → Squeeze-and-Excitation (채널 재보정)
  → Conv1×1 (4×dim → out_dim, 채널 축소)
출력 (B, out_dim, H/2, W/2)
```

#### `FFN` - Position-wise Feed-Forward Network
```
classification/model/efficientvit.py:92
```
- 1×1 Conv (ed → 2×ed) → ReLU → 1×1 Conv (2×ed → ed)
- 확장 비율 2× (기존 ViT의 4×보다 작음 → 메모리 절약)

#### `CascadedGroupAttention`
```
classification/model/efficientvit.py:104
```
```python
# 핵심 forward 로직
feats_in = x.chunk(num_heads, dim=1)  # 채널을 head 수로 분할
feat = feats_in[0]
for i, qkv_proj in enumerate(self.qkvs):
    if i > 0:
        feat = feat + feats_in[i]    # 이전 출력 + 현재 chunk (Cascade!)
    q, k, v = qkv_proj(feat).split([key_dim, key_dim, value_dim])
    q = depthwise_conv[i](q)         # Query에 DW-Conv 적용
    attn = (q.T @ k) * scale + position_bias
    feat = v @ softmax(attn).T
    feats_out.append(feat)
return projection(concat(feats_out))
```

#### `LocalWindowAttention`
```
classification/model/efficientvit.py:184
```
```python
# 윈도우 분할 → CGA 수행 → 윈도우 복원
x → reshape to (B*nH*nW, C, win_h, win_w)
  → CascadedGroupAttention
  → reshape back to (B, C, H, W)
```

#### `EfficientViTBlock` - 기본 빌딩 블록
```
classification/model/efficientvit.py:250
```
```python
def forward(self, x):
    x = self.dw0(x)    # Residual(DW-Conv 3×3)
    x = self.ffn0(x)   # Residual(FFN)
    x = self.mixer(x)  # Residual(LocalWindowAttention)
    x = self.dw1(x)    # Residual(DW-Conv 3×3)
    x = self.ffn1(x)   # Residual(FFN)
    return x
```

---

## 5. 모델 변형 (M0~M5)

`classification/model/build.py`에 6가지 변형이 정의되어 있습니다.

| 모델 | embed_dim (Stage 1/2/3) | depth (Stage 1/2/3) | num_heads (Stage 1/2/3) | kernels (CGA DW-Conv) |
|------|------------------------|--------------------|-----------------------|----------------------|
| M0   | [64, 128, 192]         | [1, 2, 3]          | [4, 4, 4]             | [5, 5, 5, 5]         |
| M1   | [128, 144, 192]        | [1, 2, 3]          | [2, 3, 3]             | [7, 5, 3, 3]         |
| M2   | [128, 192, 224]        | [1, 2, 3]          | [4, 3, 2]             | [7, 5, 3, 3]         |
| M3   | [128, 240, 320]        | [1, 2, 3]          | [4, 3, 4]             | [5, 5, 5, 5]         |
| M4   | [128, 256, 384]        | [1, 2, 3]          | [4, 4, 4]             | [7, 5, 3, 3]         |
| M5   | [192, 288, 384]        | [1, 3, 4]          | [3, 3, 4]             | [7, 5, 3, 3]         |

**모든 변형 공통 설정:**
- `img_size`: 224×224
- `patch_size`: 16 (4개의 stride-2 Conv → 총 16× 다운샘플링)
- `window_size`: [7, 7, 7] (각 Stage)
- `key_dim`: [16, 16, 16]

### 사전학습 모델 다운로드

```python
# build.py의 URL 포맷
# https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/{모델명}.pth
```

---

## 6. 환경 설정

### Classification (이미지 분류)

```bash
cd classification
pip install -r requirements.txt
```

`requirements.txt` 주요 패키지:
```
torch==1.11.0
torchvision
timm==0.5.4        # 모델 레지스트리, 데이터 증강, 옵티마이저
einops==0.4.1      # 텐서 연산
fvcore             # 공통 유틸리티
```

### Downstream (Object Detection)

```bash
cd downstream
pip install openmim
mim install mmcv-full
pip install mmdet
```

---

## 7. 학습 방법

### 7.1 단일 GPU 학습

```bash
cd classification
python main.py \
  --model EfficientViT_M4 \
  --data-path /path/to/imagenet \
  --batch-size 256 \
  --epochs 300
```

### 7.2 분산 학습 (8 GPU 권장)

```bash
cd classification
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port 12345 \
  --use_env main.py \
  --model EfficientViT_M4 \
  --data-path /path/to/imagenet \
  --dist-eval \
  --batch-size 256 \
  --epochs 300
```

### 7.3 주요 학습 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--batch-size` | 256 | 배치 크기 (GPU당) |
| `--epochs` | 300 | 총 에폭 수 |
| `--opt` | adamw | 옵티마이저 (adamw/sgd/etc.) |
| `--lr` | 1e-3 | 학습률 |
| `--weight-decay` | 0.025 | 가중치 감쇠 |
| `--sched` | cosine | 학습률 스케줄러 |
| `--warmup-epochs` | 5 | 웜업 에폭 수 |
| `--min-lr` | 1e-5 | 최소 학습률 |
| `--clip-grad` | 0.02 | 그래디언트 클리핑 |
| `--clip-mode` | agc | 그래디언트 클리핑 방식 |
| `--smoothing` | 0.1 | Label Smoothing 계수 |
| `--model-ema` | True | EMA 모델 사용 여부 |
| `--model-ema-decay` | 0.99996 | EMA decay 계수 |

### 7.4 데이터 증강 옵션

```bash
# ThreeAugment 사용 (권장)
--ThreeAugment

# Mixup / CutMix
--mixup 0.8 --cutmix 1.0

# Random Erase
--reprob 0.25

# AutoAugment
--aa rand-m9-mstd0.5-inc1
```

**ThreeAugment** (`data/threeaugment.py`):
- 이미지를 다음 3가지 변환 중 하나 적용:
  1. GaussianBlur + 색상 지터
  2. Solarization + 색상 지터
  3. Grayscale

### 7.5 Knowledge Distillation 학습

```bash
python main.py \
  --model EfficientViT_M4 \
  --distillation-type soft \
  --teacher-model deit_base_patch16_224 \
  --teacher-path /path/to/teacher.pth \
  --distillation-alpha 0.5 \
  --distillation-tau 1.0 \
  --data-path /path/to/imagenet
```

### 7.6 Fine-tuning

```bash
python main.py \
  --model EfficientViT_M4 \
  --finetune /path/to/pretrained.pth \
  --data-path /path/to/imagenet \
  --epochs 30 \
  --lr 5e-5
```

### 7.7 체크포인트에서 재개

```bash
python main.py \
  --model EfficientViT_M4 \
  --resume /path/to/checkpoint.pth \
  --data-path /path/to/imagenet
```

### 7.8 지원 데이터셋

`data/datasets.py`에서 다음 데이터셋을 지원합니다:

```bash
# ImageNet-1K
--data-set IMNET --data-path /path/to/imagenet

# CIFAR-100
--data-set CIFAR --data-path /path/to/cifar100

# Flowers-102
--data-set FLOWERS --data-path /path/to/flowers

# iNaturalist 2018/2019
--data-set INAT --data-path /path/to/inat
--data-set INAT19 --data-path /path/to/inat19
```

---

## 8. 평가 방법

### 8.1 ImageNet 분류 평가

```bash
cd classification
python main.py \
  --eval \
  --model EfficientViT_M4 \
  --resume /path/to/efficientvit_m4.pth \
  --data-path /path/to/imagenet \
  --dist-eval
```

**평가 메트릭:**
- Top-1 Accuracy (%)
- Top-5 Accuracy (%)
- Cross-Entropy Loss

**평가 루프** (`engine.py:76`):
```python
@torch.no_grad()
def evaluate(data_loader, model, device):
    model.eval()  # BN을 eval 모드로 전환 (통계 고정)
    for images, target in data_loader:
        with torch.cuda.amp.autocast():  # Mixed Precision
            output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
```

### 8.2 분산 평가

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
  --eval --model EfficientViT_M4 --resume /path/to/model.pth \
  --data-path /path/to/imagenet --dist-eval
```

---

## 9. 속도 벤치마크

```bash
cd classification
python speed_test.py
```

`speed_test.py`는 모든 모델 변형(M0~M5)의 처리속도를 측정합니다:

- **GPU 벤치마크:** CUDA 동기화 기반 측정
- **CPU 벤치마크:** 스레드 수 제어 (4 threads)
- **웜업:** 10회 반복 후 측정 시작
- **측정:** 30회 반복 평균
- **최적화:** TorchScript 트레이싱 + BatchNorm 융합

```
출력 예시:
EfficientViT_M0 cuda:0 27644.23 images/s @ batch size 2048
EfficientViT_M4 cuda:0 15914.45 images/s @ batch size 2048
```

---

## 10. Downstream Task (Object Detection)

### 10.1 지원 태스크

| 태스크 | 모델 | COCO 성능 |
|--------|------|----------|
| Object Detection | RetinaNet + M4 | 32.7 box AP |
| Instance Segmentation | Mask R-CNN + M4 | 31.0 mask AP |

### 10.2 학습

```bash
cd downstream

# 8 GPU 분산 학습 (Mask R-CNN)
bash dist_train.sh \
  configs/mask_rcnn_efficientvit_m4_fpn_1x_coco.py \
  8 \
  --cfg-options model.backbone.pretrained=/path/to/efficientvit_m4.pth

# 8 GPU 분산 학습 (RetinaNet)
bash dist_train.sh \
  configs/retinanet_efficientvit_m4_fpn_1x_coco.py \
  8 \
  --cfg-options model.backbone.pretrained=/path/to/efficientvit_m4.pth
```

### 10.3 평가

```bash
# Object Detection 평가
bash dist_test.sh \
  configs/retinanet_efficientvit_m4_fpn_1x_coco.py \
  /path/to/model.pth \
  8 \
  --eval bbox

# Instance Segmentation 평가
bash dist_test.sh \
  configs/mask_rcnn_efficientvit_m4_fpn_1x_coco.py \
  /path/to/model.pth \
  8 \
  --eval bbox segm
```

### 10.4 Detection Backbone 구조

`downstream/efficientvit.py`의 EfficientViT는 분류 버전과 유사하지만:
- **다중 스케일 특징 출력:** blocks1, blocks2, blocks3의 출력을 FPN에 전달
- **Frozen Stages 지원:** 초기 레이어 고정 후 파인튜닝
- **Attention Bias 보간:** 다른 해상도 입력에 대한 위치 인코딩 적응

`downstream/efficientvit_fpn.py`의 EfficientViTFPN:
- MMDetection에 커스텀 넥(Neck)으로 등록
- 다중 레벨 특징 맵을 처리하는 FPN 구조
- FP16 Mixed Precision 지원

---

## 11. 지식 증류 (Knowledge Distillation)

`classification/losses.py`에 `DistillationLoss` 클래스가 구현되어 있습니다.

### 11.1 손실 함수 구조

```python
# 총 손실 = (1 - α) × 기본 손실 + α × 증류 손실
total_loss = (1 - alpha) * base_loss + alpha * distillation_loss
```

### 11.2 증류 방식

**Soft Distillation (권장):**
```python
# KL Divergence를 이용한 소프트 레이블 증류
distillation_loss = KL_div(
    log_softmax(student_output / T),
    log_softmax(teacher_output / T)
) * T²
```
- `T` (temperature): 소프트맥스 분포를 더 부드럽게 만들어 정보 전달 향상

**Hard Distillation:**
```python
# 교사 모델의 예측 클래스를 하드 레이블로 사용
distillation_loss = CrossEntropy(student_output, teacher_output.argmax())
```

### 11.3 증류 모드 옵션

```bash
--distillation-type none   # 증류 없이 일반 학습
--distillation-type soft   # KL Divergence 기반 소프트 증류
--distillation-type hard   # 교사 예측 기반 하드 증류
--distillation-alpha 0.5   # 증류 손실 가중치 (0~1)
--distillation-tau 1.0     # 온도 파라미터 (soft 증류에만 해당)
```

### 11.4 증류 학습 시 모델 구조

증류 사용 시 `EfficientViT`는 **두 개의 분류 헤드**를 가집니다:
```python
self.head = BN_Linear(embed_dim[-1], num_classes)       # 기본 헤드
self.head_dist = BN_Linear(embed_dim[-1], num_classes)  # 증류 헤드

# 추론 시 두 출력의 평균 사용
if not self.training:
    x = (head_output + head_dist_output) / 2
```

---

## 12. 추론 최적화

### 12.1 BatchNorm 융합 (Conv-BN Fusion)

```python
# build.py의 replace_batchnorm 함수
def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())  # Conv+BN → 단일 Conv
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)
```

모델 빌더에서 `fuse=True`로 활성화:
```python
model = EfficientViT_M4(fuse=True)
```

### 12.2 Attention Bias 캐싱

```python
# CascadedGroupAttention.train() 오버라이드
def train(self, mode=True):
    super().train(mode)
    if mode:
        del self.ab  # 학습 시 캐시 삭제
    else:
        self.ab = self.attention_biases[:, self.attention_bias_idxs]  # eval 시 미리 계산
```

### 12.3 Mixed Precision 추론

```python
# 평가 시 자동으로 AMP 적용
with torch.cuda.amp.autocast():
    output = model(images)
```

---

## 13. 코드 구성 요소 상세

### 13.1 `utils.py` - 유틸리티

- **`SmoothedValue`:** 윈도우 기반 이동평균 메트릭 추적
- **`MetricLogger`:** 분산 환경에서 메트릭 집계 및 출력
- **`save_on_master`:** 마스터 프로세스만 파일 저장
- **`init_distributed_mode`:** 분산학습 환경 초기화

### 13.2 `data/samplers.py` - RASampler

Repeated Augmentation Sampler: 동일한 이미지에 서로 다른 증강을 3번 적용하여 서로 다른 GPU에 분배합니다. `--repeated-aug` 플래그로 활성화합니다.

### 13.3 학습 루프 흐름 (`main.py`)

```
1. 인자 파싱 및 분산학습 초기화
2. 데이터셋 및 DataLoader 구성
3. 모델 생성 (timm.models.create_model)
4. 옵티마이저 생성 (timm.optim.create_optimizer)
5. 학습률 스케줄러 생성 (timm.scheduler.create_scheduler)
6. EMA 모델 초기화
7. 손실함수 구성 (DistillationLoss 또는 CrossEntropyLoss)
8. 에폭 루프:
   a. train_one_epoch() → 학습
   b. lr_scheduler.step()
   c. evaluate() → 검증
   d. 최고 성능 모델 저장
9. 최종 결과 출력
```

### 13.4 파라미터 그룹 분리

```python
# attention_biases는 weight_decay 적용 제외
@torch.jit.ignore
def no_weight_decay(self):
    return {x for x in self.state_dict().keys() if 'attention_biases' in x}
```

---

## 참고 자료

- **논문:** [EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention (CVPR 2023)](https://arxiv.org/abs/2305.07027)
- **모델 Zoo:** https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo
- **기반 코드:** LeViT, Swin Transformer, DeiT
