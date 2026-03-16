# EfficientViT Physical Pruning 완전 가이드

> **프로젝트**: RS-2024-00339187 | 고려대학교 | 3차년도 ViT 확장 연구
> **목표**: EfficientViT M4에 Structured Pruning 적용 → **76% 모델 최적화율** 달성
> **최적화율 공식**: `100 × (B - A) / B` (B: 원본 크기, A: 압축 후 크기)
> **작성일**: 2026-03-14

---

## 목차

1. [개요](#1-개요)
2. [EfficientViT 아키텍처 분석](#2-efficientvit-아키텍처-분석)
3. [Pruning 이론](#3-pruning-이론) (Granularity, λ 비율, 제약 조건 포함)
4. [구현 과정 및 문제 해결](#4-구현-과정-및-문제-해결)
5. [최종 구현](#5-최종-구현)
6. [실행 가이드](#6-실행-가이드)
7. [하이퍼파라미터 설명](#7-하이퍼파라미터-설명)
8. [예상 결과](#8-예상-결과)

---

## 1. 개요

### 1.1 프로젝트 목표

EfficientViT M4 모델(35.2MB)을 **76% 압축**하여 약 8.4MB로 줄이는 것이 목표입니다.

```
원본 모델: 35.2 MB
목표 크기: 35.2 × (1 - 0.76) = 8.4 MB
최적화율: 76%
```

### 1.2 접근 방식 변천

| 단계 | 방식 | 결과 |
|------|------|------|
| Phase A | 모델 프로파일링 | 67개 pruning group 정의 완료 |
| Phase B v1 | Soft Masking (weight=0) | **폐기** - 실제 크기 감소 없음 |
| Phase B v2 | Physical Pruning | **채택** - 실제 Conv2d 크기 축소 |

### 1.3 핵심 개념: Soft Masking vs Physical Pruning

**Soft Masking (폐기된 방식)**
```python
# Weight를 0으로 설정하지만 텐서 크기는 그대로
weight[pruning_indices] = 0.0
# 문제: 0 × value 연산 여전히 수행, 메모리 감소 없음
```

**Physical Pruning (채택된 방식)**
```python
# Conv2d 자체를 새로운 크기로 교체
old_conv = nn.Conv2d(128, 256, 1)  # 128→256
new_conv = nn.Conv2d(128, 64, 1)   # 128→64 (실제 축소!)
new_conv.weight.data = old_conv.weight.data[keep_indices]
```

---

## 2. EfficientViT 아키텍처 분석

### 2.1 전체 구조

```
Input [B, 3, 224, 224]
  └─ OverlapPatchEmbed          → [B, C1, H/16, W/16]
       └─ Stage 1 (blocks1)
            └─ EfficientViTBlock (C=128, H=4)
       └─ SubsampleBlock        → C1→C2
       └─ Stage 2 (blocks2)
            └─ EfficientViTBlock ×2 (C=256, H=4)
       └─ SubsampleBlock        → C2→C3
       └─ Stage 3 (blocks3)
            └─ EfficientViTBlock ×3 (C=384, H=4)
       └─ AvgPool + Linear      → logits [B, 1000]
```

### 2.2 EfficientViTBlock 내부 구조

```
Input X [B, C, H, W]
  │
  ├─ DWConv (TokenInteraction)   ← groups=C, 로컬 정보 교환
  │   └─ (+residual)
  │
  ├─ FFN (Feed-Forward Network)
  │   └─ expand: Conv2d(C → C×2)  ← Pruning 대상!
  │   └─ BN → ReLU
  │   └─ shrink: Conv2d(C×2 → C)  ← Pruning 대상!
  │   └─ (+residual)
  │
  ├─ DWConv (TokenInteraction)
  │
  ├─ CGA (Cascaded Group Attention)  ← Pruning 대상!
  │   └─ (+residual)
  │
  └─ DWConv (TokenInteraction)

Output X' [B, C, H, W]
```

### 2.3 CGA (Cascaded Group Attention) 상세

```
Input X [B, C, H, W]
  │
  └─ Split into H heads: [X₀, X₁, ..., X_{H-1}], 각 [B, C/H, H, W]

Head 0:
  X₀ → QKV Conv → [Q₀, K₀, V₀]
       Q₀ → DW Conv → Q₀'
       Attention(Q₀', K₀, V₀) → Out₀

Head 1 (Cascade!):
  X₁ + Out₀ → QKV Conv → [Q₁, K₁, V₁]  ← 이전 head 출력 더함
             ...

Concat[Out₀, Out₁, ..., Out_{H-1}] → Proj → Output
```

**M4 기준 CGA 파라미터**:
| 항목 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| embed_dim (C) | 128 | 256 | 384 |
| num_heads (H) | 4 | 4 | 4 |
| head_dim (C/H) | 32 | 64 | 96 |
| key_dim | 16 | 16 | 16 |
| d (V dim) | 32 | 64 | 96 |

### 2.4 파라미터 분포 분석

```
EfficientViT M4 파라미터 분포 (약 8.8M):

FFN (expand + shrink):     ~70-80%  ← 가장 많음, 적극적 pruning!
CGA (QKV + DW + Proj):     ~15-20%
기타 (PatchEmbed, DW 등):  ~5-10%
```

**핵심 인사이트**: FFN이 대부분의 파라미터를 차지하므로, FFN을 공격적으로 pruning해야 76% 목표 달성 가능.

---

## 3. Pruning 이론

### 3.1 Structured Pruning vs Unstructured Pruning

| 구분 | Unstructured | Structured |
|------|--------------|------------|
| 단위 | 개별 weight | 채널/필터/head 단위 |
| Sparsity | 임의 위치 | 연속된 블록 |
| 하드웨어 가속 | 어려움 | 용이 |
| 본 프로젝트 | ❌ | ✅ 채택 |

### 3.2 Importance 기반 Pruning

**원리**: 중요도가 낮은 unit을 먼저 제거

```python
# FFN Importance: expand의 출력 채널과 shrink의 입력 채널 L2 norm
importance[k] = ||expand.weight[k, :, :, :]||₂ + ||shrink.weight[:, k, :, :]||₂

# Q/K Importance: Q, K, DW의 L2 norm 합
importance[d] = ||Q.weight[d, :]||₂ + ||K.weight[d, :]||₂ + ||DW.weight[d, :, :, :]||₂
```

### 3.3 두 가지 Pruning 방식

#### Physical-Only (CE Loss만 사용)
```
L_total = L_CE (CrossEntropy)

매 epoch 끝에 importance 기반 physical pruning
장점: 단순, 빠른 실험
단점: pruning 시 갑작스러운 변화 (weights가 미리 작아지지 않은 상태에서 제거)
```

#### Combined (λ Regularization + Physical) — PGM-inspired

```
L_total = L_CE + L_reg + L_mem

where:
  L_CE   = CrossEntropy(softmax(z), y)              ← 분류 손실
  L_reg  = λ_FFN·Σ||w_FFN||² + λ_QK·Σ||w_QK||² + λ_V·Σ||w_V||²   ← 정규화
  L_mem  = μ · max(0, current_size - M_max)          ← 메모리 제약
```

**PGM (Proximal Gradient Method)과의 관계:**

| 항목 | 순수 PGM | Combined (본 구현) |
|------|---------|-------------------|
| Step 1: Gradient | `w -= η·∇L_total` | `w -= η·∇(L_CE + λ·‖w‖² + μ·mem)` |
| Step 2: Proximal | Group Soft Thresholding (수학적) | Physical Pruning (importance 기반 제거) |
| Sparsity 유도 | Group Lasso (L1/L2 norm) | L2 Regularization (‖w‖²) |
| Zero 전환 | Soft thresholding이 자동으로 0 생성 | Epoch 끝에 명시적으로 작은 것 제거 |
| 구조 축소 | 학습 후 별도 extraction 필요 | **매 epoch 물리적 축소 (extraction 불필요)** |

순수 PGM의 proximal operator (Group Soft Thresholding):
```python
# PGM 원본: threshold 이하면 그룹 전체를 0으로
if ||w_g|| > η·λ_g:
    w_g *= (1 - η·λ_g / ||w_g||)  # 수축
else:
    w_g = 0                        # 그룹 제거 → 별도 extraction으로 구조 축소
```

본 구현의 대체 방식:
```python
# Combined: λ regularization이 weight를 미리 작게 유도
#           → physical pruning이 importance 하위 unit 물리적 제거
L_reg gradient: ∂L_reg/∂w = 2λw  → optimizer가 -2λw 방향으로 업데이트
                                  → 불필요한 weights가 점차 0에 가까워짐

# Epoch 끝: importance 기반 pruning
importance[k] = ||expand.weight[k]||₂ + ||shrink.weight[:,k]||₂
keep_indices = topk(importance, new_size)  # 이미 작아진 것들이 자연스럽게 제거됨
new_conv.weight = old_conv.weight[keep_indices]  # 물리적 축소!
```

**결론**: Combined 방식은 **"PGM-inspired structured pruning"** — PGM의 핵심 아이디어(regularization으로 sparsity 유도 + operator로 구조 제거)를 차용하되, proximal operator를 importance 기반 physical pruning으로 대체한 실용적 변형입니다. 학습 후 별도 structure extraction이 필요 없다는 것이 순수 PGM 대비 장점입니다.

### 3.4 λ 비율 설계 (중요!)

**비율: FFN : QK : V = 10 : 2 : 1**

| 대상 | λ 값 | 비율 | 이유 |
|------|------|------|------|
| FFN | 1e-3 | 10x | 파라미터 많음, 적극적 pruning |
| Q/K | 2e-4 | 2x | key_dim=16으로 이미 작음, 조심 |
| V | 1e-4 | 1x | 정보 손실 민감, 최소 pruning |

### 3.5 Pruning Granularity: Layer-wise vs Global

현재 구현은 **Layer-wise (Block별)** pruning을 사용합니다.

#### 현재 구현 방식

| 대상 | Granularity | Importance 범위 | 설명 |
|------|------------|----------------|------|
| FFN | **Block별 독립** | 해당 FFN의 expand+shrink | 각 block의 FFN을 개별적으로 importance 계산 후 pruning |
| QK | **Block 내 Global** | 해당 CGA의 전체 head 합산 | 한 block 내 모든 head의 importance를 합산하여 통일된 indices 결정 |
| V | 미적용 (keep=1.0) | — | — |

#### FFN — Block별 독립 Pruning

```
Block A (Stage 2, C=256):  hidden=512 → importance 계산 → top-k 선택 → 384개 유지
Block B (Stage 3, C=384):  hidden=768 → importance 계산 → top-k 선택 → 576개 유지

→ 모든 block에 동일한 keep_ratio(75%) 적용
→ 각 block에서 제거되는 neuron은 해당 block의 importance에 따라 독립 결정
```

#### QK — Block 내 Global Importance

```python
# 한 Block의 CGA 내 모든 head importance를 합산
global_importance = torch.zeros(key_dim)
for h in range(num_heads):
    head_importance = compute_qk_importance(qkvs[h], dws[h], key_dim)
    global_importance += head_importance

# 합산된 importance로 통일된 keep_indices 결정
_, keep_indices = torch.topk(global_importance, new_key_dim)

# 모든 head에 동일 indices 적용 (key_dim 공유 제약)
```

**QK가 Block 내 Global인 이유**: CGA의 모든 head가 동일한 `key_dim`을 공유하므로, head별로 다른 indices를 pruning하면 `split_with_sizes` 오류 발생 (§4.4 참조).

#### Layer-wise vs Global 비교

| 방식 | 장점 | 단점 |
|------|------|------|
| **Layer-wise (현재)** | 구현 단순, 모든 block이 균등하게 축소 | Stage간 중요도 차이 반영 못함 |
| **Global** | redundant layer를 더 공격적으로 축소 가능, 같은 압축률에서 정확도 유리 | 구현 복잡, 일부 layer가 극단적으로 축소될 위험 |

#### EfficientViT에서 Global FFN이 어려운 이유

EfficientViT는 이미 최적화된 아키텍처입니다:
- FFN expansion ratio가 **r=2** (일반 ViT의 r=4보다 이미 절반)
- 각 Stage별 channel 수(128, 256, 384)가 ablation을 통해 결정된 값
- Block 수(1, 2, 3)도 최적으로 설정됨

따라서 Global pruning으로 특정 layer를 과도하게 축소하면, 이미 최적화된 아키텍처 밸런스가 무너질 가능성이 높습니다.

> **향후 비교 과제**: Layer-wise vs Global FFN pruning의 정확도/압축률 비교 실험을 진행하여, EfficientViT 같은 최적화된 아키텍처에서 어떤 방식이 더 적합한지 검증 예정.

### 3.6 Pruning 제약 조건

**FFN 제약**:
```
expand.out_channels == shrink.in_channels (항상!)
BN도 동일 indices로 축소
```

**CGA 제약 (CRITICAL!)**:
```
모든 head가 동일한 key_dim 공유!
→ 모든 head에서 동일한 indices를 pruning해야 함
→ Global Importance 계산 필요
```

---

## 4. 구현 과정 및 문제 해결

### 4.1 Phase A: 모델 프로파일링

**목표**: Pruning 가능한 모든 그룹 식별

**결과**: 67개 pruning group
- G_PATCH: 3개 (PatchEmbed Conv)
- G_FFN: 20개 (각 block의 expand+shrink)
- G_QK: 24개 (각 CGA의 Q/K dimensions)
- G_V: 24개 (각 CGA의 V dimensions)

**파일**: `group_dict.py`, `phase_a_profile.py`

### 4.2 Phase B v1: Soft Masking (폐기)

**시도한 방식**:
```python
# pgm_loss.py
w_expand[pruning_idx, :, :, :] = 0.0
w_shrink[:, pruning_idx, :, :] = 0.0
```

**발견한 문제**:
1. 실제 연산량 감소 없음 (0 × value 연산)
2. 실제 메모리 감소 없음 (tensor 크기 동일)
3. optimizer.step()에서 0이 다시 업데이트됨
4. 76% 압축 **불가능**

**결론**: Soft Masking 방식 폐기, Physical Pruning으로 전환

### 4.3 Phase B v2: Physical Pruning

**핵심 아이디어**: Conv2d 자체를 새로운 크기로 교체

```python
# structural_pruning.py
def prune_ffn_physically(ffn, keep_ratio):
    # 1. Importance 계산
    importance = compute_ffn_importance(expand, shrink)

    # 2. Top-k indices 선택
    _, keep_indices = torch.topk(importance, new_hidden)

    # 3. 새 Conv2d 생성
    new_expand = nn.Conv2d(embed_dim, new_hidden, kernel_size=1)
    new_expand.weight.data = expand.weight.data[keep_indices]

    # 4. 기존 모듈 교체
    expand.c = new_expand
```

### 4.4 버그 수정: CGA key_dim 동기화

**문제**: Q/K pruning 후 `split_with_sizes` RuntimeError
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 62,
but got split_sizes=[16, 16, 32]
```

**원인 분석**:
```
각 head를 개별적으로 pruning하면서 key_dim 불일치 발생:
- Head 0: key_dim=15로 pruned
- Head 1~3: key_dim=16 기대
- QKV tensor: 15+15+32=62, split은 16+16+32=64 기대
```

**해결책**: Global Importance 기반 통합 Pruning
```python
def prune_efficientvit_block_cga(cga, keep_ratio):
    # 1. 모든 head의 importance를 합산
    global_importance = torch.zeros(key_dim)
    for h in range(num_heads):
        head_importance = compute_qk_importance(qkvs[h], dws[h], key_dim)
        global_importance += head_importance

    # 2. 통일된 keep_indices 결정
    _, keep_indices = torch.topk(global_importance, new_key_dim)

    # 3. 모든 head에 동일 indices 적용
    for h in range(num_heads):
        apply_pruning(qkvs[h], dws[h], keep_indices)

    # 4. CGA 속성 한 번만 업데이트
    cga.key_dim = new_key_dim
    cga.scale = new_key_dim ** -0.5
```

### 4.5 Pruning Rate 조정

**문제**: 2 epochs에 6.8%만 압축 (76% 목표 대비 너무 느림)

**원인**: 보수적인 기본 pruning rate

**해결**: 공격적인 설정으로 변경
```python
# 변경 전 (보수적)
ffn_prune_per_epoch = 0.10  # 10%
qk_prune_per_epoch = 0.02   # 2%
min_ffn_ratio = 0.10        # 최소 10%
min_qk_ratio = 0.50         # 최소 50%

# 변경 후 (공격적)
ffn_prune_per_epoch = 0.25  # 25% (2.5배 증가)
qk_prune_per_epoch = 0.05   # 5% (2.5배 증가)
min_ffn_ratio = 0.05        # 최소 5% (2배 감소)
min_qk_ratio = 0.25         # 최소 25% (2배 감소)
```

### 4.6 버그 수정: Additive → Multiplicative 누적 추적 (Critical)

**문제**: ImageNet 학습 시 46.2%만 달성 (76% 목표), FFN이 거의 pruning 안 됨
```
로그: FFN keep=1.00  ← pruning이 전혀 일어나지 않음!
```

**원인**: Additive 누적 추적 — 4 epoch 만에 `cumulative = 1.0` 도달하여 조기 중단
```python
# 버그 코드 (additive)
self.cumulative_ffn_pruned = 0.0
self.cumulative_ffn_pruned += 0.25  # epoch 1: 0.25
self.cumulative_ffn_pruned += 0.25  # epoch 2: 0.50
self.cumulative_ffn_pruned += 0.25  # epoch 3: 0.75
self.cumulative_ffn_pruned += 0.25  # epoch 4: 1.00 → 중단!
# 실제로는 0.75^4 = 31.6%만 pruned인데 100%로 인식
```

**해결**: Multiplicative 누적 추적 — 실제 남은 비율을 정확히 추적
```python
# 수정 코드 (multiplicative)
self.ffn_remaining = 1.0        # 100%에서 시작
self.ffn_remaining *= (1 - 0.25)  # epoch 1: 0.750
self.ffn_remaining *= (1 - 0.25)  # epoch 2: 0.563
self.ffn_remaining *= (1 - 0.25)  # epoch 3: 0.422
self.ffn_remaining *= (1 - 0.25)  # epoch 4: 0.316
# ...                              # epoch 10: 0.056
# min_ffn_ratio(0.05)에 도달할 때까지 계속 진행!
```

**추가 보호**: min ratio에 정확히 맞추기
```python
projected = self.ffn_remaining * (1.0 - ffn_rate)
if projected < self.min_ffn_ratio:
    ffn_rate = 1.0 - self.min_ffn_ratio / self.ffn_remaining
    ffn_rate = max(0, ffn_rate)
```

---

## 5. 최종 구현

### 5.1 파일 구조

```
classification/pruning/
├── structural_pruning.py         ← 핵심 모듈
│   ├── compute_ffn_importance()      - FFN importance 계산
│   ├── prune_ffn_physically()        - FFN 물리적 축소
│   ├── compute_qk_importance()       - Q/K importance 계산
│   ├── prune_efficientvit_block_cga() - CGA 통합 pruning (Global Importance)
│   ├── apply_iterative_physical_pruning()
│   ├── IterativePhysicalPruner       - epoch별 pruning 관리
│   ├── compute_model_size_mb()       - 모델 크기 계산
│   └── validate_model_forward()      - shape 검증
│
├── train_physical_pruning.py     ← Physical-Only (CE만)
├── train_combined_pruning.py     ← Combined (CE + λ + μ)
├── test_quick_pruning.py         ← 빠른 테스트 (Dummy 데이터)
│
├── pgm_loss.py                   ← 기존 soft masking (deprecated)
├── group_dict.py                 ← 67개 pruning group 정의
├── memory_utils.py               ← 메모리 측정 유틸리티
│
├── PHASE_B_ITERATIVE.md          ← 실행 가이드
└── PRUNING_GUIDE.md              ← 이 문서
```

### 5.2 핵심 알고리즘

```
Initialize: pretrained model M₀

=== Pruning Phase (15 epochs) ===
for epoch = 1 to N_prune:
    # Step 1: Train one epoch
    train_one_epoch(model, train_loader, optimizer, criterion)

    # Step 2: Compute FFN Importance
    for each FFN:
        importance[k] = ||expand.weight[k]||₂ + ||shrink.weight[:,k]||₂

    # Step 3: Compute CGA Global Importance
    for each CGA:
        global_importance = Σ_h (||Q_h[d]||₂ + ||K_h[d]||₂ + ||DW_h[d]||₂)

    # Step 4: Physical Pruning
    keep_indices = topk(importance, new_size)
    new_conv.weight = old_conv.weight[keep_indices]

    # Step 5: Validate forward pass
    validate_model_forward(model, device)

    # Step 6: Check target
    if compression_ratio >= 0.76:
        break

=== Fine-tuning Phase (1-10 epochs) ===
for epoch = 1 to N_finetune:
    train_one_epoch(model)  # No pruning, only training
```

### 5.3 Pruning 진행 예측

```
Epoch  | FFN 잔여율 | QK 잔여율 | 예상 압축률
-------|-----------|----------|------------
  0    |   100%    |   100%   |     0%
  1    |    75%    |    95%   |    ~8%
  2    |    56%    |    90%   |   ~19%
  5    |    24%    |    77%   |   ~45%
 10    |     5%*   |    60%   |   ~65%
 15    |     5%*   |    25%** |   ~76%

* min_ffn_ratio = 5%에서 정지
** min_qk_ratio = 25%에서 정지
```

---

## 6. 실행 가이드

### 6.1 환경 설정

```bash
conda create -n efficientvit python=3.10 -y
conda activate efficientvit
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.9.12 einops Pillow matplotlib tqdm
```

### 6.2 빠른 테스트 (Dummy 데이터)

```bash
cd /home/junsu/EfficientVIT_Compression

# Physical-Only 테스트
python -m classification.pruning.test_quick_pruning --mode physical

# Combined 테스트
python -m classification.pruning.test_quick_pruning --mode combined

# Pretrained 체크포인트 사용
python -m classification.pruning.test_quick_pruning --mode physical --resume efficientvit_m4.pth
```

### 6.3 실제 ImageNet 학습

#### Physical-Only (빠른 실험용)
```bash
python -m classification.pruning.train_physical_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --ffn-prune-per-epoch 0.25 \
    --qk-prune-per-epoch 0.15 \
    --min-ffn-ratio 0.05 \
    --min-qk-ratio 0.25 \
    --pruning-epochs 15 \
    --finetune-epochs 1 \
    --batch-size 256 \
    --lr 1e-4 \
    --output-dir classification/pruning/results/physical_76pct
```

#### Combined (최종 결과용, 권장)
```bash
python -m classification.pruning.train_combined_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --ffn-prune-per-epoch 0.25 \
    --qk-prune-per-epoch 0.15 \
    --min-ffn-ratio 0.05 \
    --min-qk-ratio 0.25 \
    --lambda-ffn 0.001 \
    --lambda-qk 0.0002 \
    --lambda-v 0.0001 \
    --mu 1.0 \
    --pruning-epochs 15 \
    --finetune-epochs 1 \
    --batch-size 256 \
    --lr 1e-4 \
    --output-dir classification/pruning/results/combined_76pct
```

### 6.4 압축률별 실행 명령어

#### 15% 압축 (33.59 MB → ~28.55 MB)

```bash
# Physical-Only 15%
python -m classification.pruning.train_physical_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.15 \
    --ffn-prune-per-epoch 0.08 \
    --qk-prune-per-epoch 0.05 \
    --min-ffn-ratio 0.70 \
    --min-qk-ratio 0.70 \
    --pruning-epochs 5 \
    --finetune-epochs 10 \
    --output-dir /results/physical_15pct

# Combined 15%
python -m classification.pruning.train_combined_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.15 \
    --ffn-prune-per-epoch 0.08 \
    --qk-prune-per-epoch 0.05 \
    --min-ffn-ratio 0.70 \
    --min-qk-ratio 0.70 \
    --lambda-ffn 5e-5 --lambda-qk 2e-5 --lambda-v 5e-6 \
    --pruning-epochs 8 \
    --finetune-epochs 5 \
    --output-dir /results/combined_15pct
```

#### 30% 압축 (33.59 MB → ~23.51 MB)

```bash
# Physical-Only 30%
python -m classification.pruning.train_physical_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.30 \
    --ffn-prune-per-epoch 0.12 \
    --qk-prune-per-epoch 0.08 \
    --min-ffn-ratio 0.50 \
    --min-qk-ratio 0.50 \
    --pruning-epochs 8 \
    --finetune-epochs 10 \
    --output-dir classification/pruning/results/physical_30pct

# Combined 30%
python -m classification.pruning.train_combined_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.30 \
    --ffn-prune-per-epoch 0.12 \
    --qk-prune-per-epoch 0.08 \
    --min-ffn-ratio 0.50 \
    --min-qk-ratio 0.50 \
    --lambda-ffn 8e-5 --lambda-qk 3e-5 --lambda-v 8e-6 \
    --pruning-epochs 8 \
    --finetune-epochs 10 \
    --output-dir results/combined_30pct
```

#### 76% 압축 (33.59 MB → ~8.06 MB) — 기존

```bash
# Physical-Only 76%
python -m classification.pruning.train_physical_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --ffn-prune-per-epoch 0.25 \
    --qk-prune-per-epoch 0.15 \
    --min-ffn-ratio 0.05 \
    --min-qk-ratio 0.25 \
    --pruning-epochs 15 \
    --finetune-epochs 10 \
    --output-dir classification/pruning/results/physical_76pct

# Combined 76%
python -m classification.pruning.train_combined_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --ffn-prune-per-epoch 0.25 \
    --qk-prune-per-epoch 0.15 \
    --min-ffn-ratio 0.05 \
    --min-qk-ratio 0.25 \
    --lambda-ffn 0.001 --lambda-qk 0.0002 --lambda-v 0.0001 \
    --mu 1.0 \
    --pruning-epochs 15 \
    --finetune-epochs 10 \
    --output-dir classification/pruning/results/combined_76pct
```

#### 압축률별 파라미터 비교표

| 파라미터 | 15% | 30% | 76% |
|----------|-----|-----|-----|
| ffn-prune-per-epoch | 0.08 | 0.12 | 0.25 |
| qk-prune-per-epoch | 0.05 | 0.08 | 0.15 |
| min-ffn-ratio | 0.70 | 0.50 | 0.05 |
| min-qk-ratio | 0.70 | 0.50 | 0.25 |
| pruning-epochs | 5 | 8 | 15 |
| finetune-epochs | 10 | 10 | 10 |
| λ_FFN (Combined) | 5e-5 | 8e-5 | 1e-3 |
| λ_QK (Combined) | 2e-5 | 3e-5 | 2e-4 |
| λ_V (Combined) | 5e-6 | 8e-6 | 1e-4 |

### 6.5 체크포인트 관련 FAQ

**Q: 다시 실행할 때 체크포인트를 지워야 하나요?**

A: 아니요! 항상 원본 pretrained weights를 사용하면 됩니다:
```bash
--resume efficientvit_m4.pth  # 원본 pretrained 사용
```

출력 디렉토리만 바꾸거나 같은 디렉토리면 덮어쓰기됩니다.

**Q: Pruning만 확인하고 싶으면?**

A: `--finetune-epochs 1`로 설정하세요.

---

## 7. 하이퍼파라미터 설명

### 7.1 Pruning 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `--target-reduction` | **0.76** | 목표 압축률 (76%) |
| `--ffn-prune-per-epoch` | **0.25** | FFN: 매 epoch 25% 제거 |
| `--qk-prune-per-epoch` | **0.15** | Q/K: 매 epoch 15% 제거 |
| `--min-ffn-ratio` | **0.05** | FFN 최소 5% 유지 (최대 95% pruning) |
| `--min-qk-ratio` | **0.25** | Q/K 최소 25% 유지 |
| `--pruning-epochs` | **15** | Pruning 진행 epoch 수 |
| `--finetune-epochs` | **1-10** | Finetune epoch 수 |

### 7.2 λ Regularization (Combined 전용)

| 파라미터 | 값 | 비율 | 설명 |
|----------|-----|------|------|
| `--lambda-ffn` | **1e-3** | 10x | FFN 적극적 축소 |
| `--lambda-qk` | **2e-4** | 2x | Q/K 조심스럽게 축소 |
| `--lambda-v` | **1e-4** | 1x | V 최소 축소 (보존) |
| `--mu` | **1.0** | - | Memory penalty 계수 |

### 7.3 Training 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `--batch-size` | **256** | GPU 메모리에 따라 조정 |
| `--lr` | **1e-4** | AdamW 학습률 |
| `--weight-decay` | **0.05** | Weight decay |
| `--clip-grad` | **1.0** | Gradient clipping |
| `--smoothing` | **0.1** | Label smoothing |

---

## 8. 예상 결과

### 8.1 Pruning 결과 예측

| Epoch | FFN 잔여 | QK 잔여 | Model Size | Reduction |
|-------|---------|---------|------------|-----------|
| 0     | 100%    | 100%    | 35.2 MB    | 0%        |
| 1     | 75%     | 95%     | 32.1 MB    | ~8%       |
| 2     | 56%     | 90%     | 28.5 MB    | ~19%      |
| 5     | 24%     | 77%     | 19.4 MB    | ~45%      |
| 10    | 5%      | 60%     | 12.3 MB    | ~65%      |
| 15    | 5%      | 25%     | 8.4 MB     | ~76%      |

### 8.2 Accuracy 예측

| Phase | Epochs | Acc@1 | Notes |
|-------|--------|-------|-------|
| Initial | 0 | 74.3% | Pretrained M4 |
| Pruning | 5 | ~65% | 45% 축소 |
| Pruning | 10 | ~58% | 65% 축소 |
| Pruning | 15 | ~55% | 76% 축소 (target) |
| Finetune | +5 | ~62% | 회복 시작 |
| Finetune | +10 | ~66% | 추가 회복 |
| + KD (Phase C) | +20 | ~70% | Teacher 도움 |

### 8.3 메모리 사용량

| 항목 | 예상 값 |
|------|---------|
| 모델 (초기) | ~35 MB |
| 모델 (76% 압축 후) | ~8.4 MB |
| GPU 메모리 (batch=256) | ~20-24 GB |
| GPU 메모리 (batch=128) | ~12-14 GB |

---

## 부록 A: 핵심 구현 코드

> 파일: `structural_pruning.py`, `train_physical_pruning.py`, `train_combined_pruning.py`

### A.1 FFN Importance 계산

```python
# structural_pruning.py: compute_ffn_importance()
def compute_ffn_importance(expand_conv, shrink_conv):
    """
    FFN hidden neuron의 importance = expand 출력 채널 L2 norm + shrink 입력 채널 L2 norm

    expand.weight: [hidden_dim, embed_dim, 1, 1]  → 출력 채널별 norm
    shrink.weight: [embed_dim, hidden_dim, 1, 1]  → 입력 채널별 norm
    """
    with torch.no_grad():
        w_expand = expand_conv.c.weight  # Conv2d_BN 내부의 Conv2d
        expand_norms = w_expand.view(w_expand.size(0), -1).norm(dim=1)  # [hidden_dim]

        w_shrink = shrink_conv.c.weight
        shrink_norms = w_shrink.view(w_shrink.size(0), -1, 1, 1).squeeze().norm(dim=0)  # [hidden_dim]

        importance = expand_norms + shrink_norms  # [hidden_dim]
    return importance
```

### A.2 FFN Physical Pruning

```python
# structural_pruning.py: prune_ffn_physically()
def prune_ffn_physically(ffn, keep_ratio, min_neurons=8):
    expand = ffn.pw1  # Conv2d_BN (expand)
    shrink = ffn.pw2  # Conv2d_BN (shrink)

    hidden_dim = expand.c.out_channels
    embed_dim = expand.c.in_channels

    # 유지할 neuron 수 계산
    new_hidden = max(min_neurons, int(hidden_dim * keep_ratio))
    new_hidden = min(new_hidden, hidden_dim)

    # Importance 기반 top-k 선택
    importance = compute_ffn_importance(expand, shrink)
    _, keep_indices = torch.topk(importance, new_hidden, largest=True)
    keep_indices = keep_indices.sort().values  # 순서 유지

    with torch.no_grad():
        # === Expand (pw1) 축소: out_channels 감소 ===
        new_expand_conv = nn.Conv2d(embed_dim, new_hidden, kernel_size=1, bias=False)
        new_expand_conv.weight.data = expand.c.weight.data[keep_indices]
        expand.c = new_expand_conv

        # BN도 동일 indices로 축소 (weight, bias, running_mean, running_var)
        new_expand_bn = nn.BatchNorm2d(new_hidden)
        new_expand_bn.weight.data = expand.bn.weight.data[keep_indices]
        new_expand_bn.bias.data = expand.bn.bias.data[keep_indices]
        new_expand_bn.running_mean.data = expand.bn.running_mean.data[keep_indices]
        new_expand_bn.running_var.data = expand.bn.running_var.data[keep_indices]
        expand.bn = new_expand_bn

        # === Shrink (pw2) 축소: in_channels 감소 ===
        new_shrink_conv = nn.Conv2d(new_hidden, embed_dim, kernel_size=1, bias=False)
        new_shrink_conv.weight.data = shrink.c.weight.data[:, keep_indices]
        shrink.c = new_shrink_conv
        # shrink의 BN은 출력(embed_dim) 기준이므로 변경 없음

    return hidden_dim, new_hidden
```

### A.3 QK Importance 계산 및 CGA Global Pruning

```python
# structural_pruning.py: compute_qk_importance()
def compute_qk_importance(qkv_conv, dw_conv, key_dim):
    """
    Q/K dim의 importance = Q norm + K norm + DW norm

    qkv.weight: [key_dim*2 + d, in_ch, 1, 1]
      Q: [0 : key_dim]
      K: [key_dim : 2*key_dim]
      V: [2*key_dim : ]  ← pruning 대상 아님
    """
    with torch.no_grad():
        w_qkv = qkv_conv.c.weight
        w_dw = dw_conv.c.weight  # [key_dim, 1, kH, kW]

        q_norms = w_qkv[:key_dim].view(key_dim, -1).norm(dim=1)
        k_norms = w_qkv[key_dim:2*key_dim].view(key_dim, -1).norm(dim=1)
        dw_norms = w_dw.view(key_dim, -1).norm(dim=1)

        importance = q_norms + k_norms + dw_norms
    return importance


# structural_pruning.py: prune_efficientvit_block_cga()
def prune_efficientvit_block_cga(block, qk_keep_ratio, v_keep_ratio=1.0):
    """
    CRITICAL: CGA의 모든 head가 동일한 key_dim을 공유
    → 모든 head의 importance를 합산하여 Global Importance 계산
    → 통일된 keep_indices를 모든 head에 적용
    """
    cga = block.mixer.m.attn  # Residual → LocalWindowAttention → CascadedGroupAttention
    key_dim = cga.key_dim
    num_heads = cga.num_heads

    new_key_dim = max(4, int(key_dim * qk_keep_ratio))

    # === Step 1: Global Importance 계산 (모든 head 합산) ===
    global_importance = torch.zeros(key_dim, device=cga.qkvs[0].c.weight.device)
    for h in range(num_heads):
        head_importance = compute_qk_importance(cga.qkvs[h], cga.dws[h], key_dim)
        global_importance += head_importance

    # === Step 2: 통일된 keep_indices 결정 ===
    _, keep_indices = torch.topk(global_importance, new_key_dim, largest=True)
    keep_indices = keep_indices.sort().values

    # === Step 3: 모든 head에 동일 indices 적용 ===
    with torch.no_grad():
        for h in range(num_heads):
            qkv = cga.qkvs[h]
            dw = cga.dws[h]
            d = cga.d
            in_ch = qkv.c.in_channels

            # QKV Conv: Q/K는 pruning, V는 유지
            w_qkv = qkv.c.weight  # [key_dim*2+d, in_ch, 1, 1]
            new_q = w_qkv[:key_dim][keep_indices]
            new_k = w_qkv[key_dim:2*key_dim][keep_indices]
            v_weights = w_qkv[2*key_dim:]  # V는 그대로

            new_qkv_weight = torch.cat([new_q, new_k, v_weights], dim=0)

            new_out_ch = new_key_dim * 2 + d
            new_qkv_conv = nn.Conv2d(in_ch, new_out_ch, kernel_size=1, bias=False)
            new_qkv_conv.weight.data = new_qkv_weight
            qkv.c = new_qkv_conv

            # BN: Q/K 구간은 keep_indices, V 구간은 전체 유지
            bn_w = qkv.bn.weight.data
            new_bn_w = torch.cat([bn_w[:key_dim][keep_indices],
                                  bn_w[key_dim:2*key_dim][keep_indices],
                                  bn_w[2*key_dim:]])
            # bias, running_mean, running_var도 동일 패턴
            # ... (생략, 동일 로직)
            qkv.bn = new_qkv_bn

            # DW Conv: groups = key_dim → new_key_dim
            new_dw_conv = nn.Conv2d(
                new_key_dim, new_key_dim,
                kernel_size=dw.c.kernel_size, padding=dw.c.padding,
                groups=new_key_dim, bias=False
            )
            new_dw_conv.weight.data = dw.c.weight[keep_indices]
            dw.c = new_dw_conv
            # DW BN도 keep_indices로 축소
            # ...

    # === Step 4: CGA 속성 업데이트 (한 번만!) ===
    cga.key_dim = new_key_dim
    cga.scale = new_key_dim ** -0.5
```

### A.4 IterativePhysicalPruner (Multiplicative 누적 추적)

```python
# structural_pruning.py: IterativePhysicalPruner
class IterativePhysicalPruner:
    def __init__(self, target_reduction=0.76, ffn_prune_per_epoch=0.25,
                 qk_prune_per_epoch=0.15, min_ffn_ratio=0.05, min_qk_ratio=0.25):
        self.target_reduction = target_reduction
        self.ffn_prune_per_epoch = ffn_prune_per_epoch
        self.qk_prune_per_epoch = qk_prune_per_epoch
        self.min_ffn_ratio = min_ffn_ratio
        self.min_qk_ratio = min_qk_ratio

        # 승산(multiplicative) 누적 추적 — 실제 남은 비율
        self.ffn_remaining = 1.0  # 100%에서 시작
        self.qk_remaining = 1.0
        self.target_reached = False

    def step(self, model, device='cuda'):
        # FFN: min_ffn_ratio 이상이면 pruning 계속
        if self.ffn_remaining > self.min_ffn_ratio:
            ffn_rate = self.ffn_prune_per_epoch
            # min 이하로 내려가지 않도록 보호
            projected = self.ffn_remaining * (1.0 - ffn_rate)
            if projected < self.min_ffn_ratio:
                ffn_rate = 1.0 - self.min_ffn_ratio / self.ffn_remaining
                ffn_rate = max(0, ffn_rate)
        else:
            ffn_rate = 0

        # QK: 동일 로직
        if self.qk_remaining > self.min_qk_ratio:
            qk_rate = self.qk_prune_per_epoch
            projected = self.qk_remaining * (1.0 - qk_rate)
            if projected < self.min_qk_ratio:
                qk_rate = 1.0 - self.min_qk_ratio / self.qk_remaining
                qk_rate = max(0, qk_rate)
        else:
            qk_rate = 0

        # Physical pruning 적용
        result = apply_iterative_physical_pruning(model, ffn_rate, qk_rate)

        # 승산 누적 업데이트 (핵심!)
        # remaining은 원본 대비 실제 남은 비율
        self.ffn_remaining *= (1.0 - ffn_rate)  # 0.75 → 0.56 → 0.42 → ...
        self.qk_remaining *= (1.0 - qk_rate)    # 0.85 → 0.72 → 0.61 → ...

        # Forward pass 검증
        validate_model_forward(model, device)

        # 목표 도달 확인
        if result['cumulative_reduction'] >= self.target_reduction:
            self.target_reached = True

        return result
```

**Multiplicative 추적이 핵심인 이유** (§4.6 버그 참조):
```
additive (버그):     0.25 + 0.25 + 0.25 + 0.25 = 1.0 → 4 epoch에서 중단!
multiplicative (수정): 0.75 × 0.75 × 0.75 × ... → 계속 감소, min_ratio까지 진행
```

### A.5 Training Loop (Physical-Only)

```python
# train_physical_pruning.py: main() 핵심 부분
for epoch in range(total_epochs):
    phase = 'PRUNE' if epoch < args.pruning_epochs else 'FINETUNE'

    # Step 1: Train one epoch (CE loss only)
    train_stats = train_one_epoch_physical(
        model, data_loader_train, optimizer, device, epoch,
        criterion, scaler, args.clip_grad
    )

    # Step 2: Physical pruning (pruning phase에만)
    if epoch < args.pruning_epochs and not pruner.target_reached:
        pruning_result = pruner.step(model, device)

        # CRITICAL: pruning 후 optimizer 재생성 (파라미터가 변경되었으므로)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer.param_groups[0]['lr'],
            weight_decay=args.weight_decay
        )

    # Step 3: Evaluate
    test_stats = evaluate(data_loader_val, model, device)
    lr_scheduler.step()

    # Step 4: Pruning 완료 시점 기록
    if epoch == args.pruning_epochs - 1 or pruner.target_reached:
        pruning_end_acc1 = test_stats['acc1']
        pruning_end_size = compute_model_size_mb(model)
```

### A.6 Combined Loss 계산 (λ Regularization + Memory Penalty)

```python
# train_combined_pruning.py: compute_lambda_regularization()
def compute_lambda_regularization(model, lambda_ffn, lambda_qk, lambda_v):
    """
    L_reg = λ_FFN·Σ||w_FFN||² + λ_QK·Σ||w_QK||² + λ_V·Σ||w_V||²

    gradient에 -2λw 항 추가 → weights를 0 방향으로 유도
    physical pruning 시 이미 작아진 weights 제거 → 안정적
    """
    reg_loss = 0.0

    for name, module in model.named_modules():
        # FFN layers (pw1 = expand, pw2 = shrink)
        if hasattr(module, 'pw1') and hasattr(module, 'pw2'):
            reg_loss += lambda_ffn * torch.sum(module.pw1.c.weight ** 2)
            reg_loss += lambda_ffn * torch.sum(module.pw2.c.weight ** 2)

        # CGA layers (qkvs, dws) — Q/K/V를 slice로 분리
        if hasattr(module, 'qkvs') and hasattr(module, 'dws'):
            for h in range(len(module.qkvs)):
                w = module.qkvs[h].c.weight
                key_dim = module.key_dim
                d = module.d

                reg_loss += lambda_qk * torch.sum(w[:key_dim] ** 2)           # Q
                reg_loss += lambda_qk * torch.sum(w[key_dim:2*key_dim] ** 2)  # K
                reg_loss += lambda_v * torch.sum(w[2*key_dim:2*key_dim+d] ** 2)  # V

                reg_loss += lambda_qk * torch.sum(module.dws[h].c.weight ** 2)  # DW

    return reg_loss


def compute_memory_penalty(current_size_mb, m_max_mb, mu):
    """μ · max(0, current_mem - m_max)"""
    return mu * max(0.0, current_size_mb - m_max_mb)


# Training loop에서:
def train_one_epoch_combined(model, data_loader, ..., lambda_ffn, lambda_qk, lambda_v, mu):
    for samples, targets in data_loader:
        outputs = model(samples)
        ce_loss = criterion(outputs, targets)
        reg_loss = compute_lambda_regularization(model, lambda_ffn, lambda_qk, lambda_v)
        mem_penalty = compute_memory_penalty(current_size_mb, m_max_mb, mu)

        # Total loss = CE + λ_reg + μ·memory_penalty
        loss = ce_loss + reg_loss + mem_penalty

        loss.backward()
        optimizer.step()
```

### A.7 전체 모델 Pruning 순회 구조

```python
# structural_pruning.py: apply_iterative_physical_pruning()
def apply_iterative_physical_pruning(model, ffn_prune_rate, qk_prune_rate):
    """전체 모델의 모든 block을 순회하며 physical pruning 적용"""
    ffn_keep = 1.0 - ffn_prune_rate
    qk_keep = 1.0 - qk_prune_rate

    # Stage 1 (blocks1): EfficientViTBlock
    for block in model.blocks1:
        if hasattr(block, 'ffn0'):  # EfficientViTBlock 판별
            prune_efficientvit_block_ffn(block, ffn_keep)
            prune_efficientvit_block_cga(block, qk_keep)

    # Stage 2 (blocks2): Subsample + EfficientViTBlock
    for block in model.blocks2:
        if isinstance(block, nn.Sequential):
            # Subsample FFN (SubPreDWFFN, SubPostDWFFN)
            for sub in block:
                if hasattr(sub, 'm') and hasattr(sub.m, 'pw1'):
                    prune_ffn_physically(sub.m, ffn_keep)
        elif hasattr(block, 'ffn0'):
            prune_efficientvit_block_ffn(block, ffn_keep)
            prune_efficientvit_block_cga(block, qk_keep)

    # Stage 3 (blocks3): 동일 패턴
    for block in model.blocks3:
        # ... Stage 2와 동일
```

---

## 부록 B: 다음 단계 (Phase C)

### Knowledge Distillation 추가

```python
# Teacher: M5 (pretrained, frozen)
# Student: Pruned M4

L_total = (1 - α) * L_CE + α * L_KD

L_KD = KL(softmax(z_T / T) || log_softmax(z_S / T)) * T²

# 권장값
α = 0.5
T = 4.0
```

### 예상 파일
```
classification/pruning/
├── train_physical_pruning.py      ← Phase B (현재)
├── train_combined_pruning.py      ← Phase B (현재)
├── train_pruning_kd.py            ← Phase C (KD 추가) - 예정
```

---

## 부록 C: 파라미터 비교 실험 — 64.6% vs 76% vs 80% 목표

> **배경**: min ratio가 pruning floor를 결정함을 발견. `target-reduction` 파라미터는 조기 종료(early stop)만 하며, 실제 압축률의 하한은 `min_ffn_ratio`와 `min_qk_ratio`가 결정함.

### C.1 원인 분석 — 왜 15 epoch(76%)와 20 epoch(80%)가 동일한 크기로 나오는가

```
pruning이 멈추는 조건 (두 가지):
  1. current_reduction >= target_reduction  → 조기 종료 (early stop)
  2. ffn_remaining <= min_ffn_ratio AND
     qk_remaining <= min_qk_ratio          → 하한선 도달, 이후 pruning 없음

목표(target_reduction)보다 하한선이 먼저 걸리면 목표와 무관하게 멈춤!

현재 설정 (min_ffn_ratio=0.05, min_qk_ratio=0.25):
  Epoch 9  : QK → 0.25 (min_qk_ratio 도달)
  Epoch 11 : FFN → 0.05 (min_ffn_ratio 도달)
  Epoch 12+ : pruning 완전 중단
  결과 : 64.6% (15 epoch = 20 epoch = 동일)
```

### C.2 QK 하드코딩 제약 (중요!)

`structural_pruning.py`의 `prune_efficientvit_block_cga()` 내부:
```python
min_dim = 4  # 하드코딩
new_key_dim = max(min_dim, int(key_dim * qk_keep_ratio))
```

M4 기준 key_dim=16 → **QK 절대 하한 = 4/16 = 25%**
→ `min_qk_ratio`를 0.25 이하로 설정해도 QK는 더 이상 pruning 안 됨
→ **추가 압축은 FFN을 더 줄이는 수밖에 없음**

### C.3 파라미터 세트 비교

| 항목 | 기존 (Exp A) | 76% 시도 (Exp B) | 80% 시도 (Exp C) |
|------|-------------|-----------------|-----------------|
| `--target-reduction` | 0.76 | 0.76 | 0.80 |
| `--min-ffn-ratio` | **0.05** | **0.02** | **0.01** |
| `--min-qk-ratio` | 0.25 | 0.10* | 0.06* |
| `--pruning-epochs` | 15 | 20 | 25 |
| `--ffn-prune-per-epoch` | 0.25 | 0.25 | 0.25 |
| `--qk-prune-per-epoch` | 0.15 | 0.15 | 0.15 |
| `--lambda-ffn` | 1e-3 | 1e-3 | 2e-3 |
| `--lambda-qk` | 2e-4 | 2e-4 | 2e-4 |
| `--lambda-v` | 1e-4 | 1e-4 | 1e-4 |
| `--finetune-epochs` | 10 | 10 | 10 |
| FFN 이론 최소 | 5% | 2% | 1% |
| QK 실제 최소 | 25% (하드코딩) | 25% (하드코딩) | 25% (하드코딩) |
| **예상 압축률** | **~65%** | **~70-73%** | **~73-76%** |
| 실측 결과 | 64.6%, 11.88MB | — | — |

*min_qk_ratio를 낮춰도 QK는 min_dim=4 제약으로 25%가 실제 floor

> **주의**: FFN을 1%까지 줄이면 accuracy 급락 위험. Phase C (KD) 없이는 회복 어려울 수 있음.
> 80% 달성을 위해서는 `structural_pruning.py`의 `min_dim=4`를 `min_dim=2`로 낮추는 코드 수정도 고려 필요.

### C.4 실행 명령어

#### Exp A — 기존 (실측: 64.6%, 60.43%)
```bash
python -m classification.pruning.train_combined_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --ffn-prune-per-epoch 0.25 \
    --qk-prune-per-epoch 0.15 \
    --min-ffn-ratio 0.05 \
    --min-qk-ratio 0.25 \
    --lambda-ffn 0.001 \
    --lambda-qk 0.0002 \
    --lambda-v 0.0001 \
    --mu 1.0 \
    --pruning-epochs 15 \
    --finetune-epochs 10 \
    --batch-size 256 \
    --lr 1e-4 \
    --output-dir checkpoints/exp_a_64pct
```

#### Exp B — 76% 시도 (min_ffn_ratio 낮춤)
```bash
python -m classification.pruning.train_combined_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --ffn-prune-per-epoch 0.25 \
    --qk-prune-per-epoch 0.15 \
    --min-ffn-ratio 0.02 \
    --min-qk-ratio 0.10 \
    --lambda-ffn 0.001 \
    --lambda-qk 0.0002 \
    --lambda-v 0.0001 \
    --mu 1.0 \
    --pruning-epochs 20 \
    --finetune-epochs 10 \
    --batch-size 256 \
    --lr 1e-4 \
    --output-dir checkpoints/exp_b_76pct
```

#### Exp C — 80% 시도 (min_ffn_ratio 최저, lambda_ffn 강화)
```bash
python -m classification.pruning.train_combined_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.80 \
    --ffn-prune-per-epoch 0.25 \
    --qk-prune-per-epoch 0.15 \
    --min-ffn-ratio 0.01 \
    --min-qk-ratio 0.06 \
    --lambda-ffn 0.002 \
    --lambda-qk 0.0002 \
    --lambda-v 0.0001 \
    --mu 1.5 \
    --pruning-epochs 20 \
    --finetune-epochs 10 \
    --batch-size 512 \
    --lr 1e-4 \
    --output-dir results/exp_c_80
```

### C.5 예상 압축률 계산 근거

```
M4 파라미터 분포 (추정):
  FFN:      ~70% × 33.59MB ≈ 23.5 MB
  QK/V/Proj: ~20% × 33.59MB ≈  6.7 MB
  기타:      ~10% × 33.59MB ≈  3.4 MB

QK는 min_dim=4 고정 → key_dim 16→4 → QK 75% pruning
  남는 QK/V/Proj ≈ 6.7 × 0.40 ≈ 2.7 MB (V, Proj 포함하므로 rough)

Exp A (min_ffn=0.05): FFN 95% pruning → FFN 잔여 ≈ 23.5 × 0.05 = 1.2 MB
  Total ≈ 1.2 + 2.7 + 3.4 = 7.3 MB → 이론상 78%... 실측 64.6% (기타 포함 더 큼)

Exp B (min_ffn=0.02): FFN 98% pruning → FFN 잔여 ≈ 23.5 × 0.02 = 0.5 MB
  추가 절감 ≈ 1.2 - 0.5 = 0.7 MB → 실측 대비 +약 5-6%p 기대

Exp C (min_ffn=0.01): FFN 99% pruning → FFN 잔여 ≈ 23.5 × 0.01 = 0.2 MB
  추가 절감 ≈ 1.2 - 0.2 = 1.0 MB → 실측 대비 +약 8-9%p 기대
```

### C.6 결과 기록표 (실험 후 채워넣기)

| 실험 | 압축률 | 최종 크기 | Pruning 후 Acc | Finetune 후 Acc | 비고 |
|------|--------|-----------|----------------|-----------------|------|
| Exp A | 64.6% | 11.88 MB | 59.51% | 60.43% | 기존 결과 |
| Exp B | — | — | — | — | 실험 예정 |
| Exp C | — | — | — | — | 실험 예정 |

---

## 부록 D: Pruned 모델 저장 및 로드

### D.1 저장 방식 비교

학습 완료 후 두 가지 형태로 저장됨:

| 파일 | 저장 방식 | 로드 가능 여부 | 용도 |
|------|----------|--------------|------|
| `best_combined.pth` | `state_dict()` | ❌ 바로 불가 | 학습 재개 전용 |
| `best_combined_model.pth` | `torch.save(model)` | ✅ 바로 가능 | 최고 Acc 시점 모델 |
| `final_pruned_model.pth` | `torch.save(model)` | ✅ 바로 가능 | Finetune 완료 모델 |

### D.2 왜 state_dict만으로는 로드가 안 되는가

Physical Pruning은 레이어 크기를 물리적으로 교체함:
```
FFN Linear: (256, 128) → (13, 128)   ← 실제로 줄어든 레이어
key_dim:     16         → 4           ← 실제로 줄어든 레이어
```

`state_dict()`는 이 줄어든 weights만 저장. 나중에 로드 시:
```python
model = EfficientViT_M4(...)     # 원본 shape (FFN=256, key_dim=16)으로 생성
model.load_state_dict(ckpt)
# ❌ RuntimeError: size mismatch for blocks1.0.ffn.pw1.weight:
#    shape torch.Size([13, 128]) vs torch.Size([256, 128])
```

### D.3 torch.save(model) 로드 시 필요 조건

`torch.save(model, ...)` 은 Python pickle로 직렬화됨.
Pickle은 weights와 함께 **"이 객체가 어떤 클래스인지"** 도 기록.

로드 시 내부 동작:
```
1. pth 파일 열기
2. "이건 EfficientViT 클래스 객체야" 확인
3. EfficientViT 클래스 정의를 현재 Python 환경에서 탐색  ← 여기서 실패 가능
4. weights 채워넣기 → 객체 복원 완료
```

3번에서 클래스 정의를 못 찾으면:
```
ModuleNotFoundError: No module named 'classification'
```

### D.4 올바른 로드 방법

```python
import sys
import torch

# 프로젝트 루트를 Python 경로에 추가 (클래스 정의 탐색 가능하게)
sys.path.insert(0, '/path/to/EfficientVIT_Compression')

# 이제 바로 로드 가능
model = torch.load('final_pruned_model.pth', map_location='cuda')
model.eval()

# 바로 inference
with torch.no_grad():
    output = model(input_tensor)  # pruned된 작은 모델로 실행
```

> **핵심**: `torch.load` 전에 프로젝트 코드가 있는 경로를 `sys.path`에 추가해야 함.
> 같은 서버에서 같은 경로로 실행하면 `sys.path.insert` 없이도 동작할 수 있음.

### D.5 다른 환경(서버/PC)으로 모델 이전 시

```bash
# 이전할 파일 목록
scp results/exp_b_76pct/final_pruned_model.pth  user@new_server:/path/
scp -r classification/  user@new_server:/path/EfficientVIT_Compression/  # 코드도 같이!
```

코드 없이 pth만 가져가면 로드 불가. **pth + 프로젝트 코드 세트**로 이전해야 함.

---

**Prepared by**: Claude Code
**Last Updated**: 2026-03-15
