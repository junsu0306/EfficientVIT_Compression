# EfficientViT Physical Pruning 완전 가이드

> **프로젝트**: RS-2024-00339187 | 고려대학교 | 3차년도 ViT 확장 연구
> **목표**: EfficientViT M4에 Structured Pruning 적용 → **76% 모델 최적화율** 달성
> **최적화율 공식**: `100 × (B - A) / B` (B: 원본 크기, A: 압축 후 크기)
> **작성일**: 2026-03-14

---

## 목차

1. [개요](#1-개요)
2. [EfficientViT 아키텍처 분석](#2-efficientvit-아키텍처-분석)
3. [Pruning 이론](#3-pruning-이론)
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
```python
L_total = L_CE (CrossEntropy)

# 매 epoch 끝에 importance 기반 physical pruning
# 장점: 단순, 빠른 실험
# 단점: pruning 시 갑작스러운 변화
```

#### Combined (λ Regularization + Physical)
```python
L_total = L_CE + λ_FFN·Σ||w_FFN||² + λ_QK·Σ||w_QK||² + λ_V·Σ||w_V||² + μ·max(0, mem - m_max)

# λ regularization이 weight를 미리 0 방향으로 유도
# physical pruning 시 이미 작아진 weight 제거 → 안정적
# memory penalty가 목표 압축률 강제
```

### 3.4 λ 비율 설계 (중요!)

**비율: FFN : QK : V = 10 : 2 : 1**

| 대상 | λ 값 | 비율 | 이유 |
|------|------|------|------|
| FFN | 1e-3 | 10x | 파라미터 많음, 적극적 pruning |
| Q/K | 2e-4 | 2x | key_dim=16으로 이미 작음, 조심 |
| V | 1e-4 | 1x | 정보 손실 민감, 최소 pruning |

### 3.5 Pruning 제약 조건

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
    --output-dir checkpoints/physical_only_76pct
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
    --output-dir checkpoints/combined_76pct
```

### 6.4 체크포인트 관련 FAQ

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

## 부록 A: 주요 코드 스니펫

### A.1 FFN Physical Pruning

```python
def prune_ffn_physically(ffn, keep_ratio, min_neurons=8):
    expand = ffn[0]  # Conv-BN-ReLU
    shrink = ffn[1]  # Conv-BN

    # Importance 계산
    expand_norms = expand.c.weight.norm(dim=(1, 2, 3))
    shrink_norms = shrink.c.weight.norm(dim=(0, 2, 3))
    importance = expand_norms + shrink_norms

    # Top-k 선택
    new_hidden = max(min_neurons, int(current_hidden * keep_ratio))
    _, keep_indices = torch.topk(importance, new_hidden, largest=True)
    keep_indices = keep_indices.sort().values

    # expand 축소 (out_channels)
    new_expand_conv = nn.Conv2d(embed_dim, new_hidden, kernel_size=1, bias=False)
    new_expand_conv.weight.data = expand.c.weight.data[keep_indices]
    expand.c = new_expand_conv

    # BN 축소
    new_expand_bn = nn.BatchNorm2d(new_hidden)
    new_expand_bn.weight.data = expand.bn.weight.data[keep_indices]
    new_expand_bn.bias.data = expand.bn.bias.data[keep_indices]
    # ... running_mean, running_var도 동일
    expand.bn = new_expand_bn

    # shrink 축소 (in_channels)
    new_shrink_conv = nn.Conv2d(new_hidden, embed_dim, kernel_size=1, bias=False)
    new_shrink_conv.weight.data = shrink.c.weight.data[:, keep_indices]
    shrink.c = new_shrink_conv
```

### A.2 CGA Global Importance Pruning

```python
def prune_efficientvit_block_cga(cga, keep_ratio):
    key_dim = cga.key_dim
    num_heads = cga.num_heads

    # Global Importance 계산 (모든 head 합산)
    global_importance = torch.zeros(key_dim)
    for h in range(num_heads):
        qkv = cga.qkvs[h]
        dw = cga.dws[h]

        # Q, K, DW norms
        q_norms = qkv.c.weight[:key_dim].norm(dim=(1, 2, 3))
        k_norms = qkv.c.weight[key_dim:2*key_dim].norm(dim=(1, 2, 3))
        dw_norms = dw.c.weight.norm(dim=(1, 2, 3))

        global_importance += q_norms + k_norms + dw_norms

    # 통일된 keep_indices
    new_key_dim = max(4, int(key_dim * keep_ratio))
    _, keep_indices = torch.topk(global_importance, new_key_dim)
    keep_indices = keep_indices.sort().values

    # 모든 head에 동일 indices 적용
    for h in range(num_heads):
        # QKV Conv 수정
        q_weights = qkv.c.weight[:key_dim][keep_indices]
        k_weights = qkv.c.weight[key_dim:2*key_dim][keep_indices]
        v_weights = qkv.c.weight[2*key_dim:]  # V는 유지
        new_qkv_weight = torch.cat([q_weights, k_weights, v_weights])
        # ... 새 Conv 생성 및 교체

        # DW Conv 수정 (groups 조정!)
        new_dw = nn.Conv2d(new_key_dim, new_key_dim,
                          kernel_size=kernel_size, groups=new_key_dim)
        new_dw.weight.data = dw.c.weight[keep_indices]

    # CGA 속성 업데이트 (한 번만!)
    cga.key_dim = new_key_dim
    cga.scale = new_key_dim ** -0.5
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

**Prepared by**: Claude Code
**Last Updated**: 2026-03-14
