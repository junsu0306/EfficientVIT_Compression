# Phase B — Iterative Physical Pruning

> **Date**: 2026-03-14
> **Status**: 구현 완료, 서버 실행 대기
> **Target**: 76% 압축률 (= 100×(B-A)/B %)
> **모델**: EfficientViT M4 (35.2 MB → 8.4 MB)

---

## 📋 목차

1. [이전 접근법 (Soft Masking) 히스토리](#-이전-접근법-히스토리)
2. [새 접근법 (Physical Pruning)](#-두-가지-버전-비교)
3. [구현 진행 상황](#-구현-진행-상황)
4. [서버 실행 가이드](#-서버-실행-가이드)

---

## 📜 이전 접근법 히스토리

### Phase B v1: Soft Masking (pgm_loss.py)

**기간**: Phase A 완료 후 ~ 2026-03-14

**접근법**:
```python
# 기존 방식: Weight를 0으로 설정 (물리적 크기 유지)
w_expand[pruning_idx, :, :, :] = 0.0
w_shrink[:, pruning_idx, :, :] = 0.0

# Gradient 막기 위해 mask 적용
w.register_hook(lambda grad: grad * mask)
```

**구현한 파일들** (현재 삭제됨):
- `PHASE_B.md` — 초기 계획
- `PHASE_B_COMPLETE.md` — soft masking 구현 완료 문서
- `PHASE_B_LOGIC_PRUNING_FIX.md` — pruning index 버그 수정
- `LAMBDA_TUNING.md` — λ 튜닝 가이드
- `CRITICAL_FIX.md` — gradient hook 이슈 수정
- `IMPLEMENTATION_SUMMARY.md` — 구현 요약
- `PGM_IMPLEMENTATION.md` — PGM 알고리즘 상세

**문제점**:
1. ❌ 실제 연산량 감소 없음 (0 × value 연산 여전히 수행)
2. ❌ 메모리 감소 없음 (tensor 크기 동일)
3. ❌ optimizer.step()에서 gradient가 0인 weight도 업데이트
4. ❌ mask 관리 복잡 (pruned_mask 추적, apply 함수 매번 호출)
5. ❌ 추론 시 0인 weight 건너뛰는 별도 로직 필요

**결론**: Soft masking은 학습 중 weight를 0으로 유도할 수 있지만, 실제 모델 크기/연산량 감소가 없어 **76% 압축 목표 달성 불가능**

---

### Phase B v2: Physical Pruning (현재)

**전환 이유**: 매 epoch 물리적으로 Conv2d/Linear 크기를 축소하여 **실제 압축** 달성

**핵심 변경**:
```python
# 새 방식: Conv2d 자체를 새 크기로 교체
new_expand_conv = nn.Conv2d(embed_dim, new_hidden, kernel_size=1)
new_expand_conv.weight.data = old_weight[keep_indices]
expand.c = new_expand_conv  # 물리적 교체
```

**장점**:
1. ✅ 실제 연산량 감소 (작은 tensor로 matmul)
2. ✅ 실제 메모리 감소 (파라미터 수 감소)
3. ✅ 자연스러운 gradient (남은 weight만 학습)
4. ✅ 단순한 로직 (mask 불필요)
5. ✅ 추론 시 그대로 사용 가능

---

## 🔥 두 가지 버전 비교

| 항목 | Physical-Only | Combined (λ + μ + Physical) |
|------|---------------|------------------------------|
| **파일** | `train_physical_pruning.py` | `train_combined_pruning.py` |
| **Loss** | CE only | CE + λ·Σ\|\|w\|\|² + μ·mem_penalty |
| **Weights 유도** | 없음 | 0 방향으로 점진적 유도 |
| **Memory Penalty** | 없음 | ✅ μ·max(0, current - m_max) |
| **Pruning 안정성** | 갑작스러움 | 부드러움 (이미 작아진 weights 제거) |
| **튜닝 복잡도** | 낮음 | 높음 (λ, μ 튜닝 필요) |
| **권장 상황** | 빠른 실험 | 최종 결과 |

---

## ⚙️ 하이퍼파라미터 설정

### 공통 설정 (두 버전 모두)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--model` | `EfficientViT_M4` | 대상 모델 |
| `--num-classes` | 1000 | ImageNet 클래스 수 |
| `--input-size` | 224 | 입력 이미지 크기 |
| `--batch-size` | **256** | 배치 크기 |
| `--num-workers` | 8 | 데이터 로더 워커 수 |
| `--lr` | **1e-4** | 학습률 |
| `--min-lr` | 1e-6 | 최소 학습률 (cosine scheduler) |
| `--weight-decay` | 0.05 | AdamW weight decay |
| `--clip-grad` | 1.0 | Gradient clipping |
| `--smoothing` | 0.1 | Label smoothing |
| `--seed` | 42 | Random seed |

### Pruning 설정 (Sparsity 비율: FFN:QK:V = 10:2:1)

**핵심 원칙**: FFN을 적극적으로 pruning하고, QK/V는 보존!
- FFN: 파라미터가 많음 → 적극적으로 pruning
- QK: `key_dim=16`으로 이미 작음 → 조심스럽게
- V: 정보 손실에 민감 → Physical pruning 안 함

| 파라미터 | 기본값 | 비율 | 설명 |
|----------|--------|------|------|
| `--target-reduction` | **0.76** | - | 목표 압축률 (76%) |
| `--ffn-prune-per-epoch` | **0.10** | 10x | FFN 매 epoch 10% 제거 (적극적) |
| `--qk-prune-per-epoch` | **0.02** | 2x | Q/K 매 epoch 2% 제거 (조심) |
| `--min-ffn-ratio` | **0.10** | - | FFN 최소 10% 유지 (최대 90% pruning) |
| `--min-qk-ratio` | **0.50** | - | Q/K 최소 50% 유지 (attention 보존) |
| `--warmup-epochs` | 1 | - | Pruning 전 warmup |
| `--pruning-epochs` | **15** | - | Pruning 진행 epoch 수 |
| `--finetune-epochs` | **10** | - | Pruning 후 finetune epoch 수 |

**Sparsity 진행 예측 (15 epochs 기준)**:
```
FFN: epoch당 10% 제거 → 15 epochs 후 약 80% pruning (min 10% 유지)
QK:  epoch당 2% 제거  → 15 epochs 후 약 26% pruning (min 50% 유지)
V:   Physical pruning 없음 (λ regularization만 적용)
```

### Combined 전용 (λ regularization + Memory Penalty)

**λ 비율 (CRITICAL): FFN : QK : V = 10 : 2 : 1**

| 파라미터 | 기본값 | 비율 | 설명 |
|----------|--------|------|------|
| `--lambda-ffn` | **1e-3** | 10x | FFN hidden 축소 (가장 적극적) |
| `--lambda-qk` | **2e-4** | 2x | Q/K dim 축소 (이미 작음, 조심) |
| `--lambda-v` | **1e-4** | 1x | V 보존 (매우 민감!) |
| `--lambda-decay` | 0.9 | - | Finetune 시 λ 감소율 |
| `--mu` | **1.0** | - | Memory penalty 계수 |

**왜 이 비율인가?**
- **FFN (10x)**: 가장 파라미터가 많음, 적극적으로 pruning 가능
- **QK (2x)**: `key_dim=16`으로 이미 작음, 과도한 pruning 시 attention 붕괴
- **V (1x)**: 정보 손실에 매우 민감, 최소한의 regularization

### Data Augmentation (build_dataset 호환)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--color-jitter` | 0.4 | Color jitter 강도 |
| `--aa` | `rand-m9-mstd0.5-inc1` | AutoAugment 정책 |
| `--train-interpolation` | `bicubic` | 이미지 보간법 |
| `--reprob` | 0.25 | Random erase 확률 |
| `--remode` | `pixel` | Random erase 모드 |
| `--recount` | 1 | Random erase 횟수 |

### 예상 학습 시간 (V100 32GB 기준)

| 단계 | Epoch 수 | 예상 시간 |
|------|----------|-----------|
| Warmup | 1 | ~30분 |
| Pruning | 15 | ~7시간 |
| Finetune | 10 | ~5시간 |
| **총계** | **26** | **~12-13시간** |

### 메모리 사용량

| 항목 | 예상 값 |
|------|---------|
| 모델 (초기) | ~35 MB |
| 모델 (76% 압축 후) | ~8.4 MB |
| GPU 메모리 (batch=256) | ~20-24 GB |
| GPU 메모리 (batch=128) | ~12-14 GB |

> **Note**: GPU 메모리가 부족하면 `--batch-size 128` 또는 `--batch-size 64`로 조정

### 실행 명령어

```bash
# Physical-Only (CE만 사용)
python -m classification.pruning.train_physical_pruning \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --output-dir checkpoints/physical_only

# Combined (λ + Memory Penalty + Physical) — 권장
python -m classification.pruning.train_combined_pruning \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --lambda-ffn 0.001 \
    --lambda-qk 0.0002 \
    --lambda-v 0.0001 \
    --mu 1.0 \
    --output-dir checkpoints/combined
```

**λ 비율 확인**: `--lambda-ffn 0.001 : --lambda-qk 0.0002 : --lambda-v 0.0001 = 10:2:1`

---

## 0. Combined 방식의 Loss 함수

```python
# Combined Loss (train_combined_pruning.py)
L_total = L_CE + λ_FFN·Σ||w_FFN||² + λ_QK·Σ||w_QK||² + λ_V·Σ||w_V||² + μ·max(0, mem - m_max)

# 각 항의 역할:
# 1. L_CE: 분류 정확도 유지
# 2. λ·||w||²: weights를 0 방향으로 유도 (gradient에 -2λw 추가)
# 3. μ·mem_penalty: 목표 메모리 초과 시 패널티

# m_max 계산:
m_max = original_size × (1 - target_reduction)
      = 35.2 MB × (1 - 0.76)
      = 8.4 MB
```

### λ 권장값 (M4 기준, 비율 10:2:1)

| Parameter | Value | 비율 | 역할 |
|-----------|-------|------|------|
| λ_FFN | **1e-3** | 10x | FFN hidden 축소 (적극적) |
| λ_QK | **2e-4** | 2x | Q/K dim 축소 (조심, 이미 작음) |
| λ_V | **1e-4** | 1x | V 보존 (매우 민감!) |
| μ | **1.0** | - | Memory penalty 계수 |
| λ_decay | 0.9 | - | Finetune 시 λ 감소 (매 epoch ×0.9) |

### Physical-Only: 목표 달성 메커니즘

```
Physical-Only는 λ regularization 없이 매 epoch importance 기반으로 직접 pruning:

1. target_reduction = 0.76 설정
2. IterativePhysicalPruner가 현재 모델 크기를 추적
3. 매 epoch:
   - current_reduction = 1 - (current_size / original_size) 계산
   - remaining = target_reduction - current_reduction
   - remaining > 0이면 pruning 계속
   - remaining ≤ 0이면 목표 도달, pruning 중단

4. Pruning rate 조절:
   - ffn_prune_per_epoch = 0.08 (8%)
   - qk_prune_per_epoch = 0.10 (10%)
   - min_ffn_ratio = 0.15 (최소 15% 유지)
   - min_qk_ratio = 0.20 (최소 20% 유지)

5. 목표 도달 시 자동 중단:
   pruner.target_reached = True
```

### Combined: 목표 달성 메커니즘

```
Combined는 λ + μ·memory_penalty + Physical pruning 조합:

1. λ regularization:
   - weights를 점진적으로 0에 가깝게 유도
   - 불필요한 units가 먼저 작아짐

2. μ·memory_penalty:
   - current_size > m_max이면 추가 패널티
   - 목표 압축률을 loss에서 강제

3. Physical pruning (epoch 끝):
   - 이미 작아진 weights 제거
   - 급격한 accuracy 손실 방지

4. λ decay (finetune phase):
   - Pruning 완료 후 λ *= 0.9 (매 epoch)
   - Regularization 점점 약해짐 → accuracy 회복에 집중
```

### 예상 Loss 값

```
초기 loss가 13이라면:

구성:
  - CE loss: ~7 (ImageNet 초기)
  - λ regularization: ~5-6 (weights가 클 때)
  - memory penalty: ~0 (아직 m_max 미만)

Epoch 진행에 따라:
  - CE loss: 7 → 6 → 5 (학습)
  - λ reg: 6 → 3 → 1 (weights 감소)
  - mem penalty: 0 → 0.5 → 0 (목표 근접 후 감소)

총 loss: 13 → 9.5 → 6 (점진적 감소)
```

---

## 1. 문제점: Soft Masking의 한계

### 기존 방식 (pgm_loss.py)
```python
# Weight를 0으로 설정
w_expand[pruning_idx, :, :, :] = 0.0
w_shrink[:, pruning_idx, :, :] = 0.0
```

**문제점:**
1. **연산량 그대로**: 0인 weight도 matmul에 포함됨
2. **메모리 그대로**: 모델 크기 변화 없음
3. **Gradient 복원**: optimizer.step()에서 0이 아닌 값으로 업데이트됨
4. **복잡한 mask 관리**: pruned_mask 추적, apply_pruned_mask 매 iter 호출 필요

### 새 방식 (structural_pruning.py)
```python
# Conv2d 자체를 새로운 크기로 교체
new_expand_conv = nn.Conv2d(embed_dim, new_hidden, kernel_size=1, bias=False)
new_expand_conv.weight.data = expand.c.weight.data[keep_indices]
expand.c = new_expand_conv  # 물리적 교체
```

**장점:**
1. **실제 연산량 감소**: 작은 텐서로 matmul
2. **실제 메모리 감소**: 파라미터 수 감소
3. **Gradient 자연스럽게 적용**: 남은 weight만 학습
4. **단순한 로직**: mask 관리 불필요

---

## 2. 알고리즘: Iterative Physical Pruning

```
Initialize: pretrained model M₀

for epoch = 1 to N_prune:
    # Step 1: Train
    M_epoch = train_one_epoch(M_{epoch-1})  # Loss 감소

    # Step 2: Compute Importance
    for each FFN layer:
        importance[k] = ||expand.weight[k]||₂ + ||shrink.weight[:,k]||₂
    for each CGA head:
        importance[d] = ||Q[d]||₂ + ||K[d]||₂ + ||DW[d]||₂

    # Step 3: Physical Pruning
    keep_indices = top_k(importance, k=target_dim)
    new_expand = Conv2d(embed, new_hidden)
    new_expand.weight = old_expand.weight[keep_indices]
    # ... 모든 관련 layer 동시 축소

    # Step 4: Validate
    assert forward_pass(M_epoch)  # Shape 일관성 확인

    # Step 5: Check Target
    if compression_ratio >= target:
        break

# Fine-tuning Phase
for epoch = 1 to N_finetune:
    train_one_epoch(M_final)
```

---

## 3. ViT Pruning의 특수 고려사항

### 3.1 FFN Pruning
```
FFN 구조:
    Input [B, C, H, W]
        → expand (Conv2d: C → hidden)
        → ReLU
        → shrink (Conv2d: hidden → C)
    Output [B, C, H, W]
```

**제약:**
- `expand.out_channels == shrink.in_channels` 항상 만족
- BN도 함께 축소 (weight, bias, running_mean, running_var)

```python
# expand 축소
new_expand.weight = expand.weight[keep_indices]
new_expand_bn.weight = expand.bn.weight[keep_indices]
new_expand_bn.running_mean = expand.bn.running_mean[keep_indices]
...

# shrink 축소 (입력 채널 방향)
new_shrink.weight = shrink.weight[:, keep_indices]
```

### 3.2 CGA Q/K Pruning
```
CGA 구조 (per head):
    Input [B, C/H, H, W]
        → qkv (Conv2d: C/H → key_dim*2 + d)
           ↓
           Q [key_dim], K [key_dim], V [d]
        → DW on Q (Conv2d: key_dim → key_dim, groups=key_dim)
        → Attention: softmax(Q^T K / sqrt(d_k))
        → Output projection
```

**제약 (CRITICAL):**
- **Q와 K는 동일한 indices로 축소** (QK^T 연산)
- DW conv도 동일 indices로 축소 (groups=key_dim → groups=new_key_dim)
- BN 모두 동기화

```python
# Q/K 동일 indices
keep_qk = topk(q_norms + k_norms + dw_norms)

# qkv weight 재구성
new_q = qkv.weight[:key_dim][keep_qk]
new_k = qkv.weight[key_dim:2*key_dim][keep_qk]  # 동일 indices!
new_v = qkv.weight[2*key_dim:]  # V는 유지
new_qkv.weight = cat([new_q, new_k, new_v])

# DW conv (groups 조정 필수)
new_dw = Conv2d(new_key_dim, new_key_dim, groups=new_key_dim)
new_dw.weight = dw.weight[keep_qk]
```

### 3.3 V Pruning (복잡)
```
V → proj 연결:
    각 head의 V output이 concat되어 proj에 입력
    proj input = [V₀, V₁, ..., V_{H-1}]  (dim = d * num_heads)
```

**어려움:**
- V를 줄이면 proj의 입력 채널도 줄여야 함
- 모든 head의 V를 동일 비율로 줄이거나, head별로 다르게 줄이면 proj 재구성 필요

**현재 구현**: V는 보존 (QK만 pruning)
- V는 정보량이 높아 pruning 시 accuracy 손실 큼
- 향후 필요 시 확장 가능

---

## 📈 구현 진행 상황

### 완료된 작업

| 단계 | 상태 | 설명 |
|------|------|------|
| Phase A | ✅ 완료 | 모델 프로파일링, 67개 pruning group 정의 |
| Soft Masking 시도 | ✅ 완료 → 폐기 | 실제 압축 없음 확인 |
| Physical Pruning 설계 | ✅ 완료 | ViT 제약 분석, 알고리즘 설계 |
| structural_pruning.py | ✅ 완료 | FFN/CGA pruning 핵심 함수 구현 |
| train_physical_pruning.py | ✅ 완료 | Physical-Only 학습 스크립트 |
| train_combined_pruning.py | ✅ 완료 | Combined (λ + μ + Physical) 학습 스크립트 |
| 버그 수정 | ✅ 완료 | Import 경로, 누락 인자 수정 |
| Memory Penalty 추가 | ✅ 완료 | μ·max(0, current - m_max) 항 추가 |
| λ 비율 조정 | ✅ 완료 | FFN:QK:V = 10:2:1 (Q/K 보호, V 보존) |

### 구현 타임라인

```
2026-03-14:
├─ 09:00  Phase B Soft Masking 한계 분석
├─ 10:00  Physical Pruning 설계 시작
├─ 11:00  structural_pruning.py 구현
│         ├─ compute_ffn_importance()
│         ├─ prune_ffn_physically()
│         ├─ compute_qk_importance()
│         └─ prune_cga_head_qk_physically()
├─ 12:00  train_physical_pruning.py (Physical-Only) 구현
├─ 13:00  train_combined_pruning.py (Combined) 구현
├─ 14:00  버그 수정
│         ├─ ImportError: EfficientViT_M4 경로
│         └─ AttributeError: build_dataset 인자 누락
├─ 15:00  기존 Phase B MD 파일 정리 (7개 삭제)
├─ 16:00  Memory Penalty 추가 (μ·max(0, mem - m_max))
├─ 16:30  λ 비율 조정 (FFN:QK:V = 10:2:1)
└─ 현재   서버 실행 대기
```

### 다음 단계

| 단계 | 상태 | 설명 |
|------|------|------|
| Physical-Only 실행 | ⏳ 대기 | 서버에서 ImageNet 학습 |
| Combined 실행 | ⏳ 대기 | λ regularization 효과 비교 |
| 결과 비교 분석 | ⏳ 대기 | 두 방식 accuracy/compression 비교 |
| Phase C (KD) | 🔜 예정 | Knowledge Distillation 추가 |

---

## 4. 파일 구조

```
classification/pruning/
├── structural_pruning.py         ← 핵심 physical pruning 모듈
│   ├── compute_ffn_importance()
│   ├── prune_ffn_physically()
│   ├── compute_qk_importance()
│   ├── prune_cga_head_qk_physically()
│   ├── IterativePhysicalPruner (class)
│   └── validate_model_forward()
│
├── train_physical_pruning.py     ← Physical-Only (CE만)
│   └── train_one_epoch_physical()
│
├── train_combined_pruning.py     ← Combined (CE + λ + μ)
│   ├── compute_lambda_regularization()
│   ├── compute_memory_penalty()
│   └── train_one_epoch_combined()
│
├── pgm_loss.py                   ← 기존 soft masking (참고용)
├── group_dict.py                 ← 67개 pruning group 정의
├── memory_utils.py               ← 메모리 측정 유틸리티
│
└── PHASE_B_ITERATIVE.md          ← 이 문서
```

---

## 🚀 서버 실행 가이드

### 환경 설정 (최초 1회)
```bash
# Conda 환경 생성
conda create -n efficientvit python=3.10 -y
conda activate efficientvit

# PyTorch 설치 (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 의존성 설치
pip install timm==0.9.12 einops Pillow matplotlib tqdm

# 작업 디렉토리 이동
cd /workspace/etri_iitp/JS/EfficientViT  # 서버 경로
```

### GPU 선택
```bash
# 단일 GPU 지정 (예: GPU 0번)
export CUDA_VISIBLE_DEVICES=0

# 또는 --device 인자로 지정
python -m ... --device cuda:0
```

### 5.1 Physical-Only (CE만 사용)
```bash
# 기본 실행 (76% 압축)
python -m classification.pruning.train_physical_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --pruning-epochs 15 \
    --finetune-epochs 10 \
    --ffn-prune-per-epoch 0.08 \
    --qk-prune-per-epoch 0.10 \
    --batch-size 256 \
    --lr 1e-4 \
    --output-dir classification/pruning/checkpoints/physical_only_76pct

# 빠른 테스트 (50% 압축)
python -m classification.pruning.train_physical_pruning \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.50 \
    --pruning-epochs 2 \
    --finetune-epochs 2 \
    --output-dir classification/pruning/checkpoints/physical_only_50pct_test
```

### 5.2 Combined (λ + μ + Physical) — 권장
```bash
# 기본 실행 (76% 압축, λ 비율 10:2:1)
python -m classification.pruning.train_combined_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --lambda-ffn 0.001 \
    --lambda-qk 0.0002 \
    --lambda-v 0.0001 \
    --mu 1.0 \
    --pruning-epochs 2 \
    --finetune-epochs 2 \
    --batch-size 256 \
    --lr 1e-4 \
    --output-dir classification/pruning/checkpoints/combined_76pct

# λ 약하게 (더 보수적 pruning)
python -m classification.pruning.train_combined_pruning \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --lambda-ffn 0.0005 \
    --lambda-qk 0.0001 \
    --lambda-v 0.00005 \
    --mu 0.5 \
    --output-dir classification/pruning/checkpoints/combined_conservative
```

### 5.3 Standalone 테스트 (데이터 없이 구조만 확인)
```bash
cd /workspace/etri_iitp/JS/EfficientViT
python -m classification.pruning.structural_pruning
```

### 5.4 학습 모니터링
```bash
# 실시간 로그 확인
tail -f classification/pruning/checkpoints/physical_only_76pct/train.log

# 체크포인트 확인
ls -la classification/pruning/checkpoints/physical_only_76pct/

# GPU 사용량 확인
watch -n 1 nvidia-smi
```

---

## 6. 예상 결과

### Pruning Rate 예시 (M4 기준)

| Epoch | FFN Keep | QK Keep | Model Size | Reduction |
|-------|----------|---------|------------|-----------|
| 0     | 100%     | 100%    | 35.2 MB    | 0%        |
| 1     | 92%      | 90%     | 32.1 MB    | 8.8%      |
| 2     | 85%      | 81%     | 29.2 MB    | 17.0%     |
| 5     | 64%      | 59%     | 22.1 MB    | 37.2%     |
| 10    | 40%      | 35%     | 14.5 MB    | 58.8%     |
| 15    | 22%      | 20%     | 8.4 MB     | 76.1%     |

### Accuracy 예상

| Phase | Epochs | Acc@1 | Notes |
|-------|--------|-------|-------|
| Initial | 0 | 74.3% | Pretrained M4 |
| Pruning | 5 | 68% | 40% 축소 |
| Pruning | 10 | 62% | 60% 축소 |
| Pruning | 15 | 58% | 76% 축소 (target) |
| Finetune | +5 | 64% | 회복 시작 |
| Finetune | +10 | 68% | 추가 회복 |
| + KD (Phase C) | +20 | 72% | Teacher 도움 |

---

## 7. Soft Masking vs Physical Pruning 비교

| 항목 | Soft Masking | Physical Pruning |
|------|--------------|------------------|
| **실제 연산 감소** | ❌ 없음 | ✅ 실제 감소 |
| **메모리 감소** | ❌ 없음 | ✅ 실제 감소 |
| **Gradient 처리** | 복잡 (mask 필요) | 자연스러움 |
| **구현 복잡도** | 높음 | 중간 |
| **Optimizer 재생성** | 불필요 | 필요 (파라미터 변경) |
| **Forward 검증** | 불필요 | 필요 (shape 확인) |
| **ViT 적용** | 어려움 | 가능 (제약 처리) |

---

## 8. 핵심 코드 설명

### 8.1 FFN Physical Pruning
```python
def prune_ffn_physically(ffn, keep_ratio, min_neurons=8):
    # Importance 계산
    importance = compute_ffn_importance(expand, shrink)
    _, keep_indices = torch.topk(importance, new_hidden, largest=True)

    # expand 교체
    new_expand_conv = nn.Conv2d(embed_dim, new_hidden, ...)
    new_expand_conv.weight.data = expand.c.weight.data[keep_indices]
    expand.c = new_expand_conv

    # shrink 교체 (입력 채널 축소)
    new_shrink_conv = nn.Conv2d(new_hidden, embed_dim, ...)
    new_shrink_conv.weight.data = shrink.c.weight.data[:, keep_indices]
    shrink.c = new_shrink_conv
```

### 8.2 CGA Q/K Physical Pruning
```python
def prune_cga_head_qk_physically(cga, head_idx, keep_ratio):
    # Q + K + DW 합산 importance
    importance = compute_qk_importance(qkv, dw, key_dim)
    _, keep_indices = torch.topk(importance, new_key_dim, largest=True)

    # Q, K 동일 indices로 축소
    new_q = q_weights[keep_indices]
    new_k = k_weights[keep_indices]  # CRITICAL: 동일 indices
    new_qkv_weight = torch.cat([new_q, new_k, v_weights])

    # DW conv (groups 조정)
    new_dw_conv = nn.Conv2d(new_key_dim, new_key_dim, groups=new_key_dim)
    new_dw_conv.weight.data = dw.c.weight[keep_indices]
```

### 8.3 Iterative Pruner
```python
class IterativePhysicalPruner:
    def step(self, model, device):
        # 목표 도달 확인
        if self.target_reached:
            return

        # 남은 pruning 여유 계산
        ffn_rate = min(self.ffn_prune_per_epoch, remaining_ffn)
        qk_rate = min(self.qk_prune_per_epoch, remaining_qk)

        # Physical pruning 적용
        apply_iterative_physical_pruning(model, ffn_rate, qk_rate)

        # Forward 검증
        validate_model_forward(model, device)
```

---

## 9. 다음 단계 (Phase C)

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

### Phase C 구현 위치
```
classification/pruning/
├── train_iterative_pruning.py      ← Phase B (현재)
├── train_iterative_pruning_kd.py   ← Phase C (KD 추가)
```

---

## 10. 결론

**Iterative Physical Pruning**은 ViT pruning의 현실적인 접근법:

1. **실제 모델 축소**: 연산량, 메모리 모두 감소
2. **간단한 학습 루프**: mask 관리 불필요
3. **ViT 제약 처리**: Q/K 동기화, DW groups 조정
4. **점진적 축소**: epoch 단위로 안정적 pruning

76% 목표 달성을 위한 권장 설정:
- FFN prune/epoch: 8%
- QK prune/epoch: 10%
- Pruning epochs: 15
- Finetune epochs: 10
- + KD (Phase C): 20 epochs

---

**Prepared by**: Claude Code
**Date**: 2026-03-14
