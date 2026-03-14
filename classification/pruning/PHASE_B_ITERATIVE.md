# Phase B — Iterative Physical Pruning

> **Date**: 2026-03-14 (Updated)
> **Status**: ✅ 테스트 완료, 서버 실행 준비 완료
> **Target**: 76% 압축률 (= 100×(B-A)/B %)
> **모델**: EfficientViT M4 (35.2 MB → 8.4 MB)

---

## 📋 목차

1. [Quick Start](#-quick-start)
2. [핵심 하이퍼파라미터 (중요!)](#-핵심-하이퍼파라미터-중요)
3. [두 가지 방식 비교](#-두-가지-방식-비교)
4. [버그 수정 내역](#-버그-수정-내역)
5. [파일 구조](#-파일-구조)
6. [서버 실행 가이드](#-서버-실행-가이드)
7. [예상 결과](#-예상-결과)
8. [알고리즘 상세](#-알고리즘-상세)
9. [이전 접근법 히스토리](#-이전-접근법-히스토리)

---

## 🚀 Quick Start

### 빠른 테스트 (Dummy 데이터)
```bash
cd /home/junsu/EfficientVIT_Compression

# Physical-Only 테스트
python -m classification.pruning.test_quick_pruning --mode physical

# Combined (λ + Physical) 테스트
python -m classification.pruning.test_quick_pruning --mode combined

# Pretrained 체크포인트 사용
python -m classification.pruning.test_quick_pruning --mode physical --resume efficientvit_m4.pth
```

### 실제 ImageNet 학습
```bash
# Physical-Only (권장: 빠른 실험)
python -m classification.pruning.train_physical_pruning \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --output-dir checkpoints/physical_only

# Combined (권장: 최종 결과)
python -m classification.pruning.train_combined_pruning \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --output-dir checkpoints/combined
```

---

## ⚙️ 핵심 하이퍼파라미터 (중요!)

### 공격적 Pruning 설정 (기본값)

**핵심 원칙**: 10-15 epochs 내에 76% 압축률 달성!

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `--target-reduction` | **0.76** | 목표 압축률 (76%) |
| `--ffn-prune-per-epoch` | **0.25** | FFN: 매 epoch **25%** 제거 (매우 공격적) |
| `--qk-prune-per-epoch` | **0.15** | Q/K: 매 epoch **15%** 제거 (공격적) |
| `--min-ffn-ratio` | **0.05** | FFN 최소 **5%** 유지 (최대 95% pruning) |
| `--min-qk-ratio` | **0.25** | Q/K 최소 **25%** 유지 (attention 보존) |
| `--pruning-epochs` | **15** | Pruning 진행 epoch 수 |
| `--finetune-epochs` | **10** | Pruning 후 finetune epoch 수 |

### 왜 이렇게 공격적인가?

```
FFN이 전체 파라미터의 대부분을 차지:
  - M4 기준 FFN 파라미터: ~70-80%
  - Q/K 파라미터: ~10-15%
  - 나머지: ~10-15%

따라서:
  - FFN을 매우 공격적으로 pruning (25%/epoch)
  - Q/K도 공격적으로 pruning (15%/epoch)

주의: 누적 추적은 승산(multiplicative)!
  remaining *= (1 - rate)  (가산 아님!)
```

### Pruning 진행 예측 (승산 추적, 15 epochs)

```
Epoch  | FFN 잔여율 | QK 잔여율 | 예상 압축률
-------|-----------|----------|------------
  0    |   100%    |   100%   |     0%
  1    |    75%    |    85%   |   ~12%
  3    |    42%    |    61%   |   ~35%
  5    |    24%    |    44%   |   ~55%
  8    |    10%    |    27%   |   ~70%
 10    |   5.6%    |   25%*  |   ~76%
 11    |    5%**   |   25%*  |   ~76% → TARGET

** min_ffn_ratio = 5%에서 정지
* min_qk_ratio = 25%에서 정지
```

### Combined 전용: λ 설정

**λ 비율 (중요!): FFN : QK : V = 10 : 2 : 1**

| 파라미터 | 값 | 비율 | 설명 |
|----------|-----|------|------|
| `--lambda-ffn` | **1e-3** | 10x | FFN hidden 축소 (가장 적극적) |
| `--lambda-qk` | **2e-4** | 2x | Q/K dim 축소 (조심, 이미 작음) |
| `--lambda-v` | **1e-4** | 1x | V 보존 (매우 민감!) |
| `--mu` | **1.0** | - | Memory penalty 계수 |
| `--lambda-decay` | **0.9** | - | Finetune 시 λ 감소율 |

---

## 🔥 두 가지 방식 비교

| 항목 | Physical-Only | Combined (λ + Physical) |
|------|---------------|-------------------------|
| **파일** | `train_physical_pruning.py` | `train_combined_pruning.py` |
| **Loss** | CE only | CE + λ·Σ\|\|w\|\|² + μ·mem_penalty |
| **Weights 유도** | 없음 | 0 방향으로 점진적 유도 |
| **Memory Penalty** | 없음 | ✅ μ·max(0, current - m_max) |
| **Pruning 안정성** | 갑작스러움 | 부드러움 (이미 작아진 weights 제거) |
| **튜닝 복잡도** | 낮음 | 높음 (λ, μ 튜닝 필요) |
| **권장 상황** | 빠른 실험 | 최종 결과, accuracy 중요 시 |

### Loss 함수 비교

```python
# Physical-Only
L_total = L_CE

# Combined
L_total = L_CE + λ_FFN·Σ||w_FFN||² + λ_QK·Σ||w_QK||² + λ_V·Σ||w_V||² + μ·max(0, mem - m_max)

# m_max 계산:
m_max = original_size × (1 - target_reduction)
      = 35.2 MB × (1 - 0.76)
      = 8.4 MB
```

---

## 🐛 버그 수정 내역

### 1. CGA key_dim 동기화 버그 (2026-03-14 수정)

**문제**: Q/K pruning 후 `split_with_sizes` RuntimeError 발생
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 62,
but got split_sizes=[16, 16, 32]
```

**원인**:
- 각 head를 개별적으로 pruning하면서 `key_dim` 불일치
- Head 0: key_dim=15로 pruned
- Head 1~3: key_dim=16 기대
- QKV tensor 크기: 15+15+32=62, 하지만 split은 16+16+32=64 기대

**해결**: Global Importance 기반 통합 Pruning
```python
# 수정 전: 각 head 개별 처리
for head_idx in range(num_heads):
    importance = compute_importance(head_idx)
    keep_indices = topk(importance)  # 각 head별 다른 indices

# 수정 후: 모든 head 동일 indices
global_importance = sum(compute_importance(h) for h in range(num_heads))
keep_indices = topk(global_importance)  # 모든 head에 동일 적용

# CGA.key_dim 업데이트 (한 번만!)
cga.key_dim = new_key_dim
cga.scale = new_key_dim ** -0.5
```

**수정 파일**: [structural_pruning.py](structural_pruning.py) - `prune_efficientvit_block_cga()`

### 2. Pruning Rate 부족 (2026-03-14 수정)

**문제**: 2 epochs에 6.8%만 압축 (76% 목표 대비 너무 느림)

**원인**: 보수적인 기본 pruning rate
```python
# 기존 (너무 보수적)
ffn_prune_per_epoch = 0.10  # epoch당 10%
qk_prune_per_epoch = 0.02   # epoch당 2%
min_ffn_ratio = 0.10        # 최소 10% 유지
min_qk_ratio = 0.50         # 최소 50% 유지
```

**해결**: 공격적인 pruning rate로 변경
```python
# 수정 후 (공격적)
ffn_prune_per_epoch = 0.25  # epoch당 25% (2.5배 증가)
qk_prune_per_epoch = 0.05   # epoch당 5% (2.5배 증가)
min_ffn_ratio = 0.05        # 최소 5% 유지 (2배 감소)
min_qk_ratio = 0.25         # 최소 25% 유지 (2배 감소)
```

**수정 파일**:
- [structural_pruning.py](structural_pruning.py) - `IterativePhysicalPruner.__init__()`
- [train_physical_pruning.py](train_physical_pruning.py) - argparse defaults
- [train_combined_pruning.py](train_combined_pruning.py) - argparse defaults
- [test_quick_pruning.py](test_quick_pruning.py) - argparse defaults

---

## 📁 파일 구조

```
classification/pruning/
├── structural_pruning.py         ← 핵심 physical pruning 모듈
│   ├── compute_ffn_importance()
│   ├── prune_ffn_physically()
│   ├── compute_qk_importance()
│   ├── prune_efficientvit_block_cga()  ← Global Importance 기반
│   ├── IterativePhysicalPruner (class)
│   ├── apply_iterative_physical_pruning()
│   ├── compute_model_size_mb()
│   └── validate_model_forward()
│
├── train_physical_pruning.py     ← Physical-Only (CE만) - ImageNet 학습
├── train_combined_pruning.py     ← Combined (CE + λ + μ) - ImageNet 학습
├── test_quick_pruning.py         ← 빠른 테스트 (Dummy 데이터)
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
conda create -n efficientvit python=3.10 -y
conda activate efficientvit
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.9.12 einops Pillow matplotlib tqdm
```

### Physical-Only 실행 (권장: 빠른 실험)
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
    --finetune-epochs 5 \
    --batch-size 256 \
    --lr 1e-4 \
    --output-dir checkpoints/physical_target76
```

### Combined 실행 (권장: 최종 결과)
```bash
python -m classification.pruning.train_combined_pruning \
    --model EfficientViT_M4 \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.8 \
    --ffn-prune-per-epoch 0.25 \
    --qk-prune-per-epoch 0.15 \
    --min-ffn-ratio 0.05 \
    --min-qk-ratio 0.25 \
    --lambda-ffn 0.001 \
    --lambda-qk 0.0002 \
    --lambda-v 0.0001 \
    --mu 1.0 \
    --pruning-epochs 20 \
    --finetune-epochs 10 \
    --batch-size 256 \
    --lr 1e-4 \
    --output-dir checkpoints/combined_target80
```

### 체크포인트 참고

**Q: 다시 실행할 때 체크포인트를 지워야 하나요?**

A: 아니요! 원본 pretrained weights를 사용하면 됩니다:
```bash
--resume efficientvit_m4.pth  # 항상 원본 pretrained 사용
```

출력 디렉토리만 바꾸거나 같은 디렉토리면 덮어쓰기됩니다.

### 학습 모니터링
```bash
# GPU 사용량 확인
watch -n 1 nvidia-smi

# 체크포인트 확인
ls -la checkpoints/physical_only_76pct/
```

---

## 📈 예상 결과

### Pruning Rate (공격적 설정, M4 기준)

| Epoch | FFN 잔여 | QK 잔여 | Model Size | Reduction |
|-------|---------|---------|------------|-----------|
| 0     | 100%    | 100%    | 35.2 MB    | 0%        |
| 1     | 75%     | 95%     | 32.1 MB    | ~8%       |
| 2     | 56%     | 90%     | 28.5 MB    | ~19%      |
| 5     | 24%     | 77%     | 19.4 MB    | ~45%      |
| 10    | 5%*     | 60%     | 12.3 MB    | ~65%      |
| 15    | 5%*     | 25%**   | 8.4 MB     | ~76%      |

\* min_ffn_ratio=0.05에서 정지
\** min_qk_ratio=0.25에서 정지

### Accuracy 예상

| Phase | Epochs | Acc@1 | Notes |
|-------|--------|-------|-------|
| Initial | 0 | 74.3% | Pretrained M4 |
| Pruning | 5 | ~65% | 45% 축소 |
| Pruning | 10 | ~58% | 65% 축소 |
| Pruning | 15 | ~55% | 76% 축소 (target) |
| Finetune | +5 | ~62% | 회복 시작 |
| Finetune | +10 | ~66% | 추가 회복 |
| + KD (Phase C) | +20 | ~70% | Teacher 도움 |

### 메모리 사용량

| 항목 | 예상 값 |
|------|---------|
| 모델 (초기) | ~35 MB |
| 모델 (76% 압축 후) | ~8.4 MB |
| GPU 메모리 (batch=256) | ~20-24 GB |
| GPU 메모리 (batch=128) | ~12-14 GB |

> **Note**: GPU 메모리가 부족하면 `--batch-size 128` 또는 `--batch-size 64`로 조정

---

## 📖 알고리즘 상세

### Iterative Physical Pruning 알고리즘

```
Initialize: pretrained model M₀

for epoch = 1 to N_prune:
    # Step 1: Train
    M_epoch = train_one_epoch(M_{epoch-1})

    # Step 2: Compute Importance (FFN)
    for each FFN layer:
        importance[k] = ||expand.weight[k]||₂ + ||shrink.weight[:,k]||₂

    # Step 3: Compute Importance (CGA) - GLOBAL across heads
    for each CGA:
        global_importance = Σ_h (||Q_h[d]||₂ + ||K_h[d]||₂ + ||DW_h[d]||₂)
        keep_indices = topk(global_importance)  # 모든 head 동일!

    # Step 4: Physical Pruning
    new_conv = Conv2d(embed, new_hidden)
    new_conv.weight = old_conv.weight[keep_indices]

    # Step 5: Validate
    assert forward_pass(M_epoch)

    # Step 6: Check Target
    if compression_ratio >= target:
        break

# Fine-tuning Phase (no pruning)
for epoch = 1 to N_finetune:
    train_one_epoch(M_final)
```

### FFN Physical Pruning
```python
def prune_ffn_physically(ffn, keep_ratio, min_neurons=8):
    # Importance 계산
    importance = ||expand.weight[k]||₂ + ||shrink.weight[:,k]||₂
    _, keep_indices = torch.topk(importance, new_hidden, largest=True)

    # expand 교체 (out_channels 축소)
    new_expand = Conv2d(embed_dim, new_hidden, ...)
    new_expand.weight = expand.weight[keep_indices]

    # shrink 교체 (in_channels 축소)
    new_shrink = Conv2d(new_hidden, embed_dim, ...)
    new_shrink.weight = shrink.weight[:, keep_indices]

    # BN도 함께 축소
    new_bn.weight = bn.weight[keep_indices]
    new_bn.running_mean = bn.running_mean[keep_indices]
    ...
```

### CGA Q/K Physical Pruning (Global Importance)
```python
def prune_efficientvit_block_cga(cga, keep_ratio):
    # CRITICAL: 모든 head에서 Global Importance 계산
    global_importance = torch.zeros(key_dim)
    for head_idx in range(num_heads):
        q_norms = qkv.weight[:key_dim].norm(dim=1)
        k_norms = qkv.weight[key_dim:2*key_dim].norm(dim=1)
        dw_norms = dw.weight.norm(dim=(1,2,3))
        global_importance += q_norms + k_norms + dw_norms

    # 동일 keep_indices를 모든 head에 적용
    _, keep_indices = torch.topk(global_importance, new_key_dim, largest=True)

    for head_idx in range(num_heads):
        # Q, K 동일 indices로 축소
        new_q = q_weights[keep_indices]
        new_k = k_weights[keep_indices]  # 동일 indices!
        new_qkv_weight = cat([new_q, new_k, v_weights])

        # DW conv (groups 조정)
        new_dw = Conv2d(new_key_dim, new_key_dim, groups=new_key_dim)
        new_dw.weight = dw.weight[keep_indices]

    # key_dim 한 번만 업데이트
    cga.key_dim = new_key_dim
    cga.scale = new_key_dim ** -0.5
```

---

## 📜 이전 접근법 히스토리

### Phase B v1: Soft Masking (폐기됨)

**접근법**:
```python
# 기존 방식: Weight를 0으로 설정 (물리적 크기 유지)
w_expand[pruning_idx, :, :, :] = 0.0
w_shrink[:, pruning_idx, :, :] = 0.0
```

**문제점**:
1. ❌ 실제 연산량 감소 없음 (0 × value 연산 여전히 수행)
2. ❌ 메모리 감소 없음 (tensor 크기 동일)
3. ❌ optimizer.step()에서 gradient가 0인 weight도 업데이트
4. ❌ 76% 압축 목표 달성 불가능

**결론**: Soft masking은 학습 중 weight를 0으로 유도할 수 있지만, 실제 모델 크기/연산량 감소가 없어 **폐기**

### Phase B v2: Physical Pruning (현재)

**핵심 변경**:
```python
# 새 방식: Conv2d 자체를 새 크기로 교체
new_expand = nn.Conv2d(embed_dim, new_hidden, kernel_size=1)
new_expand.weight.data = old_weight[keep_indices]
expand.c = new_expand  # 물리적 교체
```

**장점**:
1. ✅ 실제 연산량 감소 (작은 tensor로 matmul)
2. ✅ 실제 메모리 감소 (파라미터 수 감소)
3. ✅ 자연스러운 gradient (남은 weight만 학습)
4. ✅ 76% 압축 목표 달성 가능

---

## 🎯 다음 단계 (Phase C)

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

## 📊 구현 진행 상황

| 단계 | 상태 | 설명 |
|------|------|------|
| Phase A | ✅ 완료 | 모델 프로파일링, 67개 pruning group 정의 |
| Soft Masking 시도 | ✅ 완료 → 폐기 | 실제 압축 없음 확인 |
| Physical Pruning 설계 | ✅ 완료 | ViT 제약 분석, 알고리즘 설계 |
| structural_pruning.py | ✅ 완료 | FFN/CGA pruning 핵심 함수 구현 |
| CGA Global Importance 버그 수정 | ✅ 완료 | key_dim 동기화 문제 해결 |
| 공격적 Pruning 설정 | ✅ 완료 | 10-15 epochs 내 76% 달성 가능 |
| train_physical_pruning.py | ✅ 완료 | Physical-Only 학습 스크립트 |
| train_combined_pruning.py | ✅ 완료 | Combined (λ + μ + Physical) 학습 스크립트 |
| test_quick_pruning.py | ✅ 완료 | Dummy 데이터 테스트 스크립트 |
| ImageNet 실행 | ⏳ 대기 | 서버에서 실행 필요 |
| Phase C (KD) | 🔜 예정 | Knowledge Distillation 추가 |

---

**Prepared by**: Claude Code
**Last Updated**: 2026-03-14
