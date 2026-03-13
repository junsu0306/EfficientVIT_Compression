# EfficientViT PGM Pruning — Claude Code Reference

> **Project**: RS-2024-00339187 | 고려대학교 | 3차년도 ViT 확장 연구  
> **Goal**: EfficientViT에 PGM 기반 Structured Pruning 적용 → 모델 최적화율 **76% 달성**  
> **Optimization rate formula**: `100 × (B - A) / B` (B: 원본 모델 크기, A: 압축 후 크기)

---

## 1. Architecture Map

```
Input [B, 3, H, W]
  └─ OverlapPatchEmbed          → [B, C1, H/16, W/16]  (3× Conv-BN-ReLU)
       └─ Stage 1 (×L1)
            EfficientViTBlock
       └─ SubsampleBlock        → H/32, C1→C2
       └─ Stage 2 (×L2)
            EfficientViTBlock
       └─ SubsampleBlock        → H/64, C2→C3
       └─ Stage 3 (×L3)
            EfficientViTBlock
       └─ AvgPool + Linear      → logits
```

### Model Variants (C1,C2,C3 / L1,L2,L3 / H1,H2,H3)

| Model | Channels      | Blocks  | Heads   | Top-1 | Params |
|-------|---------------|---------|---------|-------|--------|
| M2    | 128, 192, 224 | 1, 2, 3 | 4, 3, 2 | 70.8% | 4.2M   |
| M4    | 128, 256, 384 | 1, 2, 3 | 4, 4, 4 | 74.3% | 8.8M   |
| M5    | 192, 288, 384 | 1, 3, 4 | 3, 3, 4 | 77.1% | 12.4M  |

---

## 2. EfficientViTBlock Internal Structure

```
Input X  [B, N, C]
  │
  ├─ DWConv TokenInteraction(C)       ← local inductive bias, groups=C
  │   └─ (+residual)
  │
  ├─ FFN_pre  ×N                      ← default N=1 (best by ablation)
  │   Linear(C → C*r) → BN → ReLU → Linear(C*r → C)   r=2
  │   └─ (+residual)
  │
  ├─ DWConv TokenInteraction(C)
  │
  ├─ CGA  (Cascaded Group Attention)  ← 단 1개
  │   └─ (+residual)
  │
  ├─ DWConv TokenInteraction(C)
  │
  └─ FFN_post ×N
      └─ (+residual)
Output X'  [B, N, C]
```

**Key design choices (all relevant to pruning)**
- BN instead of LN → BN can be folded into preceding linear at inference
- ReLU instead of GELU/HardSwish → faster, ONNX friendly
- N=1 FFN is optimal — increasing N hurts (ablation: N=2 → -1.1% acc, N=3 → -5.6%)

---

## 3. CGA Internals

```
Input X_i  [B, N, C]
  │
  └─ channel-wise split into h parts:  X_i = [X_i1 | ... | X_ih],  each [B, N, C/h]

  Head j=1:
    X_i1  ──→  DWConv  ──→  Q_proj(C/h → d_qk)
           ──────────────→  K_proj(C/h → d_qk)
           ──────────────→  V_proj(C/h → C/h)
                             Attn(Q,K,V)  ──→  X̃_i1

  Head j≥2:
    X_ij + X̃_i(j-1)  ──→  Q/K/V projections  ──→  Attn  ──→  X̃_ij
    ↑
    cascade: previous head output added to current head input

  Concat[X̃_i1,...,X̃_ih]  ──→  Linear(C → C)  ──→  output
```

### Q/K/V Sizes (M4 example, Stage 1: C=128, H=4)

| Projection | Shape          | d_qk / dim | Importance | λ setting |
|------------|----------------|------------|------------|-----------|
| Q          | C/H × d_qk = 32×16 | 16    | LOW        | λ_QK (large) |
| K          | C/H × d_qk = 32×16 | 16    | LOW        | λ_QK (large) |
| V          | C/H × C/H  = 32×32 | 32    | HIGH       | λ_V  (small) |
| Out proj   | C × C = 128×128    | —     | CRITICAL   | FIXED (no pruning) |

---

## 4. Pruning Groups

Every pruning "group" = set of parameters that must be zeroed together to maintain channel alignment.

| Group ID  | Parameters included                             | Removed when zero          | λ         |
|-----------|-------------------------------------------------|----------------------------|-----------|
| G_FFN     | expand.weight[k,:] + shrink.weight[:,k]         | hidden neuron k            | λ_FFN     |
| G_QK      | Q.weight[d,:] + K.weight[d,:]  (same head, same d) | attention dim d         | λ_QK      |
| G_V       | V.weight[c,:]                                   | V output channel c         | λ_V       |
| G_PATCH   | consecutive Conv filters in PatchEmbed          | low-level feature f        | λ_FFN     |

### Hard constraints (never violate)
1. **FFN**: `expand.out_features == shrink.in_features` always
2. **QK**: Q and K must use **identical** output dimension indices (QK^T requires same dim)
3. **V**: V.out_features == C/h; cascade addition requires consistent shape
4. **Output proj**: NEVER prune — breaks channel alignment for the entire block
5. **Subsample block channels**: These define stage boundary dimensions → fix in Phase 1, prune later

---

## 5. PGM Algorithm

### 5.1 Loss Function

```
L_total = L_task
        + λ_QK  · Σ_{g∈G_QK}  ‖w_g‖₂
        + λ_FFN · Σ_{g∈G_FFN} ‖w_g‖₂
        + λ_V   · Σ_{g∈G_V}   ‖w_g‖₂
        + μ · max(0, current_memory_bytes - M_max_bytes)
```

With KD (recommended):
```
L_total += α · KL( softmax(z_T/T) ‖ log_softmax(z_S/T) ) · T²
```

### 5.2 Update Steps per Iteration

```
# Step 1 — Gradient Descent  (done by optimizer.step())
w^{k+1/2} = w^k - η · ∇L_total(w^k)

# Step 2 — Group Soft Thresholding  (run AFTER optimizer.step())
for each group g:
    norm_g = ‖w_g^{k+1/2}‖₂
    if norm_g > η · λ_g:
        w_g^{k+1} = w_g^{k+1/2} · (1 - η·λ_g / norm_g)   # shrink toward zero
    else:
        w_g^{k+1} = 0                                        # entire group zeroed → will be removed
```

### 5.3 Hyperparameters

| Param     | Recommended | Search range   | Effect                          |
|-----------|-------------|----------------|---------------------------------|
| λ_QK      | 0.010       | 0.005–0.020    | Q/K dim removal speed           |
| λ_FFN     | 0.005       | 0.002–0.010    | FFN hidden removal speed        |
| λ_V       | 0.001       | 0.0005–0.003   | V preservation (keep low)       |
| μ         | 1.0         | 0.5–2.0        | Memory constraint enforcement   |
| M_max     | orig × 0.24 | orig × 0.20–0.35 | Directly controls opt. rate   |
| lr (η)    | 1e-4        | 5e-5–5e-4      | Convergence stability           |
| α (KD)    | 0.5         | 0.3–0.7        | Distillation strength           |
| T (KD)    | 4.0         | 2.0–6.0        | Soft label temperature          |
| epochs    | 100 (ablation) / 300 (full) | —  | —                         |

**λ diagnosis**: After 10 epochs, zero-group ratio should be 0–10%. If >20%, λ is too large.

---

## 6. Structure Extraction (Sparse → Dense)

After training, convert sparse weights to a smaller dense model.

### FFN extraction
```python
# identify zero rows in expand.weight
row_norms = expand.weight.norm(dim=1)               # [hidden_dim]
keep = (row_norms >= threshold).nonzero().squeeze()  # surviving indices

new_expand = Linear(C, len(keep))
new_expand.weight.data = expand.weight.data[keep]
new_expand.bias.data   = expand.bias.data[keep]

new_shrink = Linear(len(keep), C)
new_shrink.weight.data = shrink.weight.data[:, keep]
new_shrink.bias.data   = shrink.bias.data.clone()
```

### Q/K extraction
```python
# CRITICAL: use Q's zero indices for BOTH Q and K
q_norms  = Q.weight.norm(dim=1)
keep_qk  = (q_norms >= threshold).nonzero().squeeze()  # derive from Q only

# Apply same keep_qk to Q and K
new_Q.weight.data = Q.weight.data[keep_qk]
new_K.weight.data = K.weight.data[keep_qk]   # must be identical index set
```

### Validation checklist (run after every extraction)
```python
assert new_expand.out_features == new_shrink.in_features   # FFN alignment
assert new_Q.out_features == new_K.out_features            # QK dim match
assert dwconv.in_channels == dwconv.out_channels == dwconv.groups  # DW sync
_ = model(torch.zeros(1, 3, 224, 224))                     # full forward pass
opt_rate = 100 * (size_B - size_A) / size_B
assert opt_rate >= 76.0, f"Target not met: {opt_rate:.1f}%"
```

---

## 7. Memory Profiling (from Year-1 methodology)

```python
# Use torch.cuda.memory_stats — NOT nvidia-smi (includes reserved memory)
torch.cuda.reset_peak_memory_stats(device)
torch.cuda.empty_cache()
mem_before = torch.cuda.memory_allocated(device)

# ... run target layer or create dummy parameter ...

mem_after = torch.cuda.memory_allocated(device)
per_filter_memory_bytes = mem_after - mem_before
```

Expected pattern (matches report Figure 5):
- **Early stages** (Stage 1): higher per-filter memory preserved → prune conservatively
- **Later stages** (Stage 3): more redundancy → prune aggressively
- **Q/K**: lowest memory contribution, most redundant
- **V**: moderate, preserve as much as possible

---

## 8. Experiment Phases

| Phase | Target model     | What to do                              | Success criterion        |
|-------|------------------|-----------------------------------------|--------------------------|
| A     | M4               | Profile all layers, build group dict    | Memory map complete      |
| B     | M4               | PGM only (no KD), sweep λ              | Identify best λ set      |
| C     | M4 (Teacher: M5) | PGM + KD, push to 76%                  | opt_rate ≥ 76%           |
| D     | M4 vs YOLOv8n    | Compare CNN vs ViT pruning results      | Generality validated     |
| E     | M4               | Ablation: G_FFN only / +G_QK / +G_V   | Best grouping confirmed  |

---

## 9. Parameter Importance Summary

```
REMOVE aggressively   ████████████  Q projection   (d_qk already small, high λ)
                      ████████████  K projection   (must match Q dim, high λ)
REMOVE moderately     ████████░░░░  FFN hidden     (r=2 already halved)
PRESERVE mostly       ████░░░░░░░░  V projection   (low λ, check before removing)
NEVER REMOVE          ████████████  Output proj    (channel alignment critical)
PHASE 2+ only         ████░░░░░░░░  Subsample channels / Head-level removal
```

---

## 10. Key Numbers to Remember

| Item                          | Value                  |
|-------------------------------|------------------------|
| Year-3 optimization target    | **76%**                |
| Optimization rate formula     | `100×(B-A)/B`          |
| Best FFN expansion ratio      | r = 2 (not 4)          |
| Best MHSA:FFN ratio           | 20–40% MHSA            |
| Optimal N (FFN count)         | N = 1                  |
| DWConv removal accuracy drop  | −1.4% (avoid removing) |
| BN→LN swap accuracy drop      | −0.9%                  |
| Q/K optimal d_qk (M4)        | 16                     |
| V optimal ratio to embed dim  | 1.0 (= C/h)            |
| Year-2 KD result (15% compress)| mAP drop: −1.8% → −0.2% with KD |

---

## 11. 구현 진행 현황

### 완료된 작업

#### Phase A — 코드 완성 (서버 실행 대기 중)

**생성된 파일:**
```
classification/pruning/
├── __init__.py
├── group_dict.py        ← build_pruning_groups(model) : 67개 그룹 추출
├── memory_utils.py      ← compute_active_param_memory(), count_zero_groups()
├── phase_a_profile.py   ← 실행 스크립트 (argparse, JSON 리포트 생성)
├── PHASE_A.md           ← 서버 실행 명령어 + 환경 설정 문서
└── reports/             ← phase_a_profile.py 실행 후 JSON 저장 위치
```

**group_dict.py 핵심 구조:**
- `build_pruning_groups(model)` → list[dict] 반환
- 각 그룹 dict: `{id, type, lambda_rec, unit_count, modules, meta}`
- `modules` 안에 실제 `nn.Module` 참조 포함 → Phase B soft thresholding에서 직접 사용
- M4 기준 총 67개 그룹: G_PATCH 3 + G_FFN 20 + G_QK 24 + G_V 24

**M4 blocks 인덱스 (중요, 코드 수정 시 참고):**
```
blocks1: [EVBlock(ed=128)]
blocks2: [SubPreDWFFN(128), PatchMerging(128→256), SubPostDWFFN(256),
          EVBlock(256), EVBlock(256)]
blocks3: [SubPreDWFFN(256), PatchMerging(256→384), SubPostDWFFN(384),
          EVBlock(384), EVBlock(384), EVBlock(384)]

SubPreDWFFN  = Sequential(Residual(DWConv), Residual(FFN))
FFN 접근법   : blocks2[0][1].m  (Residual.m → FFN)
```

**memory_utils.py 핵심 함수:**
- `compute_active_param_memory(groups)` → 활성 unit만 계산 (PGM loss의 current_memory_bytes)
- `profile_gpu_memory(model, device)` → CLAUDE.md §7 방법으로 실제 GPU 메모리 측정
- `count_zero_groups(groups)` → type별 zero ratio 모니터링 (λ 진단에 사용)

---

### 서버에서 해야 할 일 (Phase A 완료 조건)

```bash
# 1. 환경 설정 (PHASE_A.md 참고)
conda create -n efficientvit python=3.10 -y
conda activate efficientvit
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.9.12 einops Pillow matplotlib tqdm ipykernel

# 2. Phase A 실행
cd /workspace/etri_iitp/JS/EfficientViT
python -m classification.pruning.phase_a_profile \
    --device cuda:0 \
    --pretrained EfficientViT_M4 \
    --output classification/pruning/reports/phase_a_report.json

# GPU 0번을 사용하도록 --device cuda:0으로 설정되었습니다.

# 3. 결과 확인 → M_max 값 기록
python -c "
import json
r = json.load(open('classification/pruning/reports/phase_a_report.json'))
print('M_max:', r['compression']['m_max_mb'], 'MB')
print('Total groups:', r['n_groups_total'])
"
```

Phase A 성공 기준: `phase_a_report.json` 생성, 총 67개 그룹, M_max 값 확인

---

### Phase B — Physical Pruning (구현 완료, 서버 실행 대기)

> **방식 변경**: Soft Masking → Physical Pruning
> **이유**: Soft masking은 weight를 0으로만 설정하여 실제 모델 크기/연산량 감소가 없음

#### 구현된 파일

```
classification/pruning/
├── structural_pruning.py         ← 핵심 physical pruning 모듈
│   ├── compute_ffn_importance()      - FFN importance 계산
│   ├── prune_ffn_physically()        - FFN 물리적 축소
│   ├── compute_qk_importance()       - Q/K importance 계산
│   ├── prune_cga_head_qk_physically() - CGA Q/K 물리적 축소
│   ├── IterativePhysicalPruner       - epoch별 pruning 관리
│   └── validate_model_forward()      - shape 검증
│
├── train_physical_pruning.py     ← Physical-Only (CE만)
├── train_combined_pruning.py     ← Combined (CE + λ regularization)
│
├── pgm_loss.py                   ← 기존 soft masking (참고용, deprecated)
├── group_dict.py                 ← 67개 pruning group 정의
├── memory_utils.py               ← 메모리 측정 유틸리티
│
└── PHASE_B_ITERATIVE.md          ← 상세 문서
```

#### 두 가지 버전

| 항목 | Physical-Only | Combined |
|------|---------------|----------|
| Loss | CE only | CE + λ·Σ‖w‖² |
| 파일 | `train_physical_pruning.py` | `train_combined_pruning.py` |
| 특징 | 빠른 실험용 | 안정적 pruning |

#### 하이퍼파라미터 (기본값)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| batch-size | 256 | GPU 메모리에 따라 조정 |
| lr | 1e-4 | AdamW 학습률 |
| target-reduction | 0.76 | 76% 압축 목표 |
| ffn-prune-per-epoch | 0.08 | FFN 매 epoch 8% 제거 |
| qk-prune-per-epoch | 0.10 | Q/K 매 epoch 10% 제거 |
| pruning-epochs | 15 | Pruning 진행 epoch |
| finetune-epochs | 10 | 후속 finetune epoch |
| λ_FFN (Combined) | 1e-4 | FFN 정규화 강도 |
| λ_QK (Combined) | 5e-5 | Q/K 정규화 강도 |
| λ_V (Combined) | 1e-5 | V 보존 (작게 설정) |

#### 서버 실행 명령어

```bash
# Physical-Only (CE만)
python -m classification.pruning.train_physical_pruning \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --output-dir classification/pruning/checkpoints/physical_only_76pct

# Combined (λ + Physical) — 권장
python -m classification.pruning.train_combined_pruning \
    --data-path /workspace/etri_iitp/JS/EfficientViT/data/imagenet \
    --resume efficientvit_m4.pth \
    --target-reduction 0.76 \
    --lambda-ffn 0.0001 --lambda-qk 0.00005 --lambda-v 0.00001 \
    --output-dir classification/pruning/checkpoints/combined_76pct
```

#### 예상 결과

| 단계 | Acc@1 | 모델 크기 | 압축률 |
|------|-------|-----------|--------|
| 초기 (M4) | 74.3% | 35.2 MB | 0% |
| Pruning 후 | ~58% | 8.4 MB | 76% |
| Finetune 후 | ~68% | 8.4 MB | 76% |
| + KD (Phase C) | ~72% | 8.4 MB | 76% |

---

### Phase C / D / E 예정 (Phase B 서버 실행 후)

| Phase | 상태 | 구현 예정 파일 | 핵심 내용 |
|---|---|---|---|
| C | 🔜 예정 | `train_physical_kd.py` | Physical Pruning + KD loss (M5 teacher) |
| D | 🔜 예정 | `phase_d_compare.py` | M4 vs YOLOv8n pruning 결과 비교 |
| E | 🔜 예정 | `phase_e_ablation.py` | G_FFN only / +G_QK / +G_V ablation |

Phase C 추가 구현:
- KD loss: `α · KL(softmax(z_T/T) || log_softmax(z_S/T)) · T²`
- M5 teacher 모델 로드 및 freeze
- α=0.5, T=4.0 (CLAUDE.md §5.3)

---

### Structure Extraction (Physical Pruning에서 자동 처리)

> **Note**: Physical Pruning 방식에서는 매 epoch 물리적으로 Conv2d/Linear를 축소하므로 별도 extraction 불필요

`structural_pruning.py`에서 처리:
- FFN: `prune_ffn_physically()` — expand/shrink weight 직접 축소
- QK: `prune_cga_head_qk_physically()` — Q/K 동일 인덱스로 축소
- DW conv: groups = in_channels = out_channels 자동 동기화

검증 (매 pruning step 후 자동 실행):
```python
validate_model_forward(model, device)  # forward pass 검증
opt_rate = 100 * (original_size - current_size) / original_size
# 학습 중 콘솔에 출력됨
```
