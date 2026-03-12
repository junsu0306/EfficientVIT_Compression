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
