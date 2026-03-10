# EfficientViT Structured Pruning 계획

> **대상 모델:** EfficientViT-M4 (8.8M params, 74.3% Top-1 on ImageNet-1K)
> **목표:** 정확도 손실 최소화 + 파라미터/연산량 대폭 축소

---

## 목차

1. [CNN Pruning vs Transformer Pruning — 개념적 차이](#1-cnn-pruning-vs-transformer-pruning--개념적-차이)
2. [EfficientViT 구조 분석 (Pruning 관점)](#2-efficientvit-구조-분석-pruning-관점)
3. [컴포넌트별 중요도 측정 방법](#3-컴포넌트별-중요도-측정-방법)
4. [Pruning 단위(Granularity) 4계층](#4-pruning-단위granularity-4계층)
5. [단계별 구현 계획](#5-단계별-구현-계획)
6. [Fine-tuning 전략](#6-fine-tuning-전략)
7. [예상 결과 (M4 기준)](#7-예상-결과-m4-기준)
8. [파일 구조 및 CLI 설계](#8-파일-구조-및-cli-설계)
9. [구현 로드맵](#9-구현-로드맵)

---

## 1. CNN Pruning vs Transformer Pruning — 개념적 차이

### CNN에서의 Filter Pruning (YOLO 등)

CNN에서는 Conv 레이어의 각 필터(filter)가 하나의 **독립적인 특징 검출기**입니다.
YOLO 같은 모델에서는 BatchNorm의 **γ (gamma, scale)** 파라미터를 필터 중요도로 사용합니다.

```
Conv2d  →  BatchNorm  →  ReLU
            ↑
      γ 값이 작다 = 이 필터의 출력이 거의 0에 수렴
                  = 제거해도 네트워크에 영향 없음
```

**왜 BN γ가 중요도가 되는가?**
BN의 출력은 `γ × (x - μ)/σ + β` 입니다. γ = 0이면 이 채널의 출력이 항상 β(상수)가 되어
후속 ReLU에서 음수일 경우 완전히 소멸합니다. 즉 **γ가 그 채널이 전달하는 정보량**을 직접 제어합니다.

**Network Slimming 방법론:**
학습 중 γ에 L1 정규화 페널티를 추가(`loss += λ × Σ|γᵢ|`)하여 γ를 0으로 유도한 뒤 제거합니다.

---

### Transformer에서의 Pruning — 왜 그대로 못 쓰나?

Transformer는 CNN과 구조적으로 다른 두 가지 핵심 요소가 있습니다:

| 요소 | CNN | Transformer |
|------|-----|-------------|
| 기본 연산 | Conv (지역 필터) | Self-Attention (전역 의존성) |
| 독립성 | 각 필터 독립적 | **헤드 간 의존성 존재** |
| 중요도 지표 | BN γ (직접 측정) | **간접 측정 필요** |

Attention Head를 단순히 "가중치 크기"로 판단하면 실제 손실에 대한 기여를 놓칩니다.
예를 들어, 가중치 norm이 작아도 특정 패턴을 담당하는 헤드는 제거 시 정확도가 크게 하락합니다.

---

### EfficientViT의 위치 — 혼합 구조

EfficientViT는 **CNN 블록과 Attention 블록이 공존**하는 구조이므로,
컴포넌트에 따라 다른 중요도 측정 방법을 적용해야 합니다.

```
컴포넌트            주요 연산          사용할 중요도 방법
─────────────────────────────────────────────────────────
FFN (pw1, pw2)    Conv2d_BN (1×1)   → BN γ (CNN 방식 그대로)
DW-Conv (dw0/1)   Conv2d_BN (3×3)   → BN γ 또는 Output Perturbation
Attention Head    CGA (QKV 계산)    → Taylor Expansion 또는 Attention Entropy
EfficientViTBlock 전체 블록          → Taylor Expansion 또는 Output Perturbation
```

---

## 2. EfficientViT 구조 분석 (Pruning 관점)

### 2.1 전체 데이터 흐름 (M4 기준)

```
입력 (B, 3, 224, 224)
    ↓
[patch_embed]  4× Conv2d_BN stride-2  →  (B, 128, 14, 14)
    ↓
[blocks1]
  [0] EfficientViTBlock         embed_dim=128, resolution=14
    ↓
[blocks2]
  [0] Sequential(DW-Conv, FFN)  ← Stage 전환 전처리
  [1] PatchMerging(128→256)     ← 14×14 → 7×7 다운샘플 (제거 불가)
  [2] Sequential(DW-Conv, FFN)  ← Stage 전환 후처리
  [3] EfficientViTBlock         embed_dim=256, resolution=7
  [4] EfficientViTBlock         embed_dim=256, resolution=7
    ↓
[blocks3]
  [0] Sequential(DW-Conv, FFN)  ← Stage 전환 전처리
  [1] PatchMerging(256→384)     ← 7×7 → 4×4 다운샘플 (제거 불가)
  [2] Sequential(DW-Conv, FFN)  ← Stage 전환 후처리
  [3] EfficientViTBlock         embed_dim=384, resolution=4
  [4] EfficientViTBlock         embed_dim=384, resolution=4
  [5] EfficientViTBlock         embed_dim=384, resolution=4
    ↓
[head]  GlobalAvgPool → BN_Linear → (B, 1000)
```

**Pruning 가능한 EfficientViTBlock:** blocks1[0], blocks2[3~4], blocks3[3~5] = **총 6개**

---

### 2.2 EfficientViTBlock 내부 구조

```
입력 x  (shape 유지됨 — 입출력 shape 동일)
  ↓
dw0  : Residual( Conv2d_BN 3×3 depthwise )   ← Sandwich: 앞 DW-Conv
  ↓
ffn0 : Residual( FFN: ed → 2ed → ed )        ← Sandwich: 앞 FFN
  ↓
mixer: Residual( LocalWindowAttention )       ← 핵심 Attention 블록
  ↓
dw1  : Residual( Conv2d_BN 3×3 depthwise )   ← Sandwich: 뒤 DW-Conv
  ↓
ffn1 : Residual( FFN: ed → 2ed → ed )        ← Sandwich: 뒤 FFN
  ↓
출력 x
```

모든 서브블록이 `Residual` 래퍼로 감싸져 있습니다.
→ 서브블록을 제거하면 `x + m(x)` 에서 `m(x)` 부분이 사라져 **Identity(입력 그대로 통과)** 가 됩니다.

---

### 2.3 CGA(Cascaded Group Attention) 내부 구조와 Cascade 의존성

CGA는 EfficientViT의 핵심 혁신입니다. 일반 Multi-Head Attention과 달리 **헤드 간에 의존성**이 있습니다.

**일반 Multi-Head Attention:**
```
입력 X → [복사] → Head 0: Q,K,V → output_0
                → Head 1: Q,K,V → output_1   (각 헤드가 동일한 X를 독립적으로 처리)
                → Head 2: Q,K,V → output_2
          → concat → Linear Projection
```

**Cascaded Group Attention (CGA):**
```
입력 X
  ↓ chunk(num_heads, dim=1)  ← 채널을 헤드 수만큼 균등 분할
  [chunk_0, chunk_1, chunk_2, chunk_3]  (각 shape: B, dim//num_heads, H, W)

Head 0: feat = chunk_0        → Q,K,V → attn → feat_0
Head 1: feat = chunk_1 + feat_0  ← 이전 헤드 출력 누적 (CASCADE!)
              → Q,K,V → attn → feat_1
Head 2: feat = chunk_2 + feat_1
              → Q,K,V → attn → feat_2
Head 3: feat = chunk_3 + feat_2
              → Q,K,V → attn → feat_3

→ concat([feat_0, feat_1, feat_2, feat_3]) → Projection
```

**Cascade 의존성이 Pruning에 미치는 영향:**
```
Head i를 제거하면 → Head i+1의 입력(feat_{i-1})이 달라짐
                  → 이후 모든 헤드의 출력이 연쇄적으로 변화
```

이 의존성 때문에 **중간 헤드를 제거하는 것은 매우 복잡**합니다.
가장 안전한 전략은 **마지막 헤드부터 제거(Tail Pruning)** 하는 것입니다.

**M4 CGA 차원 검증 (Cascade 덧셈 성립 조건):**

| Stage | embed_dim | num_heads | dim//num_heads | key_dim | attn_ratio | value_dim (d) | cascade 덧셈 |
|-------|-----------|-----------|----------------|---------|------------|---------------|-------------|
| 1     | 128       | 4         | 32             | 16      | 2.0        | 32            | dim//nh == d ✓ |
| 2     | 256       | 4         | 64             | 16      | 4.0        | 64            | dim//nh == d ✓ |
| 3     | 384       | 4         | 96             | 16      | 6.0        | 96            | dim//nh == d ✓ |

`feat` (value_dim=d)와 `chunk_i` (dim//num_heads)의 shape이 같아야 덧셈이 가능합니다.
Head 제거 시 이 균형이 깨지므로 embed_dim 조정이 함께 필요합니다.

---

### 2.4 컴포넌트별 Pruning 가능 여부 요약

| 컴포넌트 | Pruning 가능 | 난이도 | 핵심 제약 |
|---------|:-----------:|:------:|---------|
| EfficientViTBlock (전체) | ✅ | 쉬움 | 같은 Stage 내 I/O shape 동일 |
| FFN hidden dim (2ed) | ✅ | 쉬움 | pw1↔pw2 입출력 채널 동시 조정 |
| DW-Conv 서브블록 (dw0/dw1) | ✅ | 쉬움 | Residual → Identity 교체 |
| CGA 마지막 헤드 | ✅ | 보통 | proj 입력 채널 재구성 필요 |
| CGA 중간 헤드 | ⚠️ | 어려움 | Cascade 체인 재연결 필요 |
| embed_dim (Stage 전체) | ⚠️ | 매우 어려움 | num_heads와 연동, DW-Conv groups 제약 |
| PatchMerging | ❌ | 불가 | Stage 간 해상도 다운샘플 담당 |
| patch_embed (Stem) | ❌ | 불가 | 입력 전처리 구조 |

---

## 3. 컴포넌트별 중요도 측정 방법

### 3.1 BN γ (Gamma) — FFN과 DW-Conv 채널용

**언제 사용:** `Conv2d_BN` 기반 컴포넌트의 **채널 단위** 중요도 측정
**대상:** FFN의 pw1 중간 채널 (2×ed), dw0/dw1의 depthwise 채널

**원리:**
```
Conv2d_BN의 출력 = γ × normalize(Conv(x)) + β

γ → 0  이면  해당 채널 출력이 상수(β)에 수렴
           → 후속 ReLU에서 소멸할 가능성 높음
           → 이 채널은 다음 레이어에 유효한 정보를 전달하지 못함
```

**구현:**
```python
def get_bn_gamma_importance(model):
    """FFN의 pw1 BN gamma를 채널 중요도로 반환"""
    scores = {}
    for name, m in model.named_modules():
        if isinstance(m, FFN):
            # pw1: Conv2d_BN(ed → 2*ed)의 BN gamma
            gamma = m.pw1.bn.weight.data.abs()
            scores[name] = gamma.cpu()
    return scores  # 값이 작을수록 제거 우선
```

**Sparsity 유도 학습 (Network Slimming):**
```python
def apply_bn_sparsity_penalty(model, sparsity_lambda=1e-4):
    """학습 중 FFN BN gamma에 L1 페널티를 추가하여 0으로 유도"""
    for m in model.modules():
        if isinstance(m, FFN):
            # gamma에 L1 gradient 수동 추가 (optimizer step 전에 호출)
            gamma = m.pw1.bn.weight
            if gamma.grad is not None:
                gamma.grad.data.add_(sparsity_lambda * torch.sign(gamma.data))
```

---

### 3.2 Taylor Expansion Criterion — 블록/헤드 단위 중요도 측정

**언제 사용:** 구조 단위(Block, Attention Head)를 제거할 때의 **손실 변화량 추정**
**대상:** EfficientViTBlock 전체, CGA Attention Head

**원리:**
```
파라미터 θ를 제거했을 때 손실 변화 ΔL을 1차 테일러 근사로 추정:

ΔL ≈ (∂L/∂y) · y
        ↑          ↑
   gradient   activation

직관: gradient × activation이 크다 = 이 뉴런의 출력이 손실에 크게 기여
      → 제거하면 손실이 많이 증가 → 중요한 뉴런
```

**캘리브레이션 데이터로 측정 (실제 학습 불필요):**
```python
def compute_taylor_importance(model, calibration_loader, device, n_batches=100):
    """
    캘리브레이션 배치로 각 EfficientViTBlock의 Taylor importance 계산.
    gradient 계산이 필요하므로 model.train() 상태에서 실행.
    """
    importances = defaultdict(float)
    activations = {}

    # 1. Forward hook: 각 블록의 출력(activation) 저장
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, EfficientViTBlock):
            def save_act(mod, inp, out, n=name):
                activations[n] = out
            hooks.append(module.register_forward_hook(save_act))

    # 2. 캘리브레이션 데이터로 gradient 수집
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    for i, (images, labels) in enumerate(calibration_loader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

        for name, act in activations.items():
            if act.grad is not None:
                # |∂L/∂y · y| 의 배치 평균
                importance = (act.grad * act).abs().mean().item()
                importances[name] += importance

        optimizer.zero_grad()

    for h in hooks:
        h.remove()

    # n_batches로 나누어 평균
    return {k: v / n_batches for k, v in importances.items()}
    # 값이 작을수록 제거 우선
```

---

### 3.3 Output Perturbation (Residual Contribution) — 블록 단위 빠른 추정

**언제 사용:** gradient 계산 없이 블록 중요도를 **빠르게** 추정할 때
**대상:** EfficientViTBlock 전체

**원리:**
```
EfficientViTBlock은 Residual 구조: 출력 = 입력 + f(입력)
                                              ↑
                                         이 부분이 블록의 기여

‖f(x)‖₂ / ‖x‖₂ 비율이 작다 = f(x) ≈ 0 = 이 블록이 거의 identity
                              → 제거해도 출력이 크게 변하지 않음
```

**구현:**
```python
@torch.no_grad()
def compute_block_perturbation(model, calibration_loader, device, n_batches=50):
    """각 EfficientViTBlock의 출력 기여도(residual norm ratio) 측정"""
    scores = {}

    for name, block in model.named_modules():
        if not isinstance(block, EfficientViTBlock):
            continue

        total_ratio = 0.0
        # 해당 블록까지만 forward하는 훅 사용 (간략화)
        for i, (images, _) in enumerate(calibration_loader):
            if i >= n_batches:
                break
            x = get_input_to_block(model, images.to(device), name)  # 블록 직전 활성화
            y = block(x)
            residual = (y - x).norm(dim=(1, 2, 3))   # f(x) norm
            base = x.norm(dim=(1, 2, 3))              # x norm
            total_ratio += (residual / base).mean().item()

        scores[name] = total_ratio / n_batches

    return scores  # 값이 작을수록 제거 우선
```

---

### 3.4 Attention Entropy — CGA Head 중요도 (Gradient-free)

**언제 사용:** gradient 없이 Attention Head 중요도를 측정할 때
**대상:** CGA의 각 Attention Head

**원리:**
```
Attention weight 행렬 A (shape: N×N, softmax 이후)의 엔트로피:

H(head_i) = -Σ A_ij × log(A_ij)

엔트로피 높음 (분포 균일) = 어느 위치에도 집중하지 않음 = 특별한 정보 추출 안 함 → 덜 중요
엔트로피 낮음 (분포 날카로움) = 특정 위치에 집중 = 의미 있는 패턴 검출 중     → 중요
```

**구현:**
```python
@torch.no_grad()
def compute_head_entropy(model, calibration_loader, device, n_batches=50):
    """각 CGA head의 Attention Entropy 계산 (낮을수록 중요)"""
    head_entropies = defaultdict(list)

    def hook_fn(module, input, output, name):
        # CGA.forward() 내부 attention 행렬에 접근하려면
        # CGA를 수정하여 attention map을 저장하는 훅 필요
        pass

    # 대안: QKV 가중치의 L1 norm (gradient-free 근사)
    for name, m in model.named_modules():
        if isinstance(m, CascadedGroupAttention):
            scores = []
            for i, qkv in enumerate(m.qkvs):
                # qkv.c: Conv2d (가중치 norm이 해당 헤드의 표현력 근사)
                weight_importance = qkv.c.weight.data.abs().mean().item()
                scores.append(weight_importance)
            head_entropies[name] = scores

    return head_entropies  # 각 head별 점수 리스트, 작을수록 제거 우선
```

---

### 3.5 중요도 측정 방법 비교 요약

| 방법 | 대상 | Gradient 필요 | 계산 비용 | 정확도 |
|-----|------|:------------:|:--------:|:-----:|
| **BN γ** | FFN 채널, DW-Conv 채널 | ❌ | 매우 낮음 | 중간 |
| **Taylor Expansion** | Block, Head | ✅ | 보통 | 높음 |
| **Output Perturbation** | Block | ❌ | 낮음 | 보통 |
| **Attention Entropy** | Head | ❌ | 낮음 | 보통 |
| **QKV Weight Norm** | Head | ❌ | 매우 낮음 | 낮음 |

**권장 조합:**
- Block 제거 → **Taylor Expansion** (정확도 중요) 또는 **Output Perturbation** (빠른 탐색)
- FFN 채널 제거 → **BN γ** (Sparsity 학습 병행 시 가장 효과적)
- Head 제거 → **Taylor Expansion** (Cascade 의존성 고려)

---

## 4. Pruning 단위(Granularity) 4계층

거친 단위에서 세밀한 단위 순으로 4계층:

```
Level 1 (가장 거침): Block Pruning     — EfficientViTBlock 전체 제거
Level 2:            Head Pruning       — CGA Attention Head 제거
Level 3:            Sub-block Pruning  — Sandwich 내 서브블록을 Identity로 교체
Level 4 (가장 세밀): Channel Pruning   — FFN hidden dim 채널 제거
```

---

### Level 1: Block Pruning

**대상:** EfficientViTBlock 6개 중 일부 (주로 blocks3[3~5])

**가능한 이유:** 같은 Stage 내 모든 EfficientViTBlock의 입출력 shape이 동일하므로
`Sequential`에서 해당 블록을 꺼내도 shape 불일치가 발생하지 않습니다.

**중요도 측정:** Taylor Expansion 또는 Output Perturbation

**구현:**
```python
def drop_block(model, stage_name: str, block_idx: int):
    """
    EfficientViTBlock을 Sequential에서 제거.
    stage_name: 'blocks1', 'blocks2', 'blocks3'
    block_idx: Sequential 내 인덱스 (EfficientViTBlock이 아닌 인덱스는 제거 불가)
    """
    stage = getattr(model, stage_name)
    target = stage[block_idx]
    assert isinstance(target, EfficientViTBlock), \
        f"Index {block_idx}는 EfficientViTBlock이 아님 (PatchMerging 등)"

    new_layers = [m for i, m in enumerate(stage) if i != block_idx]
    setattr(model, stage_name, torch.nn.Sequential(*new_layers))
```

**주의사항:**
- `blocks1`은 EfficientViTBlock이 1개뿐 → 제거 시 Stage 1 Attention 완전 소멸 (주의)
- `PatchMerging`, Stage 전환 `Sequential(DW-Conv, FFN)` 은 **절대 제거 불가**

---

### Level 2: Head Pruning (CGA)

**대상:** CGA의 `num_heads`를 4 → 3으로 축소

**Cascade 의존성으로 인한 제약:**
```
Head 0: 독립 (chunk_0만 사용)
Head 1: chunk_1 + feat_0  ← Head 0에 의존
Head 2: chunk_2 + feat_1  ← Head 1에 의존
Head 3: chunk_3 + feat_2  ← Head 2에 의존 ← 가장 안전하게 제거 가능
```

**권장 전략: Tail Pruning (마지막 헤드부터 제거)**

```python
def prune_cga_last_head(cga: CascadedGroupAttention):
    """
    CGA의 마지막 헤드를 제거.
    연산 감소: qkvs[-1], dws[-1] 제거 + proj 입력 채널 축소
    """
    nh = cga.num_heads
    assert nh > 1, "헤드가 1개면 제거 불가"

    new_nh = nh - 1

    # 1. qkvs, dws에서 마지막 원소 제거
    cga.qkvs = torch.nn.ModuleList(list(cga.qkvs)[:new_nh])
    cga.dws  = torch.nn.ModuleList(list(cga.dws)[:new_nh])

    # 2. attention_biases: (num_heads, num_offsets) → (new_nh, num_offsets)
    cga.attention_biases = torch.nn.Parameter(
        cga.attention_biases[:new_nh].clone()
    )
    cga.register_buffer(
        'attention_bias_idxs',
        cga.attention_bias_idxs  # 위치 인덱스는 변경 없음
    )

    # 3. proj: 입력 채널 d*num_heads → d*new_nh
    old_conv = cga.proj[1]  # proj = Sequential(ReLU, Conv2d_BN)
    new_in_channels = cga.d * new_nh
    new_conv = Conv2d_BN(new_in_channels, old_conv.c.out_channels,
                         resolution=cga.resolution)
    # 기존 가중치의 앞 new_in_channels개만 복사
    with torch.no_grad():
        new_conv.c.weight.copy_(old_conv.c.weight[:, :new_in_channels])
    cga.proj[1] = new_conv

    # 4. num_heads 업데이트
    cga.num_heads = new_nh
```

**남는 채널 처리 (embed_dim 불일치):**
Head 1개 제거 시 입력 `x`의 채널 수(`embed_dim`)는 그대로인데 `chunk` 수가 줄어,
마지막 `dim//num_heads` 채널이 사용되지 않습니다.
→ **해결책:** embed_dim도 함께 줄이거나, 남는 채널을 마지막 chunk에 합산합니다.

---

### Level 3: Sub-block Pruning (Sandwich 서브블록)

**대상:** `dw0`, `ffn0`, `mixer`, `dw1`, `ffn1` 중 일부를 Identity로 교체

**가능한 이유:** 모두 `Residual` 래퍼로 감싸여 있어, 내부 모듈을 제거하면
`x + m(x)` → `x` 가 되어 자연스럽게 통과(pass-through)됩니다.

**제거 우선순위 (보수적 → 공격적 순):**
```
1순위: dw0, dw1  (3×3 DW-Conv, 파라미터 적고 정확도 영향 적음)
2순위: ffn0, ffn1  (1×1 Conv × 2, 파라미터 비중 큼)
3순위: mixer  (LocalWindowAttention, 제거 시 해당 블록에서 Attention 완전 소멸)
```

**구현:**
```python
def deactivate_subblock(block: EfficientViTBlock, subblock_name: str):
    """
    서브블록을 Identity Residual로 교체 (메모리 해제 + forward 단축)
    subblock_name: 'dw0', 'ffn0', 'mixer', 'dw1', 'ffn1'
    """
    residual = getattr(block, subblock_name)
    assert isinstance(residual, Residual)

    # Residual 내부 모듈을 Identity로 교체
    residual.m = torch.nn.Identity()
```

---

### Level 4: FFN Channel Pruning

**대상:** FFN의 hidden dim (ed → 2×ed 중간 채널)을 선별적으로 제거

**FFN 구조:**
```
pw1: Conv2d_BN(ed, 2*ed)   ← BN gamma가 채널 중요도
  ↓ ReLU
pw2: Conv2d_BN(2*ed, ed)   ← pw1의 출력 채널 = pw2의 입력 채널
```

**2단계 절차:**

**Step 1 — Sparsity 학습:** BN γ를 0으로 유도하는 L1 페널티 추가
```python
# engine.py의 train_one_epoch에서 optimizer.step() 직전에 호출
apply_bn_sparsity_penalty(model, sparsity_lambda=1e-4)
```

**Step 2 — 채널 제거:**
```python
def prune_ffn_channels(ffn: FFN, prune_ratio: float = 0.3):
    """
    pw1의 BN gamma 하위 prune_ratio 비율 채널을 제거.
    pw1 출력 채널과 pw2 입력 채널을 동시에 축소.
    """
    gamma = ffn.pw1.bn.weight.data.abs()
    threshold = torch.quantile(gamma, prune_ratio)
    keep_mask = (gamma > threshold)
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    new_hidden = len(keep_idx)

    ed = ffn.pw1.c.in_channels  # 원래 embed_dim

    # pw1: Conv2d_BN(ed, 2*ed) → Conv2d_BN(ed, new_hidden)
    new_pw1 = Conv2d_BN(ed, new_hidden, resolution=ffn.pw1.resolution)
    with torch.no_grad():
        new_pw1.c.weight.copy_(ffn.pw1.c.weight[keep_idx])
        new_pw1.bn.weight.copy_(ffn.pw1.bn.weight[keep_idx])
        new_pw1.bn.bias.copy_(ffn.pw1.bn.bias[keep_idx])
        new_pw1.bn.running_mean.copy_(ffn.pw1.bn.running_mean[keep_idx])
        new_pw1.bn.running_var.copy_(ffn.pw1.bn.running_var[keep_idx])

    # pw2: Conv2d_BN(2*ed, ed) → Conv2d_BN(new_hidden, ed)
    new_pw2 = Conv2d_BN(new_hidden, ed, bn_weight_init=0, resolution=ffn.pw2.resolution)
    with torch.no_grad():
        new_pw2.c.weight.copy_(ffn.pw2.c.weight[:, keep_idx])

    ffn.pw1 = new_pw1
    ffn.pw2 = new_pw2
```

---

## 5. 단계별 구현 계획

### Phase 0: 분석 기반 구축

**목표:** 모델 구조를 파싱하고 각 Pruning 단위의 중요도를 시각화합니다.

```bash
python prune.py --analyze \
  --model EfficientViT_M4 \
  --resume ./efficientvit_m4.pth \
  --data-path $IMAGENET \
  --n-calib-batches 100
```

**기대 출력:**
```
[Block Importance — Taylor Expansion]
blocks1.0  :  0.2341  (중요, 제거 비권장)
blocks2.3  :  0.1892
blocks2.4  :  0.1105
blocks3.3  :  0.1204
blocks3.4  :  0.0723
blocks3.5  :  0.0312  ← 가장 낮음 → 1순위 제거 후보

[FFN Channel Sparsity — BN gamma]
blocks3.3.ffn0 : mean=0.031, below_threshold(30%)=23%
blocks3.5.ffn0 : mean=0.019, below_threshold(30%)=41%
```

---

### Phase 1: Block Pruning

**목표:** EfficientViTBlock 1~3개 제거로 파라미터 약 15~50% 감소

**절차:**
```
1. Pretrained M4 로드
2. 캘리브레이션 데이터 (ImageNet val 1k) 로 Taylor importance 계산
3. 중요도 낮은 블록 선택 (blocks3부터 우선)
4. Sequential에서 해당 블록 제거
5. Forward pass 검증 (shape 오류 없음 확인)
6. Fine-tuning (30~50 epochs, Knowledge Distillation 병행)
```

**실험 시나리오:**

| 제거 대상 | 파라미터 | 예상 Top-1 (KD 후) |
|---------|---------|-------------------|
| 없음 (원본) | 8.8M | 74.3% |
| blocks3[5] 1개 | ~7.4M | ~73.8% |
| blocks3[4,5] 2개 | ~5.9M | ~73.0% |
| blocks3[3,4,5] 3개 | ~4.4M | ~71.5% |
| blocks2[4]+blocks3[4,5] 3개 | ~4.4M | ~70.5% |

---

### Phase 2: FFN Channel Pruning

**목표:** FFN hidden dim 축소로 파라미터 추가 10~20% 감소

**절차:**
```
1. Phase 1 pruned 모델에서 시작 (또는 pretrained 모델 직접 사용)
2. Sparsity 학습 (30 epochs):
   - train_one_epoch에 apply_bn_sparsity_penalty() 삽입
   - lambda 범위: 1e-4 ~ 1e-3 (클수록 더 많은 채널이 0으로 수렴)
3. BN gamma 분포 시각화 → threshold 결정
4. 하위 30~40% 채널 제거 (prune_ffn_channels)
5. Fine-tuning (20~30 epochs)
```

**주의사항:**
- `DW-Conv(dw0, dw1)`는 `groups=embed_dim`으로 고정되어 있어,
  embed_dim을 바꾸지 않는 한 채널 수 자체를 줄일 수 없습니다. → FFN만 대상으로
- 채널 pruning 후에도 `Conv2d_BN.fuse()` 호환성 유지됩니다.

---

### Phase 3: Head Pruning (CGA)

**목표:** num_heads를 4 → 3으로 줄여 CGA 연산 약 25% 감소

**절차:**
```
1. Phase 1+2 pruned 모델에서 시작
2. Taylor Expansion으로 각 Stage CGA의 헤드별 중요도 계산
3. 마지막 헤드(index 3) 제거 (prune_cga_last_head)
4. embed_dim을 num_heads에 맞게 조정 (embed_dim = new_nh × (dim//nh))
5. Fine-tuning (30~50 epochs, KD)
```

**embed_dim 조정 예시 (M4 Stage 1):**
```
원본: embed_dim=128, num_heads=4, dim//nh=32
제거 후: num_heads=3, 사용 채널=3×32=96
→ embed_dim 128 → 96 으로 조정 필요
  (Stage 전체: blocks1, PatchMerging 입력, blocks2... 연쇄 수정)
```

---

### Phase 4: Iterative Pruning 통합 파이프라인

```
Pretrained M4 (74.3%, 8.8M)
    ↓ [Phase 1] Block Drop: blocks3[4,5] 제거
    ↓ [Phase 1 Fine-tune] 50 epochs, KD (teacher=M4)
    ~ 73.5%, 5.9M
    ↓ [Phase 2] FFN Sparsity 학습 30 epochs → 30% 채널 제거
    ↓ [Phase 2 Fine-tune] 30 epochs, KD
    ~ 72.8%, 5.1M
    ↓ [Phase 3] Head Prune: 각 Stage CGA head 4→3
    ↓ [Phase 3 Fine-tune] 50 epochs, KD
    ~ 72.0%, 4.2M
```

목표: **M3 원본(6.9M, 73.4%) 대비 40% 작은 파라미터로 유사 정확도 달성**

---

## 6. Fine-tuning 전략

### 6.1 기존 학습 인프라 재사용

`classification/main.py`의 파이프라인을 최대한 활용합니다:
- `--finetune`: pruned 모델 가중치 로드 (옵티마이저 상태 초기화)
- `--resume`: 체크포인트에서 재개 (옵티마이저 상태 포함)
- Pruning 후에는 `--finetune` 사용이 적절합니다 (구조가 변경되었으므로)

### 6.2 Knowledge Distillation을 통한 정확도 회복

Pruning 후 정확도 회복에 가장 효과적인 방법입니다.
기존 `classification/losses.py`의 `DistillationLoss`를 그대로 사용합니다.

```bash
python main.py \
  --model EfficientViT_M4_Pruned \
  --finetune ./pruned_model.pth \
  --distillation-type soft \
  --teacher-model EfficientViT_M4 \
  --teacher-path ./efficientvit_m4.pth \
  --distillation-alpha 0.7 \
  --distillation-tau 3.0 \
  --epochs 50 \
  --lr 1e-4 \
  --data-path $IMAGENET
```

**Soft KD 손실 수식:**
```
L_total = (1 - α) × CrossEntropy(output, label)
        + α × KL_div(softmax(output/T), softmax(teacher/T)) × T²

α=0.7, T=3.0 → Teacher의 소프트 레이블에 가중치를 두어 정확도 회복 극대화
```

### 6.3 Pruning 강도에 따른 권장 하이퍼파라미터

| Pruning 강도 | 제거 대상 | LR | Epochs | KD alpha | KD tau |
|------------|---------|:--:|:------:|:--------:|:------:|
| 약 | Block 1개 | 5e-5 | 30 | 0.5 | 1.0 |
| 중 | Block 2개 + FFN 30% | 1e-4 | 50 | 0.7 | 3.0 |
| 강 | Block 2개 + FFN 40% + Head | 2e-4 | 100 | 0.8 | 5.0 |

---

## 7. 예상 결과 (M4 기준)

### Block Pruning 단독 적용

| 시나리오 | 파라미터 | Top-1 (KD fine-tune 후) |
|--------|---------|------------------------|
| M4 원본 | 8.8M | 74.3% |
| blocks3[5] 제거 | ~7.4M | ~73.8% |
| blocks3[4,5] 제거 | ~5.9M | ~73.0% |
| blocks3[3,4,5] 제거 | ~4.4M | ~71.5% |

### 복합 Pruning

| 시나리오 | 방법 | 파라미터 | Top-1 |
|--------|-----|---------|------|
| M4-Lite | Block(2개) + FFN(30%) | ~4.3M | ~72.5% |
| M4-Tiny | Block(3개) + FFN(40%) + Head(1) | ~2.8M | ~70.0% |

**비교 기준 (원본 모델):**

| 모델 | 파라미터 | Top-1 |
|-----|---------|------|
| EfficientViT-M3 | 6.9M | 73.4% |
| EfficientViT-M2 | 4.2M | 70.8% |
| **M4-Lite (목표)** | **~4.3M** | **~72.5%** |
| **M4-Tiny (목표)** | **~2.8M** | **~70.0%** |

---

## 8. 파일 구조 및 CLI 설계

```
classification/
├── pruning/
│   ├── __init__.py
│   ├── analyzer.py        # 모델 구조 파싱, prunable 단위 목록화, 중요도 시각화
│   ├── importance.py      # BN-gamma, Taylor Expansion, Output Perturbation 구현
│   ├── block_pruner.py    # Level 1: EfficientViTBlock 제거
│   ├── channel_pruner.py  # Level 4: FFN hidden dim 채널 제거
│   ├── head_pruner.py     # Level 2: CGA Attention Head 제거
│   └── subblock_pruner.py # Level 3: Sandwich 서브블록 → Identity 교체
│
├── prune.py               # 메인 Pruning 실행 스크립트
└── model/
    ├── efficientvit.py    # 원본 유지
    ├── build.py           # 원본 유지
    └── pruned_build.py    # Pruned 모델 등록 (timm registry)
```

### CLI 인터페이스

```bash
# 1. 중요도 분석 + 시각화
python prune.py --analyze \
  --model EfficientViT_M4 --resume ./efficientvit_m4.pth \
  --data-path $IMAGENET --n-calib-batches 100

# 2. Block Pruning
python prune.py \
  --model EfficientViT_M4 --resume ./efficientvit_m4.pth \
  --prune-level block --prune-ratio 0.33 \
  --output ./pruned_m4/block_pruned.pth \
  --data-path $IMAGENET

# 3. FFN Channel Pruning (Sparsity 학습 후 제거)
python prune.py \
  --model EfficientViT_M4 --resume ./pruned_m4/block_pruned.pth \
  --prune-level channel --prune-ratio 0.30 \
  --sparsity-epochs 30 --sparsity-lambda 1e-4 \
  --output ./pruned_m4/channel_pruned.pth \
  --data-path $IMAGENET

# 4. Head Pruning
python prune.py \
  --model EfficientViT_M4 --resume ./pruned_m4/channel_pruned.pth \
  --prune-level head --heads-to-remove 1 \
  --output ./pruned_m4/head_pruned.pth \
  --data-path $IMAGENET
```

---

## 9. 구현 로드맵

### Sprint 1 — 분석 기반 확립
- [ ] `pruning/analyzer.py`: blocks1/2/3에서 EfficientViTBlock 자동 탐지, prunable 단위 목록화
- [ ] `pruning/importance.py`: Output Perturbation (gradient-free, 빠름) 구현
- [ ] `prune.py --analyze`: 중요도 bar chart 출력, BN gamma 분포 히스토그램 출력

### Sprint 2 — Block Pruning MVP
- [ ] `pruning/block_pruner.py`: Sequential에서 블록 제거 + forward 검증
- [ ] `prune.py --prune-level block` 모드 구현
- [ ] Fine-tuning 검증: 50 epochs KD로 정확도 회복 여부 확인

### Sprint 3 — FFN Channel Pruning
- [ ] `pruning/importance.py`: BN gamma scoring + Sparsity 유도 학습 구현
- [ ] `pruning/channel_pruner.py`: prune_ffn_channels 구현 + fuse() 호환성 검증
- [ ] `engine.py` 수정: `--sparsity-lambda` 옵션 추가

### Sprint 4 — Head Pruning
- [ ] `pruning/head_pruner.py`: prune_cga_last_head 구현
- [ ] CGA forward 수정: 동적 num_heads 지원 (embed_dim 조정 포함)
- [ ] Attention map 시각화로 정보 손실 정도 확인

### Sprint 5 — 통합 및 검증
- [ ] 전체 파이프라인 자동화: Block → FFN → Head 순차 pruning
- [ ] Pareto curve 시각화: params vs Top-1 트레이드오프 그래프
- [ ] Downstream 검증: Pruned backbone으로 COCO detection 성능 측정

---

## 참고 논문

| 논문 | 학회 | 핵심 방법 | EfficientViT 적용 대상 |
|-----|------|---------|----------------------|
| Network Slimming | ICCV 2017 | BN γ L1 sparsity → 채널 제거 | FFN Channel (Level 4) |
| Importance Estimation for Neural Network Pruning | CVPR 2019 | Taylor Expansion Criterion | Block, Head (Level 1, 2) |
| Are Sixteen Heads Really Better than One? | NeurIPS 2019 | Attention Head 제거 가능성 분석 | Head (Level 2) |
| ViT-Slim | ECCV 2022 | ViT 다중 granularity 동시 pruning | 전체 참고 |
| DeiT | ICML 2021 | Knowledge Distillation for ViT | Fine-tuning 전략 |

---

*작성일: 2026-03-04*
*기준 코드: EfficientVIT_Compression (CVPR 2023)*
