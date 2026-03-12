# Phase A — EfficientViT M4 Pruning Group Profiler

## 목표 (CLAUDE.md §8)

| Phase | Target | Goal | Success Criterion |
|---|---|---|---|
| **A** | M4 | Profile all layers, build group dict | Memory map complete |

모든 prunable 그룹을 추출하고 파라미터/메모리 통계를 계산합니다.
이 리포트가 Phase B (λ sweep)의 기준점이 됩니다.

---

## 생성된 파일

```
classification/pruning/
├── __init__.py
├── group_dict.py        ← build_pruning_groups(model) — Phase B/C에서 재사용
├── memory_utils.py      ← PGM loss용 메모리 측정 함수
├── phase_a_profile.py   ← 실행 스크립트
└── reports/
    └── phase_a_report.json  ← 실행 후 생성됨
```

---

## 서버 환경 설정 (2080 Ti + Jupyter)

### 1단계 — 가상환경 생성 (conda 권장)

```bash
# conda 환경 생성 (Python 3.10)
conda create -n efficientvit python=3.10 -y
conda activate efficientvit
```

> venv를 선호하는 경우:
> ```bash
> python3.10 -m venv ~/envs/efficientvit
> source ~/envs/efficientvit/bin/activate
> ```

---

### 2단계 — PyTorch + CUDA 설치

RTX 2080 Ti는 CUDA 11.8까지 안정 지원합니다.

```bash
# CUDA 버전 확인 (서버에 설치된 버전)
nvidia-smi | grep "CUDA Version"

# PyTorch 설치 (CUDA 11.8 기준 — 가장 범용적)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1을 지원하는 서버라면:
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

설치 확인:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 예상: 2.0.1+cu118 / True / NVIDIA GeForce RTX 2080 Ti
```

---

### 3단계 — 프로젝트 의존성 설치

```bash
pip install timm==0.9.12
pip install einops
pip install Pillow
pip install matplotlib
pip install tqdm
```

한번에 설치:
```bash
pip install timm==0.9.12 einops Pillow matplotlib tqdm
```

---

### 4단계 — Jupyter 커널 등록

```bash
pip install ipykernel jupyter

# 현재 conda/venv 환경을 Jupyter 커널로 등록
python -m ipykernel install --user --name efficientvit --display-name "EfficientViT (Python 3.10)"
```

Jupyter Lab 실행:
```bash
# 서버에서 포트 포워딩으로 접속할 경우
jupyter lab --no-browser --port=8888 --ip=0.0.0.0
# 로컬에서: ssh -L 8888:localhost:8888 user@server
```

---

### 5단계 — 프로젝트 설정

```bash
# 프로젝트 루트로 이동
cd /path/to/EfficientVIT_Compression

# Python path 설정 (스크립트 실행 시마다 필요)
export PYTHONPATH=/path/to/EfficientVIT_Compression:$PYTHONPATH

# 또는 .bashrc에 영구 등록
echo 'export PYTHONPATH=/path/to/EfficientVIT_Compression:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 서버 실행 명령어

### 환경 준비 확인

```bash
conda activate efficientvit
cd /path/to/EfficientVIT_Compression
python -c "from classification.model.build import EfficientViT_M4; print('OK')"
```

### Option 1 — CPU만으로 구조 분석 (빠름, GPU 불필요)

```bash
python -m classification.pruning.phase_a_profile \
    --device cpu \
    --output classification/pruning/reports/phase_a_report.json
```

### Option 2 — GPU + pretrained weights (권장, 실제 메모리 측정 포함)

```bash
python -m classification.pruning.phase_a_profile \
    --device cuda \
    --pretrained EfficientViT_M4 \
    --output classification/pruning/reports/phase_a_report.json
```

> `--pretrained EfficientViT_M4` 지정 시 GitHub에서 자동 다운로드됩니다.
> 구조 분석이 목적이라면 CPU + 랜덤 초기화로도 동일한 결과를 얻을 수 있습니다.

---

### Option 3 — Jupyter Notebook에서 실행

Jupyter 노트북 셀에 아래 코드를 순서대로 실행합니다.

**셀 1 — 경로 설정**
```python
import sys, os
# 프로젝트 루트 경로를 직접 지정
PROJECT_ROOT = '/path/to/EfficientVIT_Compression'
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
```

**셀 2 — Phase A 실행**
```python
# 터미널 명령어를 노트북에서 그대로 실행
!python -m classification.pruning.phase_a_profile \
    --device cuda \
    --pretrained EfficientViT_M4 \
    --output classification/pruning/reports/phase_a_report.json
```

**셀 3 — 결과 리포트 확인**
```python
import json

report_path = 'classification/pruning/reports/phase_a_report.json'
r = json.load(open(report_path))

print(f"Model: {r['model']}")
print(f"Total params: {r['total_params']:,}  ({r['total_param_mb']} MB)")
print(f"Total groups: {r['n_groups_total']}")
print(f"Phase1 groups: {r['n_groups_phase1']}")
print(f"M_max target: {r['compression']['m_max_mb']} MB  (76% compression)")
print()
print(f"{'Type':<10} {'Groups':>7} {'Total Units':>12} {'Mem MB':>8} {'λ_rec':>7}")
print('-' * 50)
for t, s in r['type_summary'].items():
    print(f"{t:<10} {s['n_groups']:>7} {s['total_units']:>12,} {s['mem_mb']:>8.3f} {s['lambda_rec']:>7.3f}")
```

**셀 4 — 그룹별 상세 확인 (특정 타입 필터링)**
```python
# G_FFN 그룹만 확인
ffn_groups = [g for g in r['groups'] if g['type'] == 'G_FFN']
print(f"G_FFN groups ({len(ffn_groups)} total):")
print(f"{'ID':<45} {'Units':>6} {'Params/Unit':>11} {'Mem KB':>8}")
print('-' * 75)
for g in ffn_groups:
    print(f"{g['id']:<45} {g['unit_count']:>6} {g['params_per_unit']:>11,} {g['group_mem_kb']:>8.1f}")
```

---

## 예상 출력

```
==========================================================================================
Phase A — EfficientViT M4 Pruning Group Profiler
==========================================================================================

  Model    : EfficientViT_M4
  Total params : 8,820,288  (35.28 MB)

  Total pruning groups : 67
  Phase 1 groups (G_FFN+G_QK+G_V): 64

──────────────────────────────────────────────────────────────────────────────────────────
Group ID                                         Type     Units Params/Unit  Group Params   Mem KB
──────────────────────────────────────────────────────────────────────────────────────────
patch_embed_0                                    G_PATCH     16       3,168           ...      ...  ← Phase2+
patch_embed_1                                    G_PATCH     32       ...             ...      ...  ← Phase2+
patch_embed_2                                    G_PATCH     64       ...             ...      ...  ← Phase2+
stage1_block0_ffn0                               G_FFN      256         256        66,048     258.0
stage1_block0_ffn1                               G_FFN      256         256        66,048     258.0
stage1_block0_head0_qk                           G_QK        16          69         3,200      12.5
stage1_block0_head0_v                            G_V         32          32         2,112       8.3
...
stage3_block2_ffn1                               G_FFN      768         768       590,592   2,307.8
stage3_block2_head3_qk                           G_QK        16         121         5,696      22.2
stage3_block2_head3_v                            G_V         96          96        18,624      72.8
──────────────────────────────────────────────────────────────────────────────────────────

Summary by Group Type:
──────────────────────────────────────────────────────────────────────────────
Type         Groups  Total Units   Total Params   Mem MB   λ_rec
──────────────────────────────────────────────────────────────────────────────
G_FFN           20        8,448      5,505,024     22.020   0.005
G_QK            24          384        179,712      0.719   0.010
G_V             24        1,536        380,928      1.524   0.001
G_PATCH          3          112         ...          ...    0.005  (Phase2+)
──────────────────────────────────────────────────────────────────────────────

Compression Analysis:
  Phase 1 prunable params (G_FFN+G_QK+G_V) : ~6,065,664  (68.8% of total)
  76% optimization target → A ≤ 2,116,869 params  (8.47 MB)
  M_max (μ penalty threshold) = 8.47 MB
```

---

## 그룹 구조 설명

### M4 아키텍처와 그룹 매핑

```
patch_embed:
  [0] Conv2d_BN(3→16)    ← G_PATCH 0  (Phase 2+)
  [2] Conv2d_BN(16→32)   ← G_PATCH 1  (Phase 2+)
  [4] Conv2d_BN(32→64)   ← G_PATCH 2  (Phase 2+)
  [6] Conv2d_BN(64→128)  ← FIXED (embed_dim[0])

blocks1 (Stage 1, ed=128):
  EVBlock[0]:
    ffn0 ← G_FFN  (hidden=256, 256 units)
    ffn1 ← G_FFN  (hidden=256, 256 units)
    CGA:
      head0: qk ← G_QK (16 units), v ← G_V (32 units)
      head1: qk ← G_QK (16 units), v ← G_V (32 units)
      head2: qk ← G_QK (16 units), v ← G_V (32 units)
      head3: qk ← G_QK (16 units), v ← G_V (32 units)

blocks2 (Stage 1→2 경계 + Stage 2, ed=256):
  stage1_subsample_pre_ffn  ← G_FFN (hidden=256)
  PatchMerging (FIXED)
  stage2_subsample_post_ffn ← G_FFN (hidden=512)
  EVBlock[0], EVBlock[1]:  (각각 ffn0, ffn1, CGA 4 heads)

blocks3 (Stage 2→3 경계 + Stage 3, ed=384):
  stage2_subsample_pre_ffn  ← G_FFN (hidden=512)
  PatchMerging (FIXED)
  stage3_subsample_post_ffn ← G_FFN (hidden=768)
  EVBlock[0], EVBlock[1], EVBlock[2]:  (각각 ffn0, ffn1, CGA 4 heads)
```

### Hard Constraints (절대 위반 금지)

| 제약 | 설명 |
|---|---|
| FFN expand.out == shrink.in | 항상 동일 인덱스로 pruning |
| G_QK: Q와 K 동일 인덱스 | `q_slice`와 `k_slice`를 같이 제거 |
| Output proj | NEVER prune (채널 정렬 파괴) |
| embed_dim (128/256/384) | 스테이지 경계 — Phase 1에서 고정 |

---

## 출력 파일 (phase_a_report.json)

Phase B 진입 전 확인사항:

```bash
# 리포트 확인
cat classification/pruning/reports/phase_a_report.json | python -m json.tool | head -60

# 전체 그룹 수 확인
python -c "
import json
r = json.load(open('classification/pruning/reports/phase_a_report.json'))
print('Total groups:', r['n_groups_total'])
print('Phase1 groups:', r['n_groups_phase1'])
print('M_max target:', r['compression']['m_max_mb'], 'MB')
for t, s in r['type_summary'].items():
    print(f'  {t}: {s[\"n_groups\"]} groups, {s[\"total_units\"]:,} units, {s[\"mem_mb\"]} MB')
"
```

---

## Phase A 완료 기준 체크리스트

- [ ] `phase_a_report.json` 생성 확인
- [ ] 총 그룹 수 67개 확인 (G_PATCH 3 + G_FFN 20 + G_QK 24 + G_V 24)
- [ ] Phase 1 prunable params 비율 확인 (약 60~70%)
- [ ] M_max 값 확인 (전체 param MB × 0.24)
- [ ] λ 권장값 확인: λ_FFN=0.005, λ_QK=0.010, λ_V=0.001

→ 위 항목 모두 충족 시 **Phase B 진입 가능**

---

## 다음 단계 (Phase B)

Phase B에서는 이 그룹 dict를 기반으로:
1. PGM Loss 항 추가: `λ_FFN · Σ||w_g||₂ + λ_QK · Σ||w_g||₂ + λ_V · Σ||w_g||₂`
2. 메모리 패널티 추가: `μ · max(0, current_mem - M_max)`
3. optimizer.step() 직후 group soft thresholding 적용
4. λ sweep 실험으로 최적 λ 세트 탐색
