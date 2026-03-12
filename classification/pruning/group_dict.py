"""
PGM Pruning Group Dictionary — EfficientViT M4

EfficientViT M4 모델에서 pruning 가능한 모든 그룹을 추출하여 반환합니다.
Phase B (soft thresholding), Phase C (structure extraction) 모두에서 재사용됩니다.

Group types (CLAUDE.md §4):
  G_FFN   — FFN hidden neuron k: expand.weight[k,:] ↔ shrink.weight[:,k]
  G_QK    — Attention Q/K dim d: qkv.weight[d,:] and qkv.weight[key_dim+d,:]
  G_V     — Attention V channel c: qkv.weight[2*key_dim+c,:]
  G_PATCH — PatchEmbed intermediate channel f (Phase 2+ only)

Hard constraints (CLAUDE.md §4):
  - FFN: expand.out_channels == shrink.in_channels (always)
  - QK:  Q and K must share identical pruned dim indices
  - V:   V.out = C/h; cascade addition requires consistent shape
  - Output proj: NEVER prune
  - Subsample boundary channels: fix in Phase 1

M4 architecture (embed_dim=[128,256,384], depth=[1,2,3], num_heads=[4,4,4]):
  blocks1: [EVBlock(128)]
  blocks2: [SubPreDWFFN(128), PatchMerging(128→256), SubPostDWFFN(256),
            EVBlock(256), EVBlock(256)]
  blocks3: [SubPreDWFFN(256), PatchMerging(256→384), SubPostDWFFN(384),
            EVBlock(384), EVBlock(384), EVBlock(384)]
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ffn_group(ffn_residual, group_id, stage, block_label, ffn_idx):
    """
    Residual(FFN) 모듈에서 G_FFN 그룹 dict를 생성합니다.

    Parameters
    ----------
    ffn_residual : Residual(FFN)
    group_id     : str  — 고유 식별자
    stage        : int  — 스테이지 번호 (1/2/3)
    block_label  : str  — 블록 레이블 (정수 또는 'subsample_pre'/'subsample_post')
    ffn_idx      : int  — 블록 내 FFN 순서 (0=ffn0, 1=ffn1)

    Returns
    -------
    dict with keys: id, type, lambda_rec, unit_count, modules, meta
    """
    ffn = ffn_residual.m                          # FFN module
    ed  = ffn.pw1.c.in_channels                   # embedding dim
    hid = ffn.pw1.c.out_channels                  # hidden dim (= ed * 2)

    return {
        'id':         group_id,
        'type':       'G_FFN',
        'lambda_rec': 0.005,
        'unit_count': hid,                         # prunable hidden neurons
        'modules': {
            'expand': ffn.pw1,                     # Conv2d_BN(ed → hid)
            'shrink': ffn.pw2,                     # Conv2d_BN(hid → ed)
        },
        'meta': {
            'stage':    stage,
            'block':    block_label,
            'ffn_idx':  ffn_idx,
            'ed':       ed,
            'hidden_dim': hid,
        },
    }


def _cga_groups(block, stage, block_idx):
    """
    EfficientViTBlock의 CGA에서 G_QK / G_V 그룹 dicts를 생성합니다.

    각 헤드에 대해:
      - G_QK: Q dim d와 K dim d를 동일 인덱스로 함께 pruning
      - G_V:  V channel c (d차원)

    CLAUDE.md §4 hard constraints 반영:
      - QK: Q와 K는 반드시 동일 인덱스로 제거 (QK^T 계산에서 dim 일치 필요)
      - Output proj: NEVER prune (proj[1] 참조만, weight 수정 금지)
    """
    cga      = block.mixer.m.attn
    key_dim  = cga.key_dim                        # d_qk (= 16 for all M4 stages)
    d        = cga.d                              # V dim per head (= attn_ratio * key_dim)
    n_heads  = cga.num_heads

    groups = []
    for h in range(n_heads):
        qkv = cga.qkvs[h]                         # Conv2d_BN(C/H → key_dim*2+d)
        dw  = cga.dws[h]                          # Conv2d_BN(key_dim → key_dim, DW on Q)
        in_ch = qkv.c.in_channels                 # C / num_heads

        # ── G_QK ──────────────────────────────────────────────────────────
        # qkv.c.weight shape: [key_dim*2+d, in_ch, 1, 1]
        # Q: output channels [0 : key_dim]
        # K: output channels [key_dim : 2*key_dim]
        # dw.c.weight shape:  [key_dim, 1, kH, kW]  (DW: groups=key_dim)
        groups.append({
            'id':         f'stage{stage}_block{block_idx}_head{h}_qk',
            'type':       'G_QK',
            'lambda_rec': 0.010,
            'unit_count': key_dim,                # prunable QK dims
            'modules': {
                'qkv': qkv,                       # Conv2d_BN — share Q+K slice
                'dw':  dw,                        # Conv2d_BN — DW on Q
            },
            'meta': {
                'stage':    stage,
                'block':    block_idx,
                'head':     h,
                'key_dim':  key_dim,
                'd':        d,
                'in_channels': in_ch,
                'q_slice':  (0, key_dim),
                'k_slice':  (key_dim, 2 * key_dim),
                'dw_kernel': dw.c.kernel_size,
            },
        })

        # ── G_V ───────────────────────────────────────────────────────────
        # qkv.c.weight: V at output channels [2*key_dim : 2*key_dim+d]
        # proj.c.weight: [dim, d*n_heads, 1, 1]
        #   head h contributes proj input channels [h*d : (h+1)*d]
        #   → reference only (DO NOT prune proj output channels)
        groups.append({
            'id':         f'stage{stage}_block{block_idx}_head{h}_v',
            'type':       'G_V',
            'lambda_rec': 0.001,
            'unit_count': d,                      # prunable V channels
            'modules': {
                'qkv':  qkv,                      # Conv2d_BN — V slice
                'proj': cga.proj[1],              # Conv2d_BN — input reference (never prune output)
            },
            'meta': {
                'stage':    stage,
                'block':    block_idx,
                'head':     h,
                'key_dim':  key_dim,
                'd':        d,
                'in_channels': in_ch,
                'v_slice':         (2 * key_dim, 2 * key_dim + d),
                'proj_in_slice':   (h * d, (h + 1) * d),  # proj input channels for this head
            },
        })

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_pruning_groups(model):
    """
    EfficientViT M4 모델에서 모든 pruning 그룹을 추출합니다.

    Parameters
    ----------
    model : EfficientViT (M4)

    Returns
    -------
    groups : list[dict]
        각 dict의 key:
          id          — 고유 문자열 식별자
          type        — 'G_FFN' | 'G_QK' | 'G_V' | 'G_PATCH'
          lambda_rec  — CLAUDE.md 권장 λ 값
          unit_count  — 독립적으로 pruning 가능한 unit 수
          modules     — 실제 nn.Module 참조 dict (Phase B soft thresholding에 사용)
          meta        — 위치/shape 메타데이터

    Group 순서:
      1. G_PATCH  (PatchEmbed 중간 채널, Phase 2+ 전용)
      2. blocks1  (Stage 1: depth=1)
      3. blocks2  (Subsample FFNs + Stage 2: depth=2)
      4. blocks3  (Subsample FFNs + Stage 3: depth=3)
    """
    groups = []

    # ─────────────────────────────────────────────────────────────────────
    # G_PATCH: PatchEmbed 중간 채널 (Phase 2+ 전용)
    #
    # patch_embed = Sequential:
    #   [0] Conv2d_BN(3   → C/8=16)   ← 출력 prunable
    #   [1] ReLU
    #   [2] Conv2d_BN(16  → C/4=32)   ← 출력 prunable
    #   [3] ReLU
    #   [4] Conv2d_BN(32  → C/2=64)   ← 출력 prunable
    #   [5] ReLU
    #   [6] Conv2d_BN(64  → C=128)    ← 출력 고정 (embed_dim[0])
    # ─────────────────────────────────────────────────────────────────────
    patch_pairs = [
        (model.patch_embed[0], model.patch_embed[2], 0),   # 3→[16]→32
        (model.patch_embed[2], model.patch_embed[4], 1),   # 16→[32]→64
        (model.patch_embed[4], model.patch_embed[6], 2),   # 32→[64]→128
    ]
    for out_layer, in_layer, idx in patch_pairs:
        out_ch = out_layer.c.out_channels
        groups.append({
            'id':         f'patch_embed_{idx}',
            'type':       'G_PATCH',
            'lambda_rec': 0.005,
            'unit_count': out_ch,
            'modules': {
                'out_conv': out_layer,   # Conv2d_BN: 출력 채널 pruning 대상
                'in_conv':  in_layer,    # Conv2d_BN: 입력 채널 pruning 대상
            },
            'meta': {
                'layer_idx':      idx,
                'out_channels':   out_ch,
                'in_ch_next':     in_layer.c.in_channels,
                'note':           'Phase 2+ only',
            },
        })

    # ─────────────────────────────────────────────────────────────────────
    # blocks1: [EVBlock(ed=128)]  — Stage 1, depth=1
    # ─────────────────────────────────────────────────────────────────────
    blk = model.blocks1[0]
    groups.append(_ffn_group(blk.ffn0, 'stage1_block0_ffn0', stage=1, block_label=0, ffn_idx=0))
    groups.append(_ffn_group(blk.ffn1, 'stage1_block0_ffn1', stage=1, block_label=0, ffn_idx=1))
    groups.extend(_cga_groups(blk, stage=1, block_idx=0))

    # ─────────────────────────────────────────────────────────────────────
    # blocks2 layout (M4):
    #   [0] Sequential(Residual(DWConv128), Residual(FFN128))  ← subsample pre
    #   [1] PatchMerging(128 → 256)
    #   [2] Sequential(Residual(DWConv256), Residual(FFN256))  ← subsample post
    #   [3] EVBlock(256)   — Stage 2, block 0
    #   [4] EVBlock(256)   — Stage 2, block 1
    # ─────────────────────────────────────────────────────────────────────
    groups.append(_ffn_group(
        model.blocks2[0][1], 'stage1_subsample_pre_ffn',
        stage=1, block_label='subsample_pre', ffn_idx=0
    ))
    groups.append(_ffn_group(
        model.blocks2[2][1], 'stage2_subsample_post_ffn',
        stage=2, block_label='subsample_post', ffn_idx=0
    ))
    for bi in range(2):
        blk = model.blocks2[3 + bi]
        groups.append(_ffn_group(blk.ffn0, f'stage2_block{bi}_ffn0', stage=2, block_label=bi, ffn_idx=0))
        groups.append(_ffn_group(blk.ffn1, f'stage2_block{bi}_ffn1', stage=2, block_label=bi, ffn_idx=1))
        groups.extend(_cga_groups(blk, stage=2, block_idx=bi))

    # ─────────────────────────────────────────────────────────────────────
    # blocks3 layout (M4):
    #   [0] Sequential(Residual(DWConv256), Residual(FFN256))  ← subsample pre
    #   [1] PatchMerging(256 → 384)
    #   [2] Sequential(Residual(DWConv384), Residual(FFN384))  ← subsample post
    #   [3] EVBlock(384)   — Stage 3, block 0
    #   [4] EVBlock(384)   — Stage 3, block 1
    #   [5] EVBlock(384)   — Stage 3, block 2
    # ─────────────────────────────────────────────────────────────────────
    groups.append(_ffn_group(
        model.blocks3[0][1], 'stage2_subsample_pre_ffn',
        stage=2, block_label='subsample_pre', ffn_idx=0
    ))
    groups.append(_ffn_group(
        model.blocks3[2][1], 'stage3_subsample_post_ffn',
        stage=3, block_label='subsample_post', ffn_idx=0
    ))
    for bi in range(3):
        blk = model.blocks3[3 + bi]
        groups.append(_ffn_group(blk.ffn0, f'stage3_block{bi}_ffn0', stage=3, block_label=bi, ffn_idx=0))
        groups.append(_ffn_group(blk.ffn1, f'stage3_block{bi}_ffn1', stage=3, block_label=bi, ffn_idx=1))
        groups.extend(_cga_groups(blk, stage=3, block_idx=bi))

    return groups


def get_groups_by_type(groups, group_type):
    """특정 type의 그룹만 필터링하여 반환합니다."""
    return [g for g in groups if g['type'] == group_type]


def get_phase1_groups(groups):
    """Phase 1 대상 그룹 (G_FFN + G_QK + G_V, G_PATCH 제외)."""
    return [g for g in groups if g['type'] != 'G_PATCH']
