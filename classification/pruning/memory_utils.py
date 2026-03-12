"""
Memory Utilities — PGM Loss용 메모리 측정

CLAUDE.md §5.1 loss:
  μ · max(0, current_memory_bytes - M_max_bytes)

current_memory_bytes 계산 방법 (두 가지):
  1. compute_model_param_memory  : 전체 파라미터 메모리 (zero 포함)
  2. compute_active_param_memory : 활성 unit만 계산 (zero group 제외)
     → PGM loss에 사용. Reducing 후 실제 모델 크기를 반영.

CLAUDE.md §7 GPU profiling 방법:
  torch.cuda.reset_peak_memory_stats → empty_cache → measure before → run → measure after
"""

import torch
import torch.nn as nn


def compute_model_param_memory(model):
    """
    모델 전체 파라미터 메모리를 bytes로 반환합니다 (float32 기준).
    zero weight 포함 — 훈련 도중 실제 GPU 점유량.
    """
    return sum(p.numel() for p in model.parameters()) * 4


def compute_active_param_memory(groups, threshold=0.0):
    """
    활성 unit만 계산한 파라미터 메모리를 bytes로 반환합니다.

    zero norm인 unit은 이미 pruning된 것으로 간주하여 제외합니다.
    이 값이 PGM loss의 'current_memory_bytes'로 사용됩니다.

    Parameters
    ----------
    groups    : list[dict]  — build_pruning_groups() 반환값
    threshold : float       — norm이 이 값 이하면 zero (pruned)로 간주

    Returns
    -------
    memory_bytes : int
    """
    total_params = 0

    for g in groups:
        t = g['type']
        m = g['modules']
        meta = g['meta']

        if t == 'G_FFN':
            # expand.weight shape: [hidden, ed, 1, 1]
            # 활성 unit = norm > threshold인 hidden neuron 수
            w = m['expand'].c.weight                         # [hid, ed, 1, 1]
            row_norms = w.view(w.shape[0], -1).norm(dim=1)  # [hid]
            n_active = (row_norms > threshold).sum().item()
            ed = meta['ed']
            # expand: n_active * ed  +  shrink: n_active * ed
            total_params += 2 * n_active * ed

        elif t == 'G_QK':
            # qkv.weight shape: [key_dim*2+d, in_ch, 1, 1]
            # Q norms [0:key_dim] → active QK dims
            key_dim = meta['key_dim']
            w = m['qkv'].c.weight
            q_norms = w[:key_dim].view(key_dim, -1).norm(dim=1)
            n_active = (q_norms > threshold).sum().item()
            in_ch = meta['in_channels']
            kH, kW = m['dw'].c.weight.shape[2], m['dw'].c.weight.shape[3]
            # Q+K: 2 * in_ch per dim  +  DW: kH*kW per dim
            total_params += n_active * (2 * in_ch + kH * kW)

        elif t == 'G_V':
            # qkv.weight V slice: [2*key_dim : 2*key_dim+d, in_ch, 1, 1]
            key_dim = meta['key_dim']
            d = meta['d']
            w = m['qkv'].c.weight
            v_w = w[2 * key_dim: 2 * key_dim + d]
            v_norms = v_w.view(d, -1).norm(dim=1)
            n_active = (v_norms > threshold).sum().item()
            in_ch = meta['in_channels']
            # V: in_ch per channel
            total_params += n_active * in_ch

        elif t == 'G_PATCH':
            # out_conv.weight shape: [out_ch, in_ch, kH, kW]
            w = m['out_conv'].c.weight
            num_filters = w.shape[0]
            filter_norms = w.view(num_filters, -1).norm(dim=1)
            n_active = (filter_norms > threshold).sum().item()
            in_ch = w.shape[1]
            kH, kW = w.shape[2], w.shape[3]
            # out_conv filter + in_conv corresponding input
            in_next = m['in_conv'].c.weight.shape[0]  # out_channels of in_conv
            kH2, kW2 = m['in_conv'].c.weight.shape[2], m['in_conv'].c.weight.shape[3]
            total_params += n_active * (in_ch * kH * kW + in_next * kH2 * kW2)

    return total_params * 4  # float32


def profile_gpu_memory(model, device, input_shape=(1, 3, 224, 224)):
    """
    CLAUDE.md §7 방법으로 GPU 메모리를 실측합니다.

    Parameters
    ----------
    model        : nn.Module
    device       : torch.device (cuda)
    input_shape  : tuple

    Returns
    -------
    dict with keys:
      param_memory_bytes     — 파라미터 점유 메모리
      peak_forward_bytes     — forward peak 메모리 (activation 포함)
      activation_bytes       — activation만의 메모리 (peak - param)
    """
    assert device.type == 'cuda', 'GPU profiling requires CUDA device'

    model = model.to(device).eval()
    dummy = torch.zeros(input_shape, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    mem_before = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        _ = model(dummy)

    peak_mem = torch.cuda.max_memory_allocated(device)

    param_mem = sum(p.numel() * 4 for p in model.parameters())
    peak_fwd  = peak_mem - mem_before
    activation = max(0, peak_fwd - param_mem)

    return {
        'param_memory_bytes':  param_mem,
        'peak_forward_bytes':  peak_fwd,
        'activation_bytes':    activation,
    }


def count_zero_groups(groups, threshold=0.0):
    """
    zero norm인 group unit 수를 type별로 집계합니다.
    훈련 도중 pruning 진행 상황 모니터링에 사용합니다.

    Returns
    -------
    stats : dict[type -> {'total_units': int, 'zero_units': int, 'zero_ratio': float}]
    """
    stats = {}
    for g in groups:
        t = g['type']
        if t not in stats:
            stats[t] = {'total_units': 0, 'zero_units': 0}

        m    = g['modules']
        meta = g['meta']
        total = g['unit_count']

        if t == 'G_FFN':
            w = m['expand'].c.weight
            norms = w.view(w.shape[0], -1).norm(dim=1)
        elif t == 'G_QK':
            key_dim = meta['key_dim']
            w = m['qkv'].c.weight
            norms = w[:key_dim].view(key_dim, -1).norm(dim=1)
        elif t == 'G_V':
            key_dim = meta['key_dim']
            d = meta['d']
            w = m['qkv'].c.weight
            norms = w[2 * key_dim: 2 * key_dim + d].view(d, -1).norm(dim=1)
        elif t == 'G_PATCH':
            w = m['out_conv'].c.weight
            norms = w.view(w.shape[0], -1).norm(dim=1)
        else:
            continue

        zero_count = (norms <= threshold).sum().item()
        stats[t]['total_units'] += total
        stats[t]['zero_units']  += zero_count

    for t in stats:
        total = stats[t]['total_units']
        zero  = stats[t]['zero_units']
        stats[t]['zero_ratio'] = zero / total if total > 0 else 0.0

    return stats
