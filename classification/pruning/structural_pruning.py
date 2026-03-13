"""
Iterative Physical (Structural) Pruning for EfficientViT

기존 soft masking 방식의 문제점:
  - 0으로 마스킹해도 실제 연산량/메모리 그대로
  - Gradient가 0인 weight에 계속 흐름
  - Structure extraction까지 실제 효과 확인 불가

해결: 매 epoch마다 물리적으로 구조를 축소
  1. Train (1 epoch) → Loss 감소
  2. Importance 계산 (L2 norm 기반)
  3. 물리적 pruning (Conv 채널 축소, Linear dim 축소)
  4. Fine-tuning (목표 도달까지 반복)

주의사항 (ViT 특성):
  - FFN: expand.out_channels == shrink.in_channels 동기화 필수
  - CGA: Q/K는 동일 dim, V는 head별로 proj와 연결
  - DWConv: groups = in_channels = out_channels 동기화
  - Cascade 구조: head 간 채널 일관성 필요
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


# =============================================================================
# FFN Physical Pruning
# =============================================================================

def compute_ffn_importance(expand_conv: nn.Module, shrink_conv: nn.Module) -> torch.Tensor:
    """
    FFN hidden neuron의 importance 계산 (L2 norm 기반)

    expand.weight: [hidden_dim, embed_dim, 1, 1]
    shrink.weight: [embed_dim, hidden_dim, 1, 1]

    neuron k의 importance = ||expand.weight[k]||_2 + ||shrink.weight[:,k]||_2
    """
    with torch.no_grad():
        # expand의 출력 채널별 norm
        w_expand = expand_conv.c.weight  # [hidden, embed, 1, 1]
        expand_norms = w_expand.view(w_expand.size(0), -1).norm(dim=1)  # [hidden]

        # shrink의 입력 채널별 norm
        w_shrink = shrink_conv.c.weight  # [embed, hidden, 1, 1]
        shrink_norms = w_shrink.view(w_shrink.size(0), -1, 1, 1)
        shrink_norms = shrink_norms.squeeze().norm(dim=0)  # [hidden]

        # 합산
        importance = expand_norms + shrink_norms

    return importance


def prune_ffn_physically(
    ffn: nn.Module,
    keep_ratio: float,
    min_neurons: int = 8
) -> Tuple[int, int]:
    """
    FFN의 hidden dimension을 물리적으로 축소

    Args:
        ffn: FFN module (pw1=expand, pw2=shrink)
        keep_ratio: 유지할 neuron 비율 (0.0 ~ 1.0)
        min_neurons: 최소 유지 neuron 수

    Returns:
        (original_hidden, new_hidden)
    """
    expand = ffn.pw1  # Conv2d_BN
    shrink = ffn.pw2  # Conv2d_BN

    hidden_dim = expand.c.out_channels
    embed_dim = expand.c.in_channels

    # 유지할 neuron 수 계산
    # 최소 1개는 제거하되, min_neurons 이상 유지
    new_hidden = max(min_neurons, int(hidden_dim * keep_ratio))
    new_hidden = min(new_hidden, hidden_dim)  # 원본보다 클 수 없음

    # 최소 1개는 pruning (진행 보장), 단 min_neurons 제약 우선
    if new_hidden == hidden_dim and hidden_dim > min_neurons:
        new_hidden = hidden_dim - 1

    if new_hidden >= hidden_dim:
        return hidden_dim, hidden_dim  # pruning 불필요 (이미 최소)

    # Importance 계산 및 상위 neuron 선택
    importance = compute_ffn_importance(expand, shrink)
    _, keep_indices = torch.topk(importance, new_hidden, largest=True)
    keep_indices = keep_indices.sort().values  # 순서 유지

    with torch.no_grad():
        # === Expand (pw1) 축소 ===
        # Conv weight: [hidden, embed, 1, 1] → [new_hidden, embed, 1, 1]
        new_expand_weight = expand.c.weight.data[keep_indices]

        # BN parameters: [hidden] → [new_hidden]
        new_expand_bn_weight = expand.bn.weight.data[keep_indices]
        new_expand_bn_bias = expand.bn.bias.data[keep_indices]
        new_expand_bn_mean = expand.bn.running_mean.data[keep_indices]
        new_expand_bn_var = expand.bn.running_var.data[keep_indices]

        # 새 Conv2d 생성 (bias=False)
        new_expand_conv = nn.Conv2d(
            embed_dim, new_hidden, kernel_size=1, bias=False
        )
        new_expand_conv.weight.data = new_expand_weight

        # 새 BN 생성
        new_expand_bn = nn.BatchNorm2d(new_hidden)
        new_expand_bn.weight.data = new_expand_bn_weight
        new_expand_bn.bias.data = new_expand_bn_bias
        new_expand_bn.running_mean.data = new_expand_bn_mean
        new_expand_bn.running_var.data = new_expand_bn_var
        new_expand_bn.num_batches_tracked = expand.bn.num_batches_tracked.clone()

        # expand 모듈 교체
        expand.c = new_expand_conv
        expand.bn = new_expand_bn

        # === Shrink (pw2) 축소 ===
        # Conv weight: [embed, hidden, 1, 1] → [embed, new_hidden, 1, 1]
        new_shrink_weight = shrink.c.weight.data[:, keep_indices]

        # 새 Conv2d 생성
        new_shrink_conv = nn.Conv2d(
            new_hidden, embed_dim, kernel_size=1, bias=False
        )
        new_shrink_conv.weight.data = new_shrink_weight

        # BN은 출력 채널 기준이므로 그대로 유지 (embed_dim 변화 없음)

        # shrink 모듈 교체
        shrink.c = new_shrink_conv

    return hidden_dim, new_hidden


# =============================================================================
# CGA (Q/K/V) Physical Pruning
# =============================================================================

def compute_qk_importance(qkv_conv: nn.Module, dw_conv: nn.Module, key_dim: int) -> torch.Tensor:
    """
    Q/K dimension의 importance 계산

    qkv.weight: [key_dim*2 + d, in_ch, 1, 1]
      - Q: [0 : key_dim]
      - K: [key_dim : 2*key_dim]

    dim i의 importance = ||Q[i]||_2 + ||K[i]||_2 + ||DW[i]||_2
    """
    with torch.no_grad():
        w_qkv = qkv_conv.c.weight  # [key_dim*2+d, in_ch, 1, 1]
        w_dw = dw_conv.c.weight    # [key_dim, 1, kH, kW]

        # Q importance
        q_weights = w_qkv[:key_dim]
        q_norms = q_weights.view(key_dim, -1).norm(dim=1)

        # K importance
        k_weights = w_qkv[key_dim:2*key_dim]
        k_norms = k_weights.view(key_dim, -1).norm(dim=1)

        # DW importance
        dw_norms = w_dw.view(key_dim, -1).norm(dim=1)

        importance = q_norms + k_norms + dw_norms

    return importance


def compute_v_importance(qkv_conv: nn.Module, key_dim: int, d: int) -> torch.Tensor:
    """
    V dimension의 importance 계산

    qkv.weight V slice: [2*key_dim : 2*key_dim + d]
    """
    with torch.no_grad():
        w_qkv = qkv_conv.c.weight
        v_start = 2 * key_dim
        v_weights = w_qkv[v_start:v_start + d]
        v_norms = v_weights.view(d, -1).norm(dim=1)

    return v_norms


def prune_cga_head_qk_physically(
    cga: nn.Module,
    head_idx: int,
    keep_ratio: float,
    min_dim: int = 4
) -> Tuple[int, int]:
    """
    특정 head의 Q/K dimension을 물리적으로 축소

    CRITICAL: Q와 K는 반드시 동일한 dimension으로 축소 (QK^T 연산 때문)

    Args:
        cga: CascadedGroupAttention module
        head_idx: 축소할 head index
        keep_ratio: 유지할 dim 비율
        min_dim: 최소 유지 dimension

    Returns:
        (original_key_dim, new_key_dim)
    """
    qkv = cga.qkvs[head_idx]  # Conv2d_BN
    dw = cga.dws[head_idx]    # Conv2d_BN

    key_dim = cga.key_dim
    d = cga.d
    in_ch = qkv.c.in_channels

    # 새 key_dim 계산
    # 최소 1개는 제거하되, min_dim 이상 유지
    new_key_dim = max(min_dim, int(key_dim * keep_ratio))
    new_key_dim = min(new_key_dim, key_dim)

    # 최소 1개는 pruning (진행 보장), 단 min_dim 제약 우선
    if new_key_dim == key_dim and key_dim > min_dim:
        new_key_dim = key_dim - 1

    if new_key_dim >= key_dim:
        return key_dim, key_dim  # pruning 불필요 (이미 최소)

    # Importance 계산 및 상위 dim 선택
    importance = compute_qk_importance(qkv, dw, key_dim)
    _, keep_indices = torch.topk(importance, new_key_dim, largest=True)
    keep_indices = keep_indices.sort().values

    with torch.no_grad():
        # === QKV Conv 수정 ===
        w_qkv = qkv.c.weight  # [key_dim*2+d, in_ch, 1, 1]

        # Q, K, V 분리
        q_weights = w_qkv[:key_dim]           # [key_dim, in_ch, 1, 1]
        k_weights = w_qkv[key_dim:2*key_dim]  # [key_dim, in_ch, 1, 1]
        v_weights = w_qkv[2*key_dim:]         # [d, in_ch, 1, 1]

        # Q, K pruning (동일 indices)
        new_q = q_weights[keep_indices]
        new_k = k_weights[keep_indices]

        # 새 weight 조합
        new_qkv_weight = torch.cat([new_q, new_k, v_weights], dim=0)

        # 새 Conv2d 생성
        new_out_ch = new_key_dim * 2 + d
        new_qkv_conv = nn.Conv2d(in_ch, new_out_ch, kernel_size=1, bias=False)
        new_qkv_conv.weight.data = new_qkv_weight

        # BN도 수정
        bn_w = qkv.bn.weight.data
        bn_b = qkv.bn.bias.data
        bn_mean = qkv.bn.running_mean.data
        bn_var = qkv.bn.running_var.data

        # Q, K, V 분리 후 재조합
        new_bn_w = torch.cat([bn_w[:key_dim][keep_indices],
                              bn_w[key_dim:2*key_dim][keep_indices],
                              bn_w[2*key_dim:]])
        new_bn_b = torch.cat([bn_b[:key_dim][keep_indices],
                              bn_b[key_dim:2*key_dim][keep_indices],
                              bn_b[2*key_dim:]])
        new_bn_mean = torch.cat([bn_mean[:key_dim][keep_indices],
                                  bn_mean[key_dim:2*key_dim][keep_indices],
                                  bn_mean[2*key_dim:]])
        new_bn_var = torch.cat([bn_var[:key_dim][keep_indices],
                                 bn_var[key_dim:2*key_dim][keep_indices],
                                 bn_var[2*key_dim:]])

        new_qkv_bn = nn.BatchNorm2d(new_out_ch)
        new_qkv_bn.weight.data = new_bn_w
        new_qkv_bn.bias.data = new_bn_b
        new_qkv_bn.running_mean.data = new_bn_mean
        new_qkv_bn.running_var.data = new_bn_var
        new_qkv_bn.num_batches_tracked = qkv.bn.num_batches_tracked.clone()

        qkv.c = new_qkv_conv
        qkv.bn = new_qkv_bn

        # === DW Conv 수정 (Q에만 적용) ===
        w_dw = dw.c.weight  # [key_dim, 1, kH, kW]
        new_dw_weight = w_dw[keep_indices]

        kernel_size = dw.c.kernel_size
        padding = dw.c.padding

        new_dw_conv = nn.Conv2d(
            new_key_dim, new_key_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=new_key_dim,
            bias=False
        )
        new_dw_conv.weight.data = new_dw_weight

        # DW BN
        new_dw_bn = nn.BatchNorm2d(new_key_dim)
        new_dw_bn.weight.data = dw.bn.weight.data[keep_indices]
        new_dw_bn.bias.data = dw.bn.bias.data[keep_indices]
        new_dw_bn.running_mean.data = dw.bn.running_mean.data[keep_indices]
        new_dw_bn.running_var.data = dw.bn.running_var.data[keep_indices]
        new_dw_bn.num_batches_tracked = dw.bn.num_batches_tracked.clone()

        dw.c = new_dw_conv
        dw.bn = new_dw_bn

    # CRITICAL: CGA의 key_dim 속성도 업데이트 (첫 번째 head pruning 시에만)
    # 모든 head는 동일한 key_dim을 공유해야 함
    if head_idx == 0:
        cga.key_dim = new_key_dim
        # scale도 업데이트 (attention score 계산에 사용)
        cga.scale = new_key_dim ** -0.5

    return key_dim, new_key_dim


def prune_cga_head_v_physically(
    cga: nn.Module,
    head_idx: int,
    keep_ratio: float,
    min_dim: int = 4
) -> Tuple[int, int]:
    """
    특정 head의 V dimension을 물리적으로 축소

    CRITICAL: V를 줄이면 proj의 입력도 함께 수정해야 함
             하지만 proj는 모든 head를 concat하므로 복잡함
             → 단순화: V는 head별로 독립 축소 후, proj 전체 재구성

    현재 구현: V dimension은 head별로 다르게 설정 가능하도록 확장 필요
              일단 모든 head 동일 비율로 축소
    """
    qkv = cga.qkvs[head_idx]

    key_dim = cga.key_dim
    d = cga.d  # V dimension per head
    in_ch = qkv.c.in_channels

    new_d = max(min_dim, int(d * keep_ratio))
    new_d = min(new_d, d)

    if new_d == d:
        return d, d

    # V importance
    importance = compute_v_importance(qkv, key_dim, d)
    _, keep_indices = torch.topk(importance, new_d, largest=True)
    keep_indices = keep_indices.sort().values

    with torch.no_grad():
        w_qkv = qkv.c.weight

        # Q, K, V 분리
        q_weights = w_qkv[:key_dim]
        k_weights = w_qkv[key_dim:2*key_dim]
        v_weights = w_qkv[2*key_dim:2*key_dim + d]

        # V pruning
        new_v = v_weights[keep_indices]

        # 새 weight 조합
        new_qkv_weight = torch.cat([q_weights, k_weights, new_v], dim=0)

        new_out_ch = key_dim * 2 + new_d
        new_qkv_conv = nn.Conv2d(in_ch, new_out_ch, kernel_size=1, bias=False)
        new_qkv_conv.weight.data = new_qkv_weight

        # BN 수정
        bn_w = qkv.bn.weight.data
        bn_b = qkv.bn.bias.data
        bn_mean = qkv.bn.running_mean.data
        bn_var = qkv.bn.running_var.data

        new_bn_w = torch.cat([bn_w[:2*key_dim], bn_w[2*key_dim:][keep_indices]])
        new_bn_b = torch.cat([bn_b[:2*key_dim], bn_b[2*key_dim:][keep_indices]])
        new_bn_mean = torch.cat([bn_mean[:2*key_dim], bn_mean[2*key_dim:][keep_indices]])
        new_bn_var = torch.cat([bn_var[:2*key_dim], bn_var[2*key_dim:][keep_indices]])

        new_qkv_bn = nn.BatchNorm2d(new_out_ch)
        new_qkv_bn.weight.data = new_bn_w
        new_qkv_bn.bias.data = new_bn_b
        new_qkv_bn.running_mean.data = new_bn_mean
        new_qkv_bn.running_var.data = new_bn_var
        new_qkv_bn.num_batches_tracked = qkv.bn.num_batches_tracked.clone()

        qkv.c = new_qkv_conv
        qkv.bn = new_qkv_bn

    return d, new_d


def prune_cga_proj_for_v_change(
    cga: nn.Module,
    old_d_per_head: int,
    new_d_per_head: int,
    keep_indices_per_head: List[torch.Tensor]
):
    """
    V dimension 변경 후 proj layer 수정

    proj: Sequential(ReLU, Conv2d_BN(d*num_heads -> dim))

    Args:
        cga: CascadedGroupAttention
        old_d_per_head: 이전 V dimension per head
        new_d_per_head: 새 V dimension per head
        keep_indices_per_head: 각 head에서 유지할 V channel indices
    """
    proj_conv_bn = cga.proj[1]  # Conv2d_BN
    num_heads = cga.num_heads
    dim = proj_conv_bn.c.out_channels

    old_total_d = old_d_per_head * num_heads
    new_total_d = new_d_per_head * num_heads

    with torch.no_grad():
        # proj weight: [dim, old_total_d, 1, 1]
        w = proj_conv_bn.c.weight  # [dim, d*num_heads, 1, 1]

        # head별로 유지할 indices 수집
        all_keep_indices = []
        for h in range(num_heads):
            offset = h * old_d_per_head
            head_indices = keep_indices_per_head[h] + offset
            all_keep_indices.append(head_indices)

        keep_indices = torch.cat(all_keep_indices)

        # 새 weight
        new_w = w[:, keep_indices]

        new_proj_conv = nn.Conv2d(new_total_d, dim, kernel_size=1, bias=False)
        new_proj_conv.weight.data = new_w

        proj_conv_bn.c = new_proj_conv
        # BN은 출력 채널(dim) 기준이므로 변경 불필요


# =============================================================================
# Full Model Physical Pruning
# =============================================================================

def compute_model_size_mb(model: nn.Module) -> float:
    """모델 파라미터 크기 계산 (MB)"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 * 1024)  # float32 = 4 bytes


def prune_efficientvit_block_ffn(
    block: nn.Module,
    ffn_keep_ratio: float,
    verbose: bool = False
) -> Dict[str, Tuple[int, int]]:
    """
    EfficientViTBlock의 FFN들을 물리적으로 축소

    Returns:
        {ffn_name: (original_hidden, new_hidden)}
    """
    results = {}

    # ffn0
    if hasattr(block, 'ffn0'):
        ffn = block.ffn0.m  # Residual.m -> FFN
        orig, new = prune_ffn_physically(ffn, ffn_keep_ratio)
        results['ffn0'] = (orig, new)
        if verbose:
            print(f"    FFN0: {orig} -> {new} ({new/orig*100:.1f}%)")

    # ffn1
    if hasattr(block, 'ffn1'):
        ffn = block.ffn1.m
        orig, new = prune_ffn_physically(ffn, ffn_keep_ratio)
        results['ffn1'] = (orig, new)
        if verbose:
            print(f"    FFN1: {orig} -> {new} ({new/orig*100:.1f}%)")

    return results


def prune_efficientvit_block_cga(
    block: nn.Module,
    qk_keep_ratio: float,
    v_keep_ratio: float,
    verbose: bool = False
) -> Dict[str, any]:
    """
    EfficientViTBlock의 CGA Q/K dimensions를 물리적으로 축소

    CRITICAL: CGA의 모든 head는 동일한 key_dim을 공유합니다.
    따라서 모든 head에서 동일한 indices를 pruning해야 합니다.

    전략: 모든 head의 importance를 합산하여 global importance 계산,
          그 기반으로 통일된 keep_indices 결정

    Note: V pruning은 복잡성으로 인해 별도 처리 필요 (proj 연동)
          현재는 QK만 지원
    """
    results = {'qk': [], 'v': []}

    if hasattr(block, 'mixer'):
        cga = block.mixer.m.attn  # Residual.m -> LocalWindowAttention -> attn

        key_dim = cga.key_dim
        num_heads = cga.num_heads
        min_dim = 4

        # 새 key_dim 계산
        new_key_dim = max(min_dim, int(key_dim * qk_keep_ratio))
        new_key_dim = min(new_key_dim, key_dim)

        # 최소 1개는 pruning 보장 (진행 위해)
        if new_key_dim == key_dim and key_dim > min_dim:
            new_key_dim = key_dim - 1

        if new_key_dim >= key_dim:
            # Pruning 불필요
            for h in range(num_heads):
                results['qk'].append((key_dim, key_dim))
            return results

        # === GLOBAL IMPORTANCE 계산 ===
        # 모든 head의 Q/K importance를 합산
        global_importance = torch.zeros(key_dim, device=cga.qkvs[0].c.weight.device)

        for h in range(num_heads):
            qkv = cga.qkvs[h]
            dw = cga.dws[h]
            head_importance = compute_qk_importance(qkv, dw, key_dim)
            global_importance += head_importance

        # Global importance 기반으로 통일된 keep_indices 결정
        _, keep_indices = torch.topk(global_importance, new_key_dim, largest=True)
        keep_indices = keep_indices.sort().values

        if verbose:
            print(f"    CGA Global QK: {key_dim} -> {new_key_dim} (all {num_heads} heads)")

        # === 모든 HEAD에 동일한 INDICES 적용 ===
        with torch.no_grad():
            for h in range(num_heads):
                qkv = cga.qkvs[h]
                dw = cga.dws[h]
                d = cga.d
                in_ch = qkv.c.in_channels

                # QKV Conv 수정
                w_qkv = qkv.c.weight  # [key_dim*2+d, in_ch, 1, 1]

                # Q, K, V 분리
                q_weights = w_qkv[:key_dim]
                k_weights = w_qkv[key_dim:2*key_dim]
                v_weights = w_qkv[2*key_dim:]

                # Q, K pruning (동일 indices)
                new_q = q_weights[keep_indices]
                new_k = k_weights[keep_indices]

                # 새 weight 조합
                new_qkv_weight = torch.cat([new_q, new_k, v_weights], dim=0)

                # 새 Conv2d 생성
                new_out_ch = new_key_dim * 2 + d
                new_qkv_conv = nn.Conv2d(in_ch, new_out_ch, kernel_size=1, bias=False)
                new_qkv_conv.weight.data = new_qkv_weight

                # BN도 수정
                bn_w = qkv.bn.weight.data
                bn_b = qkv.bn.bias.data
                bn_mean = qkv.bn.running_mean.data
                bn_var = qkv.bn.running_var.data

                new_bn_w = torch.cat([bn_w[:key_dim][keep_indices],
                                      bn_w[key_dim:2*key_dim][keep_indices],
                                      bn_w[2*key_dim:]])
                new_bn_b = torch.cat([bn_b[:key_dim][keep_indices],
                                      bn_b[key_dim:2*key_dim][keep_indices],
                                      bn_b[2*key_dim:]])
                new_bn_mean = torch.cat([bn_mean[:key_dim][keep_indices],
                                         bn_mean[key_dim:2*key_dim][keep_indices],
                                         bn_mean[2*key_dim:]])
                new_bn_var = torch.cat([bn_var[:key_dim][keep_indices],
                                        bn_var[key_dim:2*key_dim][keep_indices],
                                        bn_var[2*key_dim:]])

                new_qkv_bn = nn.BatchNorm2d(new_out_ch)
                new_qkv_bn.weight.data = new_bn_w
                new_qkv_bn.bias.data = new_bn_b
                new_qkv_bn.running_mean.data = new_bn_mean
                new_qkv_bn.running_var.data = new_bn_var
                new_qkv_bn.num_batches_tracked = qkv.bn.num_batches_tracked.clone()

                qkv.c = new_qkv_conv
                qkv.bn = new_qkv_bn

                # DW Conv 수정
                w_dw = dw.c.weight  # [key_dim, 1, kH, kW]
                new_dw_weight = w_dw[keep_indices]

                kernel_size = dw.c.kernel_size
                padding = dw.c.padding

                new_dw_conv = nn.Conv2d(
                    new_key_dim, new_key_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=new_key_dim,
                    bias=False
                )
                new_dw_conv.weight.data = new_dw_weight

                # DW BN
                new_dw_bn = nn.BatchNorm2d(new_key_dim)
                new_dw_bn.weight.data = dw.bn.weight.data[keep_indices]
                new_dw_bn.bias.data = dw.bn.bias.data[keep_indices]
                new_dw_bn.running_mean.data = dw.bn.running_mean.data[keep_indices]
                new_dw_bn.running_var.data = dw.bn.running_var.data[keep_indices]
                new_dw_bn.num_batches_tracked = dw.bn.num_batches_tracked.clone()

                dw.c = new_dw_conv
                dw.bn = new_dw_bn

                results['qk'].append((key_dim, new_key_dim))

        # CRITICAL: CGA의 key_dim 속성 업데이트 (모든 head 처리 후)
        cga.key_dim = new_key_dim
        cga.scale = new_key_dim ** -0.5

    return results


def apply_iterative_physical_pruning(
    model: nn.Module,
    ffn_prune_rate: float = 0.1,    # 이번 iteration에서 FFN hidden 축소 비율
    qk_prune_rate: float = 0.1,     # 이번 iteration에서 QK dim 축소 비율
    verbose: bool = True
) -> Dict[str, any]:
    """
    전체 모델에 한 iteration의 physical pruning 적용

    Args:
        model: EfficientViT 모델
        ffn_prune_rate: FFN hidden을 (1 - rate) 만큼 유지
        qk_prune_rate: QK dim을 (1 - rate) 만큼 유지
        verbose: 로그 출력 여부

    Returns:
        pruning 결과 통계
    """
    ffn_keep = 1.0 - ffn_prune_rate
    qk_keep = 1.0 - qk_prune_rate

    results = {
        'blocks1': [],
        'blocks2': [],
        'blocks3': [],
        'original_size_mb': compute_model_size_mb(model),
    }

    if verbose:
        print(f"\n[Physical Pruning] FFN keep={ffn_keep:.2f}, QK keep={qk_keep:.2f}")
        print(f"  Original size: {results['original_size_mb']:.2f} MB")

    # === blocks1 (Stage 1) ===
    if verbose:
        print("\n  Stage 1:")
    for i, block in enumerate(model.blocks1):
        if hasattr(block, 'ffn0'):  # EfficientViTBlock
            if verbose:
                print(f"  Block {i}:")
            ffn_res = prune_efficientvit_block_ffn(block, ffn_keep, verbose)
            cga_res = prune_efficientvit_block_cga(block, qk_keep, 1.0, verbose)  # V는 유지
            results['blocks1'].append({'ffn': ffn_res, 'cga': cga_res})

    # === blocks2 (Stage 2) ===
    if verbose:
        print("\n  Stage 2:")
    for i, block in enumerate(model.blocks2):
        # Subsample FFN (Sequential[Residual(DWConv), Residual(FFN)])
        if isinstance(block, nn.Sequential):
            for j, sub in enumerate(block):
                if hasattr(sub, 'm') and hasattr(sub.m, 'pw1'):  # FFN inside Residual
                    orig, new = prune_ffn_physically(sub.m, ffn_keep)
                    if verbose:
                        print(f"  Subsample[{i}] FFN: {orig} -> {new}")
        # EfficientViTBlock
        elif hasattr(block, 'ffn0'):
            if verbose:
                print(f"  Block {i}:")
            ffn_res = prune_efficientvit_block_ffn(block, ffn_keep, verbose)
            cga_res = prune_efficientvit_block_cga(block, qk_keep, 1.0, verbose)
            results['blocks2'].append({'ffn': ffn_res, 'cga': cga_res})

    # === blocks3 (Stage 3) ===
    if verbose:
        print("\n  Stage 3:")
    for i, block in enumerate(model.blocks3):
        if isinstance(block, nn.Sequential):
            for j, sub in enumerate(block):
                if hasattr(sub, 'm') and hasattr(sub.m, 'pw1'):
                    orig, new = prune_ffn_physically(sub.m, ffn_keep)
                    if verbose:
                        print(f"  Subsample[{i}] FFN: {orig} -> {new}")
        elif hasattr(block, 'ffn0'):
            if verbose:
                print(f"  Block {i}:")
            ffn_res = prune_efficientvit_block_ffn(block, ffn_keep, verbose)
            cga_res = prune_efficientvit_block_cga(block, qk_keep, 1.0, verbose)
            results['blocks3'].append({'ffn': ffn_res, 'cga': cga_res})

    results['new_size_mb'] = compute_model_size_mb(model)
    results['reduction_ratio'] = 1.0 - results['new_size_mb'] / results['original_size_mb']

    if verbose:
        print(f"\n  New size: {results['new_size_mb']:.2f} MB")
        print(f"  Reduction: {results['reduction_ratio']*100:.1f}%")

    return results


def validate_model_forward(model: nn.Module, device: str = 'cuda') -> bool:
    """
    Pruning 후 모델 forward pass 검증
    """
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            out = model(dummy)
            assert out.shape == (1, 1000), f"Unexpected output shape: {out.shape}"
        return True
    except Exception as e:
        print(f"[ERROR] Forward pass failed: {e}")
        return False


# =============================================================================
# Iterative Training with Physical Pruning
# =============================================================================

class IterativePhysicalPruner:
    """
    Iterative Physical Pruning을 관리하는 클래스

    Usage:
        pruner = IterativePhysicalPruner(
            target_reduction=0.76,  # 76% 압축 목표
            ffn_prune_per_epoch=0.05,  # epoch당 FFN 5% 축소
            qk_prune_per_epoch=0.08,   # epoch당 QK 8% 축소
        )

        for epoch in range(epochs):
            train_one_epoch(model, ...)
            pruner.step(model)

            if pruner.target_reached:
                break

        # Fine-tuning
        for epoch in range(finetune_epochs):
            train_one_epoch(model, ...)
    """

    def __init__(
        self,
        target_reduction: float = 0.76,
        ffn_prune_per_epoch: float = 0.25,   # FFN: 매우 공격적 (epoch당 25% 제거)
        qk_prune_per_epoch: float = 0.05,    # QK: epoch당 5% 제거
        min_ffn_ratio: float = 0.05,         # 최소 FFN 5% 유지 (최대 95% pruning)
        min_qk_ratio: float = 0.25,          # 최소 QK 25% 유지
        warmup_epochs: int = 0,              # pruning 시작 전 warmup
        verbose: bool = True
    ):
        self.target_reduction = target_reduction
        self.ffn_prune_per_epoch = ffn_prune_per_epoch
        self.qk_prune_per_epoch = qk_prune_per_epoch
        self.min_ffn_ratio = min_ffn_ratio
        self.min_qk_ratio = min_qk_ratio
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

        self.epoch = 0
        self.original_size_mb = None
        self.current_size_mb = None
        self.target_reached = False

        # 누적 pruning 비율 추적
        self.cumulative_ffn_pruned = 0.0
        self.cumulative_qk_pruned = 0.0

        self.history = []

    def step(self, model: nn.Module, device: str = 'cuda') -> Dict:
        """
        한 epoch의 physical pruning 수행

        Returns:
            이번 epoch의 pruning 결과
        """
        self.epoch += 1

        if self.original_size_mb is None:
            self.original_size_mb = compute_model_size_mb(model)

        # Warmup 기간에는 pruning 안 함
        if self.epoch <= self.warmup_epochs:
            if self.verbose:
                print(f"[Epoch {self.epoch}] Warmup - no pruning")
            return {'status': 'warmup'}

        # 이미 목표 도달 시 pruning 중단
        if self.target_reached:
            if self.verbose:
                print(f"[Epoch {self.epoch}] Target reached - no pruning")
            return {'status': 'target_reached'}

        # 현재 compression ratio 확인
        self.current_size_mb = compute_model_size_mb(model)
        current_reduction = 1.0 - self.current_size_mb / self.original_size_mb

        if current_reduction >= self.target_reduction:
            self.target_reached = True
            if self.verbose:
                print(f"[Epoch {self.epoch}] Target reached: {current_reduction*100:.1f}% >= {self.target_reduction*100:.1f}%")
            return {'status': 'target_reached', 'reduction': current_reduction}

        # 남은 pruning 여유 계산
        remaining_ffn = 1.0 - self.cumulative_ffn_pruned - self.min_ffn_ratio
        remaining_qk = 1.0 - self.cumulative_qk_pruned - self.min_qk_ratio

        # 이번 epoch pruning rate (최대/최소 제한)
        ffn_rate = min(self.ffn_prune_per_epoch, max(0, remaining_ffn))
        qk_rate = min(self.qk_prune_per_epoch, max(0, remaining_qk))

        if ffn_rate <= 0 and qk_rate <= 0:
            if self.verbose:
                print(f"[Epoch {self.epoch}] Min ratio reached - no more pruning possible")
            return {'status': 'min_reached'}

        # Physical pruning 적용
        result = apply_iterative_physical_pruning(
            model,
            ffn_prune_rate=ffn_rate,
            qk_prune_rate=qk_rate,
            verbose=self.verbose
        )

        # 누적 업데이트
        self.cumulative_ffn_pruned += ffn_rate
        self.cumulative_qk_pruned += qk_rate

        # Forward pass 검증
        valid = validate_model_forward(model, device)
        result['valid'] = valid
        result['epoch'] = self.epoch
        result['cumulative_reduction'] = 1.0 - result['new_size_mb'] / self.original_size_mb

        self.history.append(result)

        # 목표 도달 확인
        if result['cumulative_reduction'] >= self.target_reduction:
            self.target_reached = True
            if self.verbose:
                print(f"  >>> TARGET REACHED: {result['cumulative_reduction']*100:.1f}%")

        return result

    def get_summary(self) -> Dict:
        """전체 pruning 히스토리 요약"""
        return {
            'total_epochs': self.epoch,
            'target_reduction': self.target_reduction,
            'achieved_reduction': 1.0 - self.current_size_mb / self.original_size_mb if self.current_size_mb else 0,
            'original_size_mb': self.original_size_mb,
            'final_size_mb': self.current_size_mb,
            'target_reached': self.target_reached,
            'history': self.history
        }


# =============================================================================
# Entry point for testing
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/junsu/EfficientVIT_Compression')

    from classification.model.build import EfficientViT_M4

    print("=" * 60)
    print("Iterative Physical Pruning Test")
    print("=" * 60)

    # 모델 로드
    model = EfficientViT_M4(pretrained=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(f"\nOriginal model size: {compute_model_size_mb(model):.2f} MB")

    # Pruner 초기화
    pruner = IterativePhysicalPruner(
        target_reduction=0.50,  # 테스트: 50% 압축
        ffn_prune_per_epoch=0.1,
        qk_prune_per_epoch=0.15,
        verbose=True
    )

    # 시뮬레이션: 10 epochs
    for epoch in range(10):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}")
        print('='*40)

        # (실제로는 여기서 train_one_epoch 호출)

        result = pruner.step(model, device)

        if pruner.target_reached:
            print("\nTarget reached! Stopping pruning.")
            break

    print("\n" + "=" * 60)
    print("Summary:")
    summary = pruner.get_summary()
    print(f"  Original: {summary['original_size_mb']:.2f} MB")
    print(f"  Final: {summary['final_size_mb']:.2f} MB")
    print(f"  Reduction: {summary['achieved_reduction']*100:.1f}%")
    print(f"  Target reached: {summary['target_reached']}")
