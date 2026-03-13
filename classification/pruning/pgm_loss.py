"""
Phase B: PGM Loss for EfficientViT Pruning

YOLO pruning 방식을 참고하여 ViT에 L2-norm 기반 structured pruning을 적용합니다.
그룹 단위로 동일한 알고리즘을 사용하며, 타입별 λ로 조정합니다.

참고: YOLO pruning/compression.py, pruning_common.py
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


def get_group_pruning_idx(group: Dict[str, Any], sparsity: float) -> torch.Tensor:
    """
    그룹의 units에 대해 L2 norm 기반 pruning idx 계산 (YOLO 방식 참고)

    Args:
        group: pruning group dict (group_dict.py 참고)
        sparsity: 제거할 unit 비율 (0.0 ~ 1.0)

    Returns:
        pruning_idx: 제거할 unit 인덱스
    """
    with torch.no_grad():
        gtype = group['type']
        m = group['modules']  # 딕셔너리: {'expand': module, 'shrink': module} 등
        meta = group['meta']

        # 그룹 타입별로 unit norms 계산
        unit_norms = []

        if gtype == 'G_FFN':
            # expand.weight shape: [hidden_dim, ed, 1, 1]
            w_expand = m['expand'].c.weight
            for i in range(group['unit_count']):  # hidden_dim만큼
                norm_sq = torch.norm(w_expand[i, :, :, :]) ** 2
                unit_norms.append(norm_sq ** 0.5)

        elif gtype == 'G_QK':
            # qkv.weight shape: [key_dim*2+d, in_ch, 1, 1]
            # Q slice: [0:key_dim]
            key_dim = meta['key_dim']
            w_qkv = m['qkv'].c.weight
            for i in range(group['unit_count']):  # key_dim만큼
                # Q dim i의 norm
                norm_sq = torch.norm(w_qkv[i, :, :, :]) ** 2
                unit_norms.append(norm_sq ** 0.5)

        elif gtype == 'G_V':
            # qkv.weight V slice: [2*key_dim : 2*key_dim+d]
            key_dim = meta['key_dim']
            d = meta['d']
            w_qkv = m['qkv'].c.weight
            v_start = 2 * key_dim
            for i in range(group['unit_count']):  # d만큼
                norm_sq = torch.norm(w_qkv[v_start + i, :, :, :]) ** 2
                unit_norms.append(norm_sq ** 0.5)

        elif gtype == 'G_PATCH':
            # out_conv.weight shape: [out_ch, in_ch, kH, kW]
            w_out = m['out_conv'].c.weight
            for i in range(group['unit_count']):
                norm_sq = torch.norm(w_out[i, :, :, :]) ** 2
                unit_norms.append(norm_sq ** 0.5)
        else:
            # Unknown type
            return torch.tensor([], dtype=torch.long)

        unit_norms = torch.tensor(unit_norms)
        num_pruning = int(group['unit_count'] * sparsity)

        # 최소 1개는 제거하도록 보장 (sparsity가 너무 작아 0이 되는 경우 방지)
        # 단, unit_count가 충분히 크고(>10) sparsity > 0일 때만
        if num_pruning == 0 and sparsity > 0 and group['unit_count'] > 10:
            num_pruning = max(1, int(group['unit_count'] * sparsity + 0.5))  # 반올림

        # num_pruning이 여전히 0이면 빈 텐서 반환
        if num_pruning == 0:
            return torch.tensor([], dtype=torch.long)

        # num_pruning이 unit_count보다 크면 제한
        num_pruning = min(num_pruning, group['unit_count'])

        # 가장 작은 norm 가진 units 선택
        _, pruning_idx = torch.topk(unit_norms, num_pruning, largest=False)

    return pruning_idx


def apply_group_pruning(group: Dict[str, Any], pruning_idx: torch.Tensor):
    """
    그룹의 지정된 units를 0으로 마스킹하고, gradient도 0으로 유지합니다.

    CRITICAL: Pruned weights는 0으로 설정되지만, 다음 iteration에서 gradient로 복원됩니다.
    → 해결: pruned_mask를 group에 저장하고, gradient를 매번 0으로 만듭니다.

    Args:
        group: pruning group dict
        pruning_idx: 제거할 unit 인덱스
    """
    # pruned_mask 초기화 (처음 pruning 시)
    if 'pruned_mask' not in group:
        group['pruned_mask'] = set()

    # 현재 pruning indices를 mask에 추가 (누적)
    for idx in pruning_idx.cpu().tolist():
        group['pruned_mask'].add(idx)

    with torch.no_grad():
        gtype = group['type']
        m = group['modules']
        meta = group['meta']

        if gtype == 'G_FFN':
            # expand.weight[pruning_idx] = 0 (출력 채널 제거)
            w_expand = m['expand'].c.weight
            w_expand[pruning_idx, :, :, :] = 0.0
            if hasattr(m['expand'].c, 'bias') and m['expand'].c.bias is not None:
                m['expand'].c.bias[pruning_idx] = 0.0
            # expand BN
            if hasattr(m['expand'], 'bn'):
                bn = m['expand'].bn
                bn.weight[pruning_idx] = 0.0
                bn.bias[pruning_idx] = 0.0
                bn.running_mean[pruning_idx] = 0.0
                bn.running_var[pruning_idx] = 1.0

            # shrink.weight[:, pruning_idx] = 0 (입력 채널 제거)
            w_shrink = m['shrink'].c.weight
            w_shrink[:, pruning_idx, :, :] = 0.0

        elif gtype == 'G_QK':
            # qkv Q slice [pruning_idx] = 0
            # qkv K slice [key_dim + pruning_idx] = 0
            key_dim = meta['key_dim']
            w_qkv = m['qkv'].c.weight
            w_qkv[pruning_idx, :, :, :] = 0.0  # Q
            w_qkv[key_dim + pruning_idx, :, :, :] = 0.0  # K
            # qkv BN (Q+K dims 포함)
            if hasattr(m['qkv'], 'bn'):
                bn = m['qkv'].bn
                bn.weight[pruning_idx] = 0.0
                bn.bias[pruning_idx] = 0.0
                bn.running_mean[pruning_idx] = 0.0
                bn.running_var[pruning_idx] = 1.0
                bn.weight[key_dim + pruning_idx] = 0.0
                bn.bias[key_dim + pruning_idx] = 0.0
                bn.running_mean[key_dim + pruning_idx] = 0.0
                bn.running_var[key_dim + pruning_idx] = 1.0

            # DW on Q
            w_dw = m['dw'].c.weight
            w_dw[pruning_idx, :, :, :] = 0.0
            if hasattr(m['dw'], 'bn'):
                bn = m['dw'].bn
                bn.weight[pruning_idx] = 0.0
                bn.bias[pruning_idx] = 0.0
                bn.running_mean[pruning_idx] = 0.0
                bn.running_var[pruning_idx] = 1.0

        elif gtype == 'G_V':
            # qkv V slice [2*key_dim + pruning_idx] = 0
            key_dim = meta['key_dim']
            v_start = 2 * key_dim
            w_qkv = m['qkv'].c.weight
            w_qkv[v_start + pruning_idx, :, :, :] = 0.0
            # BN V slice
            if hasattr(m['qkv'], 'bn'):
                bn = m['qkv'].bn
                bn.weight[v_start + pruning_idx] = 0.0
                bn.bias[v_start + pruning_idx] = 0.0
                bn.running_mean[v_start + pruning_idx] = 0.0
                bn.running_var[v_start + pruning_idx] = 1.0

        elif gtype == 'G_PATCH':
            # out_conv filter 제거
            w_out = m['out_conv'].c.weight
            w_out[pruning_idx, :, :, :] = 0.0
            if hasattr(m['out_conv'].c, 'bias') and m['out_conv'].c.bias is not None:
                m['out_conv'].c.bias[pruning_idx] = 0.0
            if hasattr(m['out_conv'], 'bn'):
                bn = m['out_conv'].bn
                bn.weight[pruning_idx] = 0.0
                bn.bias[pruning_idx] = 0.0
                bn.running_mean[pruning_idx] = 0.0
                bn.running_var[pruning_idx] = 1.0


def zero_pruned_gradients(groups):
    """
    이미 pruned된 units의 gradient를 0으로 설정하여 복원을 방지합니다.

    CRITICAL FIX: Pruned weights는 0으로 설정되지만, backward()에서 gradient가 생기고
    optimizer.step()에서 다시 업데이트되어 복원됩니다.
    → 해결: optimizer.step() 직전에 pruned weights의 gradient를 0으로 만듭니다.
    """
    with torch.no_grad():
        for group in groups:
            if 'pruned_mask' not in group or len(group['pruned_mask']) == 0:
                continue

            gtype = group['type']
            m = group['modules']
            meta = group['meta']
            pruned_list = list(group['pruned_mask'])

            if gtype == 'G_FFN':
                # expand.weight gradient[pruned_idx] = 0
                w_expand = m['expand'].c.weight
                if w_expand.grad is not None:
                    w_expand.grad[pruned_list, :, :, :] = 0.0
                # shrink.weight gradient[:, pruned_idx] = 0
                w_shrink = m['shrink'].c.weight
                if w_shrink.grad is not None:
                    w_shrink.grad[:, pruned_list, :, :] = 0.0

            elif gtype == 'G_QK':
                key_dim = meta['key_dim']
                w_qkv = m['qkv'].c.weight
                if w_qkv.grad is not None:
                    w_qkv.grad[pruned_list, :, :, :] = 0.0  # Q
                    # K indices: key_dim + pruned_idx
                    k_idx = [key_dim + i for i in pruned_list]
                    w_qkv.grad[k_idx, :, :, :] = 0.0  # K
                w_dw = m['dw'].c.weight
                if w_dw.grad is not None:
                    w_dw.grad[pruned_list, :, :, :] = 0.0

            elif gtype == 'G_V':
                key_dim = meta['key_dim']
                v_start = 2 * key_dim
                w_qkv = m['qkv'].c.weight
                if w_qkv.grad is not None:
                    v_idx = [v_start + i for i in pruned_list]
                    w_qkv.grad[v_idx, :, :, :] = 0.0

            elif gtype == 'G_PATCH':
                w_out = m['out_conv'].c.weight
                if w_out.grad is not None:
                    w_out.grad[pruned_list, :, :, :] = 0.0


def apply_pruned_mask(groups):
    """
    이미 pruned된 units를 매 iteration마다 0으로 강제 설정합니다.

    CRITICAL: optimizer.step()에서 gradient로 인해 pruned weights가 복원되는 것을 방지합니다.
    매 iteration마다 호출하여 pruned_mask에 저장된 indices를 계속 0으로 유지합니다.

    Args:
        groups: pruning groups list (pruned_mask 포함)
    """
    with torch.no_grad():
        for group in groups:
            if 'pruned_mask' not in group or len(group['pruned_mask']) == 0:
                continue

            gtype = group['type']
            m = group['modules']
            meta = group['meta']
            pruned_list = list(group['pruned_mask'])

            if gtype == 'G_FFN':
                # expand.weight[pruned_idx] = 0
                w_expand = m['expand'].c.weight
                w_expand[pruned_list, :, :, :] = 0.0
                if hasattr(m['expand'].c, 'bias') and m['expand'].c.bias is not None:
                    m['expand'].c.bias[pruned_list] = 0.0

                # shrink.weight[:, pruned_idx] = 0
                w_shrink = m['shrink'].c.weight
                w_shrink[:, pruned_list, :, :] = 0.0

            elif gtype == 'G_QK':
                key_dim = meta['key_dim']
                w_qkv = m['qkv'].c.weight
                w_qkv[pruned_list, :, :, :] = 0.0  # Q
                # K indices: key_dim + pruned_idx
                k_idx = [key_dim + i for i in pruned_list]
                w_qkv[k_idx, :, :, :] = 0.0  # K

                w_dw = m['dw'].c.weight
                w_dw[pruned_list, :, :, :] = 0.0

            elif gtype == 'G_V':
                key_dim = meta['key_dim']
                v_start = 2 * key_dim
                w_qkv = m['qkv'].c.weight
                v_idx = [v_start + i for i in pruned_list]
                w_qkv[v_idx, :, :, :] = 0.0

            elif gtype == 'G_PATCH':
                w_out = m['out_conv'].c.weight
                w_out[pruned_list, :, :, :] = 0.0
                if hasattr(m['out_conv'].c, 'bias') and m['out_conv'].c.bias is not None:
                    m['out_conv'].c.bias[pruned_list] = 0.0


def pgm_regularization_loss(groups: List[Dict], lambda_ffn: float, lambda_qk: float, lambda_v: float) -> torch.Tensor:
    """
    PGM 정규화 항 계산 (L2 norm 기반)

    Args:
        groups: pruning groups list
        lambda_ffn, lambda_qk, lambda_v: 그룹 타입별 λ 값

    Returns:
        loss: PGM regularization loss
    """
    loss = 0.0
    for group in groups:
        gtype = group['type']
        lambda_val = {
            'G_FFN': lambda_ffn,
            'G_QK': lambda_qk,
            'G_V': lambda_v,
            'G_PATCH': lambda_ffn  # PATCH도 FFN과 동일하게
        }.get(gtype, 0.0)

        if lambda_val > 0:
            m = group['modules']
            # 각 그룹 타입별로 관련 weights에만 regularization 적용
            if gtype == 'G_FFN':
                loss += lambda_val * torch.sum(m['expand'].c.weight ** 2)
                loss += lambda_val * torch.sum(m['shrink'].c.weight ** 2)
            elif gtype == 'G_QK':
                # Q+K slices만 (V는 제외)
                key_dim = group['meta']['key_dim']
                w = m['qkv'].c.weight
                loss += lambda_val * torch.sum(w[:2*key_dim] ** 2)  # Q+K
                loss += lambda_val * torch.sum(m['dw'].c.weight ** 2)
            elif gtype == 'G_V':
                # V slice만
                key_dim = group['meta']['key_dim']
                d = group['meta']['d']
                w = m['qkv'].c.weight
                loss += lambda_val * torch.sum(w[2*key_dim:2*key_dim+d] ** 2)
            elif gtype == 'G_PATCH':
                loss += lambda_val * torch.sum(m['out_conv'].c.weight ** 2)
                loss += lambda_val * torch.sum(m['in_conv'].c.weight ** 2)

    return loss


def memory_penalty(current_memory_mb: float, m_max_mb: float, mu: float) -> torch.Tensor:
    """
    메모리 패널티 항 계산

    Args:
        current_memory_mb: 현재 활성 파라미터 메모리 (MB)
        m_max_mb: 최대 허용 메모리 (MB)
        mu: 패널티 계수

    Returns:
        penalty: 메모리 패널티
    """
    return mu * max(0, current_memory_mb - m_max_mb)


def apply_phase1_pruning(model, groups: List[Dict], sparsities: Dict[str, float], verbose: bool = False):
    """
    Phase 1 그룹들에 대해 pruning 적용 (YOLO 방식)

    Args:
        model: EfficientViT 모델
        groups: pruning groups
        sparsities: {type: sparsity} dict
        verbose: True이면 상세 로그 출력
    """
    phase1_groups = [g for g in groups if g['type'] in ['G_FFN', 'G_QK', 'G_V']]

    # 통계 수집
    total_pruned = 0
    total_units = 0
    type_stats = {'G_FFN': {'pruned': 0, 'total': 0},
                  'G_QK': {'pruned': 0, 'total': 0},
                  'G_V': {'pruned': 0, 'total': 0}}

    for group in phase1_groups:
        sparsity = sparsities.get(group['type'], 0.0)
        if sparsity > 0:
            pruning_idx = get_group_pruning_idx(group, sparsity)

            # Pruning 적용
            if len(pruning_idx) > 0:
                apply_group_pruning(group, pruning_idx)
                total_pruned += len(pruning_idx)
                type_stats[group['type']]['pruned'] += len(pruning_idx)

                # Verbose 모드에서만 상세 로그
                if verbose:
                    print(f"  Pruned {len(pruning_idx)}/{group['unit_count']} units in {group['id']}")

            total_units += group['unit_count']
            type_stats[group['type']]['total'] += group['unit_count']

    # 요약 통계만 출력 (항상)
    if total_pruned > 0:
        print(f"  → Pruned {total_pruned}/{total_units} units total ({total_pruned/total_units*100:.1f}%)")
        for gtype in ['G_FFN', 'G_QK', 'G_V']:
            stats = type_stats[gtype]
            if stats['total'] > 0:
                ratio = stats['pruned'] / stats['total'] * 100
                print(f"    {gtype}: {stats['pruned']}/{stats['total']} ({ratio:.1f}%)")
    else:
        print(f"  → No units pruned (sparsity too small for unit counts)")


def count_zero_groups(groups: List[Dict]) -> Dict[str, int]:
    """
    각 타입별 zero 그룹 수 카운트 (YOLO reducing 참고)

    Returns:
        {type: zero_group_count}
    """
    zero_counts = {}
    for group in groups:
        gtype = group['type']
        if gtype not in zero_counts:
            zero_counts[gtype] = 0

        # 그룹의 units 중 zero인 비율 계산
        total_units = group['unit_count']
        zero_units = 0
        for module in group['modules']:
            if hasattr(module, 'weight'):
                w = module.weight
                if len(w.shape) == 4:  # Conv
                    norms = torch.norm(w.view(w.shape[0], -1), dim=1)
                elif len(w.shape) == 2:  # Linear
                    if w.shape[0] == total_units:
                        norms = torch.norm(w, dim=1)
                    else:
                        norms = torch.norm(w, dim=0)
                elif len(w.shape) == 3:  # QKV
                    norms = torch.norm(w.view(w.shape[0], -1), dim=1)
                else:
                    continue

                zero_units += (norms == 0).sum().item()

        if zero_units / total_units > 0.5:  # 50% 이상 zero면 zero 그룹으로 간주
            zero_counts[gtype] += 1

    return zero_counts