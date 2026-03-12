"""
Pruning Common Functions

L2-norm 기반 Structured Pruning을 위한 공통 함수들입니다.
"""

import torch


def get_filter_pruning_sparsity(layer, memory, max_memory):
    """
    메모리 제약 기반으로 sparsity를 계산합니다.

    Args:
        layer: Conv2d 레이어
        memory: 필터당 메모리 사용량
        max_memory: 최대 허용 메모리

    Returns:
        sparsity: 제거할 필터 비율
    """
    with torch.no_grad():
        num_filters = layer.weight.shape[0]
        survive_limit = max_memory // memory
        if survive_limit > num_filters:
            survive_limit = num_filters
        sparsity = (num_filters - survive_limit) / num_filters
    return sparsity


def filter_pruning(layer, pruning_idx):
    """
    Conv 레이어의 지정된 필터를 0으로 마스킹합니다.

    Args:
        layer: Conv2d 레이어
        pruning_idx: 제거할 필터 인덱스
    """
    with torch.no_grad():
        layer.weight[pruning_idx, :, :, :] = 0.0


def bn_pruning(layer, pruning_idx):
    """
    BatchNorm 레이어의 지정된 채널을 마스킹합니다.

    Args:
        layer: BatchNorm2d 레이어
        pruning_idx: 제거할 채널 인덱스
    """
    with torch.no_grad():
        layer.weight[pruning_idx] = 0.0
        layer.bias[pruning_idx] = 0.0
        layer.running_mean[pruning_idx] = 0.0
        layer.running_var[pruning_idx] = 1.0  # var=1로 설정하여 나눗셈 오류 방지


def get_filter_pruning_idx(layer, sparsity):

    with torch.no_grad():
        weight = layer.weight
        num_filters = weight.shape[0]
        num_pruning_filters = int(num_filters * sparsity)

        # 각 필터의 L2 norm 계산
        filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)

        # 가장 작은 norm을 가진 필터 선택
        _, pruning_idx = torch.topk(filter_norms, num_pruning_filters, largest=False)

    return pruning_idx
