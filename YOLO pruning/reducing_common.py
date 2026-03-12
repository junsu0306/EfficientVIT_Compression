"""
Reducing Common Functions

Pruning된 모델에서 0인 필터를 물리적으로 제거하기 위한 공통 함수들입니다.
"""

import torch


def get_survived_filter_idx(layer):
    """
    살아남은 필터 인덱스를 반환합니다 (norm != 0).

    Args:
        layer: Conv2d 레이어

    Returns:
        survived_filter_idx: 0이 아닌 필터의 인덱스
    """
    weight = layer.weight
    num_filters = weight.shape[0]
    filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)
    survived_filter_idx = torch.where(filter_norms != 0)[0]
    return survived_filter_idx


def conv_reduce(layer, reduced_layer, survived_out_channels_idx, survived_in_channels_idx):
    """
    Conv 레이어를 축소합니다.

    Args:
        layer: 원본 Conv2d 레이어
        reduced_layer: 축소된 Conv2d 레이어
        survived_out_channels_idx: 살아남은 출력 채널 인덱스
        survived_in_channels_idx: 살아남은 입력 채널 인덱스
    """
    # 채널 수 설정
    reduced_layer.in_channels = len(survived_in_channels_idx)
    reduced_layer.out_channels = len(survived_out_channels_idx)

    # Depthwise conv의 경우 groups 조정
    if reduced_layer.groups != 1:
        reduced_layer.groups = reduced_layer.out_channels

    # 새로운 weight 텐서 생성
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.out_channels,
            reduced_layer.in_channels,
            reduced_layer.kernel_size[0],
            reduced_layer.kernel_size[1],
        ),
        requires_grad=True,
    )

    # 검증
    assert len(survived_out_channels_idx) == reduced_layer.weight.shape[0], \
        f"out_channels mismatch: {len(survived_out_channels_idx)} vs {reduced_layer.weight.shape[0]}"
    assert len(survived_in_channels_idx) == reduced_layer.weight.shape[1], \
        f"in_channels mismatch: {len(survived_in_channels_idx)} vs {reduced_layer.weight.shape[1]}"

    # 살아남은 가중치 복사
    with torch.no_grad():
        reduced_layer.weight.copy_(
            layer.weight[survived_out_channels_idx, :, :, :][:, survived_in_channels_idx, :, :]
        )


def fc_reduce(layer, reduced_layer, survived_out_features_idx, survived_in_features_idx):
    """
    Fully Connected 레이어를 축소합니다.

    Args:
        layer: 원본 Linear 레이어
        reduced_layer: 축소된 Linear 레이어
        survived_out_features_idx: 살아남은 출력 feature 인덱스
        survived_in_features_idx: 살아남은 입력 feature 인덱스
    """
    # Feature 수 설정
    reduced_layer.in_features = len(survived_in_features_idx)
    reduced_layer.out_features = len(survived_out_features_idx)

    # 새로운 weight, bias 텐서 생성
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.out_features, reduced_layer.in_features),
        requires_grad=True,
    )
    reduced_layer.bias = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.out_features),
        requires_grad=True,
    )

    # 검증
    assert len(survived_out_features_idx) == reduced_layer.weight.shape[0], \
        f"out_features mismatch: {len(survived_out_features_idx)} vs {reduced_layer.weight.shape[0]}"
    assert len(survived_in_features_idx) == reduced_layer.weight.shape[1], \
        f"in_features mismatch: {len(survived_in_features_idx)} vs {reduced_layer.weight.shape[1]}"

    # 살아남은 가중치 복사
    with torch.no_grad():
        reduced_layer.weight.copy_(
            layer.weight[survived_out_features_idx, :][:, survived_in_features_idx]
        )
        reduced_layer.bias.copy_(layer.bias[survived_out_features_idx])


def bn_reduce(layer, reduced_layer, survived_features_idx):
    """
    BatchNorm 레이어를 축소합니다.

    Args:
        layer: 원본 BatchNorm2d 레이어
        reduced_layer: 축소된 BatchNorm2d 레이어
        survived_features_idx: 살아남은 feature 인덱스
    """
    reduced_layer.num_features = len(survived_features_idx)

    # 새로운 파라미터 텐서 생성
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.num_features),
        requires_grad=True,
    )
    reduced_layer.bias = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.num_features),
        requires_grad=True,
    )
    reduced_layer.running_mean = torch.zeros(reduced_layer.num_features)
    reduced_layer.running_var = torch.zeros(reduced_layer.num_features)

    # 검증
    assert len(survived_features_idx) == reduced_layer.weight.shape[0], "bn weight mismatch"
    assert len(survived_features_idx) == reduced_layer.bias.shape[0], "bn bias mismatch"
    assert len(survived_features_idx) == reduced_layer.running_mean.shape[0], "bn mean mismatch"
    assert len(survived_features_idx) == reduced_layer.running_var.shape[0], "bn var mismatch"

    # 살아남은 파라미터 복사
    with torch.no_grad():
        reduced_layer.weight.copy_(layer.weight[survived_features_idx])
        reduced_layer.bias.copy_(layer.bias[survived_features_idx])
        reduced_layer.running_mean.copy_(layer.running_mean[survived_features_idx])
        reduced_layer.running_var.copy_(layer.running_var[survived_features_idx])


def copy_layer(layer, reduced_layer):
    """
    레이어 파라미터를 그대로 복사합니다.

    Args:
        layer: 원본 레이어
        reduced_layer: 대상 레이어
    """
    with torch.no_grad():
        for p, reduced_p in zip(layer.parameters(), reduced_layer.parameters()):
            reduced_p.copy_(p)


def bn_copy_layer(layer, reduced_layer):
    """
    BatchNorm 레이어를 그대로 복사합니다 (running_mean, running_var 포함).

    Args:
        layer: 원본 BatchNorm 레이어
        reduced_layer: 대상 BatchNorm 레이어
    """
    with torch.no_grad():
        reduced_layer.running_mean.copy_(layer.running_mean)
        reduced_layer.running_var.copy_(layer.running_var)

        for p, reduced_p in zip(layer.parameters(), reduced_layer.parameters()):
            reduced_p.copy_(p)
