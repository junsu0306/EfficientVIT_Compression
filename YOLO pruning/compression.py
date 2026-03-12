"""
YOLOv8 Compression Module

YOLOv8 모델에 L2-norm 기반 Structured Pruning 및 Channel Reduction을 적용합니다.

주요 함수:
    - yolov8_pruning: Conv, C2f, SPPF, Detect 블록에 pruning 적용
    - yolov8_reducing: 0으로 마스킹된 필터를 물리적으로 제거
"""

import torch
from compression_src.pruning.common import (
    get_filter_pruning_idx,
    filter_pruning,
    bn_pruning,
)
from compression_src.reducing.common import (
    get_survived_filter_idx,
    conv_reduce,
    bn_reduce,
)


def yolov8_pruning(model, sparsity):
    """
    YOLOv8 모델에 structured pruning을 적용합니다.

    Args:
        model: YOLOv8 모델 (model.model 형태)
        sparsity: 제거할 필터 비율 (0.0 ~ 1.0)
    """
    block_list = [model[i] for i in range(23)]

    for i, block in enumerate(block_list):
        block_type = type(block).__name__

        # Conv 블록 처리
        if block_type == 'Conv':
            pruning_idx = get_filter_pruning_idx(block.conv, sparsity)
            filter_pruning(block.conv, pruning_idx)
            bn_pruning(block.bn, pruning_idx)

        # C2f, SPPF 블록 처리
        elif block_type in ['C2f', 'SPPF']:
            pruning_idx = get_filter_pruning_idx(block.cv2.conv, sparsity)
            filter_pruning(block.cv2.conv, pruning_idx)
            bn_pruning(block.cv2.bn, pruning_idx)

        # Detect 블록 처리
        elif block_type == 'Detect':
            for scale_idx in range(3):
                for layer_idx in range(2):
                    # cv2 레이어 pruning
                    pruning_idx = get_filter_pruning_idx(
                        block.cv2[scale_idx][layer_idx].conv, sparsity
                    )
                    filter_pruning(block.cv2[scale_idx][layer_idx].conv, pruning_idx)
                    bn_pruning(block.cv2[scale_idx][layer_idx].bn, pruning_idx)

                    # cv3 레이어 pruning
                    pruning_idx = get_filter_pruning_idx(
                        block.cv3[scale_idx][layer_idx].conv, sparsity
                    )
                    filter_pruning(block.cv3[scale_idx][layer_idx].conv, pruning_idx)
                    bn_pruning(block.cv3[scale_idx][layer_idx].bn, pruning_idx)


def yolov8_reducing(model, reduced_model):
    """
    Pruning된 YOLOv8 모델에서 0인 필터를 물리적으로 제거합니다.

    Args:
        model: Pruning이 적용된 원본 모델
        reduced_model: 축소된 가중치가 복사될 대상 모델
    """
    # 첫 번째 Conv 레이어 처리 (RGB 입력)
    survived_idx = get_survived_filter_idx(model[0].conv)
    conv_reduce(
        model[0].conv,
        reduced_model[0].conv,
        survived_idx,
        torch.arange(3),  # RGB 입력 채널
    )
    bn_reduce(model[0].bn, reduced_model[0].bn, survived_idx)

    prev_survived_idx = survived_idx

    # 나머지 블록 처리
    block_list = [model[i] for i in range(1, 23)]
    reduced_block_list = [reduced_model[i] for i in range(1, 23)]

    for i, (block, reduced_block) in enumerate(zip(block_list, reduced_block_list)):
        block_type = type(block).__name__

        # Conv 블록 처리
        if block_type == 'Conv':
            survived_idx = get_survived_filter_idx(block.conv)
            conv_reduce(
                block.conv,
                reduced_block.conv,
                survived_idx,
                prev_survived_idx,
            )
            bn_reduce(block.bn, reduced_block.bn, survived_idx)

        # C2f, SPPF 블록 처리
        elif block_type in ['C2f', 'SPPF']:
            # Concatenation 레이어의 경우 특별 처리
            if i in [11, 14, 17, 20]:
                prev_survived_idx = _get_concat_survived_idx(block_list, i)

            # cv1 레이어 처리
            survived_idx = torch.arange(block.cv1.conv.out_channels)
            conv_reduce(
                block.cv1.conv,
                reduced_block.cv1.conv,
                survived_idx,
                prev_survived_idx,
            )

            # cv2 레이어 처리
            prev_survived_idx = torch.arange(block.cv2.conv.in_channels)
            survived_idx = get_survived_filter_idx(block.cv2.conv)
            conv_reduce(
                block.cv2.conv,
                reduced_block.cv2.conv,
                survived_idx,
                prev_survived_idx,
            )
            bn_reduce(block.cv2.bn, reduced_block.cv2.bn, survived_idx)

        # Detect 블록 처리
        elif block_type == 'Detect':
            _reduce_detect_block(block, reduced_block, block_list)

        prev_survived_idx = survived_idx


def _get_concat_survived_idx(block_list, layer_idx):
    """
    Concatenation 레이어의 입력 채널 인덱스를 계산합니다.
    YOLOv8의 FPN 구조에서 여러 레이어의 출력이 연결되는 경우 처리합니다.
    """
    # 레이어 인덱스에 따른 연결 레이어 매핑
    concat_map = {
        11: (8, 5),
        14: (11, 3),
        17: (15, 11),
        20: (18, 8),
    }
    n1, n2 = concat_map[layer_idx]

    # 첫 번째 연결 레이어의 살아남은 인덱스
    if layer_idx in [17, 20]:
        p_surv_idx_1 = get_survived_filter_idx(block_list[n1].conv)
        offset = block_list[n1].conv.out_channels
    else:
        p_surv_idx_1 = get_survived_filter_idx(block_list[n1].cv2.conv)
        offset = block_list[n1].cv2.conv.out_channels

    # 두 번째 연결 레이어의 살아남은 인덱스 (offset 적용)
    p_surv_idx_2 = get_survived_filter_idx(block_list[n2].cv2.conv) + offset

    return torch.concat([p_surv_idx_1, p_surv_idx_2])


def _reduce_detect_block(block, reduced_block, block_list):
    """
    Detect 블록의 cv2, cv3 레이어를 축소합니다.
    """
    # 이전 레이어 인덱스 (FPN 출력)
    prev_indices = [14, 17, 20]

    for scale_idx, prev_idx in enumerate(prev_indices):
        for sub_layer, reduced_sub_layer in zip(
            [block.cv2, block.cv3],
            [reduced_block.cv2, reduced_block.cv3]
        ):
            prev_survived_idx = get_survived_filter_idx(block_list[prev_idx].cv2.conv)

            # Conv+BN 레이어 2개 처리
            for layer_idx in range(2):
                survived_idx = get_survived_filter_idx(sub_layer[scale_idx][layer_idx].conv)
                conv_reduce(
                    sub_layer[scale_idx][layer_idx].conv,
                    reduced_sub_layer[scale_idx][layer_idx].conv,
                    survived_idx,
                    prev_survived_idx,
                )
                bn_reduce(
                    sub_layer[scale_idx][layer_idx].bn,
                    reduced_sub_layer[scale_idx][layer_idx].bn,
                    survived_idx,
                )
                prev_survived_idx = survived_idx

            # 마지막 Conv2d 레이어 (출력 채널은 유지)
            survived_idx = get_survived_filter_idx(sub_layer[scale_idx][2])
            conv_reduce(
                sub_layer[scale_idx][2],
                reduced_sub_layer[scale_idx][2],
                survived_idx,
                prev_survived_idx,
            )
