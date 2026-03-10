# --------------------------------------------------------
# EfficientViT FPN Architecture for Downstream Tasks
# Copyright (c) 2022 Microsoft
# Adapted from mmdetection FPN and LightViT
#   mmdetection: (https://github.com/open-mmlab/mmdetection)
#   LightViT: (https://github.com/hunto/LightViT)
# Written by: Xinyu Liu
# --------------------------------------------------------
# 이 파일은 EfficientViT 백본을 객체 탐지/세그멘테이션 등 다운스트림 태스크에 연결하는
# Feature Pyramid Network(FPN) 넥(Neck) 모듈을 정의합니다.
# MMDetection 프레임워크의 NECKS 레지스트리에 등록되어 config 파일에서 type='EfficientViTFPN'으로 사용됩니다.
import warnings

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
# auto_fp16: 혼합 정밀도(Mixed Precision) 학습을 지원하는 데코레이터
# 이 데코레이터가 적용된 함수는 FP16(반정밀도) 입력을 자동으로 처리하며,
# fp16_enabled 플래그가 True일 때 FP16 연산을 활성화합니다.
# 학습 속도를 높이고 메모리 사용량을 줄이기 위해 사용됩니다.
from mmcv.runner import auto_fp16

# MMDetection의 NECKS 레지스트리: 백본과 헤드 사이의 넥 모듈을 등록/관리
# @NECKS.register_module() 데코레이터를 통해 config 파일에서 동적으로 클래스를 불러올 수 있습니다.
from mmdet.models.builder import NECKS


# MMDetection의 NECKS 레지스트리에 EfficientViTFPN 클래스를 등록합니다.
# 이를 통해 config 파일에서 type='EfficientViTFPN'으로 이 클래스를 인스턴스화할 수 있습니다.
@NECKS.register_module()
class EfficientViTFPN(nn.Module):
    r"""Feature Pyramid Network for EfficientViT.

    EfficientViT 백본에서 출력된 다중 스케일 특징 맵을 처리하는 FPN 넥 모듈입니다.
    FPN은 다양한 크기의 객체를 탐지하기 위해 여러 해상도의 특징 맵을 계층적으로 결합합니다.
    - Lateral Convolutions(측면 합성곱): 각 백본 레벨의 채널 수를 통일된 out_channels로 변환
    - Top-Down Pathway(하향 경로): 고해상도 특징 맵에 저해상도 의미 정보를 결합
    - Extra Convolutions/Transposed Convolutions: 추가 스케일 출력 생성

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        num_extra_trans_convs (int): extra transposed conv on the output
            with largest resolution. Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 num_extra_trans_convs=0,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        """EfficientViTFPN 초기화 메서드.

        백본의 각 스케일 출력을 처리하기 위한 lateral conv, fpn conv,
        그리고 선택적으로 extra conv 또는 transposed conv를 구성합니다.

        파라미터:
            in_channels (List[int]): 백본 각 스케일의 입력 채널 수 목록
                예: [64, 128, 256, 512] (4개 스케일)
            out_channels (int): FPN 모든 출력 레벨에서 사용하는 통일된 채널 수
                예: 256
            num_outs (int): FPN이 출력하는 총 스케일 수
                백본 레벨보다 많으면 extra convs 또는 max pool로 추가 레벨 생성
            start_level (int): 사용할 백본 출력의 시작 인덱스 (기본값: 0)
            end_level (int): 사용할 백본 출력의 끝 인덱스(미포함), -1이면 마지막 레벨까지
            add_extra_convs (bool|str): 최상위 레벨 위에 추가 합성곱 레이어 추가 여부
                'on_input': 백본 마지막 출력을 소스로 사용
                'on_lateral': lateral conv 출력을 소스로 사용
                'on_output': fpn conv 출력을 소스로 사용
            extra_convs_on_inputs (bool): [deprecated] add_extra_convs 대신 사용 권장
            relu_before_extra_convs (bool): extra conv 전에 ReLU 적용 여부
            no_norm_on_lateral (bool): lateral conv에 정규화 레이어 미적용 여부
            num_extra_trans_convs (int): 가장 고해상도 출력에 추가할 전치 합성곱 수
                양수이면 더 높은 해상도의 출력을 생성 (예: P2보다 큰 P1 생성)
            conv_cfg (dict): 합성곱 레이어 설정 (None이면 기본 Conv2d 사용)
            norm_cfg (dict): 정규화 레이어 설정 (예: dict(type='BN'))
            act_cfg (dict): 활성화 레이어 설정 (None이면 활성화 없음)
            upsample_cfg (dict): 업샘플링 보간법 설정 (기본: nearest 보간)
        """
        super(EfficientViTFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)       # 백본 입력 레벨 수
        self.num_outs = num_outs              # FPN 출력 레벨 수
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.num_extra_trans_convs = num_extra_trans_convs
        # FP16(혼합 정밀도) 활성화 여부 플래그. auto_fp16 데코레이터와 함께 사용됩니다.
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        # 백본 출력의 유효 끝 레벨 결정
        # end_level=-1이면 모든 백본 레벨 사용, 그렇지 않으면 지정된 레벨까지 사용
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            # 출력 수는 사용되는 백본 레벨 수 이상이어야 함 (extra levels 허용)
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            # end_level이 명시될 경우 출력 수는 정확히 사용 레벨 수와 같아야 함
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        # add_extra_convs 설정값 정규화:
        # 문자열이면 유효한 옵션('on_input', 'on_lateral', 'on_output')인지 확인
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            # [deprecated] extra_convs_on_inputs 파라미터를 add_extra_convs 문자열로 변환
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        # lateral_convs: 백본 각 레벨의 채널 수를 out_channels로 통일하는 1x1 합성곱 목록
        # fpn_convs: lateral 출력과 상위 레벨 업샘플을 합산한 후 정제하는 3x3 합성곱 목록
        # (extra conv들도 이 목록에 추가됨)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # 사용되는 각 백본 레벨에 대해 lateral conv와 fpn conv를 생성
        for i in range(self.start_level, self.backbone_end_level):
            # lateral conv: 각 백본 레벨의 채널 수(in_channels[i]) -> out_channels 로 변환하는 1x1 conv
            # 이를 통해 서로 다른 채널 수를 가진 백본 피처를 동일한 차원으로 통일
            # no_norm_on_lateral=True이면 정규화(norm) 없이 합성곱만 적용
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,               # 1x1 커널: 공간 정보는 유지하면서 채널 변환만 수행
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            # fpn conv: top-down path 결합 후 3x3 conv로 피처 정제
            # padding=1로 공간 크기를 유지하면서 채널 수는 out_channels 유지
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,               # 3x3 커널: 주변 공간 정보를 통합하여 피처 품질 향상
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        # extra_levels: 백본 레벨 수보다 더 많이 필요한 출력 스케일 수
        # transposed conv로 처리되는 레벨을 제외한 추가 레벨에 대해 stride=2 conv를 추가하여
        # 더 낮은 해상도(더 큰 수용야)의 피처 맵을 생성 (예: RetinaNet의 P6, P7)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        assert extra_levels >= num_extra_trans_convs
        extra_levels -= num_extra_trans_convs
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                # 첫 번째 extra conv의 입력 소스 결정
                # add_extra_convs='on_input'이면 백본의 마지막 레벨 출력을 입력으로 사용
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    # 두 번째 이후 extra conv는 이전 extra conv 출력을 입력으로 사용
                    in_channels = out_channels
                # stride=2 합성곱으로 해상도를 절반으로 줄이면서 채널은 out_channels 유지
                # 이를 통해 더 낮은 해상도(더 넓은 수용야)의 스케일을 추가로 생성
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,    # stride=2로 공간 크기를 절반으로 다운샘플링
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # add extra transposed convs
        # num_extra_trans_convs > 0이면 가장 고해상도 출력보다 더 큰 해상도의 피처 맵을 생성
        # TransposedConvModule의 stride=2 전치 합성곱으로 해상도를 2배로 업샘플링
        # 예: P3(1/8 해상도)에서 P2(1/4 해상도), P1(1/2 해상도) 등을 추가로 생성
        self.extra_trans_convs = nn.ModuleList()
        self.extra_fpn_convs = nn.ModuleList()
        for i in range(num_extra_trans_convs):
            # TransposedConvModule: nn.ConvTranspose2d를 이용한 학습 가능한 업샘플링
            # stride=2, kernel=2, padding=0으로 정확히 2배 해상도 증가
            extra_trans_conv = TransposedConvModule(
                out_channels,
                out_channels,
                2,               # 2x2 커널의 전치 합성곱
                stride=2,        # stride=2로 공간 크기를 2배로 업샘플링
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.extra_trans_convs.append(extra_trans_conv)
            # 전치 합성곱 직후 3x3 conv로 피처를 정제
            extra_fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.extra_fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module.

        FPN 모듈의 가중치를 Xavier 균등 분포로 초기화합니다.
        Xavier 초기화는 각 레이어의 입출력 뉴런 수를 고려하여 그래디언트 소실/폭발을 방지합니다.
        ConvModule 내부의 정규화(norm) 레이어와 활성화(activation) 레이어는
        mmcv의 기본 초기화를 따릅니다.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function.

        EfficientViT 백본에서 출력된 다중 스케일 특징 맵을 FPN으로 처리합니다.

        @auto_fp16() 데코레이터:
        - fp16_enabled=True일 때 입력 텐서를 자동으로 FP16으로 변환합니다.
        - 혼합 정밀도 학습 시 메모리 효율과 속도를 높이기 위해 사용됩니다.
        - 출력은 자동으로 FP32로 복원됩니다.

        FPN forward 처리 단계:
        1. Lateral connections: 각 백본 레벨을 1x1 conv로 채널 통일
        2. Top-down pathway: 상위(저해상도) 레벨을 업샘플하여 하위(고해상도) 레벨에 합산
        3. Extra transposed convs: 가장 고해상도 레벨을 추가로 업샘플 (선택적)
        4. FPN convs: 각 레벨을 3x3 conv로 정제하여 최종 출력 생성
        5. Extra convs/max pool: 추가 저해상도 레벨 생성 (선택적)

        파라미터:
            inputs (tuple[Tensor]): 백본에서 출력된 다중 스케일 특징 맵 튜플
                각 텐서의 형태: (배치, 채널, 높이, 너비)
                len(inputs)는 len(self.in_channels)와 같아야 함

        반환값:
            tuple[Tensor]: FPN에서 처리된 다중 스케일 특징 맵 튜플
                len(outputs) == self.num_outs
                더 작은 인덱스 = 더 높은 해상도 (예: outs[0]이 outs[-1]보다 큼)
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        # 각 백본 레벨의 특징 맵에 1x1 lateral conv를 적용하여 채널 수를 out_channels로 통일
        # inputs[start_level], inputs[start_level+1], ..., inputs[backbone_end_level-1] 사용
        # 결과: laterals[0]이 가장 작은(얕은) 레벨, laterals[-1]이 가장 깊은 레벨
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        # 가장 깊은(저해상도) 레벨부터 역순으로 진행하며 업샘플링한 후 하위 레벨에 합산
        # 이를 통해 의미론적(semantic) 정보가 고해상도 레벨에도 전달됨
        # 예: laterals[3](1/32) -> 업샘플 -> laterals[2](1/16) 에 더함
        #     laterals[2](1/16) -> 업샘플 -> laterals[1](1/8) 에 더함
        #     laterals[1](1/8)  -> 업샘플 -> laterals[0](1/4) 에 더함
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            # scale_factor가 설정된 경우: 고정 배율(예: 2배)로 업샘플링
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                # 그 외의 경우: 하위 레벨의 정확한 공간 크기에 맞춰 업샘플링
                # 이 방식은 홀수 해상도 처리 시 크기 불일치를 방지합니다.
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # extra transposed convs for outputs with extra scales
        # num_extra_trans_convs > 0이면 가장 고해상도 lateral(laterals[0])을
        # 전치 합성곱으로 추가로 업샘플링하여 더 높은 해상도의 피처 맵 생성
        # extra_laterals는 역순으로 삽입되어 최종적으로 해상도 내림차순(큰->작은) 정렬 유지
        extra_laterals = []
        if self.num_extra_trans_convs > 0:
            prev_lateral = laterals[0]
            for i in range(self.num_extra_trans_convs):
                # 전치 합성곱으로 해상도 2배 증가
                extra_lateral = self.extra_trans_convs[i](prev_lateral)
                # insert(0, ...)으로 역순 삽입: extra_laterals[0]이 가장 고해상도
                extra_laterals.insert(0, extra_lateral)
                prev_lateral = extra_lateral

        # part 1: from original levels
        # 백본 레벨에서 나온 lateral 피처에 3x3 fpn conv를 적용하여 최종 피처 정제
        # outs[0]: 가장 고해상도 출력, outs[-1]: 가장 저해상도 출력
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        # num_outs가 백본 레벨 수보다 많을 때 추가 저해상도 스케일을 생성
        if self.num_outs > len(outs) + len(extra_laterals):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            # add_extra_convs=False이면 max pool로 간단하게 추가 레벨 생성
            # stride=2 max pool로 공간 크기를 절반으로 줄여 저해상도 피처 추가
            if not self.add_extra_convs:
                for i in range(self.num_outs - len(extra_laterals) - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            # add_extra_convs가 설정된 경우 학습 가능한 conv로 추가 레벨 생성
            else:
                # extra conv의 입력 소스 선택:
                # 'on_input': 백본 마지막 레벨 원본 피처 사용
                # 'on_lateral': lateral conv 처리 후 피처 사용
                # 'on_output': fpn conv 처리 후 최종 출력 피처 사용
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                # fpn_convs에서 백본 레벨 수 이후에 추가된 extra conv를 적용
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))

                # 두 번째 이후 extra levels (현재 코드에서는 실제로 호출되지 않음)
                for i in range(used_backbone_levels + 1, self.num_outs - len(extra_laterals)): # Not called
                    print("i: {}".format(i), self.fpn_convs[i])
                    # relu_before_extra_convs=True이면 ReLU 적용 후 conv (일부 모델에서 성능 향상)
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # part 3: add extra transposed convs
        # num_extra_trans_convs > 0이면 extra_laterals에 3x3 fpn conv를 적용하여
        # 최종 고해상도 출력 피처 맵 생성
        if self.num_extra_trans_convs > 0:
            # apply 3x3 conv on the larger feat (1/8) after 3x3 trans conv
            # because the 3x3 trans conv is on the lateral
            # thus no extra 1x1 laterals are required
            # extra_fpn_convs를 통해 전치 합성곱 출력을 정제
            extra_outs = [
                self.extra_fpn_convs[i](extra_laterals[i])
                    for i in range(self.num_extra_trans_convs)
            ]
            # 1 + 4 (3+1extra) = 5
        # 최종 출력: extra_outs(고해상도) + outs(일반 FPN 출력) 순서로 연결
        # 예: [P1, P2, P3, P4, P5] (P1이 가장 고해상도, P5가 가장 저해상도)
        assert (len(extra_outs) + len(outs)) == self.num_outs, f"{len(extra_outs)} + {len(outs)} != {self.num_outs}"
        return tuple(extra_outs + outs)


class TransposedConvModule(ConvModule):
    """전치 합성곱(Transposed Convolution) 모듈.

    mmcv의 ConvModule을 상속하여 일반 Conv2d 대신 ConvTranspose2d를 사용하는 모듈입니다.
    ConvTranspose2d는 '학습 가능한 업샘플링' 레이어로, stride=2로 설정하면
    입력 피처 맵의 공간 해상도를 정확히 2배로 증가시킵니다.

    일반 보간법(bilinear, nearest)과 달리 학습을 통해 최적의 업샘플링 가중치를 습득합니다.
    FPN에서 더 고해상도의 피처 맵이 필요할 때 사용됩니다.

    ConvModule과의 차이점:
    - self.conv가 nn.Conv2d 대신 nn.ConvTranspose2d로 교체됨
    - 나머지 정규화, 활성화, 초기화 로직은 ConvModule에서 상속
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias='auto', conv_cfg=None,
                 norm_cfg=None, act_cfg=..., inplace=True,
                 **kwargs):
        """TransposedConvModule 초기화.

        파라미터:
            in_channels (int): 입력 채널 수
            out_channels (int): 출력 채널 수
            kernel_size (int): 전치 합성곱 커널 크기 (보통 2 또는 4)
            stride (int): 전치 합성곱 스트라이드 (stride=2이면 해상도 2배 증가)
            padding (int): 전치 합성곱 패딩 (padding=0이면 출력 크기 = 입력 * stride)
            dilation (int): 팽창(dilation) 계수
            groups (int): 그룹 합성곱 그룹 수
            bias (str|bool): 편향 사용 여부 ('auto'이면 norm 없을 때 자동으로 True)
            conv_cfg (dict): 합성곱 설정 (TransposedConvModule에서는 무시됨)
            norm_cfg (dict): 정규화 레이어 설정
            act_cfg (dict): 활성화 레이어 설정
            inplace (bool): 인플레이스 연산 사용 여부
            **kwargs: ConvModule에 전달할 추가 키워드 인자
        """
        # ConvModule 부모 클래스 초기화 (norm, act, bias 처리 등을 위해)
        super(TransposedConvModule, self).__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, conv_cfg,
                         norm_cfg, act_cfg, inplace, **kwargs)

        # ConvModule이 기본으로 생성한 Conv2d를 ConvTranspose2d로 교체
        # self.with_bias는 ConvModule 부모 클래스에서 bias='auto' 처리 결과로 결정됨
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=self.with_bias
        )

        # Use msra init by default
        # Xavier/MSRA 초기화를 적용하여 학습 초기 안정성을 높임
        self.init_weights()
