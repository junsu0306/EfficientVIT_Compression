# --------------------------------------------------------
# EfficientViT Model Architecture for Downstream Tasks
# Copyright (c) 2022 Microsoft
# Written by: Xinyu Liu
# --------------------------------------------------------
# [한국어 설명]
# 이 파일은 EfficientViT를 Object Detection / Segmentation 등 Downstream Task용
# MMDetection backbone으로 변환한 구현체입니다.
#
# [분류(classification) 버전과의 주요 차이점]
# 1. 출력 구조: 단일 클래스 로짓 대신 다중 스케일 특징 맵(tuple) 반환
#    - blocks1, blocks2, blocks3 각 스테이지의 출력을 FPN(Feature Pyramid Network)에 전달
# 2. 헤드 제거: 분류용 head(BN_Linear, cls_head, dist_head 등)가 없음
# 3. Frozen stages 지원: 학습 시 특정 스테이지를 고정(freeze)할 수 있음
# 4. MMDetection 등록: @BACKBONES.register_module() 데코레이터로 MMDet에 등록
# 5. 사전학습 가중치 로딩: attention bias 크기가 다를 경우 bicubic 보간 수행
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, DropPath, to_2tuple

import numpy as np
import itertools

# MMDetection / mmcv 관련 유틸리티 임포트
# load_checkpoint: 체크포인트 파일 로딩 (mmcv_custom 래퍼)
# _load_checkpoint: 내부용 체크포인트 로더 (state_dict 추출에 사용)
# load_state_dict: strict/non-strict 모드로 가중치를 모델에 로드
from mmcv_custom import load_checkpoint, _load_checkpoint, load_state_dict
# get_root_logger: MMDetection 로거 획득 (학습 중 경고/정보 출력)
from mmdet.utils import get_root_logger
# BACKBONES: MMDetection backbone 레지스트리
# 이 레지스트리에 등록된 모델은 config 파일에서 type='EfficientViT_M0' 등으로 참조 가능
from mmdet.models.builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm


# ==============================================================================
# Conv2d_BN: Conv2d + BatchNorm2d 를 하나의 Sequential로 묶은 유틸리티 모듈
# ==============================================================================
class Conv2d_BN(torch.nn.Sequential):
    """Conv2d와 BatchNorm2d를 묶은 기본 빌딩 블록.

    역할:
        - 대부분의 레이어에서 Conv → BN 순서로 사용되는 패턴을 캡슐화
        - fuse() 메서드로 추론 시 BN을 Conv에 통합하여 연산량 감소

    입력 shape: (B, a, H, W)  - a: 입력 채널 수
    출력 shape: (B, b, H', W') - b: 출력 채널 수, H'/W'는 stride에 의해 결정

    분류 버전과의 차이: 동일한 구현체 사용 (공유 구성요소)
    """
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        """
        Args:
            a (int): 입력 채널 수 (in_channels)
            b (int): 출력 채널 수 (out_channels)
            ks (int): 커널 크기 (kernel_size). 기본값 1
            stride (int): 스트라이드. 기본값 1
            pad (int): 패딩. 기본값 0
            dilation (int): 팽창률. 기본값 1
            groups (int): 그룹 수 (depthwise conv의 경우 groups=in_channels). 기본값 1
            bn_weight_init (float): BN weight(gamma) 초기값. 기본값 1
                                    Residual 연결의 끝단 레이어는 0으로 초기화하여
                                    학습 초기 항등 함수처럼 작동하게 함
            resolution (int): 입력 해상도 (현재 사용되지 않으나 확장성 위해 유지)
        """
        super().__init__()
        # 'c'라는 이름으로 Conv2d 등록 (bias=False: BN이 bias 역할 수행)
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        # 'bn'이라는 이름으로 BatchNorm2d 등록
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        # BN weight(gamma)를 bn_weight_init으로 초기화
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        # BN bias(beta)를 0으로 초기화
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """Conv2d와 BatchNorm2d를 단일 Conv2d로 융합(fuse).

        추론 시 BN의 파라미터를 Conv 가중치에 수학적으로 통합하여
        별도의 BN 연산 없이 Conv 하나로 동일한 결과를 얻음.

        수식:
            w_fused = conv_weight * (bn_weight / sqrt(bn_var + eps))
            b_fused = bn_bias - bn_mean * bn_weight / sqrt(bn_var + eps)

        반환:
            torch.nn.Conv2d: BN이 통합된 단일 Conv2d 레이어
        """
        c, bn = self._modules.values()
        # BN 스케일 인자: gamma / sqrt(var + eps)
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        # Conv 가중치에 BN 스케일 적용: [out_ch, in_ch, kH, kW] 형태 유지
        w = c.weight * w[:, None, None, None]
        # BN의 bias 항: beta - mean * gamma / sqrt(var + eps)
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        # 융합된 가중치로 새 Conv2d 생성
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# ==============================================================================
# BN_Linear: BatchNorm1d + Linear 를 묶은 유틸리티 모듈 (분류 head용)
# ==============================================================================
class BN_Linear(torch.nn.Sequential):
    """BatchNorm1d와 Linear를 묶은 분류 head용 모듈.

    역할:
        - 분류 태스크의 최종 레이어에서 BN → FC 순서로 사용
        - downstream (detection) 버전에서는 직접 사용되지 않지만 정의는 유지

    입력 shape: (B, a)
    출력 shape: (B, b)
    """
    def __init__(self, a, b, bias=True, std=0.02):
        """
        Args:
            a (int): 입력 특징 차원
            b (int): 출력 차원 (클래스 수 등)
            bias (bool): Linear의 bias 사용 여부. 기본값 True
            std (float): Linear weight 초기화에 사용하는 truncated normal 표준편차. 기본값 0.02
        """
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        # Transformer 관행에 따라 Linear weight를 truncated normal로 초기화
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """BN과 Linear를 단일 Linear로 융합.

        반환:
            torch.nn.Linear: BN이 통합된 단일 Linear 레이어
        """
        bn, l = self._modules.values()
        # BN 스케일 벡터: [in_features]
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        # BN bias 벡터 계산
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        # Linear weight에 BN 스케일 적용: [out, in] * [in] -> [out, in]
        w = l.weight * w[None, :]
        if l.bias is None:
            # bias 없는 경우: b_fused = BN_bias @ W^T
            b = b @ self.l.weight.T
        else:
            # bias 있는 경우: b_fused = W @ b + linear_bias
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

def replace_batchnorm(net):
    """네트워크 내 모든 BN을 재귀적으로 Conv에 융합(fuse)하거나 Identity로 대체.

    역할:
        - fuse() 메서드를 가진 모듈(Conv2d_BN, BN_Linear)은 융합 실행
        - 독립적인 BatchNorm2d는 Identity로 교체
        - 추론 속도 최적화에 사용 (fuse=True 옵션으로 활성화)

    Args:
        net (torch.nn.Module): 처리할 네트워크 모듈
    """
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            # Conv2d_BN 또는 BN_Linear처럼 fuse() 가능한 모듈은 융합
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            # 독립 BN2d는 Identity로 교체 (이미 fuse된 경우 잔여 BN 처리)
            setattr(net, child_name, torch.nn.Identity())
        else:
            # 재귀적으로 하위 모듈 처리
            replace_batchnorm(child)


# ==============================================================================
# PatchMerging: 다운샘플링 + 채널 차원 확장 모듈 (스테이지 전환)
# ==============================================================================
class PatchMerging(torch.nn.Module):
    """스테이지 간 공간 해상도를 절반으로 줄이고 채널 차원을 확장하는 모듈.

    역할:
        - 스테이지 전환 시 Feature Map 해상도를 2배 다운샘플링
        - 채널 수를 dim -> out_dim으로 확장
        - Conv1x1 → DW-Conv3x3(stride=2) → SE → Conv1x1 구조

    입력 shape: (B, dim, H, W)
    출력 shape: (B, out_dim, H//2, W//2)

    분류 버전과의 차이: 동일한 구현체 (공유 구성요소)
    """
    def __init__(self, dim, out_dim, input_resolution):
        """
        Args:
            dim (int): 입력 채널 수
            out_dim (int): 출력 채널 수
            input_resolution (int): 입력 공간 해상도 (H 또는 W)
        """
        super().__init__()
        # 내부 확장 채널: dim * 4 (병목 구조의 확장)
        hid_dim = int(dim * 4)
        # 1x1 Conv로 채널 확장: dim -> hid_dim
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = torch.nn.ReLU()
        # 3x3 Depthwise Conv (stride=2): 공간 해상도 절반으로 다운샘플
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        # Squeeze-and-Excitation: 채널 간 의존성 모델링 (reduction ratio 0.25)
        self.se = SqueezeExcite(hid_dim, .25)
        # 1x1 Conv로 채널 축소: hid_dim -> out_dim
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        """순전파: Conv1x1 → ReLU → DW-Conv3x3(stride=2) → ReLU → SE → Conv1x1

        Args:
            x (Tensor): 입력 특징 맵, shape (B, dim, H, W)
        Returns:
            Tensor: 다운샘플된 특징 맵, shape (B, out_dim, H//2, W//2)
        """
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


# ==============================================================================
# Residual: Stochastic Depth(Drop Path)를 지원하는 잔차 연결 래퍼
# ==============================================================================
class Residual(torch.nn.Module):
    """잔차 연결(Residual Connection)을 수행하는 래퍼 모듈.

    역할:
        - y = x + m(x) 형태의 skip connection 구현
        - 학습 시 drop > 0이면 Stochastic Depth 적용 (일부 샘플의 분기를 확률적으로 제거)

    입력/출력 shape: 내부 모듈 m과 동일 (항등 잔차이므로 shape 유지)
    """
    def __init__(self, m, drop=0.):
        """
        Args:
            m (torch.nn.Module): 잔차 분기에 적용될 서브모듈
            drop (float): Stochastic Depth drop 확률. 0이면 일반 residual. 기본값 0.
        """
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        """순전파: 학습 중 drop > 0이면 Stochastic Depth, 그 외에는 일반 잔차 연결.

        Args:
            x (Tensor): 입력 텐서
        Returns:
            Tensor: x + m(x) (또는 Stochastic Depth 적용된 결과)
        """
        if self.training and self.drop > 0:
            # Stochastic Depth: 배치 내 각 샘플에 대해 독립적으로 분기 마스킹
            # ge_(self.drop): drop 확률보다 큰 경우만 1 (나머지 0)
            # div(1 - self.drop): drop 확률로 스케일 보정 (기댓값 유지)
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


# ==============================================================================
# FFN: Position-wise Feed-Forward Network (1x1 Conv 기반)
# ==============================================================================
class FFN(torch.nn.Module):
    """1x1 Conv 기반의 Feed-Forward Network.

    역할:
        - Transformer의 FFN과 동일한 역할 (채널 차원에서 비선형 변환)
        - Conv 기반이므로 공간 정보를 보존하면서 채널 믹싱 수행
        - 구조: pw1(확장) → ReLU → pw2(축소)

    입력 shape: (B, ed, H, W)
    출력 shape: (B, ed, H, W)  -- 채널 수 유지
    """
    def __init__(self, ed, h, resolution):
        """
        Args:
            ed (int): 입력/출력 채널 수 (embedding dimension)
            h (int): 내부 확장 채널 수 (hidden dimension, 보통 ed * 2)
            resolution (int): 입력 공간 해상도
        """
        super().__init__()
        # Pointwise Conv (1x1): 채널 확장 ed -> h
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        # Pointwise Conv (1x1): 채널 축소 h -> ed
        # bn_weight_init=0: 초기에 항등 함수처럼 동작 (Residual 내에서 안정적 학습)
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        """순전파: pw1 → ReLU → pw2

        Args:
            x (Tensor): 입력 특징 맵, shape (B, ed, H, W)
        Returns:
            Tensor: 변환된 특징 맵, shape (B, ed, H, W)
        """
        x = self.pw2(self.act(self.pw1(x)))
        return x


# ==============================================================================
# CascadedGroupAttention: 그룹별 Cascaded Self-Attention 모듈
# ==============================================================================
class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    # [한국어 설명]
    # 역할:
    #   - EfficientViT의 핵심 Attention 모듈
    #   - 입력 채널을 num_heads개로 분할 후 각 헤드가 순차적으로(Cascaded) 처리
    #   - 이전 헤드의 출력이 다음 헤드의 입력에 더해짐 (그룹 간 정보 전달)
    #   - 각 헤드에서 Query에 DW-Conv를 적용하여 지역 구조 정보를 쿼리에 주입
    #   - 상대 위치 편향(attention_biases)을 통해 위치 정보 인코딩
    #
    # 입력 shape: (B, C, H, W) -- C = dim, H*W <= resolution^2
    # 출력 shape: (B, C, H, W)
    #
    # 분류 버전과의 차이:
    #   - 분류 버전은 전체 이미지 해상도에서 동작
    #   - Downstream 버전은 LocalWindowAttention 내에서 window 단위로 호출됨
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        """
        Args:
            dim (int): 입력 채널 수 (전체)
            key_dim (int): 각 헤드의 Query/Key 차원
            num_heads (int): Attention 헤드 수. 기본값 8
            attn_ratio (int): Value 차원 배율 (d = attn_ratio * key_dim). 기본값 4
            resolution (int): 처리할 공간 해상도 (윈도우 크기). 기본값 14
            kernels (List[int]): 각 헤드의 DW-Conv 커널 크기 리스트. 기본값 [5,5,5,5]
        """
        super().__init__()
        self.num_heads = num_heads
        # Attention 스케일링 인자: 1 / sqrt(key_dim)
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        # Value 차원: d = attn_ratio * key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            # 각 헤드용 QKV 프로젝션: 입력 채널을 헤드 수로 나눈 뒤 (Q+K+V) 생성
            # 출력 채널: key_dim(Q) + key_dim(K) + d(V)
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            # Query에 적용하는 Depthwise Conv: 지역 구조 정보를 Query에 주입
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        # 헤드별 QKV 프로젝션 리스트
        self.qkvs = torch.nn.ModuleList(qkvs)
        # 헤드별 Query DW-Conv 리스트
        self.dws = torch.nn.ModuleList(dws)
        # 출력 프로젝션: 모든 헤드의 Value를 합쳐서 원래 dim으로 복원
        # ReLU + Conv1x1 구조
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        # ------------------------------------------------------------------
        # 상대 위치 편향(Relative Position Bias) 인덱스 계산
        # resolution x resolution 격자의 모든 점 쌍에 대한 오프셋을 미리 계산
        # ------------------------------------------------------------------
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)  # = resolution^2
        attention_offsets = {}  # (offset_h, offset_w) -> 인덱스 매핑
        idxs = []
        for p1 in points:
            for p2 in points:
                # 절댓값 오프셋: 상대 위치의 방향을 무시하고 거리만 사용
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        # attention_biases: 학습 가능한 상대 위치 편향 파라미터
        # shape: (num_heads, len(attention_offsets))
        # -- 주요 포인트 (Detection backbone): --
        # 입력 해상도(resolution)가 사전학습과 다른 경우 이 파라미터의 크기가 달라짐
        # init_weights()에서 bicubic 보간으로 크기를 맞춤
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        # attention_bias_idxs: 각 쿼리-키 쌍에 대한 편향 인덱스 (버퍼로 등록)
        # shape: (N, N) = (resolution^2, resolution^2)
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """학습/평가 모드 전환 시 attention bias 캐시를 관리.

        Args:
            mode (bool): True면 학습 모드, False면 평가 모드

        평가 모드(mode=False)일 때:
            - attention_biases와 attention_bias_idxs를 미리 결합하여 self.ab에 캐시
            - forward에서 반복적인 인덱싱 연산을 피해 추론 속도 향상
        학습 모드(mode=True)일 때:
            - 캐시된 self.ab를 삭제 (파라미터 업데이트 후 캐시가 유효하지 않으므로)
        """
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            # 평가 모드: attention_biases를 인덱스로 미리 인덱싱하여 캐시
            # self.ab shape: (num_heads, N, N)
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        """Cascaded Group Attention 순전파.

        핵심 연산 흐름:
            1. 입력 x를 num_heads개의 그룹으로 분할 (채널 차원)
            2. 각 헤드에서 QKV 프로젝션 수행
            3. Query에 DW-Conv 적용 (지역 구조 주입)
            4. Dot-product Attention + 상대 위치 편향
            5. 모든 헤드의 Value 결합 후 프로젝션

        Cascaded 구조:
            - i번째 헤드의 출력이 (i+1)번째 헤드의 입력에 더해짐
            - 앞 헤드의 컨텍스트 정보가 뒤 헤드로 전파됨

        Args:
            x (Tensor): 입력 특징 맵, shape (B, C, H, W)
        Returns:
            Tensor: Attention 출력, shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 학습 시: 매 forward마다 attention_biases 인덱싱 (파라미터 업데이트 반영)
        # 평가 시: 캐시된 self.ab 사용 (속도 향상)
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        # 입력을 num_heads개의 그룹으로 분할: 각 feats_in[i] shape = (B, C//num_heads, H, W)
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input
                # Cascaded: 이전 헤드의 Value 출력을 현재 헤드 입력에 더함
                feat = feat + feats_in[i]
            # QKV 프로젝션: (B, C//num_heads, H, W) -> (B, key_dim*2+d, H, W)
            feat = qkv(feat)
            # Q, K, V 분리:
            #   q: (B, key_dim, H, W)
            #   k: (B, key_dim, H, W)
            #   v: (B, d, H, W)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            # Query에 DW-Conv 적용: 지역 구조 정보를 Query에 주입
            q = self.dws[i](q)
            # 공간 차원 평탄화: (B, C/h, H, W) -> (B, C/h, H*W)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            # Scaled Dot-product Attention + 상대 위치 편향
            # q^T @ k: (B, N, N), scale: 1/sqrt(key_dim)
            # attention_bias: (num_heads, N, N)[i] -> (N, N) 브로드캐스팅
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1) # BNN
            # Attention-weighted Value 집계: v @ attn^T -> (B, d, H*W) -> (B, d, H, W)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
            feats_out.append(feat)
        # 모든 헤드의 Value 결합 후 출력 프로젝션
        # torch.cat(feats_out, 1): (B, d*num_heads, H, W)
        # self.proj: (B, d*num_heads, H, W) -> (B, dim, H, W)
        x = self.proj(torch.cat(feats_out, 1))
        return x


# ==============================================================================
# LocalWindowAttention: 윈도우 분할 기반 지역 Attention 모듈
# ==============================================================================
class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    # [한국어 설명]
    # 역할:
    #   - 큰 특징 맵을 window_resolution x window_resolution 크기의 윈도우로 분할
    #   - 각 윈도우 내에서 CascadedGroupAttention 적용 (지역 Attention)
    #   - 윈도우보다 작은 입력은 전체에 Attention 직접 적용
    #
    # Detection backbone으로서의 중요성:
    #   - 다양한 입력 해상도에서 동작 가능 (윈도우 패딩 처리)
    #   - 분류 버전(고정 해상도)과 달리 임의의 H, W에서 동작
    #
    # 입력 shape: (B, dim, H, W)
    # 출력 shape: (B, dim, H, W)
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        """
        Args:
            dim (int): 입력/출력 채널 수
            key_dim (int): Query/Key 차원
            num_heads (int): Attention 헤드 수. 기본값 8
            attn_ratio (int): Value 차원 배율. 기본값 4
            resolution (int): 전체 특징 맵 해상도. 기본값 14
            window_resolution (int): 로컬 윈도우 크기. 기본값 7
                                     CascadedGroupAttention의 resolution 파라미터로 사용됨
            kernels (List[int]): 각 헤드의 DW-Conv 커널 크기. 기본값 [5,5,5,5]
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        # 내부 Attention: 윈도우 크기를 resolution으로 사용
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio,
                                resolution=window_resolution,
                                kernels=kernels,)

    def forward(self, x):
        """윈도우 분할 후 CascadedGroupAttention 적용.

        처리 흐름:
            1. 입력이 윈도우 크기 이하이면: 직접 Attention 적용
            2. 입력이 윈도우 크기 초과이면:
               a. window_resolution의 배수가 되도록 패딩
               b. BHWC -> (B*nH*nW)Chw 형태로 윈도우 분할
               c. 각 윈도우에 Attention 적용
               d. 윈도우를 원래 shape으로 복원 (reverse window partition)
               e. 패딩 제거

        Args:
            x (Tensor): 입력 특징 맵, shape (B, C, H, W)
        Returns:
            Tensor: Attention 출력, shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        if H <= self.window_resolution and W <= self.window_resolution:
            # 입력 크기가 윈도우보다 작거나 같으면 전체에 Attention 직접 적용
            x = self.attn(x)
        else:
            # 윈도우 분할을 위해 BCHW -> BHWC로 변환
            x = x.permute(0, 2, 3, 1)
            # 윈도우 크기의 배수가 되도록 패딩 계산
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                # 오른쪽, 아래쪽 방향으로 0-패딩 (C 차원에는 패딩 없음)
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r  # 패딩 후 해상도
            nH = pH // self.window_resolution  # 수직 윈도우 수
            nW = pW // self.window_resolution  # 수평 윈도우 수
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            # 윈도우 분할: 각 윈도우를 독립 배치 원소로 처리
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            # 각 윈도우에 독립적으로 Attention 적용
            # 입력/출력: (B*nH*nW, C, window_resolution, window_resolution)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            # 윈도우 복원: 분할된 윈도우를 원래 공간 구조로 재조립
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                # 패딩 제거: 원래 H, W 크기로 자름
                x = x[:, :H, :W].contiguous()

            # BHWC -> BCHW로 복원
            x = x.permute(0, 3, 1, 2)

        return x


# ==============================================================================
# EfficientViTBlock: EfficientViT의 기본 빌딩 블록 (DW-Conv + FFN + Attention 조합)
# ==============================================================================
class EfficientViTBlock(torch.nn.Module):
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    # [한국어 설명]
    # 역할:
    #   - EfficientViT의 기본 블록: DW-Conv → FFN → Attention → DW-Conv → FFN 순서
    #   - 모든 서브모듈은 Residual 래퍼로 감싸져 skip connection 포함
    #   - type='s': Self-Attention 기반 토큰 믹서 사용
    #
    # 입력 shape: (B, ed, H, W)
    # 출력 shape: (B, ed, H, W)  -- 채널/해상도 유지
    #
    # 분류 버전과의 차이:
    #   - 동일한 블록 구조이나, LocalWindowAttention이 임의 해상도를 지원
    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        """
        Args:
            type (str): 토큰 믹서 타입. 's' = Self-Attention
            ed (int): 임베딩 차원 (입력/출력 채널 수)
            kd (int): Query/Key 차원
            nh (int): Attention 헤드 수. 기본값 8
            ar (int): Value 차원 배율. 기본값 4
            resolution (int): 입력 공간 해상도. 기본값 14
            window_resolution (int): 로컬 윈도우 크기. 기본값 7
            kernels (List[int]): 각 헤드의 DW-Conv 커널 크기. 기본값 [5,5,5,5]
        """
        super().__init__()

        # dw0: Attention 이전의 Depthwise Conv (지역 특징 강화) + Residual
        # bn_weight_init=0: 초기에 항등 함수 (안정적 학습 시작)
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        # ffn0: Attention 이전의 FFN (채널 믹싱) + Residual
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            # mixer: 로컬 윈도우 Self-Attention (핵심 토큰 믹서) + Residual
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                    resolution=resolution, window_resolution=window_resolution, kernels=kernels))

        # dw1: Attention 이후의 Depthwise Conv (지역 특징 강화) + Residual
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        # ffn1: Attention 이후의 FFN (채널 믹싱) + Residual
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        """순전파: dw0 → ffn0 → mixer(Attention) → dw1 → ffn1

        모든 서브모듈은 Residual 래퍼 내에 있으므로 실제로는:
            x = x + dw0(x)
            x = x + ffn0(x)
            x = x + mixer(x)
            x = x + dw1(x)
            x = x + ffn1(x)

        Args:
            x (Tensor): 입력 특징 맵, shape (B, ed, H, W)
        Returns:
            Tensor: 출력 특징 맵, shape (B, ed, H, W)
        """
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


# ==============================================================================
# EfficientViT: Detection / Downstream Task용 EfficientViT Backbone
# ==============================================================================
class EfficientViT(torch.nn.Module):
    """Detection backbone으로서의 EfficientViT.

    역할:
        - 분류 버전과 달리 분류 head 없이 다중 스케일 특징 맵을 반환
        - FPN(Feature Pyramid Network) 등 넥(Neck)과 연동하여 객체 탐지에 사용
        - 3개 스테이지(blocks1, blocks2, blocks3)의 출력을 tuple로 반환

    입력 shape: (B, 3, H, W)  -- H, W는 img_size
    출력 shape: tuple of 3 Tensors
        - outs[0]: (B, embed_dim[0], H//patch_size,     W//patch_size)
        - outs[1]: (B, embed_dim[1], H//patch_size//2,  W//patch_size//2)
        - outs[2]: (B, embed_dim[2], H//patch_size//4,  W//patch_size//4)
        (down_ops에 따라 스케일이 달라질 수 있음)

    Detection backbone으로서의 주요 특징:
        1. 다중 스케일 출력: FPN이 소비하는 P3/P4/P5 수준의 특징 맵
        2. Frozen stages: patch_embed를 고정하여 사전학습 지식 보존
        3. BatchNorm 고정: 학습 모드에서도 BN을 eval 상태로 유지 (소량 데이터 fine-tuning 안정성)
        4. 사전학습 가중치 로딩: attention bias 크기 불일치 시 bicubic 보간 자동 처리

    분류 버전(classification/efficientvit.py)과의 차이:
        - 분류: head (BN_Linear + cls/dist head) 포함, 단일 벡터 출력
        - 탐지: head 없음, 3개 스테이지 특징 맵 tuple 출력
        - 탐지: frozen_stages, _freeze_stages(), train() 오버라이딩 추가
        - 탐지: init_weights()가 MMDetection 방식으로 구현
        - 탐지: @BACKBONES.register_module()으로 MMDet 레지스트리에 등록
    """
    def __init__(self, img_size=400,
                 patch_size=16,
                 frozen_stages=0,
                 in_chans=3,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 pretrained=None,
                 distillation=False,):
        """
        Args:
            img_size (int): 입력 이미지 크기. 기본값 400 (탐지용 대형 입력)
                            분류 버전은 224 사용
            patch_size (int): 초기 패치 임베딩의 다운샘플 비율. 기본값 16
                              4단계의 stride=2 Conv로 구현 (2^4=16)
            frozen_stages (int): 고정할 스테이지 수.
                                 현재 구현에서는 0 이상이면 patch_embed를 고정
            in_chans (int): 입력 이미지 채널 수. 기본값 3 (RGB)
            stages (List[str]): 각 스테이지의 토큰 믹서 타입. 's'=Self-Attention
            embed_dim (List[int]): 각 스테이지의 채널 수
            key_dim (List[int]): 각 스테이지의 Query/Key 차원
            depth (List[int]): 각 스테이지의 EfficientViTBlock 반복 수
            num_heads (List[int]): 각 스테이지의 Attention 헤드 수
            window_size (List[int]): 각 스테이지의 로컬 윈도우 크기
            kernels (List[int]): CGA의 헤드별 DW-Conv 커널 크기 (모든 스테이지 공유)
            down_ops (List[List]): 스테이지 간 다운샘플링 설정
                                   ['subsample', 2]: stride=2 다운샘플
                                   ['']: 다운샘플 없음 (마지막 스테이지)
            pretrained (str or None): 사전학습 가중치 파일 경로
            distillation (bool): Knowledge Distillation 사용 여부 (현재 backbone에서는 미사용)
        """
        super().__init__()

        resolution = img_size
        # ------------------------------------------------------------------
        # Patch Embedding: 4단계 stride=2 Conv로 이미지를 패치 토큰으로 변환
        # 입력: (B, 3, H, W) -> 출력: (B, embed_dim[0], H//16, W//16)
        # 구조: Conv(3->ed/8, stride=2) → Conv(ed/8->ed/4, stride=2)
        #       → Conv(ed/4->ed/2, stride=2) → Conv(ed/2->ed, stride=2)
        # 총 stride = 16 = patch_size
        # ------------------------------------------------------------------
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8))

        # patch_embed 이후 공간 해상도: img_size // patch_size
        resolution = img_size // patch_size
        # 각 스테이지의 Attention Ratio 계산: embed_dim / (key_dim * num_heads)
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        # 3개 스테이지의 블록 리스트 초기화
        self.blocks1 = []  # 스테이지 1 (가장 큰 해상도, 가장 작은 채널)
        self.blocks2 = []  # 스테이지 2 (중간 해상도)
        self.blocks3 = []  # 스테이지 3 (가장 작은 해상도, 가장 큰 채널)
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            # 각 스테이지에 depth 만큼 EfficientViTBlock 추가
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                #('Subsample' stride)
                # 다음 스테이지의 블록 리스트에 다운샘플 모듈 추가
                blk = eval('self.blocks' + str(i+2))
                # 다운샘플 후 해상도: ceil((resolution) / stride)
                resolution_ = (resolution - 1) // do[1] + 1
                # 다운샘플 전: DW-Conv + FFN (현재 채널 수 유지)
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),))
                # PatchMerging: 해상도 절반, 채널 수 확장 (embed_dim[i] -> embed_dim[i+1])
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_  # 해상도 업데이트
                # 다운샘플 후: DW-Conv + FFN (새 채널 수 적용)
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),))
        # 리스트를 Sequential로 변환 (순차 실행 가능)
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        self.frozen_stages = frozen_stages # freeze the patch embedding
        # 초기화 시 즉시 스테이지 고정 적용
        self._freeze_stages()

        if pretrained is not None:
            self.init_weights(pretrained=pretrained)

    def _freeze_stages(self):
        """지정된 frozen_stages에 따라 patch_embed를 고정(freeze).

        frozen_stages >= 0이면 patch_embed를 eval 모드로 전환하고
        모든 파라미터의 requires_grad를 False로 설정.

        Detection fine-tuning 시 사전학습된 저수준 특징 추출기를 보존하는 역할.
        """
        if self.frozen_stages >= 0:
            # patch_embed를 평가 모드로 전환 (BN의 running statistics 고정)
            self.patch_embed.eval()
            # patch_embed 내 모든 파라미터의 gradient 계산 비활성화
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # [한국어 설명]
        # 사전학습 가중치를 로딩하고 attention bias 크기 불일치를 처리.
        #
        # 처리 흐름:
        #   1. 체크포인트 파일 로딩 (_load_checkpoint)
        #   2. state_dict 추출 (다양한 포맷 지원: 'state_dict', 'model', 또는 직접)
        #   3. 'module.' prefix 제거 (DataParallel 저장 포맷 호환)
        #   4. attention_bias_idxs 키 삭제 (버퍼는 재계산됨)
        #   5. attention_biases 크기 불일치 시 bicubic 보간 (핵심 기능)
        #      - 사전학습: 224x224 → resolution=14 → biases shape (nH, 14*14=196 unique offsets 수)
        #      - Detection: 400x400 이상 → resolution 증가 → biases shape이 달라짐
        #      - S1=sqrt(L1): 사전학습 resolution, S2=sqrt(L2): 현재 resolution
        #      - 2D bicubic 보간: (1, nH, S1, S1) -> (1, nH, S2, S2)
        #   6. strict=False로 가중치 로드 (shape 불일치 키는 무시)

        if isinstance(pretrained, str):
            logger = get_root_logger()
            # 체크포인트 파일을 CPU로 로딩 (GPU 메모리 절약)
            checkpoint = _load_checkpoint(pretrained, map_location='cpu')

            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f'No state_dict found in checkpoint file {filename}')
            # get state_dict from checkpoint
            # 다양한 체크포인트 포맷 지원
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # strip prefix of state_dict
            # DataParallel로 저장된 경우 'module.' prefix 제거
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            model_state_dict = self.state_dict()
            # bicubic interpolate attention_biases if not match

            # attention_bias_idxs는 버퍼로 재계산되므로 state_dict에서 삭제
            rpe_idx_keys = [
                k for k in state_dict.keys() if "attention_bias_idxs" in k]
            for k in rpe_idx_keys:
                print("deleting key: ", k)
                del state_dict[k]

            # attention_biases 크기 불일치 처리 (Detection backbone 핵심 기능)
            relative_position_bias_table_keys = [
                k for k in state_dict.keys() if "attention_biases" in k]
            for k in relative_position_bias_table_keys:
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = model_state_dict[k]
                # 사전학습 shape: (nH1, L1), 현재 shape: (nH2, L2)
                nH1, L1 = relative_position_bias_table_pretrained.size()
                nH2, L2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    # 헤드 수 불일치: 보간 불가, 경고 출력
                    logger.warning(f"Error in loading {k} due to different number of heads")
                else:
                    if L1 != L2:
                        # 해상도 불일치: L = resolution^2에 해당하는 고유 오프셋 수
                        # S1, S2는 각각 사전학습/현재의 윈도우 해상도 추정값
                        print("resizing key {} from {} * {} to {} * {}".format(k, L1, L1, L2, L2))
                        # bicubic interpolate relative_position_bias_table if not match
                        # attention_biases를 2D 공간으로 해석하여 bicubic 보간
                        # (1, nH1, S1, S1) -> (1, nH2, S2, S2) -> (nH2, L2)
                        S1 = int(L1 ** 0.5)  # 사전학습 해상도
                        S2 = int(L2 ** 0.5)  # 현재 해상도
                        relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.view(1, nH1, S1, S1), size=(S2, S2),
                            mode='bicubic')
                        state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                            nH2, L2)

            # strict=False: shape 불일치 키는 무시하고 가능한 키만 로드
            load_state_dict(self, state_dict, strict=False, logger=logger)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Weight decay에서 제외할 파라미터 이름 집합 반환.

        Returns:
            set: 'attention_biases'를 포함하는 파라미터 키 집합

        attention_biases는 상대 위치 편향으로 정규화 대상에서 제외하는 것이 일반적.
        TorchScript에서는 무시됨 (@torch.jit.ignore).
        """
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        # [한국어 설명]
        # 학습 모드 전환 시 frozen_stages 고정 상태를 유지하고
        # 모든 BatchNorm을 강제로 eval 모드로 설정.
        #
        # Detection fine-tuning 시 소량 데이터로 BN statistics가 망가지는 것을 방지.
        # 분류 버전에는 없는 오버라이딩 (Detection backbone의 고유 기능).
        #
        # Args:
        #     mode (bool): True면 학습 모드, False면 평가 모드
        super(EfficientViT, self).train(mode)
        # 학습 모드에서도 frozen_stages는 고정 상태 유지
        self._freeze_stages()
        if mode:
            # 모든 BatchNorm을 eval 모드로 강제 설정
            # (소량의 Detection 데이터로 fine-tuning 시 BN 안정성 향상)
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        """Detection backbone 순전파: 3개의 다중 스케일 특징 맵 반환.

        처리 흐름:
            1. patch_embed: 이미지를 패치 토큰으로 변환 (stride=16)
            2. blocks1: 스테이지 1 처리 → 첫 번째 스케일 특징 맵
            3. blocks2: 스테이지 2 처리 (다운샘플 포함) → 두 번째 스케일 특징 맵
            4. blocks3: 스테이지 3 처리 (다운샘플 포함) → 세 번째 스케일 특징 맵

        FPN 연동:
            반환된 tuple은 FPN의 입력으로 사용됨
            FPN은 이를 바탕으로 P3~P6 등의 다중 스케일 특징 맵을 생성

        Args:
            x (Tensor): 입력 이미지, shape (B, 3, H, W)
        Returns:
            tuple of Tensor: 3개의 다중 스케일 특징 맵
                - outs[0]: shape (B, embed_dim[0], H//16,    W//16)    -- 가장 큰 해상도
                - outs[1]: shape (B, embed_dim[1], H//32,    W//32)    -- 중간 해상도
                - outs[2]: shape (B, embed_dim[2], H//64,    W//64)    -- 가장 작은 해상도
                (down_ops=[['subsample',2],['subsample',2],['']] 기준)
        """
        # 1단계: Patch Embedding (stride=16 다운샘플)
        x = self.patch_embed(x)
        outs = []
        # 2단계: 스테이지 1 (blocks1에는 다운샘플 모듈이 없음)
        x = self.blocks1(x)
        outs.append(x)  # 첫 번째 스케일 특징 맵 저장
        # 3단계: 스테이지 2 (blocks2 시작 부분에 PatchMerging으로 stride=2 다운샘플)
        x = self.blocks2(x)
        outs.append(x)  # 두 번째 스케일 특징 맵 저장
        # 4단계: 스테이지 3 (blocks3 시작 부분에 PatchMerging으로 stride=2 다운샘플)
        x = self.blocks3(x)
        outs.append(x)  # 세 번째 스케일 특징 맵 저장
        return tuple(outs)

# ==============================================================================
# 모델 설정 딕셔너리: 각 EfficientViT 변형(M0~M5)의 하이퍼파라미터
# ==============================================================================
# 모든 변형의 공통 설정:
#   - img_size: 224 (분류 기본값; Detection 사용 시 config에서 오버라이드)
#   - patch_size: 16 (4단계 stride=2 Conv)
#   - depth: 스테이지별 블록 수
#   - window_size: 로컬 윈도우 크기 (모두 7x7)
#
# M0 ~ M5: 채널 수(embed_dim)와 헤드 수(num_heads)가 증가하는 스케일업 시리즈

EfficientViT_m0 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [64, 128, 192],    # 스테이지별 채널 수 (가장 경량)
        'depth': [1, 2, 3],              # 스테이지별 블록 반복 수
        'num_heads': [4, 4, 4],          # 스테이지별 Attention 헤드 수
        'window_size': [7, 7, 7],        # 로컬 윈도우 크기
        'kernels': [7, 5, 3, 3],         # 헤드별 DW-Conv 커널 크기
    }

EfficientViT_m1 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 144, 192],    # M0 대비 채널 수 증가
        'depth': [1, 2, 3],
        'num_heads': [2, 3, 3],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m2 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 192, 224],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 2],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m3 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 240, 320],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],         # M3는 균일한 커널 크기 사용
    }

EfficientViT_m4 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 256, 384],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m5 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [192, 288, 384],    # 가장 큰 채널 수 (가장 고성능)
        'depth': [1, 3, 4],              # 더 깊은 스테이지 2, 3
        'num_heads': [3, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

# ==============================================================================
# MMDetection Backbone 등록 함수들
# ==============================================================================
# [MMDetection 등록 방식 설명]
# @BACKBONES.register_module() 데코레이터:
#   - MMDetection의 BACKBONES 레지스트리에 함수를 등록
#   - config 파일에서 type='EfficientViT_M0' 등으로 이 함수를 참조 가능
#   - MMDetection은 config 파싱 시 레지스트리를 통해 backbone을 동적으로 생성
#
# 예시 config (mmdetection config 파일 내):
#   backbone = dict(
#       type='EfficientViT_M0',
#       pretrained='/path/to/pretrained.pth',
#       frozen_stages=0,
#       fuse=False,
#   )
#
# 각 함수는 EfficientViT 모델을 생성하고 선택적으로 BN 융합(fuse)을 적용.
# fuse=True: 추론 시 BN을 Conv에 통합하여 속도 향상 (학습 중에는 권장하지 않음)

@BACKBONES.register_module()
def EfficientViT_M0(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m0):
    """EfficientViT-M0 Detection Backbone 생성 함수.

    Args:
        pretrained (str or False): 사전학습 가중치 경로. False면 랜덤 초기화
        frozen_stages (int): 고정할 스테이지 수 (0 이상이면 patch_embed 고정)
        distillation (bool): Knowledge Distillation 사용 여부
        fuse (bool): True면 BN을 Conv에 융합 (추론 최적화)
        pretrained_cfg: timm 호환성을 위한 파라미터 (미사용)
        model_cfg (dict): 모델 하이퍼파라미터 딕셔너리

    Returns:
        EfficientViT: M0 설정의 EfficientViT Detection backbone
    """
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

@BACKBONES.register_module()
def EfficientViT_M1(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m1):
    """EfficientViT-M1 Detection Backbone 생성 함수.

    Args:
        pretrained (str or False): 사전학습 가중치 경로
        frozen_stages (int): 고정할 스테이지 수
        distillation (bool): Knowledge Distillation 사용 여부
        fuse (bool): True면 BN을 Conv에 융합
        pretrained_cfg: timm 호환성을 위한 파라미터 (미사용)
        model_cfg (dict): 모델 하이퍼파라미터 딕셔너리

    Returns:
        EfficientViT: M1 설정의 EfficientViT Detection backbone
    """
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

@BACKBONES.register_module()
def EfficientViT_M2(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m2):
    """EfficientViT-M2 Detection Backbone 생성 함수.

    Args:
        pretrained (str or False): 사전학습 가중치 경로
        frozen_stages (int): 고정할 스테이지 수
        distillation (bool): Knowledge Distillation 사용 여부
        fuse (bool): True면 BN을 Conv에 융합
        pretrained_cfg: timm 호환성을 위한 파라미터 (미사용)
        model_cfg (dict): 모델 하이퍼파라미터 딕셔너리

    Returns:
        EfficientViT: M2 설정의 EfficientViT Detection backbone
    """
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

@BACKBONES.register_module()
def EfficientViT_M3(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m3):
    """EfficientViT-M3 Detection Backbone 생성 함수.

    Args:
        pretrained (str or False): 사전학습 가중치 경로
        frozen_stages (int): 고정할 스테이지 수
        distillation (bool): Knowledge Distillation 사용 여부
        fuse (bool): True면 BN을 Conv에 융합
        pretrained_cfg: timm 호환성을 위한 파라미터 (미사용)
        model_cfg (dict): 모델 하이퍼파라미터 딕셔너리

    Returns:
        EfficientViT: M3 설정의 EfficientViT Detection backbone
    """
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

@BACKBONES.register_module()
def EfficientViT_M4(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m4):
    """EfficientViT-M4 Detection Backbone 생성 함수.

    Args:
        pretrained (str or False): 사전학습 가중치 경로
        frozen_stages (int): 고정할 스테이지 수
        distillation (bool): Knowledge Distillation 사용 여부
        fuse (bool): True면 BN을 Conv에 융합
        pretrained_cfg: timm 호환성을 위한 파라미터 (미사용)
        model_cfg (dict): 모델 하이퍼파라미터 딕셔너리

    Returns:
        EfficientViT: M4 설정의 EfficientViT Detection backbone
    """
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

@BACKBONES.register_module()
def EfficientViT_M5(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m5):
    """EfficientViT-M5 Detection Backbone 생성 함수 (가장 큰 변형).

    Args:
        pretrained (str or False): 사전학습 가중치 경로
        frozen_stages (int): 고정할 스테이지 수
        distillation (bool): Knowledge Distillation 사용 여부
        fuse (bool): True면 BN을 Conv에 융합
        pretrained_cfg: timm 호환성을 위한 파라미터 (미사용)
        model_cfg (dict): 모델 하이퍼파라미터 딕셔너리

    Returns:
        EfficientViT: M5 설정의 EfficientViT Detection backbone
    """
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model
