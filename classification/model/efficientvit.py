# --------------------------------------------------------
# EfficientViT Model Architecture
# Copyright (c) 2022 Microsoft
# Build the EfficientViT Model
# Written by: Xinyu Liu
# --------------------------------------------------------
# 논문: "EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention"
# (CVPR 2023, https://arxiv.org/abs/2305.07027)
#
# 전체 아키텍처 개요:
#   입력 이미지 (B, 3, 224, 224)
#     -> PatchEmbedding: 4단계 stride=2 Conv로 (B, C, 14, 14) 피처맵 생성
#     -> EfficientViTBlock x N: DW Conv + FFN + LocalWindowAttention(CGA) + DW Conv + FFN
#     -> PatchMerging: 해상도 절반, 채널 증가 (다운샘플링)
#     -> GlobalAveragePool + 분류 헤드 -> (B, num_classes)
# --------------------------------------------------------
import torch
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite


# ==============================================================================
# Conv2d_BN: Conv2d + BatchNorm2d 결합 모듈
# ==============================================================================
class Conv2d_BN(torch.nn.Sequential):
    """
    [역할]
    Conv2d와 BatchNorm2d를 하나의 Sequential로 묶은 기본 빌딩 블록.
    추론 시 fuse() 메서드로 두 레이어를 하나의 Conv2d로 병합하여 속도를 향상시킨다.

    [입력/출력 shape]
    - 입력: (B, a, H, W)  -- a: 입력 채널 수
    - 출력: (B, b, H', W') -- b: 출력 채널 수, H'/W'는 stride에 의해 결정

    [파라미터]
    - a          : 입력 채널 수
    - b          : 출력 채널 수
    - ks         : 커널 크기 (kernel_size), 기본값 1
    - stride     : 스트라이드, 기본값 1
    - pad        : 패딩 크기, 기본값 0
    - dilation   : 팽창률, 기본값 1
    - groups     : 그룹 수 (depthwise conv에서 groups=채널 수로 설정), 기본값 1
    - bn_weight_init: BN 가중치 초기화 값 (0이면 잔차 블록에서 초기 출력이 0이 됨)
    - resolution : 입력 해상도 힌트 (일부 구현에서 사용)
    """
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        # Conv2d: bias=False (BN이 bias 역할을 대체)
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        # BatchNorm2d: 채널 b에 대해 정규화
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        # BN weight를 bn_weight_init으로 초기화
        # - 잔차 블록의 마지막 레이어: bn_weight_init=0 -> 학습 초기에 잔차가 0으로 시작
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)  # BN bias는 0으로 초기화

    @torch.no_grad()
    def fuse(self):
        """
        [역할]
        Conv2d와 BatchNorm2d를 하나의 Conv2d 레이어로 수학적으로 병합한다.
        추론 속도 향상에 사용된다. (별도의 BN 연산 불필요)

        [수식 설명]
        BN 공식: y = (x - mean) / sqrt(var + eps) * weight + bias
        Conv+BN을 하나의 Conv로 합치면:
          - 새 가중치 w' = w_conv * (bn_weight / sqrt(bn_var + eps))
          - 새 편향  b' = bn_bias - bn_mean * bn_weight / sqrt(bn_var + eps)

        [반환값]
        - fused Conv2d 레이어 (bias 포함)
        """
        c, bn = self._modules.values()

        # BN 스케일 인자: gamma / sqrt(running_var + eps)
        # shape: (out_channels,)
        w = bn.weight / (bn.running_var + bn.eps)**0.5

        # Conv 가중치에 BN 스케일 인자를 곱함
        # w[:, None, None, None]: (out_channels, 1, 1, 1)로 브로드캐스팅
        w = c.weight * w[:, None, None, None]

        # BN bias 계산: beta - mean * gamma / sqrt(var + eps)
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5

        # 병합된 Conv2d 레이어 생성 (원래 Conv와 동일한 stride/padding/dilation/groups 유지)
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# ==============================================================================
# BN_Linear: BatchNorm1d + Linear 결합 모듈 (분류 헤드용)
# ==============================================================================
class BN_Linear(torch.nn.Sequential):
    """
    [역할]
    BatchNorm1d와 Linear를 결합한 분류 헤드 모듈.
    GlobalAveragePool 이후의 1D 피처에 적용된다.
    fuse() 메서드로 두 레이어를 하나의 Linear로 병합 가능.

    [입력/출력 shape]
    - 입력: (B, a)  -- a: 입력 특징 차원 (마지막 스테이지 embed_dim)
    - 출력: (B, b)  -- b: 클래스 수 (num_classes)

    [파라미터]
    - a   : 입력 특징 차원
    - b   : 출력 차원 (클래스 수)
    - bias: Linear에 bias 추가 여부
    - std : 가중치 초기화에 사용할 표준편차 (truncated normal 분포)
    """
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        # 1D BatchNorm: 글로벌 풀링 이후 피처 정규화
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        # Linear: 클래스 예측 레이어
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        # 가중치를 truncated normal 분포로 초기화 (std=0.02는 ViT 계열 표준 초기화)
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)  # bias는 0으로 초기화

    @torch.no_grad()
    def fuse(self):
        """
        [역할]
        BatchNorm1d와 Linear를 하나의 Linear 레이어로 수학적으로 병합한다.

        [수식 설명]
        BN 출력: z = (x - mean) / sqrt(var + eps) * weight + bias
        Linear 출력: y = W * z + b_linear
        병합하면:
          - 새 가중치 w' = W * diag(bn_weight / sqrt(bn_var + eps))
          - 새 편향  b' = W * (bn_bias - bn_mean * bn_weight / sqrt(bn_var + eps)) + b_linear

        [반환값]
        - fused Linear 레이어
        """
        bn, l = self._modules.values()

        # BN 스케일: gamma / sqrt(var + eps), shape: (in_features,)
        w = bn.weight / (bn.running_var + bn.eps)**0.5

        # BN의 shift(평균 보정) 항: beta - mean * gamma / sqrt(var + eps)
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5

        # Linear 가중치에 BN 스케일을 곱함 (행 방향 브로드캐스팅)
        # w[None, :]: (1, in_features)로 브로드캐스팅
        w = l.weight * w[None, :]

        # bias 병합
        if l.bias is None:
            # Linear bias 없을 때: b' = bn_shift @ W^T
            b = b @ self.l.weight.T
        else:
            # Linear bias 있을 때: b' = W @ bn_shift + b_linear
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias

        # 병합된 Linear 레이어 생성
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# ==============================================================================
# PatchMerging: 다운샘플링 전환 블록
# ==============================================================================
class PatchMerging(torch.nn.Module):
    """
    [역할]
    스테이지 간 해상도를 절반으로 줄이고 채널 수를 늘리는 다운샘플링 모듈.
    1x1 Conv -> Depthwise 3x3 Conv(stride=2) -> SE -> 1x1 Conv 구조로 구성.
    논문의 Figure 2에서 "Subsample" 블록에 해당.

    [입력/출력 shape]
    - 입력: (B, dim, H, W)
    - 출력: (B, out_dim, H//2, W//2)

    [파라미터]
    - dim              : 입력 채널 수
    - out_dim          : 출력 채널 수 (다음 스테이지의 embed_dim)
    - input_resolution : 입력 피처맵의 공간 해상도 (H 또는 W)
    """
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)  # 중간 채널: 입력 채널의 4배로 확장 (inverted bottleneck)

        # 1x1 Conv: 채널 확장 (dim -> hid_dim)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = torch.nn.ReLU()

        # 3x3 Depthwise Conv (stride=2): 공간 해상도를 절반으로 다운샘플링
        # groups=hid_dim: 각 채널이 독립적으로 컨볼루션 수행 (Depthwise)
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)

        # Squeeze-and-Excitation (SE): 채널 중요도 재보정, reduction_ratio=0.25
        self.se = SqueezeExcite(hid_dim, .25)

        # 1x1 Conv: 채널 압축 (hid_dim -> out_dim)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        """
        [순전파 흐름]
        x -> conv1(확장) -> ReLU -> conv2(DW, stride=2) -> ReLU -> SE -> conv3(압축)

        [입력]  x: (B, dim, H, W)
        [출력]  x: (B, out_dim, H//2, W//2)
        """
        # 순서: 1x1확장 -> ReLU -> DW stride=2 -> ReLU -> SE채널재보정 -> 1x1압축
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


# ==============================================================================
# Residual: 잔차 연결 래퍼 (Stochastic Depth 지원)
# ==============================================================================
class Residual(torch.nn.Module):
    """
    [역할]
    임의의 모듈 m에 잔차 연결(skip connection)을 추가하는 래퍼 클래스.
    학습 시 drop > 0이면 Stochastic Depth(확률적 깊이)를 적용하여 정규화.

    [입력/출력 shape]
    - 입력/출력 shape 동일: (B, C, H, W) 또는 (B, C)

    [파라미터]
    - m   : 잔차 연결로 감쌀 서브 모듈
    - drop: Stochastic Depth의 drop rate (학습 시에만 적용), 기본값 0
    """
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        """
        [학습 모드 + drop > 0]
        Stochastic Depth 적용:
        - 각 샘플마다 독립적으로 랜덤 마스크(0 또는 1) 생성
        - ge_(drop): drop 확률로 0 (블록 비활성화), (1-drop) 확률로 1 (블록 활성화)
        - div(1-drop): 기댓값 보정 (스케일 유지)
        결과: x + m(x) * mask / (1 - drop)

        [추론 모드 or drop=0]
        일반 잔차 연결: x + m(x)
        """
        if self.training and self.drop > 0:
            # torch.rand(...).ge_(self.drop): Bernoulli 마스크 (drop보다 크면 1, 작으면 0)
            # 마스크 shape: (B, 1, 1, 1) -> 배치 단위로 블록 전체를 드롭
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)  # 일반 잔차 연결


# ==============================================================================
# FFN: Feed-Forward Network (포인트와이즈 MLP)
# ==============================================================================
class FFN(torch.nn.Module):
    """
    [역할]
    트랜스포머의 FFN(Feed-Forward Network)을 Conv2d로 구현한 모듈.
    1x1 Conv(확장) -> ReLU -> 1x1 Conv(압축) 구조.
    입력 채널을 h로 확장한 후 다시 ed로 복원.
    논문의 EfficientViT 블록 내 'FFN' 구성요소.

    [입력/출력 shape]
    - 입력: (B, ed, H, W)
    - 출력: (B, ed, H, W)  -- shape 변화 없음

    [파라미터]
    - ed         : 입력/출력 채널 수 (embedding dimension)
    - h          : FFN 내부 확장 채널 수 (보통 ed * 2)
    - resolution : 입력 피처맵의 공간 해상도
    """
    def __init__(self, ed, h, resolution):
        super().__init__()
        # 첫 번째 포인트와이즈 Conv: 채널 확장 (ed -> h)
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        # 두 번째 포인트와이즈 Conv: 채널 복원 (h -> ed)
        # bn_weight_init=0: 잔차 블록에서 초기 출력이 0이 되도록 (학습 안정성)
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        """
        [순전파]
        x -> pw1(확장) -> ReLU -> pw2(복원)

        [입력/출력] (B, ed, H, W) -> (B, ed, H, W)
        """
        x = self.pw2(self.act(self.pw1(x)))
        return x


# ==============================================================================
# CascadedGroupAttention: 계층적 그룹 어텐션 (핵심 모듈)
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

    [역할]
    EfficientViT의 핵심 어텐션 모듈. 논문 Figure 3 참조.
    기존 Multi-Head Self-Attention과의 차이점:
      1. 채널을 헤드 수로 분할 (각 헤드는 입력 채널의 1/num_heads만 처리)
      2. 이전 헤드의 출력을 다음 헤드의 입력에 누적 (Cascaded: 계단식)
      3. Query에 Depthwise Conv 적용 (로컬 공간 정보 포함)
      4. 학습 가능한 위치 편향(attention_biases) 사용

    [입력/출력 shape]
    - 입력: (B, dim, H, W)   -- H, W는 window 크기 (보통 7x7)
    - 출력: (B, dim, H, W)

    [파라미터]
    - dim         : 입력 채널 수
    - key_dim     : Query/Key의 차원 수 (헤드당)
    - num_heads   : 어텐션 헤드 수
    - attn_ratio  : Value 차원 = attn_ratio * key_dim (헤드당)
    - resolution  : 입력 공간 해상도 (윈도우 크기)
    - kernels     : 각 헤드의 Query Depthwise Conv 커널 크기 리스트
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5  # 어텐션 스케일: 1/sqrt(key_dim), QK 내적 값을 안정화
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)  # Value의 차원 (헤드당): attn_ratio * key_dim
        self.attn_ratio = attn_ratio

        qkvs = []  # 각 헤드의 Q/K/V 프로젝션 레이어 리스트
        dws = []   # 각 헤드의 Query Depthwise Conv 리스트
        for i in range(num_heads):
            # 각 헤드: 입력 채널의 1/num_heads를 받아 Q(key_dim) + K(key_dim) + V(d)로 투영
            # 출력 채널: key_dim * 2 + d = key_dim (Q) + key_dim (K) + attn_ratio*key_dim (V)
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))

            # Query에 적용되는 Depthwise Conv (로컬 컨텍스트 포함)
            # kernels[i]: 헤드마다 다른 커널 크기로 다양한 수용 영역 포착
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))

        self.qkvs = torch.nn.ModuleList(qkvs)  # 헤드별 QKV 프로젝션
        self.dws = torch.nn.ModuleList(dws)    # 헤드별 Query DW Conv

        # 출력 프로젝션: 모든 헤드의 Value를 합쳐 원래 dim으로 복원
        # ReLU -> 1x1 Conv (d * num_heads -> dim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        # 위치 편향(Position Bias) 사전 계산
        # resolution x resolution 격자에서 모든 점 쌍의 상대적 오프셋을 계산
        points = list(itertools.product(range(resolution), range(resolution)))  # 모든 (y, x) 위치
        N = len(points)  # 총 위치 수 = resolution^2

        attention_offsets = {}  # 오프셋 -> 인덱스 매핑 (중복 제거)
        idxs = []              # N*N 크기의 오프셋 인덱스 리스트
        for p1 in points:
            for p2 in points:
                # 두 위치 간의 절대 오프셋 (방향 무관, 대칭성 활용)
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)  # 새 오프셋 유형 등록
                idxs.append(attention_offsets[offset])

        # 학습 가능한 위치 편향 파라미터: (num_heads, 고유 오프셋 수)
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))

        # 위치 편향 인덱스 행렬: (N, N) -- 각 위치 쌍 (i, j)에 대한 오프셋 인덱스
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """
        [역할]
        train/eval 모드 전환 시 위치 편향 캐시를 관리.
        - 학습 모드: 매 forward마다 최신 파라미터로 계산 (캐시 삭제)
        - 평가 모드: self.ab에 미리 계산된 위치 편향을 캐시하여 추론 속도 향상
        """
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab  # 학습 모드 전환 시 캐시 삭제 (파라미터 업데이트 반영)
        else:
            # 평가 모드: 위치 편향을 미리 완전한 (N, N) 행렬로 펼쳐 캐시
            # shape: (num_heads, N, N)
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        """
        [순전파 - Cascaded Group Attention]

        각 헤드 i에 대해:
          1. 입력 x를 num_heads 개로 채널 분할
          2. i > 0이면 이전 헤드 출력을 현재 입력에 누적 (Cascaded)
          3. QKV 프로젝션 -> Q, K, V 분리
          4. Query에 DW Conv 적용 (로컬 컨텍스트)
          5. Attention 계산: softmax((Q^T @ K) * scale + position_bias)
          6. Value와 Attention 가중합으로 출력 계산

        [입력]  x: (B, C, H, W)
        [출력]  x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 학습 시 동적으로 계산, 추론 시 캐시된 값 사용
        # trainingab shape: (num_heads, N, N), N = H*W
        trainingab = self.attention_biases[:, self.attention_bias_idxs]

        # 입력을 num_heads 개의 청크로 분할
        # feats_in: tuple of (B, C//num_heads, H, W), 길이 = num_heads
        feats_in = x.chunk(len(self.qkvs), dim=1)

        feats_out = []      # 각 헤드의 Value 출력 저장
        feat = feats_in[0]  # 첫 헤드는 첫 번째 청크로 시작

        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input
                # Cascaded: 이전 헤드의 Value 출력 + 현재 헤드의 입력 채널 병합
                # 두 텐서의 채널 수가 다를 수 있으므로 주의 (feat은 self.d 채널)
                feat = feat + feats_in[i]

            # QKV 프로젝션: (B, C//num_heads, H, W) -> (B, key_dim*2+d, H, W)
            feat = qkv(feat)

            # Q, K, V 분리
            # q: (B, key_dim, H, W), k: (B, key_dim, H, W), v: (B, d, H, W)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W

            # Query에 Depthwise Conv 적용: 로컬 공간 정보 포함
            q = self.dws[i](q)

            # 공간 차원을 시퀀스로 펼침: (B, key_dim, H*W) 또는 (B, d, H*W)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N

            # Attention 스코어 계산:
            # (q^T @ k) * scale: (B, N, key_dim) @ (B, key_dim, N) = (B, N, N)
            # + 위치 편향: (num_heads, N, N)[i] -> (N, N) (브로드캐스팅)
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale  # QK^T / sqrt(d_k)
                +
                (trainingab[i] if self.training else self.ab[i])  # 학습 가능한 위치 편향
            )
            attn = attn.softmax(dim=-1) # BNN -- 행 방향 softmax로 어텐션 가중치 정규화

            # Value와 어텐션 가중합: (B, d, N) @ (B, N, N)^T = (B, d, N)
            # -> (B, d, H, W)로 reshape
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW

            feats_out.append(feat)  # 이 헤드의 출력 저장 (다음 헤드 입력으로 누적)

        # 모든 헤드의 Value 출력을 채널 방향으로 합치고 투영
        # torch.cat(feats_out, 1): (B, d * num_heads, H, W)
        # self.proj: (B, d * num_heads, H, W) -> (B, dim, H, W)
        x = self.proj(torch.cat(feats_out, 1))
        return x


# ==============================================================================
# LocalWindowAttention: 로컬 윈도우 기반 어텐션
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

    [역할]
    입력 피처맵을 로컬 윈도우로 분할한 후 각 윈도우 내에서 CascadedGroupAttention을 수행.
    Swin Transformer의 Window Attention과 유사하지만, CGA를 내부 어텐션으로 사용.
    피처맵이 윈도우 크기보다 작으면 윈도우 분할 없이 직접 어텐션 적용.

    [입력/출력 shape]
    - 입력: (B, dim, H, W)   -- H=W=resolution
    - 출력: (B, dim, H, W)

    [파라미터]
    - dim               : 입력 채널 수
    - key_dim           : Query/Key 차원 (헤드당)
    - num_heads         : 어텐션 헤드 수
    - attn_ratio        : Value 차원 배율
    - resolution        : 전체 피처맵 해상도
    - window_resolution : 로컬 윈도우 크기 (예: 7x7)
    - kernels           : 헤드별 Query DW Conv 커널 크기
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        # 실제 사용할 윈도우 크기: 피처맵보다 크면 피처맵 크기로 제한
        window_resolution = min(window_resolution, resolution)

        # 핵심 어텐션: CascadedGroupAttention (윈도우 크기에 맞게 설정)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio,
                                resolution=window_resolution,
                                kernels=kernels,)

    def forward(self, x):
        """
        [순전파 - 윈도우 파티션 -> 어텐션 -> 윈도우 역변환]

        피처맵 크기 <= 윈도우 크기: 그냥 어텐션 적용
        피처맵 크기 > 윈도우 크기:
          1. 필요시 패딩 추가 (윈도우 크기의 배수로 맞춤)
          2. 윈도우로 분할: (B, H, W, C) -> (B*nH*nW, C, wH, wW)
          3. 각 윈도우에서 어텐션 수행
          4. 윈도우를 원래 shape로 복원
          5. 패딩 제거

        [입력/출력] (B, dim, H, W) -> (B, dim, H, W)
        """
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))

        if H <= self.window_resolution and W <= self.window_resolution:
            # 피처맵이 윈도우보다 작거나 같으면 전체에 어텐션 직접 적용
            x = self.attn(x)
        else:
            # BHWC 형식으로 변환 (패딩 작업이 더 편리)
            x = x.permute(0, 2, 3, 1)

            # 윈도우 크기의 배수가 되도록 필요한 패딩 크기 계산
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution  # 하단 패딩
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution  # 우측 패딩
            padding = pad_b > 0 or pad_r > 0

            if padding:
                # 순서: (left, right, top, bottom, front, back) -- BHWC에서 W, H 방향 패딩
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r          # 패딩 후 높이/너비
            nH = pH // self.window_resolution       # 세로 방향 윈도우 수
            nW = pW // self.window_resolution       # 가로 방향 윈도우 수

            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            # 윈도우 분할: 각 윈도우를 독립적인 배치 원소로 처리
            # view: (B, nH, wH, nW, wW, C)
            # transpose(2,3): (B, nH, nW, wH, wW, C)
            # reshape: (B*nH*nW, wH, wW, C)
            # permute: (B*nH*nW, C, wH, wW) -- 어텐션 입력 형식
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)

            # 각 윈도우에 대해 독립적으로 어텐션 수행
            x = self.attn(x)

            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            # 윈도우를 원래 피처맵 shape으로 복원 (분할의 역연산)
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                # 추가한 패딩 제거: 원래 해상도 (H, W)만 남김
                x = x[:, :H, :W].contiguous()

            # BCHW 형식으로 복원
            x = x.permute(0, 3, 1, 2)
        return x


# ==============================================================================
# EfficientViTBlock: EfficientViT 기본 빌딩 블록
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

    [역할]
    EfficientViT의 기본 반복 단위. 논문 Figure 2의 'EfficientViT Block' 참조.
    구조: DW Conv -> FFN -> (Token Mixer: LocalWindowAttention) -> DW Conv -> FFN
    모든 서브 모듈은 Residual 연결로 감싸져 있음.

    [입력/출력 shape]
    - 입력: (B, ed, H, W)
    - 출력: (B, ed, H, W)  -- shape 변화 없음

    [파라미터]
    - type              : 토큰 믹서 타입 ('s': self-attention)
    - ed                : 임베딩 차원 (입력/출력 채널)
    - kd                : Query/Key 차원
    - nh                : 어텐션 헤드 수
    - ar                : Value 차원 배율
    - resolution        : 입력 피처맵 해상도
    - window_resolution : 로컬 윈도우 크기
    - kernels           : 헤드별 Query DW Conv 커널 크기
    """
    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()

        # 전처리 DW Conv: 로컬 공간 특징 추출 (3x3 depthwise, groups=ed)
        # bn_weight_init=0: 초기에 잔차가 0 -> 학습 초기 안정성
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))

        # 전처리 FFN: 채널별 특징 변환 (ed -> 2*ed -> ed)
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            # 토큰 믹서: 로컬 윈도우 어텐션 (CGA 기반)
            # 공간적 상호작용을 담당하는 핵심 모듈
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                    resolution=resolution, window_resolution=window_resolution, kernels=kernels))

        # 후처리 DW Conv: 어텐션 후 로컬 특징 정제
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))

        # 후처리 FFN: 채널별 특징 변환 (어텐션 후)
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        """
        [순전파 흐름]
        x -> dw0(DW Conv+잔차) -> ffn0(FFN+잔차) -> mixer(LWAttn+잔차)
          -> dw1(DW Conv+잔차) -> ffn1(FFN+잔차)

        [입력/출력] (B, ed, H, W) -> (B, ed, H, W)
        """
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


# ==============================================================================
# EfficientViT: 전체 모델 (최상위 클래스)
# ==============================================================================
class EfficientViT(torch.nn.Module):
    """
    [역할]
    EfficientViT 전체 이미지 분류 모델.
    논문: "EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention" (CVPR 2023)

    [전체 아키텍처]
    1. PatchEmbedding: 4단계 stride=2 Conv -> (B, embed_dim[0], H/16, W/16)
    2. Stage 1 (blocks1): EfficientViTBlock x depth[0]
    3. Subsample (PatchMerging): 해상도 절반, 채널 embed_dim[0] -> embed_dim[1]
    4. Stage 2 (blocks2): EfficientViTBlock x depth[1]
    5. Subsample (PatchMerging): 해상도 절반, 채널 embed_dim[1] -> embed_dim[2]
    6. Stage 3 (blocks3): EfficientViTBlock x depth[2]
    7. GlobalAveragePool + 분류 헤드 -> (B, num_classes)

    [입력/출력 shape]
    - 입력: (B, in_chans, img_size, img_size) -- 보통 (B, 3, 224, 224)
    - 출력: (B, num_classes) -- 보통 (B, 1000)
           or distillation 시: ((B, num_classes), (B, num_classes)) [학습] / (B, num_classes) [추론]

    [파라미터]
    - img_size          : 입력 이미지 크기 (기본 224)
    - patch_size        : 패치 크기 (기본 16, PatchEmbedding의 총 stride)
    - in_chans          : 입력 채널 수 (기본 3, RGB)
    - num_classes       : 분류 클래스 수 (기본 1000, ImageNet)
    - stages            : 각 스테이지의 토큰 믹서 타입 리스트 (예: ['s', 's', 's'])
    - embed_dim         : 각 스테이지의 임베딩 차원 (예: [64, 128, 192])
    - key_dim           : 각 스테이지의 Q/K 차원 (예: [16, 16, 16])
    - depth             : 각 스테이지의 블록 반복 횟수 (예: [1, 2, 3])
    - num_heads         : 각 스테이지의 어텐션 헤드 수 (예: [4, 4, 4])
    - window_size       : 각 스테이지의 로컬 윈도우 크기 (예: [7, 7, 7])
    - kernels           : CGA의 헤드별 Query DW Conv 커널 크기 (예: [5, 5, 5, 5])
    - down_ops          : 스테이지 간 다운샘플링 연산 설정 (예: [['subsample', 2], ['subsample', 2], ['']])
    - distillation      : 지식 증류 사용 여부 (True시 보조 헤드 추가)
    """
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False,):
        super().__init__()

        resolution = img_size  # 현재 피처맵 해상도 추적 변수

        # -------------------------------------------------------
        # Patch Embedding: 4단계 stride=2 Conv (총 stride=16)
        # (B, 3, 224, 224) -> (B, embed_dim[0], 14, 14)
        # -------------------------------------------------------
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(
            # 1단계: (B, 3, 224, 224) -> (B, embed_dim[0]//8, 112, 112)
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), torch.nn.ReLU(),
            # 2단계: (B, embed_dim[0]//8, 112, 112) -> (B, embed_dim[0]//4, 56, 56)
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), torch.nn.ReLU(),
            # 3단계: (B, embed_dim[0]//4, 56, 56) -> (B, embed_dim[0]//2, 28, 28)
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), torch.nn.ReLU(),
            # 4단계: (B, embed_dim[0]//2, 28, 28) -> (B, embed_dim[0], 14, 14)
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8))

        # PatchEmbedding 후 해상도: img_size // patch_size (예: 224 // 16 = 14)
        resolution = img_size // patch_size

        # 각 스테이지의 Attention Ratio 계산: embed_dim / (key_dim * num_heads)
        # Value 차원이 Query 차원의 몇 배인지를 나타냄
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]

        # 3개 스테이지의 블록 리스트 초기화
        self.blocks1 = []  # Stage 1 블록
        self.blocks2 = []  # Stage 2 블록 (+ 다운샘플링 포함)
        self.blocks3 = []  # Stage 3 블록 (+ 다운샘플링 포함)

        # -------------------------------------------------------
        # Build EfficientViT blocks
        # 각 스테이지: EfficientViTBlock 반복 + 스테이지 간 다운샘플링
        # -------------------------------------------------------
        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            # 현재 스테이지에 depth 개의 EfficientViTBlock 추가
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))

            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                # 다음 스테이지로 넘어가기 위한 다운샘플링 블록 구성
                #('Subsample' stride)
                blk = eval('self.blocks' + str(i+2))  # 다음 스테이지 블록 리스트

                # 다운샘플링 후의 해상도: ceil((resolution) / stride)
                # (resolution - 1) // do[1] + 1 == ceil(resolution / do[1])
                resolution_ = (resolution - 1) // do[1] + 1

                # 다운샘플링 전 전처리: DW Conv + FFN (현재 스테이지 채널)
                blk.append(torch.nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),))

                # PatchMerging: 해상도 절반, 채널 증가 (embed_dim[i] -> embed_dim[i+1])
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))

                resolution = resolution_  # 해상도 업데이트

                # 다운샘플링 후 전처리: DW Conv + FFN (다음 스테이지 채널)
                blk.append(torch.nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),))

        # 리스트를 Sequential로 변환 (forward 시 순차 실행)
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        # -------------------------------------------------------
        # Classification head
        # -------------------------------------------------------
        # Classification head
        # num_classes > 0: BN_Linear (실제 분류), 0: Identity (피처 추출용)
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation

        if distillation:
            # 지식 증류용 보조 헤드: teacher 모델의 출력을 모방하도록 학습
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        [역할]
        Weight decay 스케줄러에서 제외할 파라미터 이름 집합 반환.
        attention_biases는 위치 편향 파라미터로 weight decay를 적용하지 않는다.
        (위치 정보는 정규화 대상이 아님)

        [반환값]
        - set of str: weight decay를 적용하지 않을 파라미터 이름들
        """
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        """
        [순전파 - 전체 모델]

        [입력]  x: (B, in_chans, img_size, img_size) -- 예: (B, 3, 224, 224)
        [출력]
        - distillation=False: (B, num_classes)
        - distillation=True, 학습 시: ((B, num_classes), (B, num_classes))
        - distillation=True, 추론 시: (B, num_classes) [두 헤드 평균]

        [흐름]
        x -> PatchEmbed(14x14) -> blocks1 -> blocks2(PatchMerge포함) -> blocks3(PatchMerge포함)
          -> GlobalAvgPool -> flatten -> Head -> 분류 결과
        """
        # PatchEmbedding: (B, 3, 224, 224) -> (B, embed_dim[0], 14, 14)
        x = self.patch_embed(x)

        # Stage 1: EfficientViTBlock x depth[0]
        x = self.blocks1(x)

        # Stage 2: 다운샘플링(PatchMerging) + EfficientViTBlock x depth[1]
        x = self.blocks2(x)

        # Stage 3: 다운샘플링(PatchMerging) + EfficientViTBlock x depth[2]
        x = self.blocks3(x)

        # Global Average Pooling: (B, embed_dim[-1], H, W) -> (B, embed_dim[-1], 1, 1) -> (B, embed_dim[-1])
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)

        if self.distillation:
            # 지식 증류: 메인 헤드와 증류 헤드 둘 다 계산
            x = self.head(x), self.head_dist(x)
            if not self.training:
                # 추론 시: 두 헤드의 평균으로 최종 예측 (앙상블 효과)
                x = (x[0] + x[1]) / 2
        else:
            # 일반 분류: (B, embed_dim[-1]) -> (B, num_classes)
            x = self.head(x)
        return x
