'''
Build the EfficientViT model family
'''
# ==============================================================================
# EfficientViT 모델 패밀리 빌더
# ==============================================================================
# 이 파일의 역할:
#   - EfficientViT M0 ~ M5 총 6가지 모델 변형의 하이퍼파라미터를 정의
#   - timm 라이브러리에 모델을 등록하여 표준 인터페이스로 사용 가능하게 함
#   - 사전 학습된 가중치 로딩, BatchNorm Fuse 기능 제공
#
# 모델 규모 (작은 것 -> 큰 것):
#   M0: 최소 모델, embed_dim [64, 128, 192]
#   M1: embed_dim [128, 144, 192], 헤드 수 다양화
#   M2: embed_dim [128, 192, 224]
#   M3: embed_dim [128, 240, 320]
#   M4: embed_dim [128, 256, 384]
#   M5: 최대 모델, embed_dim [192, 288, 384], depth [1, 3, 4]
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficientvit import EfficientViT
from timm.models.registry import register_model


# ==============================================================================
# EfficientViT 모델 설정 딕셔너리 (M0 ~ M5)
# ==============================================================================
# 공통 설정:
#   - img_size: 224 (ImageNet 표준 입력 크기)
#   - patch_size: 16 (PatchEmbedding의 총 다운샘플링 stride)
#   - stages: 기본적으로 ['s', 's', 's'] (self-attention 타입)
#   - window_size: [7, 7, 7] (로컬 윈도우 크기)
#   - down_ops: [['subsample', 2], ['subsample', 2], ['']] (기본값)
#
# 각 설정의 구조:
#   embed_dim  : 3개 스테이지의 임베딩 채널 수
#   depth      : 3개 스테이지의 EfficientViTBlock 반복 횟수
#   num_heads  : 3개 스테이지의 어텐션 헤드 수
#   window_size: 3개 스테이지의 로컬 윈도우 크기
#   kernels    : CGA의 헤드별 Query DW Conv 커널 크기 (헤드 수만큼 필요)

# M0: 가장 가벼운 모델 (4개 헤드, 균일한 커널 크기)
# FLOPs: ~0.08G, Params: ~2.3M (참고용)
EfficientViT_m0 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [64, 128, 192],    # 채널: 64 -> 128 -> 192 (단계적 증가)
        'depth': [1, 2, 3],             # 블록 수: 1 -> 2 -> 3 (깊이 증가)
        'num_heads': [4, 4, 4],         # 모든 스테이지 4개 헤드
        'window_size': [7, 7, 7],       # 모든 스테이지 7x7 윈도우
        'kernels': [5, 5, 5, 5],        # 4개 헤드 모두 5x5 DW Conv
    }

# M1: 초기 스테이지 채널 증가, 헤드 수 변화
# 첫 스테이지 embed_dim을 128로 늘려 표현력 향상
EfficientViT_m1 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 144, 192],   # 첫 스테이지 채널 128 (M0의 2배)
        'depth': [1, 2, 3],
        'num_heads': [2, 3, 3],         # 헤드 수 다양화 (2, 3, 3)
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],        # 다양한 수용 영역 (7->5->3->3)
    }

# M2: 채널 수 증가, 다양한 헤드 수
EfficientViT_m2 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 192, 224],   # 마지막 스테이지 224 채널
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 2],         # 깊어질수록 헤드 수 감소 (특이한 패턴)
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

# M3: 더 넓은 채널 (240, 320)
EfficientViT_m3 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 240, 320],   # 중간/마지막 스테이지 채널 대폭 증가
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 4],         # 헤드 수: 4 -> 3 -> 4
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],        # 균일한 커널 크기
    }

# M4: 더 넓은 채널 (256, 384)
EfficientViT_m4 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 256, 384],   # 마지막 스테이지 384 채널 (M0의 2배)
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],         # 모든 스테이지 4개 헤드
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

# M5: 가장 큰 모델 (최대 표현력)
# depth를 [1, 3, 4]로 늘려 더 깊은 네트워크 구성
EfficientViT_m5 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [192, 288, 384],   # 첫 스테이지부터 192 채널 (가장 넓음)
        'depth': [1, 3, 4],             # 2, 3 스테이지 블록 수 증가 (3, 4개)
        'num_heads': [3, 3, 4],         # 헤드 수: 3 -> 3 -> 4
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }


# ==============================================================================
# 모델 빌더 함수들 (timm registry에 등록)
# ==============================================================================
# 공통 동작:
#   1. EfficientViT 모델 생성 (model_cfg + num_classes + distillation)
#   2. pretrained=True이면 사전 학습 가중치를 URL에서 로드
#   3. fuse=True이면 Conv+BN 레이어를 단일 Conv로 병합 (추론 속도 향상)
#
# [파라미터 공통 설명]
#   num_classes   : 분류 클래스 수 (기본 1000, ImageNet)
#   pretrained    : 사전 학습 가중치 이름 (False면 랜덤 초기화)
#   distillation  : 지식 증류 모드 (True면 보조 헤드 추가)
#   fuse          : BatchNorm Fuse 여부 (추론 전용 최적화)
#   pretrained_cfg: timm 호환을 위한 추가 설정 (사용 안 함)
#   model_cfg     : 모델 하이퍼파라미터 딕셔너리

@register_model  # timm 모델 레지스트리에 'EfficientViT_M0'라는 이름으로 등록
def EfficientViT_M0(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m0):
    """
    [역할]
    EfficientViT-M0 모델을 생성하고 반환하는 팩토리 함수.
    가장 가벼운 변형으로 에지 디바이스에 적합.

    [반환값]
    - EfficientViT 모델 인스턴스
    """
    # 모델 생성: model_cfg 딕셔너리를 키워드 인자로 언패킹
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)

    if pretrained:
        # 사전 학습 가중치 URL 포맷팅 및 다운로드
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')  # CPU에 먼저 로드 (메모리 효율)

        d = checkpoint['model']   # 체크포인트의 모델 가중치
        D = model.state_dict()    # 현재 모델의 state dict

        for k in d.keys():
            if D[k].shape != d[k].shape:
                # Shape 불일치 처리: 1D 가중치를 4D로 변환 (Linear -> Conv2d 호환)
                # 예: (C,) -> (C, 1, 1, 1) 로 브로드캐스팅 가능하게 변환
                d[k] = d[k][:, :, None, None]

        model.load_state_dict(d)  # 전처리된 가중치를 모델에 로드

    if fuse:
        # Conv+BN 레이어를 단일 Conv로 병합 (추론 속도 향상, 학습에는 사용 불가)
        replace_batchnorm(model)

    return model

@register_model  # timm 레지스트리에 등록
def EfficientViT_M1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m1):
    """
    [역할]
    EfficientViT-M1 모델 생성. M0보다 첫 스테이지 채널이 2배 넓음.

    [반환값]
    - EfficientViT 모델 인스턴스
    """
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                # Shape 불일치: 1D -> 4D 변환 (BN weight를 Conv weight로 적용 시 필요)
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model

@register_model
def EfficientViT_M2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m2):
    """
    [역할]
    EfficientViT-M2 모델 생성. 헤드 수가 스테이지마다 다름 (4->3->2).

    [반환값]
    - EfficientViT 모델 인스턴스
    """
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model

@register_model
def EfficientViT_M3(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m3):
    """
    [역할]
    EfficientViT-M3 모델 생성. 중간~큰 규모의 모델.

    [반환값]
    - EfficientViT 모델 인스턴스
    """
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model

@register_model
def EfficientViT_M4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m4):
    """
    [역할]
    EfficientViT-M4 모델 생성. 마지막 스테이지 384 채널로 큰 모델.

    [반환값]
    - EfficientViT 모델 인스턴스
    """
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model

@register_model
def EfficientViT_M5(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m5):
    """
    [역할]
    EfficientViT-M5 모델 생성. 가장 크고 강력한 변형.
    depth=[1,3,4]로 더 깊은 네트워크, 첫 스테이지부터 192 채널.

    [반환값]
    - EfficientViT 모델 인스턴스
    """
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model


# ==============================================================================
# replace_batchnorm: BatchNorm을 단일 연산으로 병합하는 유틸리티 함수
# ==============================================================================
def replace_batchnorm(net):
    """
    [역할]
    모델 내의 모든 Conv2d_BN 및 BN_Linear를 fuse()로 병합하고,
    독립적으로 존재하는 BatchNorm2d를 Identity로 교체한다.
    추론 속도 최적화를 위해 사용 (학습 후 추론 전에만 적용).

    [동작 원리]
    재귀적으로 모든 서브 모듈을 순회하며:
      - fuse() 메서드가 있는 모듈 (Conv2d_BN, BN_Linear):
        -> fuse()를 호출하여 단일 레이어로 병합
      - 독립 BatchNorm2d:
        -> Identity로 교체 (BN 연산 제거)
      - 그 외 모듈:
        -> 재귀적으로 자식 모듈들에 대해 동일 작업 수행

    [주의]
    - 학습 시에는 절대 사용하지 말 것 (BN의 통계 업데이트 불가)
    - fuse 후에는 원래대로 되돌릴 수 없음

    [파라미터]
    - net: fuse를 적용할 모델 또는 서브 모듈

    [예시]
    >>> model = EfficientViT_M0()
    >>> replace_batchnorm(model)  # 추론 전 최적화
    """
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            # Conv2d_BN 또는 BN_Linear: Conv+BN을 단일 Conv/Linear로 병합
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            # 독립 BatchNorm2d: 입력을 그대로 통과시키는 Identity로 교체
            # (이미 fuse된 경우 남아있는 독립 BN 처리)
            setattr(net, child_name, torch.nn.Identity())
        else:
            # 복합 모듈: 재귀적으로 자식에 대해 동일 처리
            replace_batchnorm(child)


# ==============================================================================
# 사전 학습 가중치 URL 형식
# ==============================================================================
# GitHub Release에서 모델 가중치를 다운로드하는 URL 템플릿
# 사용 예: _checkpoint_url_format.format('EfficientViT_M0')
# -> 'https://github.com/.../EfficientViT_M0.pth'
_checkpoint_url_format = \
    'https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/{}.pth'
