"""
3Augment implementation from (https://github.com/facebookresearch/deit/blob/main/augment.py)
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
Can be called by adding "--ThreeAugment" to the command line

ThreeAugment(3-Augment) 구현 모듈.
DeiT 논문에서 제안된 3단계 데이터 증강 기법으로, 아래 세 가지 변환 중 하나를
무작위로 적용한다:
  1. Grayscale (그레이스케일 변환)
  2. Solarization (솔라리제이션)
  3. Gaussian Blur (가우시안 블러)
이 방식은 DINO와 timm의 데이터 증강 전략에서 영감을 받았다.
"""
import torch
from torchvision import transforms

from timm.data.transforms import str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor

import numpy as np
from torchvision import datasets, transforms
import random



from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.

    PIL 이미지에 가우시안 블러(Gaussian Blur)를 확률적으로 적용하는 변환 클래스.

    가우시안 블러는 이미지의 고주파 노이즈를 제거하여 모델이 전체적인 구조와
    저주파 특징에 집중하도록 유도한다. 자기지도학습(DINO 등)에서 뷰 다양성을
    높이는 데 주로 활용된다.

    Args:
        p (float): 블러를 적용할 확률. 기본값 0.1 (10% 확률로 적용).
        radius_min (float): 가우시안 커널의 최소 반경. 기본값 0.1.
        radius_max (float): 가우시안 커널의 최대 반경. 기본값 2.0.
                            반경이 클수록 더 강하게 블러 처리된다.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        """
        PIL 이미지에 가우시안 블러를 적용한다.

        Args:
            img (PIL.Image): 입력 PIL 이미지.

        Returns:
            PIL.Image: 확률 p에 따라 블러 처리되거나 원본 이미지.
        """
        # self.prob 확률로 블러 적용 여부 결정
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        # [radius_min, radius_max] 범위에서 균등 분포로 반경을 무작위 선택하여 블러 적용
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.

    PIL 이미지에 솔라리제이션(Solarization) 효과를 확률적으로 적용하는 변환 클래스.

    솔라리제이션은 픽셀 값이 임계값(128)을 초과하는 픽셀을 반전시키는 변환으로,
    이미지에 독특한 고대비 효과를 만든다. 자기지도학습에서 뷰를 다양하게 만들어
    모델이 색상 강도 외 구조적 특징에 집중하도록 돕는다.
    (PIL ImageOps.solarize의 기본 임계값: 128)

    Args:
        p (float): 솔라리제이션을 적용할 확률. 기본값 0.2 (20% 확률로 적용).
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        """
        PIL 이미지에 솔라리제이션을 적용한다.

        Args:
            img (PIL.Image): 입력 PIL 이미지.

        Returns:
            PIL.Image: 확률 p에 따라 솔라리제이션이 적용되거나 원본 이미지.
        """
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Solarization to the PIL image.

    PIL 이미지에 그레이스케일(Grayscale) 변환을 확률적으로 적용하는 클래스.

    이미지를 흑백으로 변환하되, 채널 수는 3으로 유지한다 (RGB 형식 호환성 보장).
    그레이스케일 변환은 모델이 색상 정보 없이 형태(shape)와 질감(texture)만으로
    분류하도록 학습시키는 정규화 효과를 제공한다.

    주의: 클래스 docstring에 'Solarization'으로 잘못 기재되어 있으나 실제로는
    Grayscale 변환을 수행한다 (원본 코드의 오타).

    Args:
        p (float): 그레이스케일 변환을 적용할 확률. 기본값 0.2 (20% 확률로 적용).
    """
    def __init__(self, p=0.2):
        self.p = p
        # 출력 채널 수 3: RGB 텐서와의 호환성을 위해 3채널 그레이스케일로 변환
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        """
        PIL 이미지에 그레이스케일 변환을 적용한다.

        Args:
            img (PIL.Image): 입력 PIL 이미지.

        Returns:
            PIL.Image: 확률 p에 따라 그레이스케일로 변환되거나 원본 이미지.
                       그레이스케일 이미지도 3채널(RGB)로 반환된다.
        """
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img



class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.

    PIL 이미지에 수평 뒤집기(Horizontal Flip)를 확률적으로 적용하는 클래스.

    주의: 클래스 docstring에 'Solarization'으로 잘못 기재되어 있으나 실제로는
    수평 뒤집기 변환을 수행한다 (원본 코드의 오타).

    현재 new_data_aug_generator 파이프라인에서는 이 클래스 대신
    transforms.RandomHorizontalFlip()을 primary_tfl 단계에서 직접 사용한다.
    이 클래스는 별도의 보조 용도로 정의되어 있다.

    Args:
        p (float): 수평 뒤집기를 적용할 확률. 기본값 0.2 (20% 확률로 적용).
        activate_pred (bool): 예측 시 활성화 여부 플래그 (현재 미사용). 기본값 False.
    """
    def __init__(self, p=0.2, activate_pred=False):
        self.p = p
        # p=1.0으로 설정된 RandomHorizontalFlip을 래핑하여 외부 확률(self.p)로 제어
        self.transf = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, img):
        """
        PIL 이미지에 수평 뒤집기를 적용한다.

        Args:
            img (PIL.Image): 입력 PIL 이미지.

        Returns:
            PIL.Image: 확률 p에 따라 수평으로 뒤집히거나 원본 이미지.
        """
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img



def new_data_aug_generator(args=None):
    """
    ThreeAugment(3-Augment) 데이터 증강 파이프라인을 생성하여 반환하는 함수.

    세 단계(primary, secondary, final)로 구성된 증강 파이프라인을 구성한다:

    [Primary 변환 단계] - 공간적 변환 (항상 적용)
      - RandomResizedCropAndInterpolation: 이미지의 임의 영역(scale 0.08~1.0)을
        잘라내고 bicubic 보간으로 목표 크기(img_size)에 맞게 리사이즈.
        이는 모델이 다양한 스케일과 위치의 객체를 인식하도록 훈련시킨다.
      - RandomHorizontalFlip: 50% 확률로 이미지를 좌우 반전.
        이미지 분류에서 가장 기본적인 증강 기법.

    [Secondary 변환 단계] - 외형 변환 (세 가지 중 하나를 무작위 선택)
      - gray_scale(p=1.0): 이미지를 3채널 그레이스케일로 변환.
      - Solarization(p=1.0): 픽셀 값을 임계값 기준으로 반전하는 솔라리제이션.
      - GaussianBlur(p=1.0): 가우시안 블러로 이미지를 부드럽게 처리.
      RandomChoice로 세 변환 중 하나만 선택하여 적용 (각 1/3 확률).
      (선택적) ColorJitter: args.color_jitter가 0이 아닌 경우 추가로 적용.

    [Final 변환 단계] - 텐서 변환 및 정규화 (항상 적용)
      - ToTensor: PIL 이미지를 [0.0, 1.0] 범위의 float 텐서로 변환.
      - Normalize: ImageNet 평균(mean=[0.485, 0.456, 0.406])과
                   표준편차(std=[0.229, 0.224, 0.225])로 정규화.

    Args:
        args: 실험 설정 객체. 아래 속성을 사용:
            - args.input_size (int): 모델 입력 이미지 크기 (예: 224).
            - args.color_jitter (float or None): ColorJitter 강도.
              None이거나 0이면 ColorJitter를 적용하지 않는다.

    Returns:
        transforms.Compose: primary + secondary + final 변환을 순서대로
                            적용하는 합성 변환 파이프라인 객체.
    """
    img_size = args.input_size
    remove_random_resized_crop = False  # 기본적으로 RandomResizedCrop 사용
    # ImageNet 표준 정규화 파라미터
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    primary_tfl = []
    # RandomResizedCrop의 크롭 비율 범위: 원본 이미지 넓이의 8%~100%
    scale = (0.08, 1.0)
    # bicubic 보간: 크롭 후 리사이즈 시 품질이 높은 보간법 사용
    interpolation = 'bicubic'

    if remove_random_resized_crop:
        # [대안 경로] RandomResizedCrop 비활성화 시:
        # 고정 크기로 리사이즈 후 패딩(reflect 방식)을 추가하여 RandomCrop 적용
        # 이 경로는 현재 비활성화(remove_random_resized_crop=False)되어 있음
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        # [기본 경로] RandomResizedCrop + 좌우 반전
        primary_tfl = [
            # timm의 보간법 선택 기능을 지원하는 RandomResizedCrop 변형
            # scale=(0.08, 1.0): 원본 넓이의 8%~100% 범위에서 무작위 크롭
            # interpolation='bicubic': 고품질 bicubic 보간으로 리사이즈
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            # 50% 확률로 이미지 좌우 반전 (기본 RandomHorizontalFlip 동작)
            transforms.RandomHorizontalFlip()
        ]

    # [Secondary 단계] 세 가지 외형 변환 중 하나를 무작위로 선택하여 적용
    # RandomChoice: 리스트에서 하나의 변환을 균등 확률(1/3)로 선택
    # 각 변환 클래스의 p=1.0으로 설정하여 선택되면 반드시 적용
    secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
                                              Solarization(p=1.0),
                                              GaussianBlur(p=1.0)])]

    # ColorJitter가 설정된 경우 secondary 단계에 추가로 적용
    # brightness, contrast, saturation을 동일한 강도(color_jitter)로 조정
    if args.color_jitter is not None and not args.color_jitter == 0:
        secondary_tfl.append(transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter))

    # [Final 단계] 텐서 변환 및 ImageNet 통계치로 정규화
    final_tfl = [
            transforms.ToTensor(),  # PIL -> [0.0, 1.0] float 텐서
            transforms.Normalize(
                mean=torch.tensor(mean),   # ImageNet 채널별 평균
                std=torch.tensor(std))     # ImageNet 채널별 표준편차
        ]

    # 세 단계 변환을 순서대로 연결하여 최종 파이프라인 반환
    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
