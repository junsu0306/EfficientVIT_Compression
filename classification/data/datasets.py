'''
Build trainining/testing datasets
훈련/테스트 데이터셋을 구성하는 모듈.
지원 데이터셋: CIFAR-100, ImageNet (IMNET), ImageNet-EE (IMNETEE), Oxford Flowers (FLOWERS),
iNaturalist 2018 (INAT), iNaturalist 2019 (INAT19)
'''
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    # timm의 상위 버전에서는 TimmDatasetTar 대신 ImageDataset을 사용
    from timm.data import ImageDataset as TimmDatasetTar


class INatDataset(ImageFolder):
    """
    iNaturalist 데이터셋 클래스.

    torchvision의 ImageFolder를 상속하여 iNaturalist 형식의 JSON 어노테이션 파일을
    파싱하고, 지정한 생물 분류 카테고리(kingdom, phylum, class, order 등)를
    기준으로 레이블을 매핑한다.

    Args:
        root (str): 데이터셋 루트 디렉토리 경로.
                    해당 디렉토리에 train{year}.json, val{year}.json,
                    categories.json 파일이 존재해야 한다.
        train (bool): True이면 훈련 셋, False이면 검증 셋을 로드. 기본값 True.
        year (int): 사용할 iNaturalist 연도 (2018 또는 2019). 기본값 2018.
        transform: 이미지에 적용할 전처리/증강 변환. 기본값 None.
        target_transform: 레이블에 적용할 변환. 기본값 None.
        category (str): 분류 기준 생물 계층 ('kingdom', 'phylum', 'class',
                        'order', 'supercategory', 'family', 'genus', 'name').
                        기본값 'name'.
        loader: 이미지 파일을 로드하는 함수. 기본값 default_loader.

    Attributes:
        nb_classes (int): 사용된 카테고리 수 (고유 클래스 수).
        samples (list): (이미지 경로, 레이블 인덱스) 튜플의 리스트.
    """
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']

        # 훈련 또는 검증용 JSON 어노테이션 파일 경로 결정
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        # 전체 생물 카테고리 메타데이터 로드 (각 category_id에 대한 상세 분류 정보)
        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        # 레이블 인덱싱을 위해 항상 train JSON을 기준으로 사용
        # (train/val 일관된 클래스 인덱스 보장을 위해 훈련 데이터를 기준으로 targeter 구성)
        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        # targeter: 카테고리 이름 -> 정수 인덱스 매핑 딕셔너리 구성
        # 훈련 어노테이션에서 등장 순서대로 고유 클래스에 순차적 인덱스를 부여
        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        # 고유 클래스 수 저장 (외부에서 nb_classes로 참조 가능)
        self.nb_classes = len(targeter)

        # 실제 사용할 (이미지 경로, 레이블) 샘플 리스트 구성
        self.samples = []
        for elem in data['images']:
            # 파일명 파싱: "train/{category_id}/{filename}" 형식을 분리
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])  # 카테고리 ID (디렉토리명)
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            # categories.json에서 해당 카테고리의 분류 정보 조회
            categors = data_catg[target_current]
            # 선택한 계층(category)의 이름을 정수 인덱스로 변환
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder
    # __getitem__과 __len__은 ImageFolder로부터 상속받아 사용


def build_dataset(is_train, args):
    """
    데이터셋을 구성하고 반환하는 팩토리 함수.

    args.data_set에 지정된 데이터셋 종류에 따라 적절한 Dataset 객체와
    클래스 수를 반환한다. 훈련/검증 여부(is_train)에 따라 다른 변환(transform)이
    적용된다.

    Args:
        is_train (bool): True이면 훈련 데이터셋, False이면 검증 데이터셋을 반환.
        args: 실험 설정 객체. 아래 속성을 사용:
            - args.data_set (str): 데이터셋 종류
              ('CIFAR', 'IMNET', 'IMNETEE', 'FLOWERS', 'INAT', 'INAT19')
            - args.data_path (str): 데이터셋 루트 경로
            - args.inat_category (str): iNaturalist 분류 계층 (INAT/INAT19 전용)
            - 기타 build_transform에서 사용하는 속성들

    Returns:
        tuple: (dataset, nb_classes)
            - dataset: torch.utils.data.Dataset 인스턴스
            - nb_classes (int): 데이터셋의 클래스 수
    """
    # 훈련/검증 여부에 따라 적절한 이미지 변환 파이프라인 구성
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        # CIFAR-100: 100개 클래스, 32x32 이미지
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        # ImageNet: 1000개 클래스, 표준 ImageNet 데이터셋
        # .tar 압축 파일이 존재하면 TimmDatasetTar로 직접 로드 (빠른 I/O),
        # 없으면 디렉토리 구조(ImageFolder)로 로드
        prefix = 'train' if is_train else 'val'
        data_dir = os.path.join(args.data_path, f'{prefix}.tar')
        if os.path.exists(data_dir):
            dataset = TimmDatasetTar(data_dir, transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNETEE':
        # ImageNet-EE (Easy Evaluation): ImageNet의 간소화 버전, 10개 클래스
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 10
    elif args.data_set == 'FLOWERS':
        # Oxford 102 Flowers: 102개 꽃 카테고리 데이터셋
        # 훈련 시 데이터가 부족하므로 동일 데이터셋을 100번 반복 이어붙여
        # 효과적으로 에폭당 샘플 수를 100배 늘림 (소규모 데이터셋 학습 안정화)
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = torch.utils.data.ConcatDataset(
                [dataset for _ in range(100)])
        nb_classes = 102
    elif args.data_set == 'INAT':
        # iNaturalist 2018: 생물 분류 대규모 데이터셋 (2018년 버전)
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        # iNaturalist 2019: 생물 분류 대규모 데이터셋 (2019년 버전)
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    return dataset, nb_classes


def build_transform(is_train, args):
    """
    훈련/검증 단계에 맞는 이미지 전처리 변환 파이프라인을 구성하여 반환한다.

    훈련 시에는 timm의 create_transform을 활용해 데이터 증강(AutoAugment,
    RandomResizedCrop, ColorJitter, Random Erasing 등)을 포함한 파이프라인을 구성한다.
    검증 시에는 Resize -> CenterCrop -> ToTensor -> Normalize 순서의
    결정론적(deterministic) 파이프라인을 구성한다.

    Args:
        is_train (bool): True이면 훈련용 변환, False이면 검증용 변환을 반환.
        args: 실험 설정 객체. 아래 속성을 사용:
            - args.input_size (int): 모델 입력 이미지 크기 (예: 224).
              32 이하이면 CIFAR 규모로 간주하여 RandomCrop 사용.
            - args.color_jitter (float): 색상 지터 강도.
            - args.aa (str): AutoAugment 정책 문자열 (예: 'rand-m9-mstd0.5').
            - args.train_interpolation (str): 훈련 시 보간법 (예: 'bicubic').
            - args.reprob (float): Random Erasing 적용 확률.
            - args.remode (str): Random Erasing 모드.
            - args.recount (int): Random Erasing 반복 횟수.
            - args.finetune (str or bool): 파인튜닝 체크포인트 경로.
              지정 시 검증에서도 고정 크기 리사이즈 사용.

    Returns:
        transforms.Compose: 이미지 변환 파이프라인 객체.
    """
    # input_size가 32보다 크면 일반 이미지 데이터셋(ImageNet 규모)으로 간주
    # 32 이하면 CIFAR처럼 작은 이미지이므로 RandomResizedCrop 대신 RandomCrop 사용
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # 훈련 변환: timm의 create_transform으로 표준 ImageNet 훈련 증강 파이프라인 구성
        # - RandomResizedCrop + 보간법
        # - AutoAugment (args.aa 정책)
        # - ColorJitter
        # - Random Erasing (re_prob, re_mode, re_count)
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            # 소규모 이미지(CIFAR 등)는 RandomResizedCrop 대신
            # padding=4를 사용한 RandomCrop으로 교체
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    # 검증 변환 파이프라인 구성 (결정론적, 증강 없음)
    t = []
    if args.finetune:
        # 파인튜닝 시: 입력 크기에 맞게 직접 리사이즈 (aspect ratio 무시)
        t.append(
            transforms.Resize((args.input_size, args.input_size),
                                interpolation=3)
        )
    else:
        if resize_im:
            # 표준 ImageNet 검증 방식:
            # 1. 비율을 유지하며 입력 크기에 비례하는 크기로 리사이즈
            #    (224x224 기준으로 256/224 비율 적용 -> 약 14.3% 여유 확보)
            # 2. 중앙 부분을 정확한 입력 크기로 자름 (CenterCrop)
            size = int((256 / 224) * args.input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                # 224 이미지 기준 비율을 유지하기 위해 스케일 조정
                transforms.Resize(size, interpolation=3),
            )
            t.append(transforms.CenterCrop(args.input_size))

    # 픽셀값을 [0, 255] -> [0.0, 1.0] 텐서로 변환
    t.append(transforms.ToTensor())
    # ImageNet 평균(mean)과 표준편차(std)로 정규화
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
