# --------------------------------------------------------
# Efficient Main (train/validate)
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# --------------------------------------------------------
# EfficientViT 모델의 학습(train) 및 평가(evaluate)를 수행하는 메인 스크립트.
# ImageNet 등의 분류 데이터셋에 대해 분산학습, 지식 증류(distillation),
# 데이터 증강(augmentation) 등을 지원한다.

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data import Mixup                                    # Mixup/CutMix 데이터 증강 유틸리티
from timm.models import create_model
from .model.build import EfficientViT_M4                           # 모델 이름으로 모델 인스턴스 생성
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy  # 레이블 스무딩 / 소프트 타겟 손실함수
from timm.scheduler import create_scheduler                    # 학습률 스케줄러 생성
from timm.optim import create_optimizer                        # 옵티마이저 생성
from timm.utils import NativeScaler, get_state_dict, ModelEma  # AMP 스케일러, EMA 관련 유틸

# 상대 경로 대신 절대 경로로 수정 (서버 환경 호환성)
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # 맨 앞에 추가

from .data.samplers import RASampler             # Repeated Augmentation을 위한 커스텀 샘플러
from .data.datasets import build_dataset         # 데이터셋 빌드 함수
from .data.threeaugment import new_data_aug_generator  # ThreeAugment 데이터 증강 생성기
from .engine import train_one_epoch, evaluate    # 에폭 단위 학습 및 평가 함수
from .losses import DistillationLoss             # 지식 증류 손실함수 래퍼

from .model import build  # 모델 빌드 관련 모듈 (등록 목적)
from . import utils             # 분산학습 초기화, 체크포인트 저장 등 유틸리티

# PGM Pruning imports (Phase B)
from .pruning.group_dict import build_pruning_groups
from .pruning.memory_utils import compute_active_param_memory
from .pruning.pgm_loss import pgm_regularization_loss, memory_penalty


def get_args_parser():
    """
    학습 및 평가에 필요한 모든 커맨드라인 인자를 정의하고 파서를 반환하는 함수.

    반환값:
        argparse.ArgumentParser: 설정된 인자 파서 객체
    """
    parser = argparse.ArgumentParser(
        'EfficientViT training and evaluation script', add_help=False)

    # ----------------------------------------------------------------
    # 기본 학습 설정
    # ----------------------------------------------------------------
    # --batch-size: GPU 한 장당 처리할 샘플 수. 분산학습 시 전체 배치 크기는 batch_size * world_size가 됨
    parser.add_argument('--batch-size', default=256, type=int)
    # --epochs: 전체 데이터셋을 몇 번 반복해서 학습할지 지정. 기본 300 에폭
    parser.add_argument('--epochs', default=300, type=int)

    # ----------------------------------------------------------------
    # Model parameters
    # 모델 구조 관련 파라미터
    # ----------------------------------------------------------------
    # --model: 학습할 모델 이름. timm 레지스트리에 등록된 모델명 사용
    parser.add_argument('--model', default='EfficientViT_M4', type=str, metavar='MODEL',
                        help='Name of model to train')
    # --input-size: 입력 이미지 해상도 (정사각형 기준). 기본값 224x224
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')

    # --model-ema / --no-model-ema: Exponential Moving Average(EMA) 모델 사용 여부.
    #   EMA는 학습 중 모델 가중치의 지수이동평균을 별도로 유지하여 더 안정적인 평가 모델을 만듦.
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument(
        '--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    # --model-ema-decay: EMA 감쇠율. 1에 가까울수록 과거 가중치를 더 오래 유지함 (기본 0.99996)
    parser.add_argument('--model-ema-decay', type=float,
                        default=0.99996, help='')
    # --model-ema-force-cpu: EMA 모델을 GPU 대신 CPU에서 관리할지 여부.
    #   GPU 메모리가 부족할 때 활성화하면 메모리를 절약할 수 있음
    parser.add_argument('--model-ema-force-cpu',
                        action='store_true', default=False, help='')

    # ----------------------------------------------------------------
    # Optimizer parameters
    # 옵티마이저 관련 파라미터
    # ----------------------------------------------------------------
    # --opt: 사용할 옵티마이저 종류. 기본값 'adamw' (AdamW는 가중치 감쇠를 올바르게 적용하는 Adam 변형)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    # --opt-eps: AdamW의 수치 안정성을 위한 epsilon 값. 0으로 나누기를 방지
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    # --opt-betas: Adam계열 옵티마이저의 베타 계수 (1차/2차 모멘트 감쇠율). None이면 옵티마이저 기본값 사용
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    # --clip-grad: 그래디언트 클리핑 임계값. 기울기 폭발을 방지하기 위해 그래디언트 노름을 이 값 이하로 제한
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # --clip-mode: 그래디언트 클리핑 방식.
    #   'norm': L2 노름 기반, 'value': 절댓값 기반, 'agc': Adaptive Gradient Clipping
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    # --momentum: SGD 사용 시 모멘텀 계수. 이전 업데이트 방향의 관성을 얼마나 유지할지 결정
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    # --weight-decay: L2 정규화 강도. 가중치가 지나치게 커지는 것을 방지하여 과적합을 억제
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay (default: 0.025)')

    # ----------------------------------------------------------------
    # Learning rate schedule parameters
    # 학습률 스케줄 관련 파라미터
    # ----------------------------------------------------------------
    # --sched: 학습률 스케줄러 종류. 'cosine'은 학습이 진행될수록 학습률을 코사인 곡선으로 감소시킴
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    # --lr: 기본 학습률. 실제로는 선형 스케일링 규칙에 따라 배치 크기에 비례하여 조정됨
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    # --lr-noise: 학습률에 랜덤 노이즈를 추가할 에폭 구간(비율). 학습 후반부에 안장점 탈출을 돕기 위해 사용
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    # --lr-noise-pct: 학습률 노이즈의 최대 크기 비율 (0.67 = 67%)
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    # --lr-noise-std: 학습률 노이즈의 표준편차. 클수록 더 큰 노이즈가 추가됨
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    # --warmup-lr: 워밍업 구간의 시작 학습률. 매우 작은 값에서 시작하여 --lr까지 선형 증가
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    # --min-lr: 코사인 스케줄러 등 주기적 스케줄러가 0에 수렴하지 않도록 하는 학습률 하한값
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # --decay-epochs: 스텝 방식 스케줄러에서 학습률을 감소시킬 에폭 간격
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    # --warmup-epochs: 학습 시작 시 학습률을 점진적으로 높이는 워밍업 구간의 에폭 수.
    #   초기 학습 불안정성을 방지하기 위해 사용
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    # --cooldown-epochs: 주기적 스케줄이 끝난 후 최소 학습률로 유지하는 쿨다운 구간 에폭 수
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # --patience-epochs: Plateau 스케줄러에서 성능 향상이 없을 때 학습률을 낮추기 전 기다리는 에폭 수
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    # --decay-rate: 스텝 방식 스케줄러에서 각 감소 단계마다 학습률에 곱하는 비율 (기본 0.1 = 10배 감소)
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # ----------------------------------------------------------------
    # Augmentation parameters
    # 데이터 증강 관련 파라미터
    # ----------------------------------------------------------------
    # --ThreeAugment: ThreeAugment 전략 활성화 여부.
    #   세 가지 증강(Grayscale, Solarization, Gaussian Blur) 중 하나를 랜덤 적용
    parser.add_argument('--ThreeAugment', action='store_true')
    # --color-jitter: 색상(밝기, 대비, 채도, 색조) 무작위 변환 강도. 0이면 비활성화
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    # --aa: AutoAugment 정책 이름.
    #   'rand-m9-mstd0.5-inc1'은 RandAugment를 의미하며, magnitude=9, std=0.5, increasing=True를 뜻함
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    # --smoothing: 레이블 스무딩 강도. 0.1이면 정답 클래스 확률을 0.9, 나머지를 0.1/N으로 분산시킴.
    #   과적합 방지 및 모델 불확실성 표현에 도움
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    # --train-interpolation: 학습 시 이미지 리사이즈에 사용할 보간 방법.
    #   'bicubic'은 쌍삼차 보간으로 품질이 높지만 bilinear보다 느림
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # --repeated-aug / --no-repeated-aug: Repeated Augmentation 사용 여부.
    #   같은 이미지를 여러 번 다르게 증강하여 한 배치에 넣는 방식으로, 분산학습 효율을 높임
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug',
                        action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # ----------------------------------------------------------------
    # Random Erase params
    # Random Erasing 증강 파라미터: 이미지의 일부 영역을 무작위로 지워서 가림
    # ----------------------------------------------------------------
    # --reprob: Random Erasing이 적용될 확률 (기본 25%)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    # --remode: 지워진 영역을 채울 방법. 'pixel'은 랜덤 픽셀값으로, 'const'는 상수(0)로 채움
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    # --recount: 한 이미지에서 Random Erasing을 적용할 횟수
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    # --resplit: True이면 첫 번째(클린) 증강 분할에는 Random Erasing을 적용하지 않음
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # ----------------------------------------------------------------
    # Mixup params
    # Mixup / CutMix 증강 파라미터:
    #   두 이미지를 혼합하거나 잘라붙여 새로운 샘플을 생성하는 정규화 기법
    # ----------------------------------------------------------------
    # --mixup: Mixup의 알파 파라미터 (Beta 분포의 모수). 0이면 Mixup 비활성화
    #   알파가 클수록 두 이미지가 더 균등하게 혼합됨
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    # --cutmix: CutMix의 알파 파라미터. 0이면 CutMix 비활성화
    #   CutMix는 이미지의 직사각형 영역을 다른 이미지로 대체하는 방식
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    # --cutmix-minmax: CutMix에서 잘라낼 영역의 크기 비율 범위. 설정 시 알파보다 우선 적용됨
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    # --mixup-prob: 각 배치/샘플에 Mixup 또는 CutMix를 적용할 확률
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    # --mixup-switch-prob: Mixup과 CutMix가 모두 활성화된 경우, CutMix를 선택할 확률
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    # --mixup-mode: Mixup/CutMix를 적용하는 단위.
    #   'batch': 배치 전체에 동일 혼합비 적용, 'pair': 쌍별 적용, 'elem': 샘플별 적용
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # ----------------------------------------------------------------
    # Distillation parameters
    # 지식 증류(Knowledge Distillation) 관련 파라미터:
    #   큰 교사(teacher) 모델의 출력을 이용하여 작은 학생(student) 모델을 더 잘 학습시키는 기법
    # ----------------------------------------------------------------
    # --teacher-model: 교사 모델 이름. 학생 모델보다 크고 정확한 사전학습 모델 사용
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    # --teacher-path: 교사 모델의 사전학습 가중치 경로 또는 URL
    parser.add_argument('--teacher-path', type=str,
                        default='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth')
    # --distillation-type: 증류 방식 선택.
    #   'none': 증류 없음, 'soft': 소프트 타겟(교사의 확률 분포 활용), 'hard': 하드 타겟(교사의 예측 레이블 활용)
    parser.add_argument('--distillation-type', default='none',
                        choices=['none', 'soft', 'hard'], type=str, help="")
    # --distillation-alpha: 증류 손실과 분류 손실의 가중 합산 비율.
    #   alpha=0.5이면 두 손실을 동등하게 반영
    parser.add_argument('--distillation-alpha',
                        default=0.5, type=float, help="")
    # --distillation-tau: 소프트 타겟 증류 시 사용하는 온도(temperature) 파라미터.
    #   tau가 클수록 소프트 타겟 분포가 더 부드러워지고 클래스 간 관계 정보가 잘 전달됨
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # ----------------------------------------------------------------
    # Finetuning params
    # 파인튜닝 관련 파라미터
    # ----------------------------------------------------------------
    # --finetune: 파인튜닝 시작 체크포인트 경로. 빈 문자열이면 처음부터 학습
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    # --set_bn_eval: 파인튜닝 중 Batch Normalization 레이어를 eval 모드로 고정할지 여부.
    #   소규모 데이터셋 파인튜닝 시 BN 통계가 변하지 않도록 고정하면 안정적인 학습이 가능
    parser.add_argument('--set_bn_eval', action='store_true', default=False,
                        help='set BN layers to eval mode during finetuning.')

    # ----------------------------------------------------------------
    # Dataset parameters
    # 데이터셋 및 실행 환경 관련 파라미터
    # ----------------------------------------------------------------
    # --data-path: 데이터셋 루트 디렉토리 경로
    parser.add_argument('--data-path', default='/workspace/etri_iitp/JS/EfficientViT/data', type=str,
                        help='dataset path')
    # --data-set: 사용할 데이터셋 종류.
    #   CIFAR: CIFAR-10/100, IMNET: ImageNet-1K, INAT/INAT19: iNaturalist
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    # --inat-category: iNaturalist 데이터셋에서 분류 기준이 되는 분류학적 계층 수준
    #   (kingdom~name으로 갈수록 더 세밀한 분류)
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order',
                                 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    # --output_dir: 체크포인트, 로그 등 출력 파일을 저장할 디렉토리. 빈 문자열이면 저장하지 않음
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    # --device: 학습에 사용할 디바이스. 'cuda' 또는 'cpu'
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # --seed: 난수 시드. 재현성(reproducibility)을 위해 설정. 분산학습 시 rank가 더해져 GPU별로 다른 시드 사용
    parser.add_argument('--seed', default=0, type=int)
    # --resume: 중단된 학습을 재개할 체크포인트 경로. 빈 문자열이면 처음부터 학습
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # --start_epoch: 학습 재개 시 시작할 에폭 번호. --resume과 함께 사용 시 체크포인트에서 자동 설정됨
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # --eval: 이 플래그가 있으면 학습 없이 평가만 수행하고 종료
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    # --dist-eval: 분산 환경에서 검증도 병렬로 수행할지 여부.
    #   활성화 시 검증 데이터를 여러 GPU에 나누어 처리하므로 속도가 빨라짐
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    # --num_workers: DataLoader에서 데이터를 불러오는 병렬 워커(프로세스) 수.
    #   너무 크면 CPU 오버헤드가 생기고, 너무 작으면 데이터 로딩이 병목이 됨
    parser.add_argument('--num_workers', default=10, type=int)
    # --pin-mem / --no-pin-mem: DataLoader에서 CPU 메모리를 페이지 고정(pin)할지 여부.
    #   pin_memory=True이면 CPU->GPU 데이터 전송이 빨라지지만 더 많은 CPU 메모리를 사용함
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # ----------------------------------------------------------------
    # training parameters
    # 분산학습 및 저장 관련 파라미터
    # ----------------------------------------------------------------
    # --world_size: 분산학습에 참여하는 총 프로세스(GPU) 수
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # --dist_url: 분산학습 초기화에 사용할 URL.
    #   'env://'는 환경변수(MASTER_ADDR, MASTER_PORT 등)에서 설정을 읽어오는 방식
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # --save_freq: 몇 에폭마다 체크포인트를 저장할지 지정. 1이면 매 에폭 저장
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequency of model saving')

    # ----------------------------------------------------------------
    # PGM Pruning parameters (Phase B)
    # ----------------------------------------------------------------
    parser.add_argument('--pruning', action='store_true',
                        help='Enable PGM pruning during training')
    parser.add_argument('--lambda-ffn', type=float, default=0.00002,
                        help='L2 regularization strength for FFN groups (ratio 20:4:1), scaled down 1000x')
    parser.add_argument('--lambda-qk', type=float, default=0.000004,
                        help='L2 regularization strength for QK groups (ratio 20:4:1), scaled down 1000x')
    parser.add_argument('--lambda-v', type=float, default=0.000001,
                        help='L2 regularization strength for V groups (ratio 20:4:1), scaled down 1000x')
    parser.add_argument('--mu', type=float, default=1.0,
                        help='Memory penalty coefficient')
    parser.add_argument('--m-max-mb', type=float, default=7.044,
                        help='Maximum memory target in MB (80% optimization: 35.22 * 0.20 = 7.044 MB)')
    parser.add_argument('--pruning-freq', type=int, default=100,
                        help='Pruning frequency: apply pruning every N iterations')
    parser.add_argument('--target-compression', type=float, default=0.80,
                        help='Target optimization rate (fraction to REMOVE), e.g., 0.80 = remove 80%, keep 20%')

    return parser


def main(args):
    """
    EfficientViT의 학습 및 평가를 실행하는 메인 함수.

    분산학습 초기화, 데이터셋/데이터로더 구성, 모델 생성, 옵티마이저/스케줄러/손실함수 설정,
    체크포인트 로드, 학습 루프 실행, 결과 저장 등 전체 파이프라인을 담당한다.

    파라미터:
        args: get_args_parser()로 파싱된 인자 네임스페이스 객체
    """
    # 분산학습 환경을 초기화: args.distributed, args.gpu, args.rank 등의 속성이 설정됨
    utils.init_distributed_mode(args)

    # 증류와 파인튜닝을 동시에 사용하는 경우는 아직 미구현 상태임을 체크
    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError(
            "Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # 재현성 확보를 위한 난수 시드 고정.
    # 분산학습 시 각 프로세스(rank)마다 다른 시드를 사용하여 데이터 샘플링이 겹치지 않도록 함
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # cuDNN의 자동 최적화 알고리즘 탐색 활성화.
    # 입력 크기가 고정되어 있을 때 가장 빠른 연산 커널을 자동으로 선택하여 학습 속도를 높임
    cudnn.benchmark = True

    # ----------------------------------------------------------------
    # 데이터셋 빌드
    # build_dataset은 args의 data-path, data-set 등을 참고하여
    # 학습/검증 데이터셋과 클래스 수를 반환함
    # ----------------------------------------------------------------
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # ----------------------------------------------------------------
    # 분산학습 샘플러 설정
    # 분산학습에서는 각 GPU가 데이터의 서로 다른 부분을 처리해야 하므로
    # 분산 샘플러(DistributedSampler)를 사용하여 데이터를 분할함
    # ----------------------------------------------------------------
    if True:  # args.distributed:
        num_tasks = utils.get_world_size()   # 전체 GPU(프로세스) 수
        global_rank = utils.get_rank()       # 현재 프로세스의 순서(0-based index)

        if args.repeated_aug:
            # Repeated Augmentation 샘플러:
            # 같은 이미지를 여러 번 다르게 증강하여 mini-batch에 포함시킴.
            # 분산학습에서 effective batch size를 늘리는 효과가 있어 학습 안정성에 기여함
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            # 표준 분산 샘플러: 데이터셋을 GPU 수만큼 균등 분할하여 각 GPU에 할당
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        if args.dist_eval:
            # 분산 검증 모드: 검증 데이터도 여러 GPU에 나누어 처리
            # 데이터 수가 GPU 수로 나누어 떨어지지 않으면 중복 항목이 추가됨을 경고
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            # 단일 프로세스 검증: 순서대로 데이터를 순회하는 샘플러 사용
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        # 비분산 환경 (현재 코드에서는 사용되지 않음)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # ----------------------------------------------------------------
    # 학습용 DataLoader 생성
    # drop_last=True: 마지막 미니배치가 배치 크기보다 작으면 버림 (분산학습 시 크기 불일치 방지)
    # ----------------------------------------------------------------
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # ThreeAugment 적용 시 데이터로더의 transform을 ThreeAugment 전용 증강 파이프라인으로 교체.
    # ThreeAugment는 Grayscale, Solarization, Gaussian Blur 중 하나를 랜덤으로 적용하는 기법
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    # ----------------------------------------------------------------
    # 검증용 DataLoader 생성
    # 배치 크기를 1.5배로 늘려 검증 속도를 높임 (검증 시에는 역전파가 없으므로 메모리 여유가 있음)
    # drop_last=False: 검증 시에는 모든 데이터를 빠짐없이 평가해야 함
    # ----------------------------------------------------------------
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # ----------------------------------------------------------------
    # Mixup / CutMix 증강 함수 설정
    # mixup_active: mixup 또는 cutmix 중 하나라도 활성화되어 있으면 True
    # Mixup과 CutMix는 두 샘플의 이미지와 레이블을 동시에 혼합하여
    # 모델의 과적합을 방지하고 결정 경계를 부드럽게 만드는 강력한 정규화 기법
    # ----------------------------------------------------------------
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # ----------------------------------------------------------------
    # 모델 생성
    # create_model은 timm 레지스트리에서 모델 이름을 찾아 인스턴스를 생성함
    # distillation=True이면 분류 헤드 외에 증류 전용 헤드도 함께 생성됨
    # ----------------------------------------------------------------
    print(f"Creating model: {args.model}")
    # timm create_model 대신 직접 모델 생성 (호환성 문제 해결)
    model = EfficientViT_M4(
        num_classes=args.nb_classes,
        distillation=(args.distillation_type != 'none'),
        pretrained=False,
        fuse=False,
    )

    # ----------------------------------------------------------------
    # 파인튜닝: 사전학습 가중치 로드
    # 헤드 레이어의 형태가 다른 경우(클래스 수 불일치 등) 해당 키를 제거하고 로드함
    # ----------------------------------------------------------------
    if args.finetune:
        if args.finetune.startswith('https'):
            # URL에서 직접 가중치를 다운로드하여 로드
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            # 로컬 파일에서 가중치 로드 (utils.load_model은 다양한 체크포인트 형식을 처리)
            checkpoint = utils.load_model(args.finetune, model)

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        # 분류 헤드와 증류 헤드의 가중치 형태가 맞지 않으면 제거
        # (사전학습 클래스 수 != 현재 태스크 클래스 수인 경우 발생)
        for k in ['head.l.weight', 'head.l.bias',
                  'head_dist.l.weight', 'head_dist.l.bias']:
            if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint due to shape mismatch")
                del checkpoint_model[k]
            elif k in checkpoint_model and k not in state_dict:
                print(f"Removing key {k} from pretrained checkpoint (not in current model)")
                del checkpoint_model[k]

        # strict=False: 일치하지 않는 키가 있어도 무시하고 로드 (헤드를 제거했으므로 필요)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Loading pretrained weights... {msg}")

    # 모델을 지정된 디바이스(GPU/CPU)로 이동
    model.to(device)

    # ----------------------------------------------------------------
    # PGM Pruning 그룹 로드 (Phase B)
    # ----------------------------------------------------------------
    pruning_groups = None
    if args.pruning:
        print("Loading PGM pruning groups...")
        pruning_groups = build_pruning_groups(model)
        print(f"Loaded {len(pruning_groups)} pruning groups")

    # ----------------------------------------------------------------
    # EMA(Exponential Moving Average) 모델 설정
    # EMA 모델은 학습 모델의 가중치를 지수이동평균으로 유지하여
    # 더 안정적이고 일반화 성능이 높은 평가 모델을 제공함.
    # 중요: EMA 모델은 반드시 cuda() 이동 후, DDP 래핑 전에 생성해야 함
    # ----------------------------------------------------------------
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # ----------------------------------------------------------------
    # 분산학습(DDP) 래핑
    # DDP(DistributedDataParallel)는 각 GPU에서 독립적으로 순전파/역전파를 수행하고
    # 그래디언트를 자동으로 동기화(all-reduce)하여 여러 GPU를 효율적으로 활용함.
    # model_without_ddp는 DDP 래퍼를 벗긴 실제 모델로, 파라미터 접근 및 저장에 사용됨
    # ----------------------------------------------------------------
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # ----------------------------------------------------------------
    # 선형 학습률 스케일링 (Linear LR Scaling Rule)
    # 배치 크기에 비례하여 학습률을 조정하는 규칙.
    # 기준 배치 크기를 512로 설정하고, 실제 전체 배치 크기(batch_size * world_size)에 맞게 스케일링.
    # 이렇게 하면 GPU 수가 달라져도 동등한 학습 효과를 얻을 수 있음
    # ----------------------------------------------------------------
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)

    # NativeScaler: PyTorch 내장 AMP(Automatic Mixed Precision) 손실 스케일러.
    # FP16 연산에서 발생하는 언더플로를 방지하기 위해 손실값을 동적으로 스케일링함
    loss_scaler = NativeScaler()

    # 학습률 스케줄러 생성 (반환값의 두 번째 원소는 사용하지 않음)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # ----------------------------------------------------------------
    # 손실 함수 설정
    # Mixup 활성화 여부와 레이블 스무딩 여부에 따라 적절한 손실함수를 선택:
    # - Mixup 사용 시: SoftTargetCrossEntropy (혼합된 소프트 레이블에 맞는 손실)
    # - 레이블 스무딩 사용 시: LabelSmoothingCrossEntropy
    # - 그 외: 표준 CrossEntropyLoss
    # ----------------------------------------------------------------
    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        # Mixup 사용 시 레이블이 이미 소프트 타겟으로 변환되므로 SoftTargetCrossEntropy 사용
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # ----------------------------------------------------------------
    # 교사 모델 설정 (지식 증류 사용 시)
    # 교사 모델은 학습하지 않고 추론만 수행하므로 eval()로 고정
    # ----------------------------------------------------------------
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        # 교사 모델은 파라미터 업데이트 없이 소프트/하드 타겟 생성에만 사용됨
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is
    # 'none'
    # DistillationLoss는 기존 분류 손실함수를 래핑하여 증류 손실을 추가함.
    # distillation_type이 'none'이면 기존 criterion을 그대로 통과시킴
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    # ----------------------------------------------------------------
    # 출력 디렉토리 설정 및 학습 정보 저장
    # 메인 프로세스(rank 0)에서만 파일을 저장하여 중복 저장을 방지
    # ----------------------------------------------------------------
    output_dir = Path(args.output_dir)
    # 모델 구조를 텍스트 파일로 저장 (모델 구성 확인용)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
    # 실험에 사용된 모든 하이퍼파라미터를 JSON으로 저장 (재현성 확보)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")

    # ----------------------------------------------------------------
    # 체크포인트에서 학습 재개 (resume)
    # 모델 가중치 외에 옵티마이저 상태, 스케줄러 상태, EMA 상태, 스케일러 상태도 복원하여
    # 완전히 동일한 상태에서 학습을 이어갈 수 있음
    # ----------------------------------------------------------------
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'])
        print(msg)
        # 평가 모드가 아닐 때만 옵티마이저/스케줄러 상태 복원 (평가 시에는 모델 가중치만 필요)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # 다음 에폭부터 재개 (저장된 epoch는 마지막으로 완료된 에폭)
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(
                    model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    # ----------------------------------------------------------------
    # 평가 전용 모드
    # --eval 플래그가 설정되면 학습 없이 검증 데이터셋에서 정확도를 측정하고 종료
    # ----------------------------------------------------------------
    if args.eval:
        # utils.replace_batchnorm(model) # Users may choose whether to merge Conv-BN layers during eval
        print(f"Evaluating model: {args.model}")
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    # ----------------------------------------------------------------
    # 학습 루프 시작
    # start_epoch부터 epochs까지 반복하며 학습과 평가를 수행
    # ----------------------------------------------------------------
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0      # 학습 모델의 최고 Top-1 정확도 추적
    max_accuracy_ema = 0.0  # EMA 모델의 최고 Top-1 정확도 추적 (현재 미사용)
    for epoch in range(args.start_epoch, args.epochs):
        # 분산학습 시 에폭마다 샘플러에 에폭 번호를 설정하여 셔플 시드를 변경.
        # 이를 통해 에폭마다 다른 방식으로 데이터가 GPU에 분배됨
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 한 에폭 학습 수행: 모델 가중치 업데이트 및 학습 통계 반환
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, model_ema, mixup_fn,
            # set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            set_training_mode=True,
            set_bn_eval=args.set_bn_eval, # set bn to eval if finetune
            pruning_groups=pruning_groups,
            lambda_ffn=args.lambda_ffn, lambda_qk=args.lambda_qk, lambda_v=args.lambda_v,
            mu=args.mu, m_max_mb=args.m_max_mb,
            pruning_freq=args.pruning_freq, target_compression=args.target_compression,
        )

        # 현재 에폭에 맞게 학습률 스케줄 업데이트
        lr_scheduler.step(epoch)

        # 검증 데이터셋에서 현재 모델의 정확도 평가
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        # ----------------------------------------------------------------
        # 체크포인트 저장
        # save_freq 에폭마다, 또는 마지막 에폭에 저장.
        # 모델 가중치, 옵티마이저, 스케줄러, EMA, 스케일러 상태를 모두 저장하여 완전한 복원 가능
        # ----------------------------------------------------------------
        if args.output_dir:
            if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(output_dir, 'checkpoint_'+str(epoch)+'.pth')
                checkpoint_paths = [ckpt_path]
                print("Saving checkpoint to {}".format(ckpt_path))
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        # 최고 정확도 갱신 및 출력
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        # 학습 및 검증 통계를 하나의 딕셔너리로 합쳐 로그에 기록
        # 키에 'train_' 또는 'test_' 접두사를 붙여 출처를 구분
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        # 메인 프로세스에서만 로그 파일에 기록 (분산학습 시 중복 기록 방지)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # 전체 학습 소요 시간 계산 및 출력
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # 커맨드라인에서 직접 실행할 때 인자를 파싱하고 메인 함수를 호출
    parser = argparse.ArgumentParser(
        'EfficientViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # 출력 디렉토리가 지정된 경우 디렉토리를 생성 (이미 존재해도 오류 없이 진행)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
