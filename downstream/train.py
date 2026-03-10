# MMDetection 프레임워크를 사용하여 EfficientViT 기반 객체 탐지 모델을 학습하는 스크립트입니다.
# 단일 GPU 및 분산 학습(DDP: DistributedDataParallel) 두 가지 모드를 지원합니다.
# 분산 학습 런처로 PyTorch, Slurm, MPI를 선택할 수 있습니다.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed  # , train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes, update_data_root)

# mmdet_custom: 커스텀 학습 로직이 포함된 모듈 (표준 mmdetection과 다른 커스텀 train_detector)
from mmdet_custom.apis.train import train_detector
# mmcv_custom: 커스텀 epoch 기반 러너 및 옵티마이저 (EfficientViT에 특화된 설정)
import mmcv_custom.runner.epoch_based_runner
import mmcv_custom.runner.optimizer

import sys

# EfficientViT 백본 및 FPN 넥 모듈 등록:
# 이 임포트를 통해 MMDetection의 레지스트리에 커스텀 모듈이 등록되어
# config 파일에서 type='EfficientViT', type='EfficientViTFPN' 등으로 사용 가능
import efficientvit
import efficientvit_fpn


def parse_args():
    """학습 스크립트의 명령줄 인자를 파싱합니다.

    반환값:
        argparse.Namespace: 파싱된 인자 객체
            - config: 학습 config 파일 경로 (필수)
            - work_dir: 로그 및 체크포인트 저장 디렉토리
            - resume_from: 학습 재개할 체크포인트 경로
            - auto_resume: 최신 체크포인트에서 자동 재개 여부
            - no_validate: 학습 중 검증 비활성화 여부
            - gpu_id: 단일 GPU 학습 시 사용할 GPU ID
            - seed: 재현성을 위한 난수 시드
            - diff_seed: 분산 학습 시 랭크별 다른 시드 사용 여부
            - deterministic: CUDNN 결정론적 모드 활성화 여부
            - cfg_options: config 파일의 특정 설정을 CLI에서 오버라이드
            - launcher: 분산 학습 런처 종류 ('none', 'pytorch', 'slurm', 'mpi')
            - local_rank: 분산 학습에서 현재 프로세스의 로컬 GPU 인덱스
            - auto_scale_lr: 배치 크기에 따른 학습률 자동 스케일링 여부
    """
    parser = argparse.ArgumentParser(description='Train a detector')
    # config: MMDetection 설정 파일 경로 (필수 인자)
    # 모델 구조, 데이터셋, 학습 스케줄, 옵티마이저 등 모든 설정을 담음
    parser.add_argument('config', help='train config file path')
    # work_dir: 학습 중 체크포인트(.pth), 로그 파일, config 사본이 저장될 디렉토리
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    # resume_from: 중단된 학습을 이어받을 체크포인트 파일 경로
    # 모델 가중치, 옵티마이저 상태, 에폭 정보 등을 복원
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    # auto_resume: 가장 최신 체크포인트를 자동으로 찾아 재개 (resume-from 대신 사용 가능)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    # no_validate: 학습 도중 검증 단계를 건너뜀 (빠른 실험 시 유용)
    # 기본적으로는 각 에폭 종료 시 val 데이터셋으로 성능을 측정
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    # GPU 설정 (세 옵션은 상호 배타적, 하나만 선택 가능)
    group_gpus = parser.add_mutually_exclusive_group()
    # --gpus: [deprecated] 비분산 학습에서 사용할 GPU 수 (현재는 1개만 지원)
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
             '(only applicable to non-distributed training)')
    # --gpu-ids: [deprecated] 비분산 학습에서 사용할 GPU ID 목록
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    # --gpu-id: 비분산 학습 시 사용할 단일 GPU의 ID (기본값: 0)
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    # seed: 재현 가능한 학습을 위한 난수 시드 (Python, NumPy, PyTorch 모두 적용)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    # diff_seed: 분산 학습에서 각 랭크(프로세스)가 서로 다른 시드를 사용하도록 설정
    # 데이터 증강의 다양성을 높이고 과적합을 줄이는 데 도움이 됨
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    # deterministic: CUDNN 백엔드의 결정론적 알고리즘 사용 여부
    # True로 설정하면 완전한 재현성을 보장하지만 속도가 다소 느려질 수 있음
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # options: [deprecated] config 파일 설정을 오버라이드 (--cfg-options로 대체)
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    # cfg-options: config 파일의 특정 키를 CLI에서 직접 오버라이드
    # 예: --cfg-options model.backbone.depth=50 data.samples_per_gpu=4
    # 중첩 딕셔너리, 리스트 등 복잡한 타입도 지원
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    # launcher: 분산 학습 런처 선택
    # 'none': 단일 GPU 학습
    # 'pytorch': torch.distributed 사용 (torchrun 또는 python -m torch.distributed.launch)
    # 'slurm': SLURM 클러스터 환경에서의 분산 학습
    # 'mpi': MPI(Message Passing Interface) 기반 분산 학습
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # local_rank: 분산 학습에서 현재 프로세스가 사용하는 로컬 GPU 인덱스
    # torch.distributed.launch가 자동으로 설정하며, LOCAL_RANK 환경변수와 연동됨
    parser.add_argument('--local_rank', type=int, default=0)
    # auto_scale_lr: 배치 크기에 비례하여 학습률을 자동으로 스케일링
    # 분산 학습에서 전체 배치 크기가 커지면 그에 맞게 LR도 조정 (Linear Scaling Rule)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    args = parser.parse_args()
    # LOCAL_RANK 환경변수 설정: torch.distributed 런타임이 이 변수를 사용
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # --options와 --cfg-options는 동시에 사용 불가 (상호 배타적)
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    # [deprecated] --options를 --cfg-options로 자동 변환
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    """학습 메인 함수.

    처리 순서:
    1. 인자 파싱 및 config 파일 로드
    2. 학습률 자동 스케일링, 멀티프로세스 설정
    3. 작업 디렉토리 및 로거 초기화
    4. 분산 학습 환경 초기화 (해당하는 경우)
    5. 난수 시드 설정 (재현성 보장)
    6. 모델 빌드 및 가중치 초기화
    7. 데이터셋 빌드
    8. 학습 실행 (train_detector 호출)
    """
    args = parse_args()

    # mmcv Config 객체로 config 파일 로드
    # Config 파일은 Python 딕셔너리 형태로 모델, 데이터셋, 스케줄 등 모든 설정을 포함
    cfg = Config.fromfile(args.config)

    # update data root according to MMDET_DATASETS
    # MMDET_DATASETS 환경변수가 설정된 경우 데이터 루트 경로를 자동으로 업데이트
    update_data_root(cfg)

    # CLI에서 제공된 cfg-options로 config 설정을 오버라이드
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 학습률 자동 스케일링 활성화:
    # config에 auto_scale_lr 설정이 있으면 enable=True로 변경
    # 분산 학습 시 전체 배치 크기에 맞게 LR을 자동 조정 (Linear Scaling Rule)
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

    # set multi-process settings
    # 데이터 로딩 워커 프로세스의 공유 메모리 및 파일 디스크립터 설정을 최적화
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    # cudnn_benchmark=True: 입력 크기가 고정된 경우 CUDNN이 최적의 알고리즘을 미리 탐색
    # 학습 초기에 약간의 오버헤드가 있지만 이후 학습 속도가 향상됨
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    # 작업 디렉토리 결정 우선순위: CLI 인자 > config 파일 설정 > config 파일명 기반 자동 생성
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        # config 파일명(확장자 제외)을 디렉토리명으로 사용: ./work_dirs/config_name/
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # 학습 재개 설정: 지정된 체크포인트에서 모델 가중치 및 옵티마이저 상태 복원
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    # [deprecated] GPU 관련 설정 처리
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    # 분산 학습 환경 초기화 (로거보다 먼저 초기화해야 dist_info를 로깅에 사용 가능)
    if args.launcher == 'none':
        # 단일 GPU 비분산 학습 모드
        distributed = False
    else:
        # 분산 학습 모드 초기화:
        # init_dist: 프로세스 그룹을 초기화하고 각 프로세스에 GPU를 할당
        # cfg.dist_params: 분산 학습 백엔드 설정 (보통 backend='nccl')
        # NCCL: NVIDIA Collective Communications Library, GPU 간 통신에 최적화
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        # 분산 학습 시 world_size(전체 프로세스 수)를 GPU ID 범위로 설정
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    # 작업 디렉토리 생성 (이미 존재하면 그대로 유지)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # 현재 사용된 config를 work_dir에 사본으로 저장 (재현성을 위한 기록)
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    # 타임스탬프 기반 로그 파일 생성 (예: 20240101_120000.log)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # meta 딕셔너리: 실험 재현성을 위해 환경 정보, 시드, config 등을 기록
    meta = dict()
    # log env info
    # 환경 정보 수집 및 로깅: Python 버전, PyTorch 버전, CUDA 버전, GPU 모델 등
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # 현재 사용 가능한 디바이스 탐지 (cuda, cpu, mlu 등)
    cfg.device = get_device()
    # set random seeds
    # 난수 시드 초기화: 분산 학습 시 마스터 랭크의 시드를 모든 프로세스에 동기화
    seed = init_random_seed(args.seed, device=cfg.device)
    # diff_seed=True이면 각 분산 랭크마다 서로 다른 시드 사용 (데이터 다양성 증가)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    # Python random, NumPy, PyTorch의 시드를 모두 동일하게 설정
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # MMDetection 레지스트리를 통해 config에서 모델을 동적으로 빌드
    # cfg.model: 모델 구조 설정 (백본, 넥, 헤드 등)
    # train_cfg: 학습 단계의 앵커 생성, NMS 등 설정
    # test_cfg: 추론 단계의 NMS, 신뢰도 임계값 등 설정
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # 모델 가중치 초기화: 각 서브모듈의 init_weights() 메서드를 호출
    # FPN: Xavier 초기화, 백본: 사전학습 가중치 로드 등
    model.init_weights()

    # 학습 데이터셋 빌드 (config의 data.train 설정 사용)
    datasets = [build_dataset(cfg.data.train)]
    # workflow가 [('train', 1), ('val', 1)]처럼 2개인 경우 검증 데이터셋도 준비
    # 검증 데이터셋에 학습 파이프라인을 적용하여 augmentation 일관성 유지
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        # 체크포인트에 메타데이터 저장: mmdet 버전, 클래스명 등
        # 나중에 체크포인트에서 클래스 정보를 복원할 때 사용
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    # 시각화 편의를 위해 모델에 클래스 이름 목록 추가 (예: ['person', 'car', ...])
    model.CLASSES = datasets[0].CLASSES
    # 학습 실행: 커스텀 train_detector로 EfficientViT 특화 학습 수행
    # distributed=True이면 DistributedDataParallel로 다중 GPU 학습
    # validate=True이면 각 에폭 후 검증 수행하여 mAP 등 성능 지표 측정
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
