# MMDetection 프레임워크를 사용하여 학습된 EfficientViT 기반 객체 탐지 모델을 평가하는 스크립트입니다.
# 단일 GPU 및 다중 GPU 분산 평가를 모두 지원하며, 다음 기능을 제공합니다:
# - 탐지 결과를 pkl 파일로 저장 (--out)
# - COCO mAP 등 표준 평가 지표 계산 (--eval)
# - 시각화된 탐지 결과 저장 또는 화면 출력 (--show, --show-dir)
# - 서버 제출용 결과 포맷 변환 (--format-only)
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
# fuse_conv_bn: Conv2d와 BatchNorm2d 레이어를 하나로 융합하여 추론 속도 향상
from mmcv.cnn import fuse_conv_bn
# MMDataParallel: 단일 GPU 또는 CPU 추론을 위한 래퍼
# MMDistributedDataParallel: 다중 GPU 분산 추론을 위한 래퍼
# wrap_fp16_model: 모델을 FP16 혼합 정밀도 모드로 변환
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

# single_gpu_test: 단일 GPU에서 데이터로더 전체를 추론하고 결과를 수집
# multi_gpu_test: 다중 GPU에서 데이터를 분할하여 병렬 추론 후 결과를 합산
from mmdet.apis import multi_gpu_test, single_gpu_test
# replace_ImageToTensor: 배치 크기 > 1일 때 'ImageToTensor'를 'DefaultFormatBundle'로 교체
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

import sys

# EfficientViT 백본 및 FPN 넥 모듈 등록:
# 이 임포트를 통해 MMDetection의 레지스트리에 커스텀 모듈이 등록되어
# config 파일에서 type='EfficientViT', type='EfficientViTFPN' 등으로 사용 가능
import efficientvit
import efficientvit_fpn


def parse_args():
    """테스트/평가 스크립트의 명령줄 인자를 파싱합니다.

    반환값:
        argparse.Namespace: 파싱된 인자 객체
            - config: 모델 설정 파일 경로 (필수)
            - checkpoint: 평가에 사용할 체크포인트 파일 경로 (필수)
            - work_dir: 평가 지표 JSON 파일을 저장할 디렉토리
            - out: 탐지 결과를 저장할 pkl 파일 경로
            - fuse_conv_bn: Conv+BN 융합으로 추론 속도 최적화 여부
            - format_only: 평가 없이 결과를 특정 포맷으로 변환만 수행
            - eval: 계산할 평가 지표 목록 (예: bbox, segm)
            - show: 탐지 결과를 화면에 시각화
            - show_dir: 시각화 결과 이미지를 저장할 디렉토리
            - show_score_thr: 시각화 시 표시할 최소 신뢰도 점수 (기본값: 0.3)
            - gpu_collect: 분산 평가에서 GPU 메모리로 결과 수집 여부
            - tmpdir: 분산 평가 결과 수집 시 사용할 임시 디렉토리
            - cfg_options: config 파일 설정을 CLI에서 오버라이드
            - eval_options: dataset.evaluate() 함수에 전달할 추가 옵션
            - launcher: 분산 평가 런처 ('none', 'pytorch', 'slurm', 'mpi')
            - local_rank: 현재 프로세스의 로컬 GPU 인덱스
    """
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    # config: 학습에 사용한 것과 동일한 MMDetection 설정 파일 (필수)
    parser.add_argument('config', help='test config file path')
    # checkpoint: 평가할 학습된 모델 가중치 파일 (필수, .pth 형식)
    parser.add_argument('checkpoint', help='checkpoint file')
    # work_dir: 평가 결과(JSON 파일)를 저장할 디렉토리 (랭크 0 프로세스만 저장)
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    # out: 모든 추론 결과를 pickle 형식으로 저장 (나중에 재평가 시 재사용 가능)
    # 파일 확장자는 반드시 .pkl 또는 .pickle이어야 함
    parser.add_argument('--out', help='output result file in pickle format')
    # fuse-conv-bn: Conv2d와 BatchNorm2d를 수학적으로 동일한 단일 Conv2d로 합침
    # 추론 시 약간의 속도 향상을 제공하지만 학습에는 사용 불가
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    # format-only: 실제 평가 없이 결과를 서버 제출 형식으로만 변환
    # COCO test-dev 서버 제출 시 유용 (GT 어노테이션이 없는 경우)
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    # eval: 계산할 평가 지표 선택
    # COCO: 'bbox'(바운딩박스 mAP), 'segm'(세그멘테이션 mAP), 'proposal'(제안 영역 AR)
    # PASCAL VOC: 'mAP', 'recall'
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    # show: 추론 결과를 실시간으로 화면에 표시 (OpenCV 창)
    parser.add_argument('--show', action='store_true', help='show results')
    # show-dir: 탐지 결과가 그려진 이미지를 지정 디렉토리에 저장
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    # show-score-thr: 시각화 시 이 임계값 이상의 신뢰도를 가진 탐지 결과만 표시
    # 낮은 신뢰도의 노이즈 탐지를 필터링하는 데 사용
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    # gpu-collect: 분산 평가 시 각 GPU의 결과를 GPU 메모리에서 직접 수집
    # False이면 tmpdir을 통해 디스크 경유로 결과 수집 (메모리 절약)
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    # tmpdir: gpu-collect=False일 때 각 프로세스가 결과를 저장할 임시 디렉토리
    # 마스터 프로세스(랭크 0)가 모든 임시 결과를 합산한 후 삭제
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    # cfg-options: config 파일의 특정 설정을 CLI에서 오버라이드
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
    # options: [deprecated] eval-options로 대체됨
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    # eval-options: dataset.evaluate() 함수에 전달할 추가 키워드 인자
    # 예: --eval-options classwise=True iou_thrs=0.5
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    # launcher: 분산 평가 런처 선택 (train.py와 동일한 옵션)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # local_rank: 분산 평가에서 현재 프로세스의 로컬 GPU 인덱스
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # LOCAL_RANK 환경변수 설정: torch.distributed 런타임이 이 변수를 사용
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # --options와 --eval-options는 동시에 사용 불가 (상호 배타적)
    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    # [deprecated] --options를 --eval-options로 자동 변환
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    """평가 메인 함수.

    처리 순서:
    1. 인자 파싱 및 유효성 검사 (최소 하나의 출력 옵션 필수)
    2. config 파일 로드 및 테스트 모드 설정
    3. 분산 평가 환경 초기화 (해당하는 경우)
    4. 테스트 데이터셋 및 데이터로더 빌드
    5. 모델 빌드, 체크포인트 로드, FP16/Conv-BN 융합 최적화
    6. 단일 또는 다중 GPU 추론 실행
    7. 결과 저장/평가/포맷 변환 (랭크 0 프로세스만 수행)
    """
    args = parse_args()

    # 최소 하나의 출력 옵션이 필요:
    # --out(pkl 저장), --eval(평가), --format-only(포맷 변환),
    # --show(화면 출력), --show-dir(이미지 저장) 중 하나 이상 지정 필수
    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    # --eval과 --format-only는 동시에 사용 불가 (상호 배타적)
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    # 출력 파일은 반드시 pkl 또는 pickle 형식이어야 함
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # MMDetection config 파일 로드
    cfg = Config.fromfile(args.config)
    # CLI에서 지정한 설정으로 config 오버라이드
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    # config에 custom_imports가 있으면 해당 모듈을 동적으로 임포트
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    # 추론 시에도 cudnn_benchmark=True로 성능 향상 가능 (입력 크기가 고정된 경우)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 평가 시에는 사전학습 가중치 로드 불필요 (체크포인트에서 직접 로드)
    cfg.model.pretrained = None
    # 넥 모듈(FPN 등)의 RFP 백본 사전학습 가중치도 제거
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    # 테스트 모드 활성화 및 배치 크기(samples_per_gpu) 설정
    # 배치 크기 > 1이면 ImageToTensor를 DefaultFormatBundle로 교체 (배치 처리 지원)
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        # 단일 테스트 데이터셋 설정
        cfg.data.test.test_mode = True   # test_mode=True: 어노테이션 로드 없이 추론만 수행
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            # 단일 이미지용 변환(ImageToTensor)을 배치 처리 가능한 변환으로 교체
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        # 다수의 테스트 데이터셋이 연결된 경우 (ConcatDataset)
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    # 분산 평가 환경 초기화 (train.py와 동일한 방식)
    if args.launcher == 'none':
        # 단일 GPU 비분산 평가 모드
        distributed = False
    else:
        # 분산 평가 모드: 데이터를 각 GPU에 분할하여 병렬 추론
        # NCCL 백엔드로 GPU 간 결과 수집
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # 현재 프로세스의 랭크와 전체 프로세스 수 조회
    rank, _ = get_dist_info()
    # allows not to create
    # 랭크 0 프로세스만 work_dir 생성 및 타임스탬프 로그 파일 경로 설정
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # 평가 결과(지표)를 JSON 파일로 저장: eval_20240101_120000.json
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    # 테스트 데이터셋 빌드 (config의 data.test 설정 사용)
    dataset = build_dataset(cfg.data.test)
    # 테스트용 데이터로더:
    # - shuffle=False: 평가 시 이미지 순서를 유지해야 결과를 이미지에 매핑 가능
    # - 분산 모드이면 DistributedSampler를 사용하여 각 GPU에 데이터 분할
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    # 평가 모드에서는 train_cfg=None으로 설정하여 학습 전용 레이어/로직 비활성화
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # FP16 설정이 있으면 모델을 혼합 정밀도로 변환 (추론 속도 향상)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # 체크포인트 로드: map_location='cpu'로 먼저 CPU에 로드하여 GPU 메모리 관리 효율화
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # Conv+BN 레이어를 하나의 Conv 레이어로 수학적으로 융합하여 추론 가속화
    # 학습 완료 후 추론 전용으로만 사용 가능 (역전파 불가)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # 체크포인트에서 클래스 정보 복원 (이전 버전 호환성 처리)
    # 새 체크포인트: meta['CLASSES']에서 클래스명 로드
    # 구 체크포인트: 데이터셋의 CLASSES 속성에서 클래스명 로드
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        # 단일 GPU 추론:
        # MMDataParallel: device_ids=[0]으로 첫 번째 GPU에서 추론
        # single_gpu_test: 데이터로더 전체를 순회하며 결과를 리스트로 수집
        # args.show: 각 이미지 추론 후 시각화 창 표시
        # args.show_dir: 시각화 이미지 저장 경로
        # args.show_score_thr: 표시할 최소 신뢰도 임계값
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        # 다중 GPU 분산 추론:
        # MMDistributedDataParallel: 현재 프로세스의 GPU에서 할당된 데이터를 추론
        # broadcast_buffers=False: BN 통계량을 GPU 간에 동기화하지 않음 (추론 시 불필요)
        # multi_gpu_test: 각 GPU의 결과를 수집하여 마스터 랭크(0)에서 통합
        # tmpdir: 디스크 기반 결과 수집 경로 (gpu_collect=False일 때 사용)
        # gpu_collect: GPU 메모리 기반 결과 수집 (빠르지만 메모리 소모 큼)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    # 결과 저장/평가는 랭크 0 프로세스에서만 수행 (분산 환경에서 중복 방지)
    rank, _ = get_dist_info()
    if rank == 0:
        # 추론 결과를 pkl 파일로 저장
        # outputs: 각 이미지의 탐지 결과 리스트 (클래스별 바운딩박스 배열)
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        # 결과를 서버 제출 형식으로 변환 (예: COCO JSON 형식)
        # format_results: 결과를 zip 파일 등으로 패키징하여 제출 준비
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        # 평가 지표 계산:
        # COCO의 경우 AP@0.5:0.95, AP@0.5, AP@0.75, AP_s, AP_m, AP_l 등 계산
        # eval_kwargs: config의 evaluation 설정에서 EvalHook 전용 키를 제거한 설정
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            # EvalHook 전용 인자(학습 중 검증에만 사용)를 제거하고 dataset.evaluate()에 전달
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            # dataset.evaluate(): 추론 결과와 GT 어노테이션을 비교하여 지표 계산
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            # 평가 결과와 config 경로를 JSON 파일로 저장 (실험 기록 관리)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
