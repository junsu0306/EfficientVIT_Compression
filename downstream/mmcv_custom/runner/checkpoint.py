# Copyright (c) Open-MMLab. All rights reserved.
#
# [커스터마이즈 이유]
# 이 파일은 mmcv 원본 runner/checkpoint.py를 기반으로 NVIDIA Apex AMP(Automatic Mixed Precision)
# 지원을 추가한 커스터마이즈 버전입니다.
#
# 주요 변경 사항:
#   - 표준 mmcv의 save_checkpoint()는 AMP state_dict를 저장하지 않습니다.
#   - 이 커스텀 버전은 체크포인트에 'amp' 키를 추가하여 apex.amp의 내부 상태(스케일러 등)를
#     함께 저장합니다.
#   - 이를 통해 FP16 혼합 정밀도 학습을 중단하고 재개할 때 AMP 상태가 올바르게 복원됩니다.
#
# 의존 관계:
#   - mmcv.runner.checkpoint의 weights_to_cpu, get_state_dict를 재사용합니다.
#   - epoch_based_runner.py의 EpochBasedRunnerAmp에서 이 함수를 호출합니다.

import os.path as osp
import time
from tempfile import TemporaryDirectory

import torch
from torch.optim import Optimizer

import mmcv
from mmcv.parallel import is_module_wrapper
# mmcv 원본의 weights_to_cpu, get_state_dict를 재사용합니다.
# (이 두 함수는 AMP와 무관하므로 커스터마이즈 불필요)
from mmcv.runner.checkpoint import weights_to_cpu, get_state_dict

# apex: NVIDIA의 혼합 정밀도(AMP) 및 분산 학습 라이브러리
# FP16 학습을 위해 필요하며, 설치되지 않은 환경에서도 다른 기능은 사용 가능합니다.
try:
    import apex
except:
    print('apex is not installed')


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 4 fields: ``meta``, ``state_dict`` and
    ``optimizer``, ``amp``. By default ``meta`` will contain version
    and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.

    [한국어 설명]
    AMP(Automatic Mixed Precision) 상태를 포함하여 체크포인트를 저장합니다.
    mmcv 원본 save_checkpoint()에 비해 'amp' 필드가 추가되어 총 4개의 필드를 저장합니다:
      - 'meta'       : mmcv 버전, 저장 시각 등 메타 정보
      - 'state_dict' : CPU로 복사된 모델 가중치
      - 'optimizer'  : 옵티마이저 상태 (학습 재개 시 필요)
      - 'amp'        : apex.amp 내부 상태 (loss scaler 등, FP16 학습 재개 시 필요)

    AMP 상태를 저장하는 이유:
      FP16 학습에서 apex.amp는 동적 loss scaling을 수행합니다.
      학습을 재개할 때 이 스케일러 값이 복원되지 않으면 초기 수렴이 불안정해질 수 있으므로
      체크포인트에 AMP 상태를 함께 저장합니다.

    Args:
        model (Module): 저장할 모델.
        filename (str): 저장할 파일 경로 또는 'pavi://...' URI.
        optimizer (Optimizer | dict | None): 저장할 옵티마이저.
        meta (dict | None): 체크포인트에 저장할 추가 메타 정보.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    # mmcv 버전과 현재 시각을 메타 정보에 추가합니다.
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    # DDP 등 래퍼 모듈이면 내부 실제 모듈을 꺼냅니다.
    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        # 탐지/분류 모델의 클래스 이름을 메타에 저장합니다.
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        # 모델 가중치를 CPU로 옮겨 GPU 종속성을 제거한 후 저장합니다.
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        # 단일 옵티마이저의 state_dict를 저장합니다.
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        # 복수 옵티마이저(예: GAN의 Generator/Discriminator)의 state_dict를 저장합니다.
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    # save amp state dict in the checkpoint
    # [핵심 추가 기능] apex.amp의 상태(loss scaler 등)를 체크포인트에 저장합니다.
    # 이 줄이 mmcv 원본과의 핵심 차이점입니다.
    checkpoint['amp'] = apex.amp.state_dict()

    if filename.startswith('pavi://'):
        # PAVI 모델 클라우드에 체크포인트를 업로드합니다.
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except NodeNotFoundError:
            # 디렉터리가 없으면 새로 생성합니다.
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            # 임시 파일에 저장 후 PAVI에 업로드합니다.
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        # 로컬 파일 시스템에 체크포인트를 저장합니다.
        mmcv.mkdir_or_exist(osp.dirname(filename))
        # immediately flush buffer
        # 버퍼를 즉시 디스크에 기록하여 파일이 완전히 저장되도록 합니다.
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()
