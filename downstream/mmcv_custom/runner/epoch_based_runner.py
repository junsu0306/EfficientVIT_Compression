# Copyright (c) Open-MMLab. All rights reserved.
#
# [커스터마이즈 이유]
# 이 파일은 mmcv의 EpochBasedRunner를 상속하여 NVIDIA Apex AMP(Automatic Mixed Precision)
# 지원을 추가한 커스텀 Runner 클래스를 정의합니다.
#
# 주요 변경 사항:
#   1. save_checkpoint(): 커스텀 save_checkpoint 함수(AMP state_dict 포함)를 사용합니다.
#   2. resume(): 체크포인트 재개 시 'amp' 키가 있으면 apex.amp 상태를 복원합니다.
#
# MMDetection 통합:
#   - @RUNNERS.register_module() 데코레이터를 통해 mmcv의 Runner 레지스트리에 등록됩니다.
#   - 설정 파일(config)에서 runner.type = 'EpochBasedRunnerAmp'로 지정하면 이 클래스가 사용됩니다.
#   - EpochBasedRunner의 train(), val(), run() 등 핵심 학습 루프는 변경 없이 그대로 사용합니다.

import os.path as osp
import platform
import shutil

import torch
from torch.optim import Optimizer

import mmcv
# mmcv의 Runner 레지스트리와 기본 EpochBasedRunner를 임포트합니다.
from mmcv.runner import RUNNERS, EpochBasedRunner
# AMP 상태를 포함하여 저장하는 커스텀 save_checkpoint 함수를 사용합니다.
from .checkpoint import save_checkpoint

# apex: NVIDIA의 혼합 정밀도(AMP) 학습 라이브러리
try:
    import apex
except:
    print('apex is not installed')


@RUNNERS.register_module()
class EpochBasedRunnerAmp(EpochBasedRunner):
    """Epoch-based Runner with AMP support.

    This runner train models epoch by epoch.

    [한국어 설명]
    에폭(Epoch) 단위로 학습을 수행하는 Runner에 AMP(Automatic Mixed Precision) 지원을 추가한 클래스입니다.

    mmcv의 EpochBasedRunner를 상속하며, 다음 두 가지 메서드를 오버라이드합니다:
      - save_checkpoint(): AMP state_dict를 포함한 체크포인트 저장
      - resume(): AMP state_dict를 복원하는 체크포인트 재개

    MMDetection과의 관계:
      - RUNNERS.register_module() 데코레이터로 mmcv 레지스트리에 'EpochBasedRunnerAmp'라는 이름으로 등록됩니다.
      - 설정 파일에서 cfg.runner.type = 'EpochBasedRunnerAmp'로 지정하면 이 클래스가 사용됩니다.
      - train_detector() (mmdet_custom/apis/train.py)에서 build_runner()를 통해 자동으로 생성됩니다.

    사용 예시 (config 파일):
        runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
        optimizer_config = dict(type='DistOptimizerHook', use_fp16=True)
    """

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.

        [한국어 설명]
        현재 에폭의 체크포인트를 저장합니다.
        mmcv 원본과 달리 커스텀 save_checkpoint 함수(AMP state_dict 포함)를 사용합니다.

        저장되는 파일:
          - {out_dir}/epoch_{N}.pth : 현재 에폭 N의 체크포인트
          - {out_dir}/latest.pth    : 최신 체크포인트를 가리키는 심볼릭 링크 (Linux/macOS)
                                      또는 파일 복사본 (Windows)

        메타 정보:
          - meta['epoch']: 현재 에폭 번호 (1-indexed, self.epoch는 0-indexed)
          - meta['iter']: 현재 총 이터레이션 수

        Args:
            out_dir (str): 체크포인트를 저장할 디렉터리 경로.
            filename_tmpl (str): 파일명 템플릿. '{}'에 에폭 번호가 들어갑니다.
            save_optimizer (bool): True이면 옵티마이저 상태도 함께 저장합니다.
            meta (dict | None): 추가 메타 정보. 에폭/이터레이션 정보가 자동으로 추가됩니다.
            create_symlink (bool): True이면 'latest.pth' 심볼릭 링크를 생성합니다.
        """
        if meta is None:
            # 메타 정보가 없으면 현재 에폭/이터레이션 정보로 초기화합니다.
            # self.epoch는 0-indexed이므로 +1하여 1-indexed로 저장합니다.
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            # 메타 정보가 있으면 에폭/이터레이션 정보를 업데이트합니다.
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            # runner 레벨의 메타 정보(예: 설정 파일 정보)도 함께 저장합니다.
            meta.update(self.meta)

        # 파일명을 에폭 번호로 포맷팅합니다. 예: 'epoch_12.pth'
        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        # save_optimizer=False이면 옵티마이저를 저장하지 않습니다 (파일 크기 절약).
        optimizer = self.optimizer if save_optimizer else None
        # 커스텀 save_checkpoint를 호출합니다 (AMP state_dict 포함).
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            # 'latest.pth'가 최신 체크포인트를 가리키도록 심볼릭 링크를 생성합니다.
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                # Linux/macOS에서는 심볼릭 링크를 생성합니다.
                mmcv.symlink(filename, dst_file)
            else:
                # Windows는 심볼릭 링크를 지원하지 않을 수 있어 파일을 복사합니다.
                shutil.copy(filepath, dst_file)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        """체크포인트로부터 학습을 재개합니다.

        mmcv 원본 resume()에 비해 apex.amp 상태 복원 기능이 추가되었습니다.
        'amp' 키가 체크포인트에 존재하면 apex.amp.load_state_dict()를 호출하여
        loss scaler 등 AMP 내부 상태를 복원합니다.

        Args:
            checkpoint (str): 재개할 체크포인트 파일 경로.
            resume_optimizer (bool): True이면 옵티마이저 상태도 복원합니다. 기본값: True.
            map_location (str | callable): 텐서를 로드할 장치.
                'default'이면 현재 CUDA 장치를 자동으로 선택합니다.

        흐름:
          1. 체크포인트 파일을 로드합니다.
          2. 에폭/이터레이션 카운터를 복원합니다.
          3. 옵티마이저 상태를 복원합니다 (resume_optimizer=True인 경우).
          4. AMP 상태를 복원합니다 ('amp' 키가 있는 경우).
        """
        if map_location == 'default':
            if torch.cuda.is_available():
                # 현재 CUDA 장치로 체크포인트를 로드합니다.
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                # CUDA가 없으면 기본 위치(CPU)에 로드합니다.
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            # 사용자가 지정한 위치에 로드합니다.
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        # 체크포인트에 저장된 에폭/이터레이션 카운터를 복원합니다.
        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        # 옵티마이저 상태를 복원합니다 (학습률 스케줄러 상태 등 포함).
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                # 단일 옵티마이저의 state_dict를 복원합니다.
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                # 복수 옵티마이저 각각의 state_dict를 복원합니다.
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'amp' in checkpoint:
            # [핵심 추가 기능] AMP(Automatic Mixed Precision) 상태를 복원합니다.
            # apex.amp의 loss scaler 값 등이 복원되어 FP16 학습이 안정적으로 재개됩니다.
            apex.amp.load_state_dict(checkpoint['amp'])
            self.logger.info('load amp state dict')

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)
