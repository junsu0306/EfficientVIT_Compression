# Copyright (c) Open-MMLab. All rights reserved.
#
# [커스터마이즈 이유]
# 이 파일은 mmcv 원본 checkpoint.py를 기반으로 EfficientVIT 및 Swin Transformer 계열
# 모델을 위해 커스터마이즈된 버전입니다.
# 주요 변경 사항:
#   1. load_checkpoint()에서 'model' 키를 추가로 지원 (다양한 체크포인트 포맷 호환)
#   2. MoBY 자기지도학습 모델의 online branch encoder 가중치 로딩 지원
#   3. absolute_pos_embed (절대 위치 임베딩) 크기 불일치 시 경고 처리
#   4. relative_position_bias_table (상대 위치 편향 테이블) 크기 불일치 시 bicubic 보간으로 자동 리사이즈
# 이를 통해 사전 학습된 모델을 다양한 해상도/구조의 downstream 태스크에 전이학습할 수 있습니다.

import io
import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory

import torch
import torchvision
from torch.optim import Optimizer
from torch.utils import model_zoo
from torch.nn import functional as F

import mmcv
from mmcv.fileio import FileClient
from mmcv.fileio import load as load_file
from mmcv.parallel import is_module_wrapper
from mmcv.utils import mkdir_or_exist
from mmcv.runner import get_dist_info

# 환경 변수 이름 및 기본 캐시 경로 정의
ENV_MMCV_HOME = 'MMCV_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


def _get_mmcv_home():
    """mmcv 홈 디렉터리 경로를 반환합니다.

    환경 변수 MMCV_HOME이 설정되어 있으면 해당 경로를 사용하고,
    없으면 XDG_CACHE_HOME/mmcv 또는 ~/.cache/mmcv 를 기본값으로 사용합니다.
    디렉터리가 없으면 자동으로 생성합니다.

    Returns:
        str: mmcv 홈 디렉터리의 절대 경로.
    """
    mmcv_home = os.path.expanduser(
        os.getenv(
            ENV_MMCV_HOME,
            os.path.join(
                os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmcv')))

    mkdir_or_exist(mmcv_home)
    return mmcv_home


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.

    [한국어 설명]
    모델(module)에 state_dict(가중치 딕셔너리)를 로드합니다.
    PyTorch 기본 load_state_dict와 달리 strict=False가 기본값이며,
    키 불일치 발생 시 경고만 출력하고 계속 진행합니다.

    파라미터:
        module (Module): 가중치를 로드할 대상 모델.
        state_dict (OrderedDict): 불러올 가중치 딕셔너리.
        strict (bool): True이면 키가 완전히 일치해야 하며, False이면 일부 불일치를 허용합니다.
        logger: 오류 메시지를 출력할 로거. None이면 print()를 사용합니다.
    """
    unexpected_keys = []   # 체크포인트에는 있으나 모델에 없는 키 목록
    all_missing_keys = []  # 모델에는 있으나 체크포인트에 없는 키 목록
    err_msg = []           # 오류 메시지 누적 리스트

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        # DDP(DistributedDataParallel) 등 래퍼 모듈이면 내부 실제 모듈을 꺼냅니다.
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        # PyTorch 내부 메서드를 사용해 해당 모듈의 파라미터/버퍼를 로드합니다.
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        # 하위 모듈에 대해 재귀적으로 로드를 수행합니다.
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle (순환 참조 방지를 위해 load 함수 해제)

    # ignore "num_batches_tracked" of BN layers
    # BatchNorm의 num_batches_tracked는 필수 키가 아니므로 missing_keys에서 제외합니다.
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    # 불필요한 키(unexpected)와 누락 키(missing)가 있으면 오류 메시지를 작성합니다.
    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    # rank 0 (메인 프로세스)에서만 경고 메시지를 출력합니다.
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            # strict=True이면 키 불일치 시 예외를 발생시킵니다.
            raise RuntimeError(err_msg)
        elif logger is not None:
            # logger가 있으면 warning 레벨로 출력합니다.
            logger.warning(err_msg)
        else:
            # logger가 없으면 print로 출력합니다.
            print(err_msg)


def load_url_dist(url, model_dir=None):
    """In distributed setting, this function only download checkpoint at local
    rank 0.

    [한국어 설명]
    분산 학습 환경에서 URL로부터 체크포인트를 다운로드합니다.
    LOCAL_RANK 0인 프로세스만 실제 다운로드를 수행하고,
    다른 프로세스들은 barrier에서 대기한 뒤 캐시된 파일을 로드합니다.
    이를 통해 중복 다운로드를 방지합니다.

    Args:
        url (str): 체크포인트 파일의 URL.
        model_dir (str, optional): 모델을 저장할 로컬 디렉터리. None이면 기본 캐시 경로 사용.

    Returns:
        dict: 로드된 체크포인트 딕셔너리.
    """
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        # rank 0만 먼저 다운로드합니다.
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        # 분산 환경에서는 rank 0의 다운로드가 완료될 때까지 다른 프로세스가 대기합니다.
        torch.distributed.barrier()
        if rank > 0:
            # rank 0이 다운로드한 뒤에 다른 rank들이 캐시에서 로드합니다.
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint


def load_pavimodel_dist(model_path, map_location=None):
    """In distributed setting, this function only download checkpoint at local
    rank 0.

    [한국어 설명]
    PAVI(SenseTime 모델 클라우드 서비스)에서 체크포인트를 분산 방식으로 다운로드합니다.
    rank 0만 임시 디렉터리에 다운로드하고, 다른 rank들은 barrier 후 동일하게 로드합니다.

    Args:
        model_path (str): PAVI 모델 클라우드의 모델 경로.
        map_location: torch.load의 map_location 파라미터와 동일.

    Returns:
        dict: 로드된 체크포인트 딕셔너리.
    """
    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        model = modelcloud.get(model_path)
        with TemporaryDirectory() as tmp_dir:
            downloaded_file = osp.join(tmp_dir, model.name)
            model.download(downloaded_file)
            checkpoint = torch.load(downloaded_file, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            model = modelcloud.get(model_path)
            with TemporaryDirectory() as tmp_dir:
                downloaded_file = osp.join(tmp_dir, model.name)
                model.download(downloaded_file)
                checkpoint = torch.load(
                    downloaded_file, map_location=map_location)
    return checkpoint


def load_fileclient_dist(filename, backend, map_location):
    """In distributed setting, this function only download checkpoint at local
    rank 0.

    [한국어 설명]
    Ceph 등의 분산 파일 시스템(backend)으로부터 체크포인트를 로드합니다.
    분산 학습 환경에서 rank 0만 먼저 파일을 읽고, 나머지는 barrier 후 동일하게 로드합니다.
    현재 지원하는 백엔드는 'ceph'만입니다.

    Args:
        filename (str): 파일 경로 (예: 's3://bucket/model.pth').
        backend (str): 파일 클라이언트 백엔드. 현재 'ceph'만 지원.
        map_location: torch.load의 map_location 파라미터와 동일.

    Returns:
        dict: 로드된 체크포인트 딕셔너리.
    """
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    allowed_backends = ['ceph']
    if backend not in allowed_backends:
        raise ValueError(f'Load from Backend {backend} is not supported.')
    if rank == 0:
        fileclient = FileClient(backend=backend)
        buffer = io.BytesIO(fileclient.get(filename))
        checkpoint = torch.load(buffer, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


def get_torchvision_models():
    """torchvision에 등록된 모든 사전 학습 모델의 URL 딕셔너리를 반환합니다.

    torchvision.models 패키지를 순회하여 각 모듈의 model_urls 속성을 수집합니다.

    Returns:
        dict: 모델 이름을 키로, URL을 값으로 하는 딕셔너리.
    """
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def get_external_models():
    """mmcv의 open-mmlab 모델 zoo에서 외부 모델 URL 딕셔너리를 반환합니다.

    mmcv 기본 open_mmlab.json을 로드하고, mmcv 홈 디렉터리에 사용자 정의
    open_mmlab.json이 존재하면 이를 병합하여 반환합니다.

    Returns:
        dict: 모델 이름을 키로, URL 또는 로컬 경로를 값으로 하는 딕셔너리.
    """
    mmcv_home = _get_mmcv_home()
    default_json_path = osp.join(mmcv.__path__[0], 'model_zoo/open_mmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmcv_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        # 사용자가 커스텀 모델을 추가한 경우 기본 URL 딕셔너리를 업데이트합니다.
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)

    return default_urls


def get_mmcls_models():
    """mmcls(MMClassification) 모델 zoo의 URL 딕셔너리를 반환합니다.

    Returns:
        dict: mmcls 모델 이름을 키로, URL을 값으로 하는 딕셔너리.
    """
    mmcls_json_path = osp.join(mmcv.__path__[0], 'model_zoo/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)

    return mmcls_urls


def get_deprecated_model_names():
    """mmcv에서 더 이상 사용되지 않는(deprecated) 모델 이름 딕셔너리를 반환합니다.

    deprecated.json을 로드하여 구버전 모델 이름을 새 이름으로 매핑하는 딕셔너리를 반환합니다.

    Returns:
        dict: 구버전 모델 이름을 키로, 신버전 이름을 값으로 하는 딕셔너리.
    """
    deprecate_json_path = osp.join(mmcv.__path__[0],
                                   'model_zoo/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)

    return deprecate_urls


def _process_mmcls_checkpoint(checkpoint):
    """mmcls 체크포인트에서 backbone 부분만 추출하여 새로운 체크포인트를 만듭니다.

    mmcls 체크포인트의 state_dict 키는 'backbone.'으로 시작합니다.
    이 함수는 해당 prefix를 제거하여 backbone 가중치만 담은 state_dict를 반환합니다.
    이를 통해 mmcls로 학습한 backbone을 mmdet 등 다른 프레임워크에서 바로 사용할 수 있습니다.

    Args:
        checkpoint (dict): mmcls 형식의 체크포인트 딕셔너리.

    Returns:
        dict: 'state_dict' 키만 포함하는 새로운 체크포인트 딕셔너리.
              키에서 'backbone.' prefix가 제거됩니다.
    """
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            # 'backbone.' prefix (9글자)를 제거하여 저장합니다.
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)

    return new_checkpoint


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.

    [한국어 설명]
    다양한 소스(로컬 파일, URL, 모델 zoo, Pavi, S3/Ceph)에서 체크포인트를 로드합니다.
    filename의 prefix에 따라 적절한 로드 방법을 자동으로 선택합니다:
      - 'modelzoo://'  : torchvision 모델 zoo (deprecated, 'torchvision://'으로 대체됨)
      - 'torchvision://': torchvision 사전 학습 모델
      - 'open-mmlab://' : Open-MMLab 공식 모델 zoo
      - 'mmcls://'      : MMClassification 모델 zoo
      - 'http://', 'https://': 일반 URL
      - 'pavi://'       : SenseTime PAVI 모델 클라우드
      - 's3://'         : Ceph 분산 파일 시스템
      - 그 외           : 로컬 파일 경로

    Args:
        filename (str): 체크포인트 파일 경로 또는 URI.
        map_location: 텐서를 로드할 장치. torch.load의 map_location과 동일.

    Returns:
        dict | OrderedDict: 로드된 체크포인트.
    """
    if filename.startswith('modelzoo://'):
        # 'modelzoo://' 접두사는 deprecated, torchvision://을 사용해야 함
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please '
                      'use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        # torchvision 공식 사전 학습 모델 로드
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        # Open-MMLab 공식 모델 zoo에서 로드
        model_urls = get_external_models()
        model_name = filename[13:]
        deprecated_urls = get_deprecated_model_names()
        if model_name in deprecated_urls:
            # deprecated 모델 이름이면 새 이름으로 안내합니다.
            warnings.warn(f'open-mmlab://{model_name} is deprecated in favor '
                          f'of open-mmlab://{deprecated_urls[model_name]}')
            model_name = deprecated_urls[model_name]
        model_url = model_urls[model_name]
        # check if is url
        if model_url.startswith(('http://', 'https://')):
            checkpoint = load_url_dist(model_url)
        else:
            # 로컬 경로인 경우 mmcv 홈 디렉터리 기준으로 파일을 로드합니다.
            filename = osp.join(_get_mmcv_home(), model_url)
            if not osp.isfile(filename):
                raise IOError(f'{filename} is not a checkpoint file')
            checkpoint = torch.load(filename, map_location=map_location)
    elif filename.startswith('mmcls://'):
        # MMClassification 모델 zoo에서 로드하고 backbone prefix를 제거합니다.
        model_urls = get_mmcls_models()
        model_name = filename[8:]
        checkpoint = load_url_dist(model_urls[model_name])
        checkpoint = _process_mmcls_checkpoint(checkpoint)
    elif filename.startswith(('http://', 'https://')):
        # 일반 HTTP/HTTPS URL에서 분산 방식으로 로드합니다.
        checkpoint = load_url_dist(filename)
    elif filename.startswith('pavi://'):
        # PAVI 모델 클라우드에서 로드합니다.
        model_path = filename[7:]
        checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
    elif filename.startswith('s3://'):
        # Ceph(S3 호환) 분산 파일 시스템에서 로드합니다.
        checkpoint = load_fileclient_dist(
            filename, backend='ceph', map_location=map_location)
    else:
        # 로컬 파일 경로로부터 로드합니다.
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.

    [한국어 설명]
    모델에 체크포인트를 로드하는 메인 함수입니다. mmcv 원본 대비 다음이 추가되었습니다:
      1. 'model' 키 지원: 일부 체크포인트는 'state_dict' 대신 'model' 키를 사용합니다.
      2. MoBY encoder 지원: 자기지도학습 모델(MoBY)의 online encoder 가중치를 추출합니다.
      3. absolute_pos_embed 처리: 절대 위치 임베딩 크기가 다르면 경고 후 스킵합니다.
      4. relative_position_bias_table 보간: 상대 위치 편향 테이블 크기가 다르면
         bicubic 보간으로 현재 모델 크기에 맞게 리사이즈합니다.
         (예: 224x224로 학습된 모델을 384x384 입력에 맞게 조정)

    Args:
        model (Module): 가중치를 로드할 대상 모델.
        filename (str): 체크포인트 파일 경로 또는 URI.
        map_location (str): 텐서를 로드할 장치. 기본값은 'cpu'.
        strict (bool): True이면 키가 완전히 일치해야 합니다.
        logger: 오류/경고 메시지를 출력할 로거.

    Returns:
        dict | OrderedDict: 로드된 체크포인트 딕셔너리.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    # 체크포인트 딕셔너리에서 실제 가중치를 담은 state_dict를 추출합니다.
    if 'state_dict' in checkpoint:
        # 표준 mmcv/mmdet 체크포인트 형식
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # 일부 모델(예: Swin Transformer 공식 체크포인트)은 'model' 키를 사용합니다.
        state_dict = checkpoint['model']
    else:
        # 키 없이 state_dict 자체가 체크포인트인 경우
        state_dict = checkpoint
    # strip prefix of state_dict
    # DataParallel/DistributedDataParallel 사용 시 'module.' prefix가 붙으므로 제거합니다.
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    # MoBY는 self-supervised learning 방법으로, online/target 두 개의 encoder를 사용합니다.
    # downstream fine-tuning 시에는 online branch(encoder)의 가중치만 사용합니다.
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    # reshape absolute position embedding
    # Swin Transformer 등에서 사용하는 절대 위치 임베딩(absolute_pos_embed)을 처리합니다.
    # 사전 학습 시의 이미지 크기와 fine-tuning 시의 크기가 다른 경우 크기 불일치가 발생합니다.
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H*W:
            # 크기가 맞지 않으면 경고를 출력하고 해당 키는 로드하지 않습니다.
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            # 크기가 맞으면 (N, L, C) -> (N, H, W, C) -> (N, C, H, W) 형태로 변환합니다.
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)

    # interpolate position bias table if needed
    # Swin Transformer의 relative_position_bias_table을 처리합니다.
    # 사전 학습 해상도와 fine-tuning 해상도가 다르면 윈도우 크기도 달라져 테이블 크기가 다릅니다.
    # 이때 bicubic 보간을 통해 현재 모델 크기에 맞게 리사이즈합니다.
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]   # 사전 학습된 편향 테이블
        table_current = model.state_dict()[table_key]  # 현재 모델의 편향 테이블
        L1, nH1 = table_pretrained.size()  # (테이블 크기, 헤드 수) - 사전 학습
        L2, nH2 = table_current.size()     # (테이블 크기, 헤드 수) - 현재 모델
        if nH1 != nH2:
            # 헤드 수가 다르면 보간 불가 - 경고 후 스킵합니다.
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                # 테이블 크기가 다르면 bicubic 보간으로 현재 크기에 맞게 조정합니다.
                # (L, nH) -> (nH, S1, S1) -> bicubic -> (nH, S2, S2) -> (L2, nH2)
                S1 = int(L1 ** 0.5)  # 사전 학습 윈도우 크기 (1D -> 2D)
                S2 = int(L2 ** 0.5)  # 현재 모델 윈도우 크기
                table_pretrained_resized = F.interpolate(
                     table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                     size=(S2, S2), mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # load state_dict
    # 처리된 state_dict를 모델에 로드합니다.
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.

    [한국어 설명]
    GPU에 있는 모델 가중치를 CPU로 복사합니다.
    체크포인트를 저장할 때 GPU 텐서를 직접 저장하면 특정 GPU에 종속되므로,
    CPU로 이동한 후 저장하는 것이 일반적입니다.

    Args:
        state_dict (OrderedDict): GPU에 위치한 모델 가중치 딕셔너리.

    Returns:
        OrderedDict: CPU로 복사된 모델 가중치 딕셔너리.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.

    [한국어 설명]
    모듈의 파라미터와 버퍼(buffer)를 destination 딕셔너리에 저장합니다.
    PyTorch 기본 _save_to_state_dict와 달리, _non_persistent_buffers_set 체크를 제거하여
    nn.BatchNorm2d의 모든 버퍼(running_mean, running_var, num_batches_tracked)가
    항상 저장되도록 수정되었습니다.

    Args:
        module (nn.Module): state_dict를 생성할 모듈.
        destination (dict): 상태가 저장될 딕셔너리.
        prefix (str): 파라미터와 버퍼 키에 붙일 prefix.
        keep_vars (bool): True이면 텐서를 Variable로 유지, False이면 detach합니다.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        # _non_persistent_buffers_set 체크를 제거하여 모든 버퍼를 저장합니다.
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.

    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.

    Returns:
        dict: A dictionary containing a whole state of the module.

    [한국어 설명]
    모듈 전체의 state_dict를 반환합니다. PyTorch 기본 state_dict()와 달리
    DDP 등 중첩된 래퍼 모듈도 재귀적으로 처리합니다.
    이는 복잡한 구조(예: nn.Module(nn.Module(DDP)))를 가진 모델 저장 시 필요합니다.

    Args:
        module (nn.Module): state_dict를 생성할 모듈.
        destination (OrderedDict): 결과를 저장할 딕셔너리. None이면 새로 생성합니다.
        prefix (str): 키에 붙일 prefix.
        keep_vars (bool): True이면 Variable 속성을 유지합니다. 기본값: False.

    Returns:
        dict: 모듈 전체 상태를 담은 딕셔너리.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    # DDP 등 래퍼 모듈이면 내부 실제 모듈을 꺼냅니다.
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    # 버전 정보를 메타데이터에 저장합니다.
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    # 현재 모듈의 파라미터와 버퍼를 저장합니다.
    _save_to_state_dict(module, destination, prefix, keep_vars)
    # 하위 모듈에 대해 재귀적으로 state_dict를 수집합니다.
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars)
    # state_dict hook을 실행합니다 (사용자 정의 후처리 지원).
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.

    [한국어 설명]
    모델 체크포인트를 파일로 저장합니다.
    저장되는 체크포인트의 구성:
      - 'meta': mmcv 버전, 저장 시각, 클래스 이름 등 메타 정보
      - 'state_dict': CPU로 복사된 모델 가중치 (GPU 종속성 제거)
      - 'optimizer': 옵티마이저 상태 (학습 재개 시 필요)

    파일 저장 위치:
      - 'pavi://'로 시작하면 PAVI 모델 클라우드에 업로드합니다.
      - 그 외에는 로컬 파일 시스템에 저장합니다.

    Args:
        model (Module): 저장할 모델.
        filename (str): 저장할 파일 경로 또는 'pavi://...' URI.
        optimizer (Optimizer | dict | None): 저장할 옵티마이저. 복수 옵티마이저는 dict로 전달.
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
        # 탐지/분류 모델의 클래스 이름을 메타에 저장합니다 (모델 재사용 시 활용).
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        # 모델 가중치를 CPU로 옮긴 후 state_dict로 저장합니다.
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        # 단일 옵티마이저인 경우 state_dict를 직접 저장합니다.
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        # 복수 옵티마이저(예: GAN)인 경우 각각의 state_dict를 딕셔너리로 저장합니다.
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

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
