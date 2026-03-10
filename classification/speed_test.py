"""
Testing the speed of different models
다양한 EfficientViT 모델 변형(M0~M5)의 추론 처리량(throughput)을 측정하는 벤치마크 스크립트.
CPU와 CUDA(GPU) 환경 모두에서 처리량(images/sec)을 측정한다.
"""
import os
import torch
import torchvision
import time
import timm
from model.build import EfficientViT_M0, EfficientViT_M1, EfficientViT_M2, EfficientViT_M3, EfficientViT_M4, EfficientViT_M5
import torchvision
import utils

# 그래디언트 계산 비활성화: 추론(inference) 전용 벤치마크이므로 역전파 불필요
# 메모리 사용량 감소 및 연산 속도 향상 효과
torch.autograd.set_grad_enabled(False)


# 웜업(warmup) 단계 지속 시간: 10초
# JIT 컴파일, CUDA 커널 로딩, 메모리 할당 등 초기 오버헤드를 제거하기 위해
# 실제 측정 전 충분히 워밍업
T0 = 10
# 실제 측정 단계 지속 시간: 60초
# 충분히 긴 측정 시간으로 통계적으로 안정된 평균 처리량을 얻음
T1 = 60


def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    """
    CPU 환경에서 모델의 추론 처리량(throughput)을 측정하는 함수.

    [측정 방법]
    1. 웜업(Warmup) 단계 (T0=10초):
       CPU 상태 안정화 및 PyTorch 내부 최적화(JIT 트레이싱 등)가
       완전히 적용될 때까지 반복 추론을 수행한다.
       초기 몇 번의 실행은 캐싱, 메모리 할당 등으로 느릴 수 있으므로
       이 단계의 결과는 측정에서 제외한다.

    2. 측정 단계 (T1=60초 누적):
       실제 추론 시간을 반복 측정하여 타이밍 리스트에 축적한다.
       누적 시간이 T1(60초)에 도달할 때까지 반복하여
       충분한 샘플 수로 안정적인 평균을 구한다.

    3. 처리량 계산:
       batch_size / mean(timing) = 초당 처리 이미지 수 (images/s)

    주의사항:
    - CPU 벤치마크는 스레드 수를 1로 설정(torch.set_num_threads(1))한 상태에서
      실행된다 (메인 루프에서 설정). 멀티스레드 성능이 아닌 단일 스레드 성능을 측정.
    - CPU에서는 CUDA synchronize가 불필요하므로 time.time()으로 직접 측정.

    Args:
        name (str): 모델 이름 (출력용, 예: 'EfficientViT_M0').
        model (torch.nn.Module): 벤치마크할 모델 (eval 모드, JIT 트레이싱 완료).
        device (str): 디바이스 문자열 (예: 'cpu').
        batch_size (int): 한 번에 처리할 이미지 수. CPU에서는 16으로 고정.
        resolution (int): 입력 이미지 해상도 (정방형). 기본값 224.

    Returns:
        None. 결과를 stdout에 출력:
            "<name> <device> <throughput> images/s @ batch size <batch_size>"
    """
    # 무작위 입력 텐서 생성: (batch_size, 3, resolution, resolution)
    # 실제 이미지 데이터가 아닌 랜덤 텐서를 사용하여 데이터 로딩 오버헤드 제거
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)

    # warmup
    # 웜업 단계: T0(10초) 동안 추론을 반복하여 초기 오버헤드 제거
    # CPU JIT 최적화, 메모리 할당 안정화를 위해 필수
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    # 실제 측정 단계: 누적 측정 시간이 T1(60초)에 도달할 때까지 반복
    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        # 각 배치의 처리 시간을 리스트에 추가
        timing.append(time.time() - start)

    # 측정 결과를 PyTorch 텐서로 변환하여 통계 계산
    timing = torch.as_tensor(timing, dtype=torch.float32)
    # 처리량 = 배치 크기 / 평균 처리 시간 (images/s)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)


def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    """
    CUDA(GPU) 환경에서 모델의 추론 처리량(throughput)을 측정하는 함수.

    [CUDA 벤치마크의 핵심: torch.cuda.synchronize()]
    GPU는 비동기적으로 연산을 수행한다. CPU에서 model(inputs)를 호출하면
    GPU에 작업을 제출(enqueue)하고 즉시 반환되므로, GPU 연산이 완료되지 않은
    시점에 time.time()을 측정하면 부정확한 결과가 나온다.
    torch.cuda.synchronize()는 GPU의 모든 보류 중인 CUDA 커널이 완료될 때까지
    CPU를 블록(block)하여 정확한 종단간(end-to-end) 지연시간을 측정하게 한다.

    [AMP (Automatic Mixed Precision)]
    torch.cuda.amp.autocast()를 사용하여 FP16 혼합 정밀도로 실행한다.
    - Tensor Core가 있는 GPU에서 FP16 연산은 FP32보다 2~8배 빠름
    - 메모리 사용량 감소로 더 큰 배치 크기 처리 가능
    - 실제 배포 환경과 동일한 조건에서 측정

    [측정 방법]
    1. 캐시 초기화: torch.cuda.empty_cache()로 GPU 메모리 단편화 제거
    2. 동기화: torch.cuda.synchronize()로 타이머 시작 전 GPU 상태 안정화
    3. 웜업(T0=10초): CUDA 커널 컴파일/캐싱, cuDNN 알고리즘 선택 완료까지 반복
       (CUDA 첫 실행 시 커널 JIT 컴파일로 인한 초기 지연 제거)
    4. 동기화 후 측정(T1=60초):
       - 각 배치 후 synchronize()로 정확한 GPU 완료 시간 확보
       - 누적 시간 T1 달성까지 반복 측정

    주의사항:
    - GPU 벤치마크에서 synchronize() 없이 측정하면 CPU-GPU 비동기성으로 인해
      실제보다 훨씬 빠른(과대 추정된) 처리량이 측정될 수 있다.
    - 배치 크기 2048은 GPU 메모리가 충분한 경우의 최대 처리량 측정용이다.
      실제 학습에서는 그래디언트와 중간 활성화 저장으로 훨씬 작은 배치를 사용.
    - torch.cuda.empty_cache()는 CUDA 캐시된 메모리를 해제하여 이전 모델의
      메모리가 현재 측정에 영향을 주지 않도록 한다.

    Args:
        name (str): 모델 이름 (출력용, 예: 'EfficientViT_M0').
        model (torch.nn.Module): 벤치마크할 모델 (eval 모드, JIT 트레이싱 완료).
        device (str): 디바이스 문자열 (예: 'cuda:0').
        batch_size (int): 한 번에 처리할 이미지 수. GPU에서는 2048로 설정.
        resolution (int): 입력 이미지 해상도 (정방형). 기본값 224.

    Returns:
        None. 결과를 stdout에 출력:
            "<name> <device> <throughput> images/s @ batch size <batch_size>"
    """
    # 무작위 입력 텐서를 GPU 메모리에 직접 생성
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # 이전 측정에서 남은 GPU 캐시 메모리 해제 (메모리 상태 초기화)
    torch.cuda.empty_cache()
    # GPU 동기화: 타이머 시작 전 모든 GPU 연산이 완료됨을 보장
    torch.cuda.synchronize()
    start = time.time()

    # 웜업 단계 (AMP 활성화): T0(10초) 동안 반복하여 CUDA 커널 초기화
    # CUDA 커널의 JIT 컴파일, cuDNN 컨볼루션 알고리즘 선택, GPU 클럭 안정화
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)

    timing = []
    # 웜업 완료 후 GPU 동기화: 측정 시작 전 웜업의 모든 연산 완료 대기
    if device == 'cuda:0':
        torch.cuda.synchronize()

    # 실제 측정 단계 (AMP 활성화): 누적 시간 T1(60초) 달성까지 반복
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            # 핵심: 각 배치 후 GPU 동기화로 실제 GPU 완료 시간 정확히 측정
            # 이 synchronize() 없이는 GPU 비동기 실행으로 인해 시간이 과소 측정됨
            torch.cuda.synchronize()
            timing.append(time.time() - start)

    # 측정 결과를 텐서로 변환하여 평균 계산
    timing = torch.as_tensor(timing, dtype=torch.float32)
    # 처리량 = 배치 크기 / 평균 처리 시간 (images/s)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)


# ==============================================================================
# 메인 벤치마크 루프
# CUDA와 CPU 순서로 각 EfficientViT 모델 변형(M0~M5)의 처리량을 측정한다.
# ==============================================================================
for device in ['cuda:0', 'cpu']:

    # CUDA 디바이스가 요청되었지만 사용 불가능한 경우 건너뜀
    if 'cuda' in device and not torch.cuda.is_available():
        print("no cuda")
        continue

    if device == 'cpu':
        # CPU 환경 설정:
        # 시스템의 프로세서 수와 CPU 모델명을 출력하여 벤치마크 환경 기록
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 cpu thread')
        # 단일 스레드로 제한: 공정한 비교를 위해 CPU 코어 수의 영향 제거
        # 단일 스레드 성능은 엣지 디바이스나 서버의 단일 코어 배포 시나리오와 유사
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu
    else:
        # GPU 환경: 현재 CUDA 디바이스명 출력하여 벤치마크 환경 기록
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        compute_throughput = compute_throughput_cuda

    # EfficientViT 모델 변형 목록과 각 설정 (이름, 배치 크기, 해상도)
    # 모든 모델은 224x224 입력 해상도와 GPU에서 배치 크기 2048 사용
    for n, batch_size0, resolution in [
        ('EfficientViT_M0', 2048, 224),
        ('EfficientViT_M1', 2048, 224),
        ('EfficientViT_M2', 2048, 224),
        ('EfficientViT_M3', 2048, 224),
        ('EfficientViT_M4', 2048, 224),
        ('EfficientViT_M5', 2048, 224),
    ]:

        if device == 'cpu':
            # CPU는 메모리 제약과 처리 속도를 고려하여 배치 크기 16으로 축소
            batch_size = 16
        else:
            # GPU는 최대 처리량 측정을 위해 대용량 배치(2048) 사용
            batch_size = batch_size0
            # 이전 모델의 GPU 캐시 메모리를 해제하여 메모리 부족 방지
            torch.cuda.empty_cache()

        # 더미 입력 텐서 생성 (JIT 트레이싱에 사용)
        inputs = torch.randn(batch_size, 3, resolution,
                             resolution, device=device)

        # 모델 이름 문자열로 해당 모델 클래스 인스턴스화 (eval()로 평가 모드 설정)
        # num_classes=1000: ImageNet 1000 클래스 분류 설정
        model = eval(n)(num_classes=1000)

        # BatchNorm을 추론 최적화된 형태로 교체 (예: BN을 고정된 선형 연산으로 변환)
        # 추론 시 BatchNorm의 이동 평균 통계를 사용하는 최적화
        utils.replace_batchnorm(model)

        model.to(device)   # 모델을 지정 디바이스(CPU/GPU)로 이동
        model.eval()       # Dropout, BatchNorm 등을 추론 모드로 전환

        # TorchScript JIT 트레이싱: 모델을 정적 그래프로 컴파일
        # - Python 인터프리터 오버헤드 제거
        # - 연산자 융합(operator fusion) 및 최적화 적용
        # - 특히 CPU에서 큰 속도 향상 효과
        # 주의: trace는 고정된 입력 형태(shape)에만 유효하며,
        #       동적 분기(if문 등)는 트레이싱 시의 경로로 고정됨
        model = torch.jit.trace(model, inputs)

        # 벤치마크 실행: 웜업 후 T1초 동안 처리량 측정
        compute_throughput(n, model, device,
                           batch_size, resolution=resolution)
