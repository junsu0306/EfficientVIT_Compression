"""
Misc functions, including distributed helpers and model loaders
Also include a model loader specified for finetuning EfficientViT

분산 학습 지원 유틸리티, 모델 로더, 메트릭 추적 클래스 등을 포함하는 모듈.
EfficientViT 파인튜닝을 위한 특수 모델 로더(attention bias 보간)도 포함한다.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.

    [한국어 설명]
    역할:
        시계열 수치(손실값, 정확도 등)를 추적하며 이동 윈도우 통계와
        전체 평균을 계산하는 경량 통계 추적기.

    핵심 동작:
        - deque(maxlen=window_size): 최근 N개의 값만 유지하는 고정 크기 큐
          오래된 값은 자동으로 제거되어 최근 트렌드를 반영한 통계 제공
        - total/count: 전체 누적 합계와 카운트 (글로벌 평균 계산용)
          분산 학습에서 all_reduce로 모든 프로세스의 값을 합산할 때 사용

    입력:
        window_size: 이동 윈도우 크기 (최근 N개 값만 사용, 기본값 20)
        fmt: 문자열 출력 포맷 (기본값: "median (global_avg)")

    출력:
        median, avg, global_avg, max, value 프로퍼티로 다양한 통계값 접근 가능
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            # 기본 출력 포맷: "윈도우 중앙값 (전체 평균)" 형태로 두 통계 동시 표시
            fmt = "{median:.4f} ({global_avg:.4f})"
        # deque: 최대 window_size개 원소만 유지하는 양방향 큐
        # maxlen 초과 시 왼쪽(오래된) 원소를 자동 제거 -> O(1) 삽입/삭제
        self.deque = deque(maxlen=window_size)
        self.total = 0.0   # 전체 누적 합계 (global_avg 계산용)
        self.count = 0     # 전체 누적 샘플 수 (n 가중 카운트)
        self.fmt = fmt     # __str__ 출력 포맷 문자열

    def update(self, value, n=1):
        """
        [한국어 설명]
        새 값을 추적기에 추가한다.

        파라미터:
            value: 추가할 새 수치 (예: 배치 손실값, 배치 정확도)
            n    : 이 값이 대표하는 샘플 수 (기본값 1).
                   정확도처럼 배치 크기로 가중 평균을 낼 때 batch_size를 전달.
                   total += value * n, count += n 으로 저장하여 나중에
                   global_avg = total / count로 올바른 가중 평균 계산 가능.
        """
        # 이동 윈도우 큐에 최신 값 추가 (maxlen 초과 시 오래된 값 자동 제거)
        self.deque.append(value)
        # 전체 누적 카운트와 합계 업데이트 (n으로 배치 가중치 반영)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!

        [한국어 설명]
        역할:
            분산 학습 시 모든 GPU 프로세스 간에 count와 total을 동기화한다.
            각 프로세스는 자신이 처리한 배치만의 통계를 보유하므로,
            전체 데이터셋 기준의 global_avg를 구하려면 합산이 필요하다.

        주의사항:
            deque(이동 윈도우)는 동기화하지 않는다!
            median, avg 같은 윈도우 기반 통계는 프로세스마다 다를 수 있음.
            global_avg만이 분산 동기화 후 정확한 전체 평균값이 된다.

        분산 학습에서 이렇게 하는 이유:
            - 각 GPU가 전체 데이터의 일부만 처리하므로 단순 평균은 부정확함
            - dist.all_reduce(SUM)로 모든 프로세스의 count/total을 합산하면
              마치 단일 프로세스가 전체 데이터를 처리한 것과 동일한 통계를 얻음
            - 예) GPU0: count=100, total=50 / GPU1: count=100, total=60
              -> all_reduce 후: count=200, total=110 -> global_avg=0.55
        """
        if not is_dist_avail_and_initialized():
            # 분산 학습이 초기화되지 않은 경우(단일 GPU 학습) 동기화 불필요
            return
        # count와 total을 하나의 CUDA 텐서에 묶어 한 번의 통신으로 처리
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        # dist.barrier(): 모든 프로세스가 이 지점에 도달할 때까지 대기 (동기화 포인트)
        dist.barrier()
        # dist.all_reduce(t): 기본 연산은 SUM - 모든 프로세스의 t를 합산하여 각 프로세스에 결과 반영
        dist.all_reduce(t)
        t = t.tolist()
        # 합산된 전체 count와 total로 업데이트
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """
        [한국어 설명]
        이동 윈도우 내 값들의 중앙값 반환.
        이상치(outlier)에 강인한 통계값으로, 학습 초반 급등하는 손실값에 영향받지 않음.
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """
        [한국어 설명]
        이동 윈도우 내 값들의 산술 평균 반환.
        최근 window_size개 배치의 평균 - 전체 평균보다 최근 트렌드를 더 빠르게 반영.
        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """
        [한국어 설명]
        학습 시작부터 현재까지의 전체 가중 평균 반환.
        total / count로 계산되며, 분산 동기화 후에는 전체 GPU의 통합 평균이 된다.
        에폭 종료 후 최종 성능 지표로 사용된다.
        """
        return self.total / self.count

    @property
    def max(self):
        """
        [한국어 설명]
        이동 윈도우 내 최댓값 반환.
        학습률 스케줄러 디버깅이나 손실 스파이크 감지에 유용.
        """
        return max(self.deque)

    @property
    def value(self):
        """
        [한국어 설명]
        가장 최근에 추가된 값 반환 (deque의 마지막 원소).
        학습률처럼 이동 평균이 아닌 현재값을 그대로 표시할 때 사용.
        SmoothedValue(window_size=1, fmt='{value:.6f}')와 함께 lr 추적에 활용.
        """
        return self.deque[-1]

    def __str__(self):
        """
        [한국어 설명]
        fmt 포맷 문자열에 따라 통계값을 문자열로 반환.
        MetricLogger.__str__에서 각 메트릭의 현재 상태를 출력할 때 호출된다.
        """
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """
    [한국어 설명]
    역할:
        학습/평가 중 다양한 메트릭(손실, 정확도, 학습률 등)을 수집하고
        주기적으로 콘솔에 출력하는 통합 메트릭 관리자.

    핵심 동작:
        - defaultdict(SmoothedValue): 새 메트릭 이름이 처음 등록될 때 자동으로
          SmoothedValue 인스턴스를 생성하므로 사전 등록 없이 바로 사용 가능
        - log_every: 데이터 로더 이터레이터를 감싸 배치 처리 시간, ETA, 메모리 사용량 등
          풍부한 진행 정보를 주기적으로 출력하는 제너레이터 함수

    입력:
        delimiter: 여러 메트릭을 출력할 때 사용하는 구분자 (기본값 탭 "\t")
    """

    def __init__(self, delimiter="\t"):
        # defaultdict: 존재하지 않는 키 접근 시 SmoothedValue()를 자동 생성
        # 이를 통해 update(loss=0.5) 같은 호출로 즉시 새 메트릭 등록 가능
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter  # 출력 시 메트릭 간 구분자

    def update(self, **kwargs):
        """
        [한국어 설명]
        역할:
            키워드 인자로 전달된 메트릭값들을 해당 SmoothedValue에 업데이트.

        파라미터:
            **kwargs: 메트릭 이름과 값의 쌍 (예: loss=0.5, lr=0.001)
                      값이 텐서인 경우 .item()으로 스칼라 변환 후 저장.

        사용 예:
            metric_logger.update(loss=loss_value, lr=current_lr)
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                # 텐서에서 파이썬 스칼라로 변환 (GPU 텐서도 CPU로 이동하여 값 추출)
                v = v.item()
            assert isinstance(v, (float, int))
            # defaultdict이므로 새 메트릭은 자동으로 SmoothedValue()로 초기화됨
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        [한국어 설명]
        역할:
            메트릭 이름으로 직접 속성처럼 접근할 수 있게 해주는 매직 메서드.

        동작:
            metric_logger.acc1 -> metric_logger.meters['acc1'] (SmoothedValue 반환)
            metric_logger.loss -> metric_logger.meters['loss'] (SmoothedValue 반환)

        이렇게 하는 이유:
            engine.py에서 metric_logger.acc1.global_avg 형태로 간결하게 접근 가능.
            meters 딕셔너리를 직접 인덱싱하지 않아도 되어 코드 가독성 향상.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """
        [한국어 설명]
        모든 메트릭을 delimiter로 연결한 문자열 반환.
        log_every 내부에서 진행 상황 출력 시 {meters} 자리에 삽입된다.
        출력 예: "loss: 0.3421 (0.4123)  lr: 0.000100"
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        [한국어 설명]
        역할:
            등록된 모든 메트릭(SmoothedValue)의 분산 동기화를 수행.
            에폭 종료 후 호출하여 모든 GPU의 통계를 합산한다.

        분산 학습에서 이렇게 하는 이유:
            각 GPU는 전체 데이터의 1/N만 처리하므로 에폭 평균은 부정확하다.
            동기화 후에는 모든 GPU에서 동일한 전체 평균값을 사용할 수 있다.
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        [한국어 설명]
        역할:
            특정 설정의 SmoothedValue를 수동으로 등록한다.
            defaultdict의 기본 SmoothedValue 대신 커스텀 설정(window_size, fmt)이
            필요한 메트릭(예: lr)을 등록할 때 사용한다.

        사용 예:
            metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            -> lr은 이동 평균 대신 현재값을 6자리 소수로 출력
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        [한국어 설명]
        역할:
            데이터 로더 이터레이터를 감싸 배치 처리 진행 상황을 주기적으로 출력하는 제너레이터.
            각 배치를 yield하기 전후로 타이밍을 측정하여 데이터 로딩 시간과 처리 시간을 분리 추적.

        파라미터:
            iterable  : 데이터 로더 (len()을 지원해야 ETA 계산 가능)
            print_freq: 몇 번째 배치마다 진행 상황을 출력할지 결정하는 빈도
            header    : 출력 맨 앞에 붙이는 레이블 (예: 'Epoch: [5]', 'Test:')

        제너레이터 동작 원리:
            1. end = time.time() 기록
            2. yield obj: 호출자(학습 루프)에 배치 데이터 전달 후 일시 중지
            3. 호출자가 배치를 처리하고 다음 next()를 호출하면 재개
            4. iter_time.update(time.time() - end): 배치 처리 시간 측정
            5. data_time.update(time.time() - end): 다음 배치 데이터 로딩 시간 측정

        시간 측정 구조:
            [데이터 로딩] -> yield -> [배치 처리(학습/평가)] -> [다음 데이터 로딩] -> ...
            |<-- data_time -->|          |<---------- iter_time ---------->|
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()  # 전체 에폭 시작 시간 (총 소요 시간 계산용)
        end = time.time()         # 배치 단위 타이밍 기준점

        # iter_time: 배치 1개 처리에 걸리는 전체 시간 (데이터 로딩 + 순전파 + 역전파)
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        # data_time: 데이터 로더에서 배치를 가져오는 데 걸리는 시간
        #            data_time이 크면 데이터 로딩이 병목임을 의미 -> num_workers 증가 검토
        data_time = SmoothedValue(fmt='{avg:.4f}')

        # 배치 인덱스 출력 포맷: 전체 배치 수의 자릿수에 맞춰 오른쪽 정렬
        # 예) 전체 100 배치라면 space_fmt=':3d' -> '  1/100', ' 10/100', '100/100'
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        # 출력 메시지 템플릿 구성
        log_msg = [
            header,                          # 에폭 레이블
            '[{0' + space_fmt + '}/{1}]',   # 현재 배치 / 전체 배치
            'eta: {eta}',                    # 예상 잔여 시간 (Estimated Time of Arrival)
            '{meters}',                      # MetricLogger의 모든 메트릭 문자열
            'time: {time}',                  # 배치 처리 평균 시간
            'data: {data}'                   # 데이터 로딩 평균 시간
        ]
        if torch.cuda.is_available():
            # CUDA 환경에서는 최대 GPU 메모리 사용량도 추가 출력 (MB 단위)
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0  # 바이트 -> MB 변환 상수

        for obj in iterable:
            # yield 직전: 데이터 로더가 배치를 준비하는 데 걸린 시간 측정
            data_time.update(time.time() - end)

            # 호출자(학습 루프)에 배치 데이터 전달
            # 이 지점에서 제어권이 호출자로 넘어가 실제 학습/평가가 수행됨
            yield obj

            # yield 복귀 후: 배치 전체 처리 시간 측정 (데이터 로딩 + 학습/평가)
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                # ETA 계산: 평균 배치 처리 시간 * 남은 배치 수
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        # torch.cuda.max_memory_allocated(): 현재까지 최대 GPU 메모리 사용량 (바이트)
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()  # 다음 배치 타이밍 기준점 갱신

        # 에폭 전체 소요 시간 출력
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object

    [한국어 설명]
    역할:
        이미 메모리에 로드된 체크포인트 딕셔너리를 ModelEma._load_checkpoint에
        전달하기 위한 우회(workaround) 함수.

    문제:
        ModelEma._load_checkpoint는 파일 경로(path) 또는 파일 객체(file-like object)만
        인자로 받을 수 있어, 이미 로드된 딕셔너리를 직접 전달할 수 없다.

    해결책:
        io.BytesIO()를 인메모리 파일 버퍼로 사용:
        1. torch.save(checkpoint, mem_file): 딕셔너리를 바이트 스트림으로 직렬화
        2. mem_file.seek(0): 읽기 포인터를 처음으로 이동 (쓴 후 읽으려면 필수)
        3. model_ema._load_checkpoint(mem_file): 메모리 버퍼를 파일처럼 전달

    파라미터:
        model_ema  : EMA 모델 인스턴스 (timm.utils.ModelEma)
        checkpoint : 이미 torch.load()로 로드된 체크포인트 딕셔너리
    """
    mem_file = io.BytesIO()              # 디스크 I/O 없이 메모리에서 파일처럼 사용
    torch.save(checkpoint, mem_file)     # 딕셔너리를 pickle 형식으로 바이트 스트림에 저장
    mem_file.seek(0)                     # 읽기 커서를 스트림 시작점으로 이동
    model_ema._load_checkpoint(mem_file) # 메모리 버퍼를 파일 객체로 전달하여 EMA 가중치 로드


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process

    [한국어 설명]
    역할:
        분산 학습에서 마스터 프로세스(rank=0)만 콘솔 출력을 수행하도록 설정.
        나머지 워커 프로세스들의 출력을 모두 억제하여 로그가 중복 출력되지 않게 한다.

    분산 학습에서 이렇게 하는 이유:
        N개의 GPU가 동시에 실행되면 동일한 print() 문이 N번 출력되어 로그가 혼잡해진다.
        rank=0인 마스터 프로세스만 출력을 허용하면 깔끔한 단일 로그 스트림을 유지할 수 있다.
        force=True로 전달하면 워커 프로세스에서도 강제 출력 가능 (긴급 디버깅 등에 활용).

    파라미터:
        is_master: True면 마스터 프로세스 (rank=0), False면 워커 프로세스

    동작 원리:
        Python의 builtins.print를 래퍼 함수로 교체하여
        is_master=False일 때는 아무것도 출력하지 않는 방식으로 동작.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print  # 원본 print 함수 저장

    def print(*args, **kwargs):
        # force 키워드: True면 워커 프로세스에서도 강제 출력 가능
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    # 전역 print 함수를 래퍼로 교체 - 이후 모든 print() 호출은 이 래퍼를 통함
    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    [한국어 설명]
    역할:
        분산 학습 환경이 사용 가능하고 초기화되었는지 확인하는 헬퍼 함수.

    반환값:
        True  - torch.distributed가 사용 가능하고 초기화된 상태 (분산 학습 중)
        False - 단일 프로세스 환경 또는 분산 초기화 전

    이렇게 하는 이유:
        단일 GPU와 다중 GPU 학습을 동일한 코드로 처리하기 위해
        분산 연산(all_reduce, barrier 등) 전에 항상 이 함수로 환경을 확인한다.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    [한국어 설명]
    역할:
        분산 학습에 참여하는 전체 프로세스(GPU) 수 반환.

    반환값:
        분산 학습 중: dist.get_world_size() (예: 8개 GPU면 8 반환)
        단일 GPU 또는 미초기화 환경: 1

    활용 예:
        effective_batch_size = batch_size_per_gpu * get_world_size()
        학습률 스케일링: lr = base_lr * get_world_size() (Linear Scaling Rule)
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    [한국어 설명]
    역할:
        현재 프로세스의 순위(rank) 반환.

    반환값:
        분산 학습 중: 0부터 world_size-1까지의 정수 (각 GPU의 고유 번호)
        단일 GPU 또는 미초기화 환경: 0

    활용 예:
        rank=0인 마스터 프로세스에서만 체크포인트 저장, 로그 출력 등을 수행하여
        파일 충돌이나 중복 출력을 방지한다.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    [한국어 설명]
    역할:
        현재 프로세스가 마스터 프로세스(rank=0)인지 확인하는 헬퍼 함수.

    반환값:
        True  - rank=0인 마스터 프로세스 (또는 단일 GPU 환경)
        False - 워커 프로세스 (rank != 0)

    활용 예:
        체크포인트 저장, wandb/tensorboard 로깅, 콘솔 출력 등을
        마스터 프로세스에서만 수행할 때 사용한다.
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    [한국어 설명]
    역할:
        마스터 프로세스(rank=0)에서만 torch.save()를 실행하는 래퍼 함수.

    분산 학습에서 이렇게 하는 이유:
        모든 GPU가 동시에 동일한 파일에 저장을 시도하면 파일 충돌이 발생한다.
        마스터 프로세스 하나만 저장을 수행하면 안전하게 체크포인트를 기록할 수 있다.
        모든 GPU는 동기화되어 같은 파라미터를 보유하므로 rank=0에서의 저장으로 충분하다.

    파라미터:
        *args, **kwargs: torch.save()에 그대로 전달되는 인자들
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    [한국어 설명]
    역할:
        분산 학습 환경을 초기화한다. 환경 변수에서 분산 설정을 읽어
        각 프로세스에 rank, world_size, GPU를 할당하고 프로세스 그룹을 초기화한다.

    파라미터:
        args: 분산 설정이 저장될 네임스페이스 객체 (argparse.Namespace 등)
              초기화 후 args.rank, args.world_size, args.gpu, args.distributed 등이 설정됨

    지원하는 실행 환경:
        1. torchrun/torch.distributed.launch:
           환경 변수 RANK, WORLD_SIZE, LOCAL_RANK를 자동으로 설정함
        2. SLURM 클러스터:
           SLURM_PROCID 환경 변수로 rank를 결정, GPU는 node의 GPU 수로 순환 할당
        3. 기타(로컬 단일 GPU):
           분산 모드 비활성화

    분산 학습 초기화 과정:
        1. GPU 할당: torch.cuda.set_device(args.gpu)
        2. 백엔드 설정: NCCL (NVIDIA GPU 간 최적화된 집합 통신 라이브러리)
        3. 프로세스 그룹 초기화: init_process_group으로 모든 프로세스 연결
        4. 배리어: 모든 프로세스가 초기화 완료할 때까지 대기
        5. 출력 제한: 마스터 프로세스(rank=0)만 로그 출력하도록 설정
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun 또는 torch.distributed.launch로 실행된 경우
        # 환경 변수에서 분산 설정 읽기
        args.rank = int(os.environ["RANK"])           # 전체 프로세스 중 현재 프로세스의 순위
        args.world_size = int(os.environ['WORLD_SIZE']) # 전체 프로세스(GPU) 수
        args.gpu = int(os.environ['LOCAL_RANK'])       # 현재 노드(서버) 내에서의 GPU 번호
    elif 'SLURM_PROCID' in os.environ:
        # SLURM 클러스터 환경 (HPC 슈퍼컴퓨터 등)에서 실행된 경우
        args.rank = int(os.environ['SLURM_PROCID'])
        # 노드당 GPU 수로 나눈 나머지로 로컬 GPU 번호 결정 (순환 할당)
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # 분산 환경 변수가 없는 경우 - 단일 GPU 또는 CPU 학습
        print('Not using distributed mode')
        args.distributed = False
        return

    # 분산 학습 활성화 플래그 설정
    args.distributed = True

    # 현재 프로세스가 사용할 GPU 지정 (다른 프로세스와 충돌 방지)
    torch.cuda.set_device(args.gpu)

    # NCCL 백엔드: NVIDIA GPU 간 통신에 최적화된 라이브러리
    # Gloo(CPU), MPI 등 다른 백엔드도 있지만 CUDA 환경에서는 NCCL이 가장 빠름
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    # 프로세스 그룹 초기화: 모든 프로세스를 하나의 통신 그룹으로 연결
    # init_method: 프로세스들이 서로를 찾는 방법 (예: 'env://', 'tcp://ip:port')
    # 이 호출이 완료되면 dist.all_reduce, dist.broadcast 등을 사용 가능
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)

    # 모든 프로세스가 초기화를 완료할 때까지 대기 (동기화 포인트)
    # 일부 프로세스가 느려도 모두 준비된 후 학습을 시작하도록 보장
    torch.distributed.barrier()

    # 마스터 프로세스(rank=0)만 로그를 출력하도록 print 함수 교체
    setup_for_distributed(args.rank == 0)


def replace_batchnorm(net):
    """
    [한국어 설명]
    역할:
        네트워크의 모든 BatchNorm2d 레이어를 Identity로 교체하거나
        fuse() 메서드가 있는 레이어는 융합(fusion)하여 추론 속도를 향상시킨다.

    사용 이유:
        - 배포/추론 시 BatchNorm을 이전 Conv 레이어에 수학적으로 융합하면
          별도의 BN 연산이 제거되어 추론 속도가 빨라진다.
        - fuse() 메서드: Conv + BN을 하나의 Conv 연산으로 합치는 메서드
          (BN의 gamma, beta, mean, var를 Conv의 weight, bias에 흡수)
        - fuse()가 없는 BN은 Identity()로 교체하여 완전히 제거

    파라미터:
        net: BatchNorm을 교체할 대상 네트워크 (재귀적으로 모든 하위 모듈 처리)
    """
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            # fuse() 메서드 존재: Conv+BN 융합 레이어 - 융합된 새 모듈로 교체
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            # 순수 BatchNorm2d: 항등 함수(Identity)로 교체하여 연산 제거
            setattr(net, child_name, torch.nn.Identity())
        else:
            # 그 외 모듈: 재귀적으로 하위 모듈도 동일하게 처리
            replace_batchnorm(child)


def replace_layernorm(net):
    """
    [한국어 설명]
    역할:
        네트워크의 모든 LayerNorm을 NVIDIA Apex의 FusedLayerNorm으로 교체한다.
        FusedLayerNorm은 CUDA 커널 레벨에서 최적화되어 표준 LayerNorm보다 빠르다.

    사용 이유:
        Transformer 기반 모델(EfficientViT 등)은 LayerNorm을 많이 사용하며,
        Apex의 FusedLayerNorm은 메모리 접근을 최소화하여 학습/추론 속도를 향상시킨다.

    파라미터:
        net: LayerNorm을 교체할 대상 네트워크 (재귀적으로 모든 하위 모듈 처리)

    의존성:
        NVIDIA Apex 라이브러리 필요 (pip install apex)
    """
    import apex
    for child_name, child in net.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            # child.weight.size(0): LayerNorm의 정규화 차원 수 (hidden_dim)
            setattr(net, child_name, apex.normalization.FusedLayerNorm(
                child.weight.size(0)))
        else:
            # 재귀적으로 하위 모듈도 동일하게 처리
            replace_layernorm(child)


def load_model(modelpath, model):
    '''
    A function to load model from a checkpoint, which is used
    for fine-tuning on a different resolution.

    [한국어 설명]
    역할:
        사전 학습된 체크포인트를 로드하여 다른 해상도의 파인튜닝에 사용하는 함수.
        EfficientViT에서 사용하는 attention_biases(상대 위치 편향)를 입력 해상도에 맞게
        바이큐빅 보간(bicubic interpolation)으로 크기를 조정한다.

    파라미터:
        modelpath: 체크포인트 파일 경로 (torch.save()로 저장된 딕셔너리)
        model    : 가중치를 로드할 대상 모델 인스턴스

    반환값:
        attention_biases가 현재 모델 해상도에 맞게 보간된 체크포인트 딕셔너리

    핵심 처리 과정:
        1. attention_bias_idxs 키 제거:
           인덱스 맵은 해상도에 의존하므로 새 해상도에서 재계산되어야 함
           -> 로드하지 않고 삭제하면 모델이 자체적으로 재생성

        2. attention_biases 보간:
           사전학습 시 (nH, L1) 크기였던 편향을 현재 모델의 (nH, L2) 크기로 조정
           L = S * S (S = 윈도우 크기 또는 시퀀스 길이의 제곱근)
           바이큐빅 보간으로 (nH, S1, S1) -> (nH, S2, S2) 후 (nH, L2)로 변환
    '''
    # 체크포인트를 CPU로 로드 (GPU 메모리 절약, 이후 model.to(device)로 이동)
    checkpoint = torch.load(modelpath, map_location='cpu')
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()
        # bicubic interpolate attention_biases if not match

    # [Step 1] attention_bias_idxs 키 제거
    # attention_bias_idxs는 각 쿼리-키 쌍의 상대 위치 인덱스 맵으로,
    # 입력 해상도(시퀀스 길이)가 달라지면 완전히 다른 인덱스가 필요하므로 삭제
    rpe_idx_keys = [
        k for k in state_dict.keys() if "attention_bias_idxs" in k]
    for k in rpe_idx_keys:
        print("deleting key: ", k)
        del state_dict[k]

    # [Step 2] attention_biases 보간
    # attention_biases: EfficientViT의 상대 위치 편향 파라미터 (nH x L 행렬)
    # nH = 헤드 수, L = 위치 관계 종류 수 (시퀀스 길이의 제곱에 비례)
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "attention_biases" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]        # 사전학습 편향 (nH1, L1)
        relative_position_bias_table_current = model_state_dict[k]     # 현재 모델 편향 (nH2, L2)
        nH1, L1 = relative_position_bias_table_pretrained.size()       # 사전학습 헤드 수, 위치 수
        nH2, L2 = relative_position_bias_table_current.size()          # 현재 모델 헤드 수, 위치 수
        if nH1 != nH2:
            # 헤드 수가 다르면 보간 불가능 - 경고 후 건너뜀 (키는 유지되어 로드 시 오류 발생 가능)
            logger.warning(f"Error in loading {k} due to different number of heads")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                # 사전학습과 현재 모델의 입력 해상도(시퀀스 길이)가 다른 경우 보간
                #
                # L = S * S (S = 윈도우 내 토큰 수의 제곱근)
                # 예) L1=49 (7x7 윈도우) -> L2=196 (14x14 윈도우) 로 업스케일링
                S1 = int(L1 ** 0.5)  # 사전학습 윈도우 크기 (예: 7)
                S2 = int(L2 ** 0.5)  # 현재 모델 윈도우 크기 (예: 14)

                # 바이큐빅 보간:
                # (nH1, L1) -> (1, nH1, S1, S1): 4D로 변환하여 interpolate에 전달
                # interpolate로 (1, nH1, S2, S2)로 업샘플링 (bicubic: 3차 보간)
                # -> (nH2, L2)로 다시 펼침
                # 바이큐빅은 bilinear보다 더 부드러운 보간을 제공하며
                # 위치 편향 같은 연속적인 2D 신호에 적합
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                # 보간된 편향을 현재 모델 크기 (nH2, L2)로 변환하여 state_dict에 저장
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2)
    # 수정된 state_dict를 체크포인트에 반영하여 반환
    checkpoint['model'] = state_dict
    return checkpoint
