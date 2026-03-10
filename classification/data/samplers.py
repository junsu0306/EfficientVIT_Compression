'''
Build samplers for data loading
데이터 로딩을 위한 샘플러 모듈.
분산 학습(DDP, DistributedDataParallel) 환경에서 반복 증강(Repeated Augmentation)을
지원하는 RASampler를 제공한다.
'''
import torch
import torch.distributed as dist
import math


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler

    분산 학습 환경에서 반복 증강(Repeated Augmentation, RA)을 지원하는 샘플러.

    [동작 원리]
    반복 증강(Repeated Augmentation)은 각 샘플을 한 에폭 내에서 3번 반복하되,
    서로 다른 GPU(프로세스)가 동일 샘플의 서로 다른 증강 버전을 처리하도록 한다.
    이 방식을 사용하는 이유:
      - 데이터 다양성 증가: 동일 이미지에 서로 다른 랜덤 증강을 적용하므로,
        모델이 더 다양한 변형을 학습하여 일반화 성능이 향상된다.
      - GPU 간 증강 다양성: 같은 이미지의 다른 증강 버전이 다른 GPU에 할당되므로
        배치 내 샘플 다양성이 유지된다.
      - DeiT 논문에서 제안된 방식으로, 적은 에폭으로도 좋은 성능을 달성한다.

    [인덱스 구성 과정]
    1. 전체 데이터셋 인덱스를 에폭 시드로 결정론적으로 셔플
    2. 각 인덱스를 3번 반복 ([0,1,2,...] -> [0,0,0,1,1,1,2,2,2,...])
    3. total_size에 맞게 패딩 (나누어 떨어지도록)
    4. rank 기준으로 num_replicas 간격으로 서브샘플
       (rank=0이면 0,3,6,...번째; rank=1이면 1,4,7,...번째 인덱스 선택)
       -> 연속된 3개의 복사본이 각각 다른 GPU에 분산됨
    5. num_selected_samples만큼만 실제로 사용 (256의 배수로 정렬)

    [num_selected_samples 계산 방식]
    전체 데이터셋 크기를 256으로 내림(floor)하여 256의 배수로 맞춘 후
    num_replicas로 나눔. 배치 크기(256)의 배수를 보장하여 마지막 배치의
    패딩을 방지한다.

    Heavily based on torch.utils.data.DistributedSampler
    PyTorch의 DistributedSampler를 기반으로 RA 기능을 추가한 확장 구현이다.

    Args:
        dataset (torch.utils.data.Dataset): 샘플링할 데이터셋.
        num_replicas (int, optional): 분산 학습에 참여하는 프로세스(GPU) 수.
            None이면 dist.get_world_size()로 자동 설정.
        rank (int, optional): 현재 프로세스의 순위(0부터 시작).
            None이면 dist.get_rank()로 자동 설정.
        shuffle (bool): True이면 에폭마다 에폭 번호를 시드로 하여 결정론적 셔플.
            False이면 원래 순서 유지. 기본값 True.

    Attributes:
        num_samples (int): 이 프로세스가 한 에폭에서 처리할 총 샘플 수
                           (3배 반복 후 num_replicas로 나눈 값, ceil).
        total_size (int): 전체 프로세스의 총 샘플 수 (num_samples * num_replicas).
        num_selected_samples (int): 실제로 반환할 샘플 수
                                    (256의 배수로 정렬, num_replicas로 나눔).
        epoch (int): 현재 에폭 번호 (셔플 시드로 사용). set_epoch()으로 갱신.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        # 분산 패키지 사용 가능 여부 확인 후 num_replicas (world_size) 자동 설정
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        # 현재 프로세스의 rank 자동 설정
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas  # 총 GPU 수 (world_size)
        self.rank = rank                  # 현재 GPU 번호 (0-indexed)
        self.epoch = 0                    # 에폭 번호 (셔플 재현성 보장을 위한 시드)

        # 각 프로세스가 처리할 샘플 수: 전체 데이터셋 * 3(반복) / GPU 수 (올림)
        # 3배 반복은 반복 증강(Repeated Augmentation)의 핵심: 각 샘플을 3번 사용
        self.num_samples = int(
            math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        # 전체 프로세스의 합산 샘플 수 (나누어 떨어지도록 패딩 기준)
        self.total_size = self.num_samples * self.num_replicas

        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        # 실제 반환할 샘플 수: 배치 크기 256의 배수로 내림하여 마지막 배치 패딩 방지
        # 예: 데이터셋 1281167개, 8 GPU -> floor(1281167 // 256 * 256 / 8) = 160000
        self.num_selected_samples = int(math.floor(
            len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        """
        이 프로세스(rank)에 할당된 샘플 인덱스의 이터레이터를 반환한다.

        [인덱스 생성 단계]
        1. 에폭 번호를 시드로 결정론적 셔플 (재현성 보장)
        2. 각 인덱스를 3번 반복하여 반복 증강 구현
        3. total_size를 맞추기 위해 앞부분을 이어 붙여 패딩
        4. rank 기준으로 서브샘플 (stride=num_replicas)
        5. num_selected_samples 개수만큼 슬라이싱하여 반환

        Returns:
            iterator: 이 프로세스에 할당된 데이터셋 인덱스의 이터레이터.
        """
        # deterministically shuffle based on epoch
        # 에폭마다 다른 시드를 사용하여 결정론적으로 셔플 (재현 가능)
        # 모든 프로세스가 동일한 시드를 사용하므로 일관된 셔플 보장
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            # 전체 데이터셋 인덱스를 무작위 순열로 생성
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            # 셔플 비활성화: 원래 순서(0, 1, 2, ...) 유지
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        # 반복 증강(Repeated Augmentation): 각 인덱스를 3번 반복
        # [0, 1, 2] -> [0, 0, 0, 1, 1, 1, 2, 2, 2]
        # 연속된 3개 복사본이 rank 0, 1, 2에 각각 하나씩 배분됨
        indices = [ele for ele in indices for i in range(3)]
        # total_size가 되도록 앞부분 인덱스를 이어 붙여 패딩
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # rank 기준으로 num_replicas 간격으로 서브샘플
        # rank=0: 0, num_replicas, 2*num_replicas, ... 번째 인덱스 선택
        # rank=1: 1, 1+num_replicas, 1+2*num_replicas, ... 번째 인덱스 선택
        # 이 방식으로 연속된 3개 반복본이 서로 다른 GPU에 하나씩 분산됨
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # num_selected_samples만큼만 사용 (256의 배수로 정렬된 실제 사용 샘플 수)
        # 전체 num_samples 중 앞부분만 사용하여 마지막 배치의 불완전함을 방지
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        """
        이 프로세스가 한 에폭에서 반환할 샘플 수를 반환한다.

        Returns:
            int: num_selected_samples (256의 배수로 정렬된 실제 사용 샘플 수).
        """
        return self.num_selected_samples

    def set_epoch(self, epoch):
        """
        현재 에폭 번호를 설정한다.

        에폭마다 이 메서드를 호출하여 셔플 시드를 갱신해야 한다.
        동일한 에폭 번호를 사용하면 모든 프로세스에서 동일한 셔플 순서가
        보장되어 분산 학습의 일관성이 유지된다.

        Args:
            epoch (int): 현재 에폭 번호. __iter__에서 torch.Generator의 시드로 사용.

        사용 예시:
            sampler.set_epoch(epoch)  # 매 에폭 시작 전 호출 필수
            for batch in dataloader:
                ...
        """
        self.epoch = epoch
