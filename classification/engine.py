"""
Train and eval functions used in main.py
main.py에서 호출되는 학습(train) 및 평가(evaluate) 핵심 함수 모음.
분산 학습(DistributedDataParallel), 혼합 정밀도(AMP), 지식 증류(KD),
Mixup 증강, EMA(지수 이동 평균) 등을 통합하여 한 에폭 학습을 수행한다.
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
from . import utils


def set_bn_state(model):
    """
    [한국어 설명]
    역할:
        모델 내 모든 BatchNorm 레이어를 강제로 eval() 모드로 설정한다.

    사용 이유:
        일부 학습 시나리오(예: 매우 작은 배치 사이즈, 파인튜닝)에서 BatchNorm의
        running_mean/running_var 업데이트를 막고 싶을 때 사용한다.
        set_bn_eval=True 옵션과 함께 model.train()으로 전환한 후에도
        BatchNorm만 eval 상태로 유지하여 통계값이 변경되지 않도록 고정한다.

    파라미터:
        model: BatchNorm 상태를 고정할 PyTorch 모델
    """
    for m in model.modules():
        # torch.nn.modules.batchnorm._BatchNorm은 BatchNorm1d/2d/3d의 공통 부모 클래스
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()  # running_mean, running_var 업데이트 중단; 학습 중에도 eval 통계 사용


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False,
                    pruning_groups=None, lambda_ffn=0.00002, lambda_qk=0.000004, lambda_v=0.000001,
                    mu=1.0, m_max_mb=7.044, pruning_freq=100, target_compression=0.80):
    """
    [한국어 설명]
    역할:
        한 에폭(epoch) 동안 전체 데이터셋을 순회하며 모델을 학습시킨다.
        지식 증류, Mixup 데이터 증강, 혼합 정밀도(AMP), Gradient Clipping,
        EMA 업데이트, 분산 학습 통계 동기화를 모두 처리한다.

    파라미터:
        model           : 학습할 Student 모델 (nn.Module)
        criterion       : DistillationLoss - 기본 분류 손실 + 증류 손실 결합 함수
        data_loader     : 이터러블 데이터 로더 (배치 단위로 (samples, targets) 반환)
        optimizer       : 파라미터 업데이트에 사용할 옵티마이저 (예: AdamW)
        device          : 연산 디바이스 (cuda / cpu)
        epoch           : 현재 에폭 번호 (로깅용)
        loss_scaler     : 혼합 정밀도(AMP)를 위한 GradScaler 래퍼.
                          FP16 언더플로우를 방지하기 위해 손실값을 스케일링하고
                          optimizer.step()을 안전하게 호출한다.
        clip_grad       : Gradient Clipping 임계값. 0이면 클리핑 비활성화.
                          너무 큰 그래디언트로 인한 학습 발산을 방지한다.
        clip_mode       : 그래디언트 클리핑 방식 ('norm': L2 노름 클리핑, 'value': 값 클리핑)
        model_ema       : EMA(지수 이동 평균) 모델. 각 배치 후 파라미터를 지수 평균으로 업데이트.
                          EMA 모델은 검증 시 더 안정적인 성능을 제공한다.
        mixup_fn        : Mixup/CutMix 데이터 증강 함수. 두 샘플을 혼합하여 새로운 학습 샘플 생성.
        set_training_mode: True면 model.train() 호출, False면 model.eval() 상태 유지.
        set_bn_eval     : True면 학습 중에도 BatchNorm을 eval 모드로 고정 (통계 업데이트 방지).

    반환값:
        Dict[str, float]: 에폭 전체의 평균 통계 {'loss': ..., 'lr': ...}
    """
    # 학습 모드 설정: set_training_mode=True면 드롭아웃 활성화, BN 통계 업데이트
    model.train(set_training_mode)
    if set_bn_eval:
        # 학습 모드여도 BatchNorm만 eval로 고정 (예: 사전학습 BN 통계를 파인튜닝에 재사용)
        set_bn_state(model)

    # MetricLogger: 학습 통계(loss, lr 등)를 수집하고 주기적으로 출력하는 유틸리티
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 학습률(lr)은 window_size=1로 최신값만 표시 (이동 평균 불필요)
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100  # 100 배치마다 한 번씩 진행 상황 출력

    # Pruning 카운터 초기화
    iteration_counter = 0
    # Prunable groups의 초기 메모리 계산 (첫 epoch 시작 시 한 번만)
    if pruning_groups is not None:
        from .pruning.memory_utils import compute_active_param_memory
        original_prunable_mb = compute_active_param_memory(pruning_groups) / 1e6
    else:
        original_prunable_mb = 0.0

    # 데이터 로더를 순회하며 배치별 학습 수행
    # metric_logger.log_every: 이터레이터를 감싸 ETA, 처리 시간 등을 주기적으로 출력
    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        iteration_counter += 1

        # 데이터를 GPU로 비동기 전송 (non_blocking=True: CPU-GPU 전송과 연산을 병렬 수행)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            # Mixup/CutMix 증강: 두 샘플을 베타 분포 가중치로 혼합하여 새 (샘플, 소프트레이블) 생성
            # 예) samples = lambda * img1 + (1-lambda) * img2
            #     targets = [lambda, class1, class2] 형태의 소프트 레이블로 변환
            samples, targets = mixup_fn(samples, targets)

        if True:  # with torch.cuda.amp.autocast():
            # 순전파(Forward Pass): 모델에 배치 입력 후 로짓 출력 획득
            # outputs는 Tensor 또는 Tuple[Tensor, Tensor] (DeiT의 cls+dist 토큰)
            outputs = model(samples)
            # 손실 계산: DistillationLoss가 base_loss + distill_loss를 결합하여 반환
            loss = criterion(samples, outputs, targets)

            # PGM Pruning regularization loss 추가 (Phase B)
            if pruning_groups is not None:
                from .pruning.pgm_loss import pgm_regularization_loss, memory_penalty
                from .pruning.memory_utils import compute_active_param_memory
                pgm_loss = pgm_regularization_loss(pruning_groups, lambda_ffn, lambda_qk, lambda_v)
                current_mem_mb = compute_active_param_memory(pruning_groups) / 1e6
                mem_penalty = memory_penalty(current_mem_mb, m_max_mb, mu)
                loss += pgm_loss + mem_penalty

        # 스칼라 값으로 변환하여 로깅 (텐서에서 파이썬 float로)
        loss_value = loss.item()

        # NaN 또는 Inf 손실이 발생하면 학습이 발산한 것이므로 즉시 중단
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 이전 배치의 그래디언트 초기화 (누적 방지)
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        # Adahessian처럼 2차 미분(헤시안)이 필요한 옵티마이저 여부 확인
        # create_graph=True이면 역전파 그래프를 유지하여 2차 미분 계산 가능
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order

        # loss_scaler: AMP(자동 혼합 정밀도) GradScaler 래퍼
        # 내부 동작:
        #   1. loss * scale_factor로 그래디언트 스케일링 (FP16 언더플로우 방지)
        #   2. loss.backward()로 역전파 수행
        #   3. Gradient Clipping 적용 (clip_grad > 0인 경우)
        #   4. optimizer.step()으로 파라미터 업데이트
        #   5. NaN/Inf 그래디언트 감지 시 해당 스텝 건너뜀 (scale_factor 감소)
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        # CRITICAL FIX: Pruned weights를 매 iteration마다 0으로 재설정
        # optimizer.step()에서 gradient로 복원되는 것을 방지
        if pruning_groups is not None:
            from .pruning.pgm_loss import apply_pruned_mask
            apply_pruned_mask(pruning_groups)

        # Phase B: Iteration-level pruning (주기적 실행 + 목표치 제한)
        # pruning_freq 마다 한 번씩 실행 (예: 100 iterations마다)
        if pruning_groups is not None and iteration_counter % pruning_freq == 0:
            from .pruning.pgm_loss import apply_phase1_pruning
            from .pruning.memory_utils import compute_active_param_memory
            current_mem_mb = compute_active_param_memory(pruning_groups) / 1e6

            # 현재 제거 비율 계산 (prunable 부분 기준)
            removed_ratio = (original_prunable_mb - current_mem_mb) / original_prunable_mb

            # 목표 제거 비율 도달 확인 (target_compression = 제거할 비율)
            # 예: target_compression=0.80이면 80% 제거, 20% 유지
            if removed_ratio < target_compression:
                # 모니터링 출력
                print(f"\n[Pruning Iter {iteration_counter}]")
                print(f"  Prunable Memory: {current_mem_mb:.2f} / {original_prunable_mb:.2f} MB ({(1-removed_ratio)*100:.1f}% remaining)")
                print(f"  Removed: {removed_ratio*100:.1f}% (Target: {target_compression*100:.0f}%)")

                # 동적 sparsity 계산 (주기적 실행이므로 비율 증가)
                sparsities = {}
                for gtype in ['G_FFN', 'G_QK', 'G_V']:
                    # 주기당 기본 제거 비율 (pruning_freq=100 기준)
                    # 80% 목표를 위해 증가: 기존 대비 2배 증가
                    # ImageNet-1K: ~5000 iter/epoch, 50 pruning/epoch
                    # 80% 도달: ~2 epochs (50 * 0.02 * 2 = 2.0 = 200% removed from some groups)
                    base_sparsity_per_freq = {
                        'G_FFN': 0.020,  # 2.0% per 100 iterations (ratio 20)
                        'G_QK':  0.004,  # 0.4% per 100 iterations (ratio 4)
                        'G_V':   0.001   # 0.1% per 100 iterations (ratio 1)
                    }[gtype]

                    # 메모리 초과 시 sparsity 증가
                    if current_mem_mb > m_max_mb:
                        extra = min(0.05, (current_mem_mb - m_max_mb) / m_max_mb * 0.1)
                        sparsities[gtype] = min(0.15, base_sparsity_per_freq + extra)
                    else:
                        sparsities[gtype] = base_sparsity_per_freq

                # Pruning 실행 (verbose=False로 요약만 출력)
                apply_phase1_pruning(model, pruning_groups, sparsities, verbose=False)
            else:
                # 목표 도달 시 알림 (처음 1회만)
                if iteration_counter % (pruning_freq * 10) == 0:  # 1000 iterations마다
                    print(f"\n[Pruning Iter {iteration_counter}] Target reached: {removed_ratio*100:.1f}% removed (goal: {target_compression*100:.0f}%)")

        # GPU 연산이 모두 완료될 때까지 대기 (비동기 CUDA 연산 동기화)
        # 이후의 EMA 업데이트 등이 올바른 파라미터값으로 수행되도록 보장
        torch.cuda.synchronize()

        if model_ema is not None:
            # EMA 파라미터 업데이트:
            # ema_param = decay * ema_param + (1 - decay) * current_param
            # 배치마다 현재 모델 파라미터를 지수 이동 평균으로 EMA 모델에 반영
            # EMA 모델은 검증/추론에서 더 안정적인(less noisy) 성능을 보임
            model_ema.update(model)

        # 현재 배치의 손실값과 학습률을 MetricLogger에 기록
        metric_logger.update(loss=loss_value)
        # param_groups[0]: 첫 번째 파라미터 그룹의 현재 학습률 기록
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    # 분산 학습 시 모든 GPU 프로세스의 통계를 수집하여 전체 평균값 계산
    # 내부적으로 dist.all_reduce를 사용하여 count와 total을 합산
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Phase B: Epoch 종료 후 통계 출력 (pruning은 주기적으로 수행됨)
    if pruning_groups is not None:
        from .pruning.pgm_loss import count_zero_groups
        from .pruning.memory_utils import compute_active_param_memory

        # 현재 메모리 및 zero 그룹 통계 출력
        current_mem_mb = compute_active_param_memory(pruning_groups) / 1e6
        removed_ratio = (original_prunable_mb - current_mem_mb) / original_prunable_mb
        zero_counts = count_zero_groups(pruning_groups)

        print(f"\n{'='*70}")
        print(f"[Epoch {epoch} Pruning Summary]")
        print(f"{'='*70}")
        print(f"  Prunable Memory (Active):  {current_mem_mb:.2f} MB / {original_prunable_mb:.2f} MB")
        print(f"  Removed from Prunable:     {removed_ratio*100:.1f}% (Target: {target_compression*100:.0f}%)")
        print(f"  Target Prunable Memory:    {m_max_mb:.2f} MB")
        print(f"  Total Iterations:          {iteration_counter}")
        print(f"  Zero Groups by Type:")
        for gtype in ['G_FFN', 'G_QK', 'G_V', 'G_PATCH']:
            if gtype in zero_counts:
                stats = zero_counts[gtype]
                print(f"    {gtype:8s}: {stats['zero_units']:4d}/{stats['total_units']:4d} units ({stats['zero_ratio']*100:5.1f}%)")
        print(f"{'='*70}")

    # 각 메트릭의 에폭 전체 평균값을 딕셔너리로 반환
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """
    [한국어 설명]
    역할:
        검증/테스트 데이터셋에서 모델 성능을 평가한다.
        Top-1 정확도, Top-5 정확도, CrossEntropy 손실을 계산하여 반환한다.

    데코레이터 @torch.no_grad():
        평가 시에는 그래디언트 계산이 불필요하므로 비활성화하여
        메모리 절약 및 연산 속도를 높인다.

    파라미터:
        data_loader : 평가용 데이터 로더 (배치 단위로 (images, target) 반환)
        model       : 평가할 모델 (eval 모드로 전환됨)
        device      : 연산 디바이스 (cuda / cpu)

    반환값:
        Dict[str, float]: 전체 검증 세트의 평균 통계
            {'loss': 평균 CE 손실, 'acc1': Top-1 정확도(%), 'acc5': Top-5 정확도(%)}
    """
    # 평가는 항상 단순 CrossEntropy 손실 사용 (증류 없음, 소프트 레이블 없음)
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    # eval 모드: 드롭아웃 비활성화, BatchNorm이 학습된 running_mean/var 사용
    model.eval()

    # 10 배치마다 진행 상황 출력 (학습보다 자주 출력)
    for images, target in metric_logger.log_every(data_loader, 10, header):
        # 데이터를 GPU로 비동기 전송
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # torch.cuda.amp.autocast(): FP16 혼합 정밀도로 순전파 수행
        # 학습 때와 달리 GradScaler는 사용하지 않음 (그래디언트 업데이트 없음)
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # Top-K 정확도 계산:
        # acc1: 모델의 Top-1 예측이 정답과 일치하는 비율 (%)
        # acc5: 모델의 Top-5 예측 중 정답이 포함된 비율 (%)
        # ImageNet-1K 표준 지표: acc1(top-1), acc5(top-5) 보고
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        # 배치 손실값 기록 (기본 n=1이므로 배치 수로 가중 평균하지 않음; global_avg 참고)
        metric_logger.update(loss=loss.item())
        # acc1, acc5는 배치 크기(n=batch_size)로 가중 평균 계산
        # global_avg = total_correct / total_samples 방식으로 정확한 전체 정확도 집계
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    # 분산 평가 시 모든 GPU의 통계를 합산하여 전체 정확도 계산
    # 단일 GPU 환경에서는 아무 동작도 하지 않음
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # 각 메트릭의 전체 평균값을 딕셔너리로 반환
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
