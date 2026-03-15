"""
Combined Pruning Training Script (λ Regularization + Memory Penalty + Physical Pruning)

결합 방식:
  - Loss: CE + λ regularization + μ·memory_penalty
  - λ regularization: weights를 0 방향으로 유도
  - Memory penalty: 목표 메모리 초과 시 패널티 (μ * max(0, current_mem - m_max))
  - Epoch 끝에 physical pruning (작아진 weights 제거)

λ 비율 (CRITICAL):
  - FFN : QK : V = 10 : 2 : 1
  - QK는 이미 많이 pruning되어 있음 (key_dim=16으로 작음)
  - V는 pruning에 매우 민감함 (정보 손실 큼)

장점:
  - λ가 불필요한 weights를 미리 작게 만듦
  - Memory penalty가 목표 압축률을 강제
  - Physical pruning 시 더 안정적 (이미 작은 weights 제거)

Usage:
    python -m classification.pruning.train_combined_pruning \
        --model EfficientViT_M4 \
        --data-path /path/to/imagenet \
        --resume efficientvit_m4.pth \
        --target-reduction 0.76 \
        --lambda-ffn 0.001 \
        --lambda-qk 0.0002 \
        --lambda-v 0.0001 \
        --mu 1.0 \
        --output-dir ./results/combined
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification.model.build import EfficientViT_M4
from classification.data.datasets import build_dataset
from classification.engine import evaluate
from classification import utils
from classification.pruning.structural_pruning import (
    IterativePhysicalPruner,
    compute_model_size_mb,
    validate_model_forward
)

from timm.data import Mixup


def get_args_parser():
    parser = argparse.ArgumentParser('Combined Pruning (λ + Physical)', add_help=False)

    # Model
    parser.add_argument('--model', default='EfficientViT_M4', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--num-classes', default=1000, type=int)

    # Data
    parser.add_argument('--data-path', default='/workspace/etri_iitp/JS/EfficientViT/data/imagenet', type=str)
    parser.add_argument('--data-set', default='IMNET', type=str)
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true', default=True)

    # Pruning (공격적 설정 - 15 epochs 내 76% 달성)
    # FFN이 파라미터의 대부분을 차지하므로 매우 공격적으로 pruning
    parser.add_argument('--target-reduction', default=0.76, type=float)
    parser.add_argument('--ffn-prune-per-epoch', default=0.25, type=float,
                        help='FFN: 매 epoch 25%% 제거 (매우 공격적)')
    parser.add_argument('--qk-prune-per-epoch', default=0.15, type=float,
                        help='QK: 매 epoch 15%% 제거 (공격적)')
    parser.add_argument('--min-ffn-ratio', default=0.05, type=float,
                        help='FFN 최소 5%% 유지 (최대 95%% pruning)')
    parser.add_argument('--min-qk-ratio', default=0.25, type=float,
                        help='QK 최소 25%% 유지')
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--pruning-epochs', default=15, type=int)
    parser.add_argument('--finetune-epochs', default=10, type=int)

    # Lambda regularization (FFN:QK:V = 10:2:1 비율!)
    parser.add_argument('--lambda-ffn', default=1e-3, type=float,
                        help='L2 regularization for FFN weights (largest, 10x)')
    parser.add_argument('--lambda-qk', default=2e-4, type=float,
                        help='L2 regularization for Q/K weights (2x, already small)')
    parser.add_argument('--lambda-v', default=1e-4, type=float,
                        help='L2 regularization for V weights (1x, very sensitive)')
    parser.add_argument('--lambda-decay', default=0.9, type=float,
                        help='Lambda decay per epoch during finetune (reduce regularization)')

    # Memory penalty (PGM loss)
    parser.add_argument('--mu', default=1.0, type=float,
                        help='Memory penalty coefficient: μ * max(0, current_mem - m_max)')

    # Training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min-lr', default=1e-6, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--clip-grad', default=1.0, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float)

    # Output
    parser.add_argument('--output-dir', default='./results/combined', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)

    # Data augmentation (required by build_dataset)
    parser.add_argument('--color-jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',
                        help='AutoAugment policy')
    parser.add_argument('--train-interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25,
                        help='Random erase prob')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count')
    parser.add_argument('--finetune', default='', type=str,
                        help='finetune from checkpoint')

    return parser


def compute_lambda_regularization(model, lambda_ffn, lambda_qk, lambda_v):
    """
    모델 전체의 λ regularization loss 계산

    L_reg = λ_FFN * Σ||w_FFN||² + λ_QK * Σ||w_QK||² + λ_V * Σ||w_V||²

    이 loss를 CE에 더하면 gradient에 -2λw 항이 추가되어
    weights를 0 방향으로 유도함
    """
    reg_loss = 0.0

    # Iterate through all EfficientViTBlocks
    for name, module in model.named_modules():
        # FFN layers (pw1 = expand, pw2 = shrink)
        if hasattr(module, 'pw1') and hasattr(module, 'pw2'):
            # FFN expand (pw1)
            if hasattr(module.pw1, 'c'):
                reg_loss += lambda_ffn * torch.sum(module.pw1.c.weight ** 2)
            # FFN shrink (pw2)
            if hasattr(module.pw2, 'c'):
                reg_loss += lambda_ffn * torch.sum(module.pw2.c.weight ** 2)

        # CGA layers (qkvs, dws)
        if hasattr(module, 'qkvs') and hasattr(module, 'dws'):
            for h in range(len(module.qkvs)):
                qkv = module.qkvs[h]
                dw = module.dws[h]
                key_dim = module.key_dim
                d = module.d

                if hasattr(qkv, 'c'):
                    w = qkv.c.weight
                    # Q slice: [0:key_dim]
                    reg_loss += lambda_qk * torch.sum(w[:key_dim] ** 2)
                    # K slice: [key_dim:2*key_dim]
                    reg_loss += lambda_qk * torch.sum(w[key_dim:2*key_dim] ** 2)
                    # V slice: [2*key_dim:2*key_dim+d]
                    reg_loss += lambda_v * torch.sum(w[2*key_dim:2*key_dim+d] ** 2)

                # DW conv on Q
                if hasattr(dw, 'c'):
                    reg_loss += lambda_qk * torch.sum(dw.c.weight ** 2)

    return reg_loss


def compute_memory_penalty(current_size_mb: float, m_max_mb: float, mu: float) -> float:
    """
    메모리 패널티 계산: μ * max(0, current_mem - m_max)

    Args:
        current_size_mb: 현재 모델 크기 (MB)
        m_max_mb: 목표 최대 크기 (MB)
        mu: 패널티 계수

    Returns:
        penalty: 메모리 초과 시 패널티 값
    """
    return mu * max(0.0, current_size_mb - m_max_mb)


def train_one_epoch_combined(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    criterion: nn.Module,
    lambda_ffn: float,
    lambda_qk: float,
    lambda_v: float,
    mu: float,
    current_size_mb: float,
    m_max_mb: float,
    scaler=None,
    clip_grad: float = 1.0,
    print_freq: int = 100
):
    """
    Combined: CE loss + λ regularization + μ·memory_penalty
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('reg', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('mem_pen', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = f'Epoch: [{epoch}]'

    # Memory penalty (constant for the epoch, computed from model size)
    mem_penalty = compute_memory_penalty(current_size_mb, m_max_mb, mu)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            ce_loss = criterion(outputs, targets)

            # λ regularization 추가
            reg_loss = compute_lambda_regularization(model, lambda_ffn, lambda_qk, lambda_v)

            # Total loss = CE + λ_reg + μ·memory_penalty
            loss = ce_loss + reg_loss + mem_penalty

        loss_value = loss.item()
        reg_value = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping")
            sys.exit(1)

        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(reg=reg_value)
        metric_logger.update(mem_pen=mem_penalty)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    print(f"\n{'='*70}")
    print("Combined Pruning for EfficientViT")
    print("(λ Regularization + Memory Penalty + Physical Pruning)")
    print(f"{'='*70}")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    print("\n[1/5] Loading dataset...")
    dataset_train, _ = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=torch.utils.data.RandomSampler(dataset_train),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    # Model
    print("\n[2/5] Loading model...")
    model = EfficientViT_M4(num_classes=args.num_classes, pretrained=False)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded: {args.resume}")
    model = model.to(device)

    original_size = compute_model_size_mb(model)
    m_max_mb = original_size * (1.0 - args.target_reduction)  # 목표 메모리
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Target: {args.target_reduction*100:.1f}% reduction")
    print(f"  M_max: {m_max_mb:.2f} MB (target size)")

    # Training setup
    print("\n[3/5] Setup training...")
    print(f"  λ_FFN: {args.lambda_ffn} (10x)")
    print(f"  λ_QK:  {args.lambda_qk} (2x)")
    print(f"  λ_V:   {args.lambda_v} (1x)")
    print(f"  λ ratio: FFN:QK:V = {args.lambda_ffn/args.lambda_v:.0f}:{args.lambda_qk/args.lambda_v:.0f}:1")
    print(f"  μ (memory penalty): {args.mu}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    total_epochs = args.pruning_epochs + args.finetune_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=args.min_lr
    )

    # Pruner
    print("\n[4/5] Initialize pruner...")
    pruner = IterativePhysicalPruner(
        target_reduction=args.target_reduction,
        ffn_prune_per_epoch=args.ffn_prune_per_epoch,
        qk_prune_per_epoch=args.qk_prune_per_epoch,
        min_ffn_ratio=args.min_ffn_ratio,
        min_qk_ratio=args.min_qk_ratio,
        warmup_epochs=args.warmup_epochs,
        verbose=True
    )

    # Initial eval
    print("\n[5/5] Initial evaluation...")
    test_stats = evaluate(data_loader_val, model, device)
    print(f"  Initial Acc@1: {test_stats['acc1']:.2f}%")

    # Training loop
    print("\n" + "="*70)
    print("Starting Combined Pruning (λ + Physical)")
    print("="*70)

    history = {'train_loss': [], 'reg_loss': [], 'mem_penalty': [], 'val_acc1': [], 'model_size_mb': [], 'reduction': []}
    best_acc1 = 0.0
    pruning_end_acc1 = 0.0
    pruning_end_size = 0.0
    pruning_end_reduction = 0.0
    start_time = time.time()

    # Epoch별 핵심 로그 파일 초기화
    log_path = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("Combined Pruning (λ + Physical) Training Log\n")
        f.write("=" * 100 + "\n")
        f.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Original size: {original_size:.2f} MB\n")
        f.write(f"Target reduction: {args.target_reduction*100:.0f}%\n")
        f.write(f"M_max: {m_max_mb:.2f} MB\n")
        f.write(f"FFN prune/epoch: {args.ffn_prune_per_epoch*100:.0f}%\n")
        f.write(f"QK prune/epoch: {args.qk_prune_per_epoch*100:.0f}%\n")
        f.write(f"Min FFN ratio: {args.min_ffn_ratio*100:.0f}%\n")
        f.write(f"Min QK ratio: {args.min_qk_ratio*100:.0f}%\n")
        f.write(f"λ_FFN: {args.lambda_ffn}, λ_QK: {args.lambda_qk}, λ_V: {args.lambda_v}\n")
        f.write(f"λ ratio: FFN:QK:V = {args.lambda_ffn/args.lambda_v:.0f}:{args.lambda_qk/args.lambda_v:.0f}:1\n")
        f.write(f"μ (memory penalty): {args.mu}\n")
        f.write(f"Pruning epochs: {args.pruning_epochs}\n")
        f.write(f"Finetune epochs: {args.finetune_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Initial Acc@1: {test_stats['acc1']:.2f}%\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Epoch':>5} | {'Phase':>8} | {'Loss':>8} | {'Reg':>7} | {'MemPen':>7} | "
                f"{'Acc@1':>7} | {'Acc@5':>7} | {'Size(MB)':>9} | {'Reduced':>8} | {'Best':>5} | {'Time':>10}\n")
        f.write("-" * 100 + "\n")

    # Current lambda values (will decay during finetune)
    current_lambda_ffn = args.lambda_ffn
    current_lambda_qk = args.lambda_qk
    current_lambda_v = args.lambda_v

    for epoch in range(total_epochs):
        is_pruning_phase = epoch < args.pruning_epochs
        epoch_start = time.time()
        phase = 'PRUNE' if is_pruning_phase else 'FINETUNE'

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{total_epochs} [{phase}]")
        print(f"  λ_FFN={current_lambda_ffn:.2e}, λ_QK={current_lambda_qk:.2e}, λ_V={current_lambda_v:.2e}")
        print('='*60)

        # Current model size for memory penalty
        current_size = compute_model_size_mb(model)

        # Train (CE + λ regularization + memory penalty)
        train_stats = train_one_epoch_combined(
            model, data_loader_train, optimizer, device, epoch,
            criterion, current_lambda_ffn, current_lambda_qk, current_lambda_v,
            args.mu, current_size, m_max_mb,
            scaler, args.clip_grad
        )

        # Physical pruning (only during pruning phase)
        if is_pruning_phase and not pruner.target_reached:
            pruning_result = pruner.step(model, device)

            # Rebuild optimizer (parameters changed)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer.param_groups[0]['lr'],
                weight_decay=args.weight_decay
            )
        elif not is_pruning_phase:
            # Finetune phase: decay lambda (reduce regularization)
            current_lambda_ffn *= args.lambda_decay
            current_lambda_qk *= args.lambda_decay
            current_lambda_v *= args.lambda_decay

        # Evaluate
        test_stats = evaluate(data_loader_val, model, device)
        lr_scheduler.step()

        # Stats
        current_size = compute_model_size_mb(model)
        current_reduction = 1.0 - current_size / original_size
        epoch_time = str(datetime.timedelta(seconds=int(time.time() - epoch_start)))
        reg_loss = train_stats.get('reg', 0)
        mem_pen = train_stats.get('mem_pen', 0)

        history['train_loss'].append(train_stats['loss'])
        history['reg_loss'].append(reg_loss)
        history['mem_penalty'].append(mem_pen)
        history['val_acc1'].append(test_stats['acc1'])
        history['model_size_mb'].append(current_size)
        history['reduction'].append(current_reduction)

        print(f"\n[Summary] Loss: {train_stats['loss']:.4f} (reg: {reg_loss:.4f}, mem: {mem_pen:.4f}) | "
              f"Acc@1: {test_stats['acc1']:.2f}% | Size: {current_size:.2f}MB ({current_reduction*100:.1f}% reduced)")

        # Pruning phase 마지막 epoch 기록
        if epoch == args.pruning_epochs - 1 or (epoch < args.pruning_epochs and pruner.target_reached):
            pruning_end_acc1 = test_stats['acc1']
            pruning_end_size = current_size
            pruning_end_reduction = current_reduction

        # Save best
        is_best = False
        if test_stats['acc1'] > best_acc1:
            best_acc1 = test_stats['acc1']
            is_best = True
            # state_dict 체크포인트 (학습 재개용)
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'acc1': best_acc1, 'size_mb': current_size,
            }, os.path.join(args.output_dir, 'best_combined.pth'))
            # 완전 모델 저장 (pruned 아키텍처 + weights, 바로 로드 가능)
            torch.save(model, os.path.join(args.output_dir, 'best_combined_model.pth'))
            print(f"  >>> New best: {best_acc1:.2f}%")

        # Epoch별 핵심 로그 기록
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1:>5} | {phase:>8} | {train_stats['loss']:>8.4f} | "
                    f"{reg_loss:>7.4f} | {mem_pen:>7.4f} | "
                    f"{test_stats['acc1']:>6.2f}% | {test_stats['acc5']:>6.2f}% | "
                    f"{current_size:>8.2f} | {current_reduction*100:>6.1f}% | "
                    f"{'*' if is_best else ' ':>5} | {epoch_time:>10}\n")

    # Final summary
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    final_size = compute_model_size_mb(model)

    final_acc1 = test_stats['acc1']
    final_acc5 = test_stats['acc5']

    print(f"\n{'='*70}")
    print("Combined Pruning Complete!")
    print(f"{'='*70}")
    print(f"  Method: Combined (λ + Memory Penalty + Physical)")
    print(f"  λ_FFN: {args.lambda_ffn}, λ_QK: {args.lambda_qk}, λ_V: {args.lambda_v}")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Final: {final_size:.2f} MB ({(1-final_size/original_size)*100:.1f}% reduced)")
    print(f"  Pruning 후 Acc@1: {pruning_end_acc1:.2f}% (size: {pruning_end_size:.2f}MB, {pruning_end_reduction*100:.1f}% reduced)")
    print(f"  Finetune 후 Acc@1: {final_acc1:.2f}% (최종)")
    print(f"  Time: {total_time}")

    # 최종 요약도 로그에 기록
    with open(log_path, 'a') as f:
        f.write("-" * 100 + "\n")
        f.write(f"\nFINAL RESULTS\n")
        f.write(f"  Method: Combined (λ + Memory Penalty + Physical)\n")
        f.write(f"  λ_FFN: {args.lambda_ffn}, λ_QK: {args.lambda_qk}, λ_V: {args.lambda_v}\n")
        f.write(f"  λ ratio: FFN:QK:V = {args.lambda_ffn/args.lambda_v:.0f}:{args.lambda_qk/args.lambda_v:.0f}:1\n")
        f.write(f"  μ: {args.mu}, M_max: {m_max_mb:.2f} MB\n")
        f.write(f"  Original size: {original_size:.2f} MB\n")
        f.write(f"  Final size: {final_size:.2f} MB ({(1-final_size/original_size)*100:.1f}% reduced)\n")
        f.write(f"\n")
        f.write(f"  [Pruning 완료 시점]\n")
        f.write(f"    Acc@1: {pruning_end_acc1:.2f}%\n")
        f.write(f"    Size: {pruning_end_size:.2f} MB ({pruning_end_reduction*100:.1f}% reduced)\n")
        f.write(f"\n")
        f.write(f"  [Finetune 완료 시점 (최종)]\n")
        f.write(f"    Acc@1: {final_acc1:.2f}%\n")
        f.write(f"    Acc@5: {final_acc5:.2f}%\n")
        f.write(f"    Size: {final_size:.2f} MB ({(1-final_size/original_size)*100:.1f}% reduced)\n")
        f.write(f"\n")
        f.write(f"  Accuracy 회복: {pruning_end_acc1:.2f}% → {final_acc1:.2f}% ({final_acc1-pruning_end_acc1:+.2f}%)\n")
        f.write(f"  Total time: {total_time}\n")
        f.write(f"  End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Save summary JSON
    summary = {
        'method': 'combined_lambda_memory_physical',
        'lambda_ffn': args.lambda_ffn,
        'lambda_qk': args.lambda_qk,
        'lambda_v': args.lambda_v,
        'lambda_ratio': f"{args.lambda_ffn/args.lambda_v:.0f}:{args.lambda_qk/args.lambda_v:.0f}:1",
        'mu': args.mu,
        'm_max_mb': m_max_mb,
        'original_size_mb': original_size,
        'final_size_mb': final_size,
        'reduction': 1 - final_size/original_size,
        'pruning_end_acc1': pruning_end_acc1,
        'pruning_end_size_mb': pruning_end_size,
        'pruning_end_reduction': pruning_end_reduction,
        'final_acc1': final_acc1,
        'final_acc5': final_acc5,
        'acc_recovery': final_acc1 - pruning_end_acc1,
        'history': history,
        'args': vars(args),
    }
    with open(os.path.join(args.output_dir, 'summary_combined.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # 최종 모델 저장 (pruned 아키텍처 + weights, 바로 로드 가능)
    torch.save(model, os.path.join(args.output_dir, 'final_pruned_model.pth'))
    print(f"  Final pruned model saved: {args.output_dir}/final_pruned_model.pth")
    print(f"  Load with: model = torch.load('final_pruned_model.pth')")

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Training log: {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combined Pruning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
