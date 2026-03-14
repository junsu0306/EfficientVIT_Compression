"""
Physical-Only Pruning Training Script

순수 Physical Pruning 방식:
  - Loss: CE만 사용 (λ regularization 없음)
  - Epoch 끝에 importance 기반으로 물리적 pruning
  - Weights를 0으로 유도하지 않고, 직접 작은 것들 제거

장점:
  - 단순한 loss 함수
  - 학습 중 weights가 자유롭게 학습됨

단점:
  - Pruning 시 갑작스러운 변화 (weights가 미리 작아지지 않음)

Usage:
    python -m classification.pruning.train_physical_pruning \
        --model EfficientViT_M4 \
        --data-path /path/to/imagenet \
        --resume efficientvit_m4.pth \
        --target-reduction 0.76 \
        --output-dir ./checkpoints/physical_only
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
    parser = argparse.ArgumentParser('Physical-Only Pruning', add_help=False)

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

    # Training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min-lr', default=1e-6, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--clip-grad', default=1.0, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float)

    # Output
    parser.add_argument('--output-dir', default='./checkpoints/physical_only', type=str)
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


def train_one_epoch_physical(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    criterion: nn.Module,
    scaler=None,
    clip_grad: float = 1.0,
    print_freq: int = 100
):
    """
    Physical-Only: 순수 CE loss만 사용
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            # 순수 CE loss만 사용 (λ regularization 없음)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

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
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    print(f"\n{'='*70}")
    print("Physical-Only Pruning for EfficientViT")
    print("(No λ regularization, pure importance-based pruning)")
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
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Target: {args.target_reduction*100:.1f}% reduction")

    # Training setup
    print("\n[3/5] Setup training...")
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
    print("Starting Physical-Only Pruning")
    print("="*70)

    history = {'train_loss': [], 'val_acc1': [], 'model_size_mb': [], 'reduction': []}
    best_acc1 = 0.0
    pruning_end_acc1 = 0.0   # Pruning 마지막 epoch의 Acc@1
    pruning_end_size = 0.0   # Pruning 마지막 epoch의 모델 크기
    pruning_end_reduction = 0.0
    start_time = time.time()

    # Epoch별 핵심 로그 파일 초기화
    log_path = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Physical-Only Pruning Training Log\n")
        f.write("=" * 80 + "\n")
        f.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Original size: {original_size:.2f} MB\n")
        f.write(f"Target reduction: {args.target_reduction*100:.0f}%\n")
        f.write(f"FFN prune/epoch: {args.ffn_prune_per_epoch*100:.0f}%\n")
        f.write(f"QK prune/epoch: {args.qk_prune_per_epoch*100:.0f}%\n")
        f.write(f"Min FFN ratio: {args.min_ffn_ratio*100:.0f}%\n")
        f.write(f"Min QK ratio: {args.min_qk_ratio*100:.0f}%\n")
        f.write(f"Pruning epochs: {args.pruning_epochs}\n")
        f.write(f"Finetune epochs: {args.finetune_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Initial Acc@1: {test_stats['acc1']:.2f}%\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':>5} | {'Phase':>8} | {'Loss':>8} | {'Acc@1':>7} | {'Acc@5':>7} | "
                f"{'Size(MB)':>9} | {'Reduced':>8} | {'Best':>5} | {'Time':>10}\n")
        f.write("-" * 80 + "\n")

    for epoch in range(total_epochs):
        epoch_start = time.time()
        phase = 'PRUNE' if epoch < args.pruning_epochs else 'FINETUNE'

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{total_epochs} [{phase}]")
        print('='*60)

        # Train (CE loss only)
        train_stats = train_one_epoch_physical(
            model, data_loader_train, optimizer, device, epoch,
            criterion, scaler, args.clip_grad
        )

        # Physical pruning (only during pruning phase)
        if epoch < args.pruning_epochs and not pruner.target_reached:
            pruning_result = pruner.step(model, device)

            # Rebuild optimizer (parameters changed)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer.param_groups[0]['lr'],
                weight_decay=args.weight_decay
            )

        # Evaluate
        test_stats = evaluate(data_loader_val, model, device)
        lr_scheduler.step()

        # Stats
        current_size = compute_model_size_mb(model)
        current_reduction = 1.0 - current_size / original_size
        epoch_time = str(datetime.timedelta(seconds=int(time.time() - epoch_start)))

        history['train_loss'].append(train_stats['loss'])
        history['val_acc1'].append(test_stats['acc1'])
        history['model_size_mb'].append(current_size)
        history['reduction'].append(current_reduction)

        print(f"\n[Summary] Loss: {train_stats['loss']:.4f} | Acc@1: {test_stats['acc1']:.2f}% | "
              f"Size: {current_size:.2f}MB ({current_reduction*100:.1f}% reduced)")

        # Pruning phase 마지막 epoch 기록
        if epoch == args.pruning_epochs - 1 or (epoch < args.pruning_epochs and pruner.target_reached):
            pruning_end_acc1 = test_stats['acc1']
            pruning_end_size = current_size
            pruning_end_reduction = current_reduction

        # Save best (pruning 후 기준)
        is_best = False
        if test_stats['acc1'] > best_acc1:
            best_acc1 = test_stats['acc1']
            is_best = True
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'acc1': best_acc1, 'size_mb': current_size,
            }, os.path.join(args.output_dir, 'best_physical.pth'))
            print(f"  >>> New best: {best_acc1:.2f}%")

        # Epoch별 핵심 로그 기록
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1:>5} | {phase:>8} | {train_stats['loss']:>8.4f} | "
                    f"{test_stats['acc1']:>6.2f}% | {test_stats['acc5']:>6.2f}% | "
                    f"{current_size:>8.2f} | {current_reduction*100:>6.1f}% | "
                    f"{'*' if is_best else ' ':>5} | {epoch_time:>10}\n")

    # Final summary
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    final_size = compute_model_size_mb(model)

    final_acc1 = test_stats['acc1']
    final_acc5 = test_stats['acc5']

    print(f"\n{'='*70}")
    print("Physical-Only Pruning Complete!")
    print(f"{'='*70}")
    print(f"  Method: Physical-Only (CE loss)")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Final: {final_size:.2f} MB ({(1-final_size/original_size)*100:.1f}% reduced)")
    print(f"  Pruning 후 Acc@1: {pruning_end_acc1:.2f}% (size: {pruning_end_size:.2f}MB, {pruning_end_reduction*100:.1f}% reduced)")
    print(f"  Finetune 후 Acc@1: {final_acc1:.2f}% (최종)")
    print(f"  Time: {total_time}")

    # 최종 요약도 로그에 기록
    with open(log_path, 'a') as f:
        f.write("-" * 80 + "\n")
        f.write(f"\nFINAL RESULTS\n")
        f.write(f"  Method: Physical-Only (CE loss)\n")
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
        'method': 'physical_only',
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
    with open(os.path.join(args.output_dir, 'summary_physical.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Training log: {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Physical-Only Pruning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
