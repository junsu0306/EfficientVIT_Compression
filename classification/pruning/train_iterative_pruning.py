"""
Iterative Physical Pruning Training Script

기존 soft masking 방식 대신, 매 epoch마다 물리적으로 구조를 축소하는 방식.

알고리즘:
  1. Train 1 epoch (loss 감소)
  2. Importance 계산 (L2 norm 기반)
  3. Physical pruning (Conv 채널 축소, Linear dim 축소)
  4. Forward pass 검증
  5. 목표 compression ratio까지 반복
  6. Fine-tuning (추가 epochs)

Usage:
    python -m classification.pruning.train_iterative_pruning \
        --model EfficientViT_M4 \
        --data-path /path/to/imagenet \
        --resume efficientvit_m4.pth \
        --target-reduction 0.76 \
        --pruning-epochs 15 \
        --finetune-epochs 5 \
        --output-dir ./checkpoints/iterative_pruning
"""

import argparse
import datetime
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification.model import EfficientViT_M4
from classification.data.datasets import build_dataset
from classification.engine import evaluate
from classification import utils
from classification.pruning.structural_pruning import (
    IterativePhysicalPruner,
    compute_model_size_mb,
    validate_model_forward
)

from timm.utils import accuracy
from timm.data import Mixup


def get_args_parser():
    parser = argparse.ArgumentParser('Iterative Physical Pruning Training', add_help=False)

    # Model
    parser.add_argument('--model', default='EfficientViT_M4', type=str,
                        choices=['EfficientViT_M4'], help='Model architecture')
    parser.add_argument('--resume', default='', type=str,
                        help='Pretrained model checkpoint path')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='Number of classes')

    # Data
    parser.add_argument('--data-path', default='/workspace/etri_iitp/JS/EfficientViT/data/imagenet',
                        type=str, help='Dataset path')
    parser.add_argument('--data-set', default='IMNET', type=str,
                        choices=['CIFAR10', 'CIFAR100', 'IMNET', 'IMNET100', 'INAT', 'INAT19'],
                        help='Dataset type')
    parser.add_argument('--input-size', default=224, type=int, help='Image size')
    parser.add_argument('--batch-size', default=256, type=int, help='Batch size per GPU')
    parser.add_argument('--num-workers', default=8, type=int, help='Data loader workers')
    parser.add_argument('--pin-mem', action='store_true', help='Pin memory in DataLoader')
    parser.set_defaults(pin_mem=True)

    # Pruning
    parser.add_argument('--target-reduction', default=0.76, type=float,
                        help='Target compression ratio (e.g., 0.76 = 76% reduction)')
    parser.add_argument('--ffn-prune-per-epoch', default=0.08, type=float,
                        help='FFN hidden dimension pruning rate per epoch')
    parser.add_argument('--qk-prune-per-epoch', default=0.10, type=float,
                        help='Q/K dimension pruning rate per epoch')
    parser.add_argument('--min-ffn-ratio', default=0.15, type=float,
                        help='Minimum FFN hidden ratio to keep')
    parser.add_argument('--min-qk-ratio', default=0.20, type=float,
                        help='Minimum Q/K dimension ratio to keep')
    parser.add_argument('--warmup-epochs', default=0, type=int,
                        help='Warmup epochs before pruning starts')
    parser.add_argument('--pruning-epochs', default=15, type=int,
                        help='Maximum epochs for pruning phase')
    parser.add_argument('--finetune-epochs', default=10, type=int,
                        help='Fine-tuning epochs after pruning')

    # Training
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--min-lr', default=1e-6, type=float, help='Minimum LR for cosine')
    parser.add_argument('--weight-decay', default=0.05, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Total epochs (pruning + finetune)')
    parser.add_argument('--clip-grad', default=1.0, type=float, help='Gradient clipping norm')

    # Mixup / Cutmix
    parser.add_argument('--mixup', default=0.0, type=float, help='Mixup alpha')
    parser.add_argument('--cutmix', default=0.0, type=float, help='Cutmix alpha')
    parser.add_argument('--mixup-prob', default=0.0, type=float, help='Mixup probability')
    parser.add_argument('--mixup-switch-prob', default=0.5, type=float)
    parser.add_argument('--mixup-mode', default='batch', type=str)
    parser.add_argument('--smoothing', default=0.1, type=float, help='Label smoothing')

    # Output
    parser.add_argument('--output-dir', default='./checkpoints/iterative_pruning',
                        type=str, help='Output directory')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)

    # Distributed (placeholder)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False)

    return parser


def train_one_epoch_simple(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    criterion,
    mixup_fn=None,
    clip_grad: float = 1.0,
    scaler=None,
    print_freq: int = 100
):
    """
    간단한 1 epoch 학습 함수 (pruning 없이, loss 감소만)
    """
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Forward
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            loss = criterion(samples, outputs, targets) if hasattr(criterion, '__call__') and not isinstance(criterion, nn.Module) else criterion(outputs, targets)

        loss_value = loss.item()

        if not torch.isfinite(torch.tensor(loss_value)):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()

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
    print("Iterative Physical Pruning for EfficientViT")
    print(f"{'='*70}")

    # Setup
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # 1. Build Dataset
    # =========================================================================
    print("\n[1/6] Loading dataset...")

    dataset_train, _ = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"  Train samples: {len(dataset_train)}")
    print(f"  Val samples: {len(dataset_val)}")

    # =========================================================================
    # 2. Build Model
    # =========================================================================
    print("\n[2/6] Loading model...")

    model = EfficientViT_M4(num_classes=args.num_classes, pretrained=False)

    if args.resume:
        print(f"  Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)

    original_size = compute_model_size_mb(model)
    print(f"  Original model size: {original_size:.2f} MB")
    print(f"  Target reduction: {args.target_reduction*100:.1f}%")
    print(f"  Target size: {original_size * (1 - args.target_reduction):.2f} MB")

    # =========================================================================
    # 3. Setup Training
    # =========================================================================
    print("\n[3/6] Setting up training...")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    # Mixup
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            cutmix_minmax=None,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes
        )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.pruning_epochs + args.finetune_epochs,
        eta_min=args.min_lr
    )

    # =========================================================================
    # 4. Initialize Pruner
    # =========================================================================
    print("\n[4/6] Initializing pruner...")

    pruner = IterativePhysicalPruner(
        target_reduction=args.target_reduction,
        ffn_prune_per_epoch=args.ffn_prune_per_epoch,
        qk_prune_per_epoch=args.qk_prune_per_epoch,
        min_ffn_ratio=args.min_ffn_ratio,
        min_qk_ratio=args.min_qk_ratio,
        warmup_epochs=args.warmup_epochs,
        verbose=True
    )

    print(f"  FFN prune rate/epoch: {args.ffn_prune_per_epoch*100:.1f}%")
    print(f"  QK prune rate/epoch: {args.qk_prune_per_epoch*100:.1f}%")
    print(f"  Min FFN ratio: {args.min_ffn_ratio*100:.1f}%")
    print(f"  Min QK ratio: {args.min_qk_ratio*100:.1f}%")

    # =========================================================================
    # 5. Initial Evaluation
    # =========================================================================
    print("\n[5/6] Initial evaluation...")

    test_stats = evaluate(data_loader_val, model, device)
    print(f"  Initial Acc@1: {test_stats['acc1']:.2f}%")
    print(f"  Initial Acc@5: {test_stats['acc5']:.2f}%")

    # =========================================================================
    # 6. Training Loop
    # =========================================================================
    print("\n[6/6] Starting training...")

    history = {
        'train_loss': [],
        'val_acc1': [],
        'val_acc5': [],
        'model_size_mb': [],
        'reduction_ratio': [],
    }

    best_acc1 = 0.0
    start_time = time.time()

    total_epochs = args.pruning_epochs + args.finetune_epochs

    for epoch in range(total_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{total_epochs}")
        print(f"{'='*70}")

        # ----- Train one epoch -----
        train_stats = train_one_epoch_simple(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            criterion=criterion,
            mixup_fn=mixup_fn,
            clip_grad=args.clip_grad,
            scaler=scaler,
            print_freq=100
        )

        # ----- Physical Pruning (only during pruning phase) -----
        current_size = compute_model_size_mb(model)
        current_reduction = 1.0 - current_size / original_size

        if epoch < args.pruning_epochs and not pruner.target_reached:
            pruning_result = pruner.step(model, device)

            # Rebuild optimizer after pruning (parameters changed)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer.param_groups[0]['lr'],  # Keep current LR
                weight_decay=args.weight_decay
            )
            # 새 optimizer에 맞게 scheduler 재생성 (남은 epochs)
            remaining_epochs = total_epochs - epoch - 1
            if remaining_epochs > 0:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=remaining_epochs,
                    eta_min=args.min_lr
                )
        else:
            if epoch == args.pruning_epochs:
                print("\n>>> Pruning phase complete. Starting fine-tuning...")

        # Update current size after potential pruning
        current_size = compute_model_size_mb(model)
        current_reduction = 1.0 - current_size / original_size

        # ----- Evaluate -----
        test_stats = evaluate(data_loader_val, model, device)

        # ----- Scheduler step -----
        lr_scheduler.step()

        # ----- Save history -----
        history['train_loss'].append(train_stats['loss'])
        history['val_acc1'].append(test_stats['acc1'])
        history['val_acc5'].append(test_stats['acc5'])
        history['model_size_mb'].append(current_size)
        history['reduction_ratio'].append(current_reduction)

        # ----- Print summary -----
        print(f"\n[Epoch {epoch + 1} Summary]")
        print(f"  Train Loss: {train_stats['loss']:.4f}")
        print(f"  Val Acc@1: {test_stats['acc1']:.2f}%")
        print(f"  Val Acc@5: {test_stats['acc5']:.2f}%")
        print(f"  Model Size: {current_size:.2f} MB ({current_reduction*100:.1f}% reduced)")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ----- Save best model -----
        if test_stats['acc1'] > best_acc1:
            best_acc1 = test_stats['acc1']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc1': best_acc1,
                'model_size_mb': current_size,
                'reduction_ratio': current_reduction,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  >>> New best model saved! Acc@1: {best_acc1:.2f}%")

        # ----- Save checkpoint -----
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth'))

    # =========================================================================
    # Final Summary
    # =========================================================================
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"  Total time: {total_time_str}")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Final size: {current_size:.2f} MB")
    print(f"  Final reduction: {current_reduction*100:.1f}%")
    print(f"  Best Acc@1: {best_acc1:.2f}%")
    print(f"  Target: {args.target_reduction*100:.1f}%")
    print(f"  Target reached: {pruner.target_reached}")

    # Save final summary
    summary = {
        'original_size_mb': original_size,
        'final_size_mb': current_size,
        'reduction_ratio': current_reduction,
        'target_reduction': args.target_reduction,
        'target_reached': pruner.target_reached,
        'best_acc1': best_acc1,
        'total_epochs': total_epochs,
        'total_time': total_time_str,
        'history': history,
        'args': vars(args),
    }

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Iterative Physical Pruning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
