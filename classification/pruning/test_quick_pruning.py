"""
Quick Test Script for Physical Pruning
- 카테고리별 10개 이미지만 사용 (빠른 테스트)
- Pruning + Finetuning 전체 플로우 검증

Usage:
    python -m classification.pruning.test_quick_pruning --mode physical
    python -m classification.pruning.test_quick_pruning --mode combined
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification.model.build import EfficientViT_M4
from classification.pruning.structural_pruning import (
    IterativePhysicalPruner,
    compute_model_size_mb,
    validate_model_forward,
    apply_iterative_physical_pruning
)


def create_dummy_dataloader(batch_size=8, num_batches=10, num_classes=1000):
    """
    테스트용 더미 데이터로더 생성
    실제 이미지 없이 random tensor로 테스트
    """
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size, num_classes):
            self.size = size
            self.num_classes = num_classes

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Random image and label
            img = torch.randn(3, 224, 224)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return img, label

    dataset = DummyDataset(batch_size * num_batches, num_classes)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )


def train_one_epoch_dummy(model, data_loader, optimizer, device, criterion):
    """간단한 1 epoch 학습 (테스트용)"""
    model.train()
    total_loss = 0
    num_batches = 0

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_dummy(model, data_loader, device):
    """간단한 평가 (테스트용)"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = targets.to(device)

            outputs = model(samples)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / max(total, 1)


def compute_lambda_regularization(model, lambda_ffn, lambda_qk, lambda_v):
    """λ regularization 계산 (Combined용)"""
    reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, module in model.named_modules():
        # FFN (pw1, pw2)
        if 'pw1' in name and hasattr(module, 'c'):
            reg_loss += lambda_ffn * torch.sum(module.c.weight ** 2)
        if 'pw2' in name and hasattr(module, 'c'):
            reg_loss += lambda_ffn * torch.sum(module.c.weight ** 2)

        # QKV
        if 'qkvs' in name and hasattr(module, 'c'):
            reg_loss += lambda_qk * torch.sum(module.c.weight ** 2)

        # DW
        if 'dws' in name and hasattr(module, 'c'):
            reg_loss += lambda_qk * torch.sum(module.c.weight ** 2)

    return reg_loss


def test_physical_only(args):
    """Physical-Only 방식 테스트"""
    print("\n" + "="*60)
    print("Physical-Only Quick Test")
    print("="*60)

    device = torch.device(args.device)

    # 모델 로드
    print("\n[1] Loading model...")
    model = EfficientViT_M4(pretrained=False)
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded: {args.resume}")
    model = model.to(device)

    original_size = compute_model_size_mb(model)
    print(f"  Original size: {original_size:.2f} MB")

    # 데이터로더
    print("\n[2] Creating dummy dataloader...")
    train_loader = create_dummy_dataloader(args.batch_size, args.num_batches)
    val_loader = create_dummy_dataloader(args.batch_size, 5)
    print(f"  Train batches: {args.num_batches}, Val batches: 5")

    # Pruner 초기화
    print("\n[3] Initializing pruner...")
    pruner = IterativePhysicalPruner(
        target_reduction=args.target_reduction,
        ffn_prune_per_epoch=args.ffn_prune_per_epoch,
        qk_prune_per_epoch=args.qk_prune_per_epoch,
        min_ffn_ratio=args.min_ffn_ratio,
        min_qk_ratio=args.min_qk_ratio,
        verbose=True
    )
    print(f"  Target: {args.target_reduction*100:.0f}% reduction")
    print(f"  FFN prune/epoch: {args.ffn_prune_per_epoch*100:.0f}%")
    print(f"  QK prune/epoch: {args.qk_prune_per_epoch*100:.0f}%")

    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print("\n[4] Starting training loop...")
    total_epochs = args.pruning_epochs + args.finetune_epochs
    start_time = time.time()

    for epoch in range(total_epochs):
        is_pruning = epoch < args.pruning_epochs
        phase = "[PRUNING]" if is_pruning else "[FINETUNE]"

        # Train
        loss = train_one_epoch_dummy(model, train_loader, optimizer, device, criterion)

        # Pruning (pruning phase only)
        if is_pruning and not pruner.target_reached:
            pruner.step(model, device)
            # Rebuild optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # Eval
        acc = evaluate_dummy(model, val_loader, device)
        current_size = compute_model_size_mb(model)
        reduction = 1.0 - current_size / original_size

        print(f"Epoch {epoch+1:2d}/{total_epochs} {phase} | "
              f"Loss: {loss:.4f} | Acc: {acc:.1f}% | "
              f"Size: {current_size:.2f}MB ({reduction*100:.1f}% reduced)")

        if pruner.target_reached and is_pruning:
            print("  >>> Target reached!")

    # Summary
    elapsed = time.time() - start_time
    final_size = compute_model_size_mb(model)
    final_reduction = 1.0 - final_size / original_size

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Final: {final_size:.2f} MB")
    print(f"  Reduction: {final_reduction*100:.1f}%")
    print(f"  Target reached: {pruner.target_reached}")
    print(f"  Time: {elapsed:.1f}s")

    # Forward validation
    valid = validate_model_forward(model, device)
    print(f"  Forward pass valid: {valid}")

    return model


def test_combined(args):
    """Combined 방식 테스트 (λ + Physical)"""
    print("\n" + "="*60)
    print("Combined (λ + Physical) Quick Test")
    print("="*60)

    device = torch.device(args.device)

    # 모델 로드
    print("\n[1] Loading model...")
    model = EfficientViT_M4(pretrained=False)
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded: {args.resume}")
    model = model.to(device)

    original_size = compute_model_size_mb(model)
    print(f"  Original size: {original_size:.2f} MB")

    # 데이터로더
    print("\n[2] Creating dummy dataloader...")
    train_loader = create_dummy_dataloader(args.batch_size, args.num_batches)
    val_loader = create_dummy_dataloader(args.batch_size, 5)

    # Pruner 초기화
    print("\n[3] Initializing pruner...")
    pruner = IterativePhysicalPruner(
        target_reduction=args.target_reduction,
        ffn_prune_per_epoch=args.ffn_prune_per_epoch,
        qk_prune_per_epoch=args.qk_prune_per_epoch,
        min_ffn_ratio=args.min_ffn_ratio,
        min_qk_ratio=args.min_qk_ratio,
        verbose=True
    )

    # Lambda 설정
    lambda_ffn = args.lambda_ffn
    lambda_qk = args.lambda_qk
    lambda_v = args.lambda_v
    print(f"  λ_FFN: {lambda_ffn}, λ_QK: {lambda_qk}, λ_V: {lambda_v}")
    print(f"  λ ratio: {lambda_ffn/lambda_v:.0f}:{lambda_qk/lambda_v:.0f}:1")

    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print("\n[4] Starting training loop...")
    total_epochs = args.pruning_epochs + args.finetune_epochs
    start_time = time.time()

    for epoch in range(total_epochs):
        is_pruning = epoch < args.pruning_epochs
        phase = "[PRUNING]" if is_pruning else "[FINETUNE]"

        # Train with λ regularization
        model.train()
        total_loss = 0
        total_reg = 0
        num_batches = 0

        for samples, targets in train_loader:
            samples = samples.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(samples)
            ce_loss = criterion(outputs, targets)
            reg_loss = compute_lambda_regularization(model, lambda_ffn, lambda_qk, lambda_v)
            loss = ce_loss + reg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reg += reg_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_reg = total_reg / max(num_batches, 1)

        # Pruning (pruning phase only)
        if is_pruning and not pruner.target_reached:
            pruner.step(model, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # Eval
        acc = evaluate_dummy(model, val_loader, device)
        current_size = compute_model_size_mb(model)
        reduction = 1.0 - current_size / original_size

        print(f"Epoch {epoch+1:2d}/{total_epochs} {phase} | "
              f"Loss: {avg_loss:.4f} (reg: {avg_reg:.4f}) | Acc: {acc:.1f}% | "
              f"Size: {current_size:.2f}MB ({reduction*100:.1f}% reduced)")

    # Summary
    elapsed = time.time() - start_time
    final_size = compute_model_size_mb(model)
    final_reduction = 1.0 - final_size / original_size

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print(f"  Method: Combined (λ + Physical)")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Final: {final_size:.2f} MB")
    print(f"  Reduction: {final_reduction*100:.1f}%")
    print(f"  Target reached: {pruner.target_reached}")
    print(f"  Time: {elapsed:.1f}s")

    valid = validate_model_forward(model, device)
    print(f"  Forward pass valid: {valid}")

    return model


def main():
    parser = argparse.ArgumentParser('Quick Pruning Test')

    # Mode
    parser.add_argument('--mode', default='physical', choices=['physical', 'combined'],
                        help='Test mode: physical or combined')

    # Model
    parser.add_argument('--resume', default='', type=str,
                        help='Pretrained checkpoint (optional)')

    # Quick test settings
    parser.add_argument('--batch-size', default=8, type=int,
                        help='Batch size for quick test')
    parser.add_argument('--num-batches', default=10, type=int,
                        help='Number of batches per epoch (10 batches × 8 = 80 samples)')

    # Pruning (FFN:QK:V = 10:2:1)
    parser.add_argument('--target-reduction', default=0.50, type=float,
                        help='Target reduction (0.5 = 50%% for quick test)')
    parser.add_argument('--ffn-prune-per-epoch', default=0.10, type=float)
    parser.add_argument('--qk-prune-per-epoch', default=0.02, type=float)
    parser.add_argument('--min-ffn-ratio', default=0.10, type=float)
    parser.add_argument('--min-qk-ratio', default=0.50, type=float)

    # Epochs (빠른 테스트)
    parser.add_argument('--pruning-epochs', default=5, type=int,
                        help='Pruning epochs (quick test: 5)')
    parser.add_argument('--finetune-epochs', default=2, type=int,
                        help='Finetune epochs (quick test: 2)')

    # Lambda (Combined only)
    parser.add_argument('--lambda-ffn', default=1e-3, type=float)
    parser.add_argument('--lambda-qk', default=2e-4, type=float)
    parser.add_argument('--lambda-v', default=1e-4, type=float)

    # Training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Quick Pruning Test")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches/epoch: {args.num_batches}")
    print(f"Target reduction: {args.target_reduction*100:.0f}%")
    print(f"Epochs: {args.pruning_epochs} pruning + {args.finetune_epochs} finetune")

    if args.mode == 'physical':
        test_physical_only(args)
    else:
        test_combined(args)


if __name__ == '__main__':
    main()
