#!/usr/bin/env python3
"""
Visualize Fire-ViT training results

Usage:
    # View training curves
    python scripts/visualize_results.py \
        --log-dir experiments/mac_m1_full/logs \
        --output results/training_curves.png

    # View predictions
    python scripts/visualize_results.py \
        --mode predictions \
        --checkpoint experiments/mac_m1_full/checkpoints/best_model.pth \
        --images data/FireTiny/images/test \
        --output results/predictions.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re


def parse_log_file(log_file):
    """Parse training log file to extract metrics"""

    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    with open(log_file, 'r') as f:
        for line in f:
            # Extract epoch number
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            # Extract training loss
            train_loss_match = re.search(r'loss=([\d.]+)', line)
            if train_loss_match and 'Epoch' in line:
                loss = float(train_loss_match.group(1))
                if current_epoch not in metrics['epochs']:
                    metrics['epochs'].append(current_epoch)
                    metrics['train_loss'].append(loss)

            # Extract learning rate
            lr_match = re.search(r'lr=([\d.e-]+)', line)
            if lr_match and len(metrics['learning_rate']) < len(metrics['train_loss']):
                lr = float(lr_match.group(1))
                metrics['learning_rate'].append(lr)

    return metrics


def plot_training_curves(log_dir, output_file):
    """Plot training curves from log directory"""

    log_dir = Path(log_dir)

    # Find log file
    log_files = list(log_dir.glob('train_*.log'))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    log_file = log_files[0]
    print(f"Parsing log file: {log_file}")

    metrics = parse_log_file(log_file)

    if not metrics['epochs']:
        print("No metrics found in log file")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot loss
    axes[0].plot(metrics['epochs'], metrics['train_loss'],
                 marker='o', label='Training Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot learning rate
    if metrics['learning_rate']:
        axes[1].plot(metrics['epochs'][:len(metrics['learning_rate'])],
                     metrics['learning_rate'],
                     marker='o', color='orange', label='Learning Rate', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

    plt.tight_layout()

    # Save
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves to {output_file}")

    # Also show statistics
    print()
    print("Training Statistics:")
    print(f"  Epochs: {len(metrics['epochs'])}")
    print(f"  Initial Loss: {metrics['train_loss'][0]:.4f}")
    print(f"  Final Loss: {metrics['train_loss'][-1]:.4f}")
    print(f"  Best Loss: {min(metrics['train_loss']):.4f}")
    print(f"  Improvement: {(metrics['train_loss'][0] - metrics['train_loss'][-1]) / metrics['train_loss'][0] * 100:.1f}%")


def show_checkpoint_info(checkpoint_dir):
    """Show information about saved checkpoints"""

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return

    checkpoints = list(checkpoint_dir.glob('*.pth'))

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print()
    print("="*80)
    print("Saved Checkpoints")
    print("="*80)

    for ckpt in sorted(checkpoints):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"  {ckpt.name:30s} - {size_mb:6.1f} MB")

    print()
    print("Use checkpoints for inference:")
    print(f"  python inference.py \\")
    print(f"      --checkpoint {checkpoint_dir}/best_model.pth \\")
    print(f"      --input image.jpg \\")
    print(f"      --save-vis --show")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Fire-ViT training results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--log-dir', help='Directory containing training logs')
    parser.add_argument('--checkpoint-dir', help='Directory containing checkpoints')
    parser.add_argument('--output', default='results/training_curves.png',
                       help='Output file for plots')
    args = parser.parse_args()

    if args.log_dir:
        plot_training_curves(args.log_dir, args.output)

    if args.checkpoint_dir:
        show_checkpoint_info(args.checkpoint_dir)

    if not args.log_dir and not args.checkpoint_dir:
        print("Please specify --log-dir or --checkpoint-dir")
        print()
        print("Examples:")
        print("  # View training curves")
        print("  python scripts/visualize_results.py \\")
        print("      --log-dir experiments/mac_m1_full/logs")
        print()
        print("  # View checkpoint info")
        print("  python scripts/visualize_results.py \\")
        print("      --checkpoint-dir experiments/mac_m1_full/checkpoints")
        print()
        print("  # Both")
        print("  python scripts/visualize_results.py \\")
        print("      --log-dir experiments/mac_m1_full/logs \\")
        print("      --checkpoint-dir experiments/mac_m1_full/checkpoints")


if __name__ == '__main__':
    main()
