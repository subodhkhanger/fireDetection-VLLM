"""
Training script with FULL monitoring - predictions + gradients

This enhanced version monitors:
1. Class predictions at each epoch (fire vs smoke)
2. Gradient norms at each step
3. Training health checks
4. Automatic issue detection

Usage:
    python train_with_full_monitoring.py \
        --config configs/mac_m1_config.yaml \
        --data-dir data/FireTiny \
        --train-ann data/FireTiny/annotations/instances_train.json \
        --val-ann data/FireTiny/annotations/instances_val.json \
        --output-dir experiments/full_monitoring \
        --device cpu \
        --check-gradients-every 50  # Check gradients every 50 steps
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from models.fire_vit import build_fire_vit
from data import create_dataloaders
from losses import CompositeLoss
from utils.target_encoder import FCOSTargetEncoder
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils.epoch_monitor import EpochMonitor, quick_validation_check
from utils.gradient_monitor import GradientMonitor, check_gradients_quick


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Fire-ViT with Full Monitoring')

    parser.add_argument('--config', type=str, default='configs/mac_m1_config.yaml')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--train-ann', type=str, required=True)
    parser.add_argument('--val-ann', type=str, required=True)

    parser.add_argument('--output-dir', type=str, default='./experiments')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--num-workers', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=None)

    # Monitoring options
    parser.add_argument('--check-gradients-every', type=int, default=50,
                       help='Check gradients every N steps')
    parser.add_argument('--monitor-epoch-interval', type=int, default=1,
                       help='Run full monitoring every N epochs')

    return parser.parse_args()


def train_epoch_with_monitoring(
    model, train_loader, optimizer, scheduler, loss_fn, target_encoder,
    device, epoch, grad_monitor, use_amp=False, grad_clip_norm=1.0,
    gradient_accumulation_steps=1, check_gradients_every=50
):
    """
    Train for one epoch with gradient monitoring

    Returns:
        metrics dict
    """
    model.train()

    total_loss = 0.0
    loss_components = {}
    num_batches = len(train_loader)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    progress_bar = tqdm(
        enumerate(train_loader),
        total=num_batches,
        desc=f"Epoch {epoch}"
    )

    global_step = 0

    for batch_idx, (images, targets) in progress_bar:
        # Move to device
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in targets]

        # Forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                predictions = model(images)
                encoded_targets = target_encoder.encode(targets, predictions)
                loss, loss_dict = loss_fn(predictions, encoded_targets, epoch=epoch)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
        else:
            predictions = model(images)
            encoded_targets = target_encoder.encode(targets, predictions)
            loss, loss_dict = loss_fn(predictions, encoded_targets, epoch=epoch)
            loss = loss / gradient_accumulation_steps
            loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # ============================================================
            # GRADIENT MONITORING (before clipping)
            # ============================================================
            if global_step % check_gradients_every == 0:
                grad_info = grad_monitor.log_gradients(
                    step=global_step,
                    clip_norm=grad_clip_norm,
                    verbose=True
                )

                # Check for critical gradient issues
                if grad_info['health']['status'] == 'critical':
                    print("\n" + "!"*80)
                    print("CRITICAL GRADIENT ISSUE DETECTED!")
                    print("!"*80)
                    for warning in grad_info['health']['warnings']:
                        print(f"  {warning}")
                    print("\nConsider:")
                    print("  - Lowering learning rate")
                    print("  - Checking for data issues")
                    print("  - Reducing model complexity")
                    print("!"*80 + "\n")

            # Clip gradients
            if use_amp:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            # Optimizer step
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            global_step += 1

        # Accumulate metrics
        total_loss += loss.item() * gradient_accumulation_steps

        for key, value in loss_dict.items():
            if key not in loss_components:
                loss_components[key] = 0.0
            if isinstance(value, (int, float)):
                loss_components[key] += value

        # Update progress bar with gradient info
        if batch_idx % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']

            # Quick gradient check
            grad_status = check_gradients_quick(model)
            grad_emoji = {
                'ok': '✓',
                'vanishing': '⬇️',
                'exploding': '⬆️',
                'nan': '❌',
                'none': '-'
            }.get(grad_status, '?')

            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.6f}',
                'grad': grad_emoji
            })

    # Step scheduler
    if scheduler is not None:
        scheduler.step()

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    metrics = {
        'total': avg_loss,
        **avg_components
    }

    return metrics


def main():
    """Main training function"""
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    monitor_dir = output_dir / 'monitoring'
    grad_monitor_dir = output_dir / 'gradient_monitoring'

    # Setup logger
    logger = setup_logger('fire_vit_full_monitor', log_dir=log_dir)
    logger.info("="*80)
    logger.info("Fire-ViT Training with FULL Monitoring")
    logger.info("="*80)
    logger.info(f"Monitoring predictions AND gradients")
    logger.info(f"Output: {output_dir}")

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")

    # For FireTiny dataset, images are in data_dir/images/
    # The create_dataloaders expects the parent images directory
    data_dir_path = Path(args.data_dir)
    if (data_dir_path / 'images').exists():
        # FireTiny structure: data/FireTiny/images/train/
        image_dir = data_dir_path / 'images'
    else:
        # Standard structure
        image_dir = data_dir_path

    dataloaders = create_dataloaders(
        data_dir=str(image_dir),
        train_annotation=args.train_ann,
        val_annotation=args.val_ann,
        test_annotation=None,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers,
        img_size=config['model']['input_size'][0],
        pin_memory=(device.type == 'cuda')
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Val: {len(val_loader.dataset)} samples")

    # Build model
    logger.info("Building model...")
    model = build_fire_vit(config).to(device)

    # Create loss, optimizer, scheduler
    logger.info("Creating training components...")

    # Get strides from config
    fpn_strides = config['model'].get('fpn_strides', [8, 16, 32, 64])

    loss_fn = CompositeLoss(
        focal_alpha=config['loss']['focal_alpha'],
        focal_gamma=config['loss']['focal_gamma'],
        loss_weights={
            'focal': config['loss']['focal_weight'],
            'ciou': config['loss']['ciou_weight'],
            'dice': config['loss']['dice_weight'],
            'centerness': 1.0,  # Centerness loss weight
            'attention': config['loss'].get('attention_weight', 0.0),
            'auxiliary': config['loss'].get('auxiliary_weight', 0.0)
        },
        use_attention_loss=(config['loss'].get('attention_weight', 0.0) > 0),
        use_auxiliary_loss=(config['loss'].get('auxiliary_weight', 0.0) > 0),
        strides=fpn_strides  # Pass strides for bbox normalization
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['base_lr'],
        weight_decay=config['training']['weight_decay']
    )

    total_epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=config['training'].get('min_lr', 1e-6)
    )

    # Target encoder
    target_encoder = FCOSTargetEncoder(
        num_classes=config['model']['num_classes']
    )

    # ============================================================
    # SETUP MONITORING
    # ============================================================
    logger.info("Setting up monitoring...")

    # Prediction monitor
    pred_monitor = EpochMonitor(
        num_classes=config['model']['num_classes'] - 1,
        save_dir=monitor_dir,
        class_names={0: 'fire', 1: 'smoke'}
    )

    # Gradient monitor
    grad_monitor = GradientMonitor(
        model=model,
        save_dir=grad_monitor_dir
    )

    logger.info(f"✓ Gradient monitoring every {args.check_gradients_every} steps")
    logger.info(f"✓ Prediction monitoring every {args.monitor_epoch_interval} epoch(s)")

    # Training loop
    logger.info("="*80)
    logger.info("Starting training...")
    logger.info("="*80)

    best_val_loss = float('inf')
    start_epoch = 0

    for epoch in range(start_epoch, total_epochs):
        logger.info(f"\nEpoch {epoch+1}/{total_epochs}")
        logger.info("-"*80)

        # Train with gradient monitoring
        train_metrics = train_epoch_with_monitoring(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            target_encoder=target_encoder,
            device=device,
            epoch=epoch+1,
            grad_monitor=grad_monitor,
            use_amp=config['training'].get('use_amp', False),
            grad_clip_norm=config['training'].get('clip_grad_norm', 1.0),
            gradient_accumulation_steps=config['training'].get('accumulation_steps', 1),
            check_gradients_every=args.check_gradients_every
        )

        train_loss = train_metrics['total']
        logger.info(f"Train Loss: {train_loss:.4f}")

        # Simple validation (just loss, no full metrics)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in t.items()} for t in targets]

                predictions = model(images)
                encoded_targets = target_encoder.encode(targets, predictions)
                loss, _ = loss_fn(predictions, encoded_targets, epoch=epoch+1)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logger.info(f"Val Loss: {val_loss:.4f}")

        # ============================================================
        # PREDICTION MONITORING
        # ============================================================
        if (epoch + 1) % args.monitor_epoch_interval == 0:
            logger.info("\nRunning prediction monitoring...")

            val_predictions, val_targets = quick_validation_check(
                model, val_loader, device,
                conf_threshold=0.3,
                max_batches=10
            )

            pred_monitor.log_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_predictions=val_predictions,
                val_targets=val_targets,
                conf_threshold=0.3,
                verbose=True
            )

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"✓ New best: {val_loss:.4f}")

        metrics_dict = {**train_metrics, 'val_total': val_loss}
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            metrics_dict,
            checkpoint_dir,
            is_best=is_best
        )

    # Save monitoring history
    logger.info("\n" + "="*80)
    logger.info("Training complete!")
    logger.info("="*80)

    pred_monitor.save_history()
    grad_monitor.save_history()

    try:
        pred_monitor.plot_history()
        grad_monitor.plot_gradients()
        logger.info(f"✓ Plots saved")
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

    logger.info(f"\nMonitoring saved in:")
    logger.info(f"  Predictions: {monitor_dir}")
    logger.info(f"  Gradients: {grad_monitor_dir}")


if __name__ == '__main__':
    main()
