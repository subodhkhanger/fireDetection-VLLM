"""
Training script with epoch-by-epoch monitoring

This is an enhanced version of train.py that monitors model predictions
at every epoch to catch issues early (like predicting only one class).

Usage:
    python train_with_monitoring.py \
        --config configs/mac_m1_config.yaml \
        --data-dir data/FireTiny \
        --train-ann data/FireTiny/annotations/instances_train.json \
        --val-ann data/FireTiny/annotations/instances_val.json \
        --output-dir experiments/monitored_training \
        --device cpu
"""

import os
import argparse
import yaml
import torch
from pathlib import Path

from models.fire_vit import build_fire_vit
from data import create_dataloaders
from losses import CompositeLoss
from utils.trainer import Trainer
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.logger import setup_logger, TensorBoardLogger
from utils.epoch_monitor import EpochMonitor, quick_validation_check


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Fire-ViT with Monitoring')

    parser.add_argument('--config', type=str, default='configs/mac_m1_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--train-ann', type=str, required=True,
                       help='Path to training annotations (COCO format)')
    parser.add_argument('--val-ann', type=str, required=True,
                       help='Path to validation annotations')

    parser.add_argument('--output-dir', type=str, default='./experiments',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')

    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')

    parser.add_argument('--monitor-interval', type=int, default=1,
                       help='Run monitoring every N epochs')
    parser.add_argument('--monitor-conf-threshold', type=float, default=0.3,
                       help='Confidence threshold for monitoring')

    return parser.parse_args()


def main():
    """Main training function with monitoring"""
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    monitor_dir = output_dir / 'monitoring'

    # Setup logger
    logger = setup_logger('fire_vit_monitored', log_dir=log_dir)
    logger.info("="*80)
    logger.info("Fire-ViT Training with Monitoring")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {output_dir}")

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

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Build model
    logger.info("Building model...")
    model = build_fire_vit(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: Fire-ViT")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create loss function
    logger.info("Creating loss function...")
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
        use_auxiliary_loss=(config['loss'].get('auxiliary_weight', 0.0) > 0)
    )

    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['base_lr'],
        weight_decay=config['training']['weight_decay']
    )

    # Create scheduler
    logger.info("Creating scheduler...")
    total_epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=config['training'].get('min_lr', 1e-6)
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        use_amp=config['training'].get('use_amp', False),
        use_ema=config['training'].get('use_ema', False),
        gradient_accumulation_steps=config['training'].get('accumulation_steps', 1),
        grad_clip_norm=config['training'].get('clip_grad_norm', 1.0),
        num_classes=config['model']['num_classes']
    )

    # ============================================================
    # MONITORING SETUP
    # ============================================================
    logger.info("Setting up epoch monitoring...")
    monitor = EpochMonitor(
        num_classes=config['model']['num_classes'] - 1,  # Exclude background
        save_dir=monitor_dir,
        class_names={0: 'fire', 1: 'smoke'}
    )

    # Resume if requested
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1

    # Training loop
    logger.info("="*80)
    logger.info("Starting training with monitoring...")
    logger.info("="*80)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, total_epochs):
        logger.info(f"\nEpoch {epoch+1}/{total_epochs}")
        logger.info("-"*80)

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch+1)
        train_loss = train_metrics['total']

        # Validate
        val_metrics = trainer.validate(val_loader)
        val_loss = val_metrics['total']

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")

        # ============================================================
        # EPOCH MONITORING
        # ============================================================
        if (epoch + 1) % args.monitor_interval == 0:
            logger.info("\nRunning epoch monitoring...")

            # Get predictions on validation set
            val_predictions, val_targets = quick_validation_check(
                model,
                val_loader,
                device,
                conf_threshold=args.monitor_conf_threshold,
                max_batches=10  # Check first 10 batches
            )

            # Analyze and log
            monitor_result = monitor.log_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_predictions=val_predictions,
                val_targets=val_targets,
                conf_threshold=args.monitor_conf_threshold,
                verbose=True
            )

            # Check for critical issues
            if monitor_result['health']['status'] == 'critical':
                logger.warning("⚠️  CRITICAL ISSUES DETECTED!")
                logger.warning("Consider stopping training and investigating the issues.")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"✓ New best validation loss: {val_loss:.4f}")

        # Combine metrics
        all_metrics = {**train_metrics, **val_metrics}

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            all_metrics,
            checkpoint_dir,
            is_best=is_best
        )

        # Update scheduler
        scheduler.step()

    # Final monitoring summary
    logger.info("\n" + "="*80)
    logger.info("Training complete!")
    logger.info("="*80)

    # Save monitoring history
    monitor.save_history()
    logger.info(f"Monitoring history saved to: {monitor_dir}")

    # Create plots
    try:
        monitor.plot_history()
        logger.info(f"Monitoring plots saved to: {monitor_dir}/training_plots.png")
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

    logger.info("\nFinal Summary:")
    logger.info(f"  Best validation loss: {best_val_loss:.4f}")
    logger.info(f"  Checkpoints saved in: {checkpoint_dir}")
    logger.info(f"  Monitoring logs in: {monitor_dir}")


if __name__ == '__main__':
    main()
