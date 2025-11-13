"""
Training script for Fire-ViT

Complete training pipeline with:
- Data loading
- Model initialization
- Training loop with AMP and EMA
- Validation
- Checkpointing
- Logging
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path

from models.fire_vit import build_fire_vit
from data import create_dataloaders
from losses import CompositeLoss
from utils.trainer import Trainer
from utils.metrics import COCOEvaluator, compute_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.logger import setup_logger, TensorBoardLogger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Fire-ViT')

    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--train-ann', type=str, required=True,
                       help='Path to training annotations (COCO format)')
    parser.add_argument('--val-ann', type=str, required=True,
                       help='Path to validation annotations')
    parser.add_argument('--test-ann', type=str, default=None,
                       help='Path to test annotations (optional)')

    parser.add_argument('--output-dir', type=str, default='./experiments',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers')

    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')

    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (fewer batches)')

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['base_lr'] = args.lr

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'

    # Setup logger
    logger = setup_logger('fire_vit', log_dir=log_dir)
    logger.info("="*80)
    logger.info("Fire-ViT Training")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {output_dir}")

    # Setup TensorBoard
    tb_logger = TensorBoardLogger(log_dir / 'tensorboard')

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        train_annotation=args.train_ann,
        val_annotation=args.val_ann,
        test_annotation=args.test_ann,
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
    model_info = model.get_model_info()

    logger.info(f"Model: {model_info['model_name']}")
    logger.info(f"Total parameters: {model_info['total_params']:,}")
    logger.info(f"  Backbone: {model_info['backbone_params']:,}")
    logger.info(f"  Neck: {model_info['neck_params']:,}")
    logger.info(f"  Head: {model_info['head_params']:,}")

    # Create loss function
    logger.info("Creating loss function...")

    # Get strides from config
    fpn_strides = config['model'].get('fpn_strides', [8, 16, 32, 64])

    loss_fn = CompositeLoss(
        focal_alpha=config['loss']['focal_alpha'],
        focal_gamma=config['loss']['focal_gamma'],
        loss_weights={
            'focal': config['loss']['focal_weight'],
            'ciou': config['loss']['ciou_weight'],
            'dice': config['loss']['dice_weight'],
            'attention': config['loss']['attention_weight'],
            'centerness': 1.0
        },
        use_attention_loss=(config['loss']['attention_weight'] > 0),
        strides=fpn_strides,  # Pass strides for bbox normalization
        use_l1_loss=True  # Use L1 loss for stability instead of CIoU
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
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config['training']['warmup_lr'] / config['training']['base_lr'],
        end_factor=1.0,
        total_iters=config['training']['warmup_epochs']
    )

    # Main scheduler
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=config['training']['min_lr']
    )

    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config['training']['warmup_epochs']]
    )

    fpn_strides = config['model'].get('fpn_strides', [8, 16, 32, 64])

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        use_amp=config['training']['use_amp'],
        use_ema=config['training']['use_ema'],
        ema_decay=config['training']['ema_decay'],
        gradient_accumulation_steps=config['training']['accumulation_steps'],
        grad_clip_norm=config['training']['clip_grad_norm'],
        log_interval=config['logging']['log_interval'],
        num_classes=config['model'].get('num_classes', 3),
        strides=fpn_strides
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, metrics = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            ema=trainer.ema if trainer.use_ema else None,
            device=device
        )
        best_val_loss = metrics.get('val_loss', float('inf'))
        start_epoch += 1

    # Evaluation only
    if args.eval_only:
        logger.info("Running evaluation...")
        val_metrics = trainer.validate(val_loader, epoch=0)

        logger.info("Validation Metrics:")
        for key, value in val_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        if 'test' in dataloaders:
            logger.info("Running test evaluation...")
            test_loader = dataloaders['test']
            test_metrics = trainer.validate(test_loader, epoch=0)

            logger.info("Test Metrics:")
            for key, value in test_metrics.items():
                logger.info(f"  {key}: {value:.4f}")

        return

    # Training loop
    logger.info("="*80)
    logger.info("Starting training...")
    logger.info("="*80)

    num_epochs = config['training']['epochs']

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 80)

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_metrics = trainer.validate(val_loader, epoch)

        # Log metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")

        # TensorBoard logging
        for key, value in train_metrics.items():
            tb_logger.log_scalar(f'train/{key}', value, epoch)

        for key, value in val_metrics.items():
            tb_logger.log_scalar(f'val/{key}', value, epoch)

        tb_logger.log_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            logger.info(f"✓ New best validation loss: {best_val_loss:.4f}")

        if (epoch + 1) % config['logging']['save_interval'] == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={**train_metrics, **val_metrics},
                checkpoint_dir=checkpoint_dir,
                is_best=is_best,
                ema=trainer.ema if trainer.use_ema else None
            )

        # Debug mode: only train for a few epochs
        if args.debug and epoch >= 2:
            logger.info("Debug mode: stopping early")
            break

    # Final evaluation on test set (if available)
    if 'test' in dataloaders:
        logger.info("="*80)
        logger.info("Final evaluation on test set...")
        logger.info("="*80)

        test_loader = dataloaders['test']
        test_metrics = trainer.validate(test_loader, epoch=num_epochs)

        logger.info("Test Metrics:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    # Close loggers
    tb_logger.close()

    logger.info("="*80)
    logger.info("Training complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
