"""
Checkpoint management utilities
"""

import torch
import os
from pathlib import Path
import glob


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    checkpoint_dir,
    filename=None,
    is_best=False,
    ema=None
):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename (default: checkpoint_epoch_{epoch}.pth)
        is_best: Whether this is the best model so far
        ema: EMA model (optional)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"

    checkpoint_path = checkpoint_dir / filename

    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
    }

    # Add EMA if available
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved checkpoint: {checkpoint_path}")

    # Save as best model if applicable
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model: {best_path}")

    # Also save latest
    latest_path = checkpoint_dir / "latest.pth"
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    ema=None,
    device='cuda'
):
    """
    Load model checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: LR scheduler (optional)
        ema: EMA model (optional)
        device: Device to load checkpoint on

    Returns:
        epoch: Epoch number from checkpoint
        metrics: Metrics from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load EMA
    if ema is not None and 'ema_shadow' in checkpoint:
        ema.shadow = checkpoint['ema_shadow']

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"✓ Loaded checkpoint from epoch {epoch}")

    return epoch, metrics


def get_latest_checkpoint(checkpoint_dir):
    """
    Get path to latest checkpoint

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        latest_checkpoint: Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Check for latest.pth
    latest_path = checkpoint_dir / "latest.pth"
    if latest_path.exists():
        return str(latest_path)

    # Find all checkpoints
    checkpoints = glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.pth"))

    if not checkpoints:
        return None

    # Sort by epoch number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    return checkpoints[-1] if checkpoints else None


def cleanup_checkpoints(checkpoint_dir, keep_last_n=5, keep_best=True):
    """
    Remove old checkpoints, keeping only the most recent ones

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to always keep best_model.pth
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Find all epoch checkpoints
    checkpoints = glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.pth"))

    if len(checkpoints) <= keep_last_n:
        return

    # Sort by epoch (oldest first)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Remove oldest checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        os.remove(checkpoint)
        print(f"Removed old checkpoint: {checkpoint}")


if __name__ == "__main__":
    # Test checkpoint utilities
    print("Testing checkpoint utilities...")

    import tempfile
    from torch import nn

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy model and optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Save checkpoint
        metrics = {'train_loss': 0.5, 'val_loss': 0.6}
        save_checkpoint(
            model, optimizer, scheduler,
            epoch=10,
            metrics=metrics,
            checkpoint_dir=tmpdir,
            is_best=True
        )

        print(f"\n✓ Checkpoint saved to {tmpdir}")

        # Load checkpoint
        model2 = nn.Linear(10, 10)
        optimizer2 = torch.optim.Adam(model2.parameters())
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10)

        latest = get_latest_checkpoint(tmpdir)
        print(f"✓ Latest checkpoint: {latest}")

        epoch, loaded_metrics = load_checkpoint(
            latest, model2, optimizer2, scheduler2
        )

        print(f"✓ Loaded epoch: {epoch}")
        print(f"✓ Loaded metrics: {loaded_metrics}")

        # Verify weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

        print("\n✅ Checkpoint tests passed!")
