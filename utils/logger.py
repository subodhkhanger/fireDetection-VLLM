"""
Logging utilities for training
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name='fire_vit', log_dir=None, level=logging.INFO):
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level

    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'train_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


class TensorBoardLogger:
    """
    TensorBoard logger wrapper

    Provides simple interface for logging scalars, images, etc.
    """

    def __init__(self, log_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
            print(f"✓ TensorBoard logging enabled: {log_dir}")
        except ImportError:
            print("⚠️  TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False

    def log_scalar(self, tag, value, step):
        """Log scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars"""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag, image, step):
        """Log image"""
        if self.enabled:
            self.writer.add_image(tag, image, step)

    def log_histogram(self, tag, values, step):
        """Log histogram"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def log_text(self, tag, text, step):
        """Log text"""
        if self.enabled:
            self.writer.add_text(tag, text, step)

    def close(self):
        """Close writer"""
        if self.enabled and self.writer is not None:
            self.writer.close()


class WandBLogger:
    """
    Weights & Biases logger wrapper

    Optional integration with wandb for experiment tracking
    """

    def __init__(self, project_name, config=None, name=None):
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project_name,
                config=config,
                name=name
            )
            self.enabled = True
            print(f"✓ WandB logging enabled: {project_name}")
        except ImportError:
            print("⚠️  WandB not available. Install with: pip install wandb")
            self.enabled = False
            self.wandb = None

    def log(self, metrics, step=None):
        """Log metrics"""
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def log_image(self, key, image, caption=None):
        """Log image"""
        if self.enabled:
            self.wandb.log({key: self.wandb.Image(image, caption=caption)})

    def finish(self):
        """Finish run"""
        if self.enabled and self.run is not None:
            self.run.finish()


if __name__ == "__main__":
    # Test logger
    print("Testing logger setup...")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup logger
        logger = setup_logger('test_logger', log_dir=tmpdir)

        logger.info("This is an info message")
        logger.warning("This is a warning")
        logger.error("This is an error")

        print("✓ Logger test passed")

        # Test TensorBoard logger
        print("\nTesting TensorBoard logger...")
        tb_logger = TensorBoardLogger(tmpdir)

        tb_logger.log_scalar('test/loss', 0.5, 0)
        tb_logger.log_scalars('metrics', {'train': 0.5, 'val': 0.6}, 0)

        tb_logger.close()

        print("✓ TensorBoard logger test passed")

    print("\n✅ All logger tests passed!")
