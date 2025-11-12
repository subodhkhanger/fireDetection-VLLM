"""
Training utilities for Fire-ViT

Implements:
- Exponential Moving Average (EMA)
- Training epoch loop
- Validation loop
- Mixed precision training
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import copy


class EMA:
    """
    Exponential Moving Average for model parameters

    Maintains shadow parameters that are updated as exponential moving average
    of the model parameters. Improves generalization.

    Args:
        model: PyTorch model
        decay: Decay rate (typically 0.9999)
    """

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow parameters to model (for validation/inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """
    Training manager for Fire-ViT

    Handles:
    - Training loop with mixed precision
    - Validation
    - EMA updates
    - Gradient accumulation
    - Learning rate scheduling
    - Logging
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device='cuda',
        use_amp=True,
        use_ema=True,
        ema_decay=0.9999,
        gradient_accumulation_steps=1,
        grad_clip_norm=1.0,
        log_interval=50
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        # Move model to device BEFORE initializing EMA
        self.model.to(device)

        # EMA (initialized after model is on device)
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(model, decay=ema_decay)
        else:
            self.ema = None

    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        loss_components = {}
        num_batches = len(train_loader)

        progress_bar = tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc=f"Epoch {epoch}"
        )

        for batch_idx, (images, targets) in progress_bar:
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    predictions = self.model(images)
                    loss, loss_dict = self.loss_fn(predictions, targets, epoch=epoch)

                    # Normalize loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                predictions = self.model(images)
                loss, loss_dict = self.loss_fn(predictions, targets, epoch=epoch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm
                    )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update EMA
                if self.use_ema:
                    self.ema.update()

            # Accumulate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps

            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                if isinstance(value, (int, float)):
                    loss_components[key] += value

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.optimizer.param_groups[0]['lr']

                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.6f}'
                })

        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        metrics = {
            'train_loss': avg_loss,
            **{f'train_{k}': v for k, v in avg_components.items()}
        }

        return metrics

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """
        Validate the model

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            metrics: Dictionary of validation metrics
        """
        # Apply EMA if available
        if self.use_ema:
            self.ema.apply_shadow()

        self.model.eval()

        total_loss = 0.0
        loss_components = {}
        num_batches = len(val_loader)

        progress_bar = tqdm(val_loader, desc=f"Validation")

        for images, targets in progress_bar:
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]

            # Forward pass
            if self.use_amp:
                with autocast():
                    predictions = self.model(images)
                    loss, loss_dict = self.loss_fn(predictions, targets, epoch=epoch)
            else:
                predictions = self.model(images)
                loss, loss_dict = self.loss_fn(predictions, targets, epoch=epoch)

            # Accumulate metrics
            total_loss += loss.item()

            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                if isinstance(value, (int, float)):
                    loss_components[key] += value

        # Restore original parameters
        if self.use_ema:
            self.ema.restore()

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        metrics = {
            'val_loss': avg_loss,
            **{f'val_{k}': v for k, v in avg_components.items()}
        }

        return metrics


if __name__ == "__main__":
    # Test EMA
    print("Testing EMA...")

    model = nn.Linear(10, 10)
    ema = EMA(model, decay=0.999)

    # Simulate training
    for _ in range(10):
        # Forward/backward (simulated)
        loss = model(torch.randn(5, 10)).sum()
        loss.backward()

        # Update EMA
        ema.update()

    # Apply shadow
    original_weight = model.weight.data.clone()
    ema.apply_shadow()
    shadow_weight = model.weight.data.clone()

    print(f"✓ Original weight norm: {original_weight.norm().item():.4f}")
    print(f"✓ Shadow weight norm: {shadow_weight.norm().item():.4f}")

    # Restore
    ema.restore()
    restored_weight = model.weight.data.clone()

    assert torch.allclose(original_weight, restored_weight)
    print("✓ Restoration successful")

    print("\n✅ EMA test passed!")
