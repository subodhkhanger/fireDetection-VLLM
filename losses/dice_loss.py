"""
Dice Loss for Fire-ViT

Particularly effective for small objects and class imbalance.
Provides region-based optimization complementary to focal loss.

Reference:
- V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary/multi-class segmentation-style optimization

    L_dice = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)

    where:
    - P: predictions
    - T: targets
    - ε: smoothing factor

    Particularly effective for:
    - Small fire instances
    - Class imbalance
    - Smooth gradient flow

    Args:
        smooth (float): Smoothing factor to avoid division by zero
        reduction (str): 'mean', 'sum', or 'none'
    """

    def __init__(self, smooth=1.0, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] predicted probabilities (after softmax/sigmoid)
            target: [B, H, W] ground truth labels or [B, C, H, W] one-hot

        Returns:
            loss: Scalar
        """
        # Convert target to one-hot if needed
        if target.dim() == 3:  # [B, H, W]
            num_classes = pred.shape[1]
            target_one_hot = F.one_hot(target, num_classes=num_classes)
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        else:
            target_one_hot = target

        # Apply softmax if pred contains logits
        if pred.requires_grad:
            pred = F.softmax(pred, dim=1)

        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)  # [B, C, H*W]
        target_flat = target_one_hot.reshape(target_one_hot.shape[0], target_one_hot.shape[1], -1)

        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
        pred_sum = pred_flat.sum(dim=2)  # [B, C]
        target_sum = target_flat.sum(dim=2)  # [B, C]

        # Dice coefficient
        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Dice loss
        loss = 1 - dice

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss: Generalization of Dice loss

    Allows weighting false positives and false negatives differently.
    Useful when precision and recall have different importance.

    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives
        smooth: Smoothing factor
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, H, W] or [B, C, H, W]

        Returns:
            loss: Scalar
        """
        # Convert target to one-hot if needed
        if target.dim() == 3:
            num_classes = pred.shape[1]
            target = F.one_hot(target, num_classes=num_classes)
            target = target.permute(0, 3, 1, 2).float()

        # Softmax on predictions
        if pred.requires_grad:
            pred = F.softmax(pred, dim=1)

        # Flatten
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)

        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return (1 - tversky).mean()


if __name__ == "__main__":
    # Unit test
    print("Testing DiceLoss...")

    B, C, H, W = 4, 3, 64, 64

    # Create dummy data
    pred_logits = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))

    # Test Dice loss
    dice_loss = DiceLoss()
    loss = dice_loss(pred_logits, target)

    print(f"  Prediction shape: {pred_logits.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Dice Loss: {loss.item():.4f}")
    assert loss.item() >= 0 and loss.item() <= 1
    print(f"  ✓ Loss in valid range [0, 1]")

    # Test gradient
    pred_logits.requires_grad = True
    loss = dice_loss(pred_logits, target)
    loss.backward()
    print(f"  ✓ Gradient computed: {pred_logits.grad.shape}")

    # Test Tversky loss
    print(f"\nTesting TverskyLoss...")
    tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
    loss_t = tversky_loss(pred_logits.detach(), target)
    print(f"  Tversky Loss: {loss_t.item():.4f}")

    print("\n✅ All Dice loss tests passed!")
