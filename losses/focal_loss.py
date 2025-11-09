"""
Focal Loss for Fire-ViT

Addresses class imbalance by down-weighting well-classified examples.
Critical for fire detection where 99% of pixels are background.

Reference:
- Focal Loss for Dense Object Detection (Lin et al., ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Key properties:
    - Reduces loss for well-classified examples (high p_t)
    - Focuses training on hard negatives (low p_t)
    - α_t: class balancing weight (typically 0.25 for positive class)
    - γ: focusing parameter (typically 2.0)

    Args:
        alpha (float): Weighting factor for positive class
        gamma (float): Focusing parameter
        reduction (str): 'none', 'mean', or 'sum'
        ignore_index (int): Index to ignore in target
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        reduction='mean',
        ignore_index=-100
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] or [B, C] predicted logits
            target: [B, H, W] or [B] ground truth class labels

        Returns:
            loss: Scalar or [B, H, W] depending on reduction
        """
        # Get number of classes
        num_classes = pred.shape[1]

        # Compute cross-entropy loss without reduction
        ce_loss = F.cross_entropy(
            pred,
            target,
            reduction='none',
            ignore_index=self.ignore_index
        )

        # Get probability of true class: p_t
        # Convert logits to probabilities
        p = F.softmax(pred, dim=1)

        # Gather probabilities for true class
        if pred.dim() == 4:  # [B, C, H, W]
            B, C, H, W = pred.shape
            # Reshape for gathering
            p = p.transpose(1, 2).transpose(2, 3).reshape(-1, C)  # [B*H*W, C]
            target_flat = target.reshape(-1)  # [B*H*W]

            # Gather
            p_t = p.gather(1, target_flat.unsqueeze(1)).squeeze(1)  # [B*H*W]
            p_t = p_t.reshape(B, H, W)  # [B, H, W]
            ce_loss_original_shape = ce_loss
        else:  # [B, C]
            p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
            ce_loss_original_shape = ce_loss

        # Focal term: (1 - p_t)^γ
        focal_term = (1 - p_t) ** self.gamma

        # Alpha balancing
        # Create alpha weights based on target class
        if pred.dim() == 4:
            alpha_t = torch.where(
                target > 0,
                torch.tensor(self.alpha, device=pred.device),
                torch.tensor(1 - self.alpha, device=pred.device)
            )
        else:
            alpha_t = torch.where(
                target > 0,
                torch.tensor(self.alpha, device=pred.device),
                torch.tensor(1 - self.alpha, device=pred.device)
            )

        # Mask ignored indices
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            focal_term = focal_term * mask
            alpha_t = alpha_t * mask

        # Final focal loss
        loss = alpha_t * focal_term * ce_loss_original_shape

        # Apply reduction
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                loss = loss.sum() / mask.sum().clamp(min=1.0)
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss for joint classification and quality estimation

    Used when predictions include both class and quality (IoU) scores.
    Useful for combining classification with centerness/IoU prediction.

    Reference:
    - Generalized Focal Loss (Li et al., NeurIPS 2020)
    """

    def __init__(self, beta=2.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, quality_target):
        """
        Args:
            pred: [B, C, H, W] predicted class probabilities (after sigmoid)
            target: [B, H, W] ground truth class labels (0 or 1)
            quality_target: [B, H, W] quality scores (e.g., IoU with GT)

        Returns:
            loss: Scalar
        """
        # Quality focal loss
        # For positive samples: |(y - σ)|^β * log(σ)
        # For negative samples: |σ|^β * log(1 - σ)

        pred_sigmoid = torch.sigmoid(pred) if pred.requires_grad else pred

        # Flatten
        if pred.dim() == 4:
            B, C, H, W = pred.shape
            pred_sigmoid = pred_sigmoid.permute(0, 2, 3, 1).reshape(-1, C)
            target_flat = target.reshape(-1)
            quality_target_flat = quality_target.reshape(-1)
        else:
            pred_sigmoid = pred_sigmoid
            target_flat = target
            quality_target_flat = quality_target

        # Split positive and negative samples
        pos_mask = target_flat > 0
        neg_mask = ~pos_mask

        # Positive loss
        if pos_mask.sum() > 0:
            pred_pos = pred_sigmoid[pos_mask]
            quality_pos = quality_target_flat[pos_mask]

            # |y - σ|^β * log(σ)
            pos_loss = torch.abs(quality_pos - pred_pos.squeeze()) ** self.beta * \
                      F.binary_cross_entropy(pred_pos.squeeze(), quality_pos, reduction='none')
        else:
            pos_loss = torch.tensor(0.0, device=pred.device)

        # Negative loss
        if neg_mask.sum() > 0:
            pred_neg = pred_sigmoid[neg_mask]

            # |σ|^β * log(1 - σ)
            neg_loss = pred_neg.squeeze() ** self.beta * \
                      F.binary_cross_entropy(pred_neg.squeeze(),
                                           torch.zeros_like(pred_neg.squeeze()),
                                           reduction='none')
        else:
            neg_loss = torch.tensor(0.0, device=pred.device)

        # Combine
        if self.reduction == 'mean':
            pos_loss = pos_loss.mean() if pos_mask.sum() > 0 else pos_loss
            neg_loss = neg_loss.mean() if neg_mask.sum() > 0 else neg_loss
            loss = pos_loss + neg_loss
        else:
            loss = pos_loss.sum() + neg_loss.sum()

        return loss


if __name__ == "__main__":
    # Unit test
    print("Testing FocalLoss...")

    batch_size = 4
    num_classes = 3
    H, W = 64, 64

    # Create dummy data
    pred = torch.randn(batch_size, num_classes, H, W)
    target = torch.randint(0, num_classes, (batch_size, H, W))

    # Create focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    # Compute loss
    loss = focal_loss(pred, target)

    print(f"  Input shape: {pred.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Loss: {loss.item():.4f}")
    assert loss.item() >= 0
    print(f"  ✓ Loss is non-negative")

    # Test with 1D input
    pred_1d = torch.randn(batch_size, num_classes)
    target_1d = torch.randint(0, num_classes, (batch_size,))

    loss_1d = focal_loss(pred_1d, target_1d)
    print(f"\n  1D Input shape: {pred_1d.shape}")
    print(f"  1D Loss: {loss_1d.item():.4f}")

    # Compare with standard cross-entropy
    ce_loss = F.cross_entropy(pred, target)
    print(f"\n  Standard CE loss: {ce_loss.item():.4f}")
    print(f"  Focal loss: {loss.item():.4f}")
    print(f"  Ratio: {loss.item() / ce_loss.item():.4f}")

    print("\n✅ Focal Loss tests passed!")
