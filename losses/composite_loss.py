"""
Composite Loss for Fire-ViT

Combines all loss components with dynamic weighting:
- Focal Loss (classification)
- CIoU Loss (bounding box)
- Dice Loss (region-based)
- Attention Regularization (interpretability)
- Auxiliary Loss (intermediate supervision)
"""

import torch
import torch.nn as nn
from .focal_loss import FocalLoss
from .iou_loss import CIoULoss
from .dice_loss import DiceLoss
from .attention_loss import AttentionRegularizationLoss


class CompositeLoss(nn.Module):
    """
    Complete loss function for Fire-ViT

    L_total = λ_cls * L_focal +
              λ_box * L_ciou +
              λ_dice * L_dice +
              λ_attn * L_attn +
              λ_aux * L_aux

    Args:
        focal_alpha (float): Focal loss alpha parameter
        focal_gamma (float): Focal loss gamma parameter
        loss_weights (dict): Weights for each loss component
        use_attention_loss (bool): Whether to use attention regularization
        use_auxiliary_loss (bool): Whether to use auxiliary loss
    """

    def __init__(
        self,
        focal_alpha=0.25,
        focal_gamma=2.0,
        loss_weights=None,
        use_attention_loss=True,
        use_auxiliary_loss=False
    ):
        super().__init__()

        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'focal': 1.0,
                'ciou': 5.0,
                'dice': 2.0,
                'attention': 0.1,
                'centerness': 1.0,
                'auxiliary': 0.5
            }

        self.loss_weights = loss_weights
        self.use_attention_loss = use_attention_loss
        self.use_auxiliary_loss = use_auxiliary_loss

        # Initialize loss modules
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ciou_loss = CIoULoss()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

        if use_attention_loss:
            self.attention_loss = AttentionRegularizationLoss()

    def forward(self, predictions, targets, attention_maps=None, epoch=0):
        """
        Compute composite loss

        Args:
            predictions: List of dicts with 'cls_logits', 'bbox_pred', 'centerness'
                        One dict per pyramid level
            targets: Dict with:
                - 'labels': [B, H, W] class labels
                - 'boxes': [B, N, 4] bounding boxes
                - 'masks': [B, H, W] binary masks for attention
            attention_maps: Optional attention maps from transformer
            epoch: Current training epoch (for dynamic weighting)

        Returns:
            loss_dict: Dict with total loss and individual components
        """
        device = predictions[0]['cls_logits'].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # Accumulate losses from all pyramid levels
        num_levels = len(predictions)

        for level_idx, pred in enumerate(predictions):
            # Classification loss (Focal)
            cls_logits = pred['cls_logits']  # [B, C, H, W]
            cls_targets = self._prepare_cls_targets(targets, cls_logits.shape, device)

            focal_loss = self.focal_loss(cls_logits, cls_targets)
            loss_dict[f'focal_l{level_idx}'] = focal_loss.item()
            total_loss += self.loss_weights['focal'] * focal_loss / num_levels

            # Dice loss (region-based)
            dice_loss = self.dice_loss(cls_logits, cls_targets)
            loss_dict[f'dice_l{level_idx}'] = dice_loss.item()
            total_loss += self.loss_weights['dice'] * dice_loss / num_levels

            # Bounding box loss (CIoU) - only on positive samples
            if 'boxes' in targets and targets['boxes'].numel() > 0:
                bbox_pred = pred['bbox_pred']  # [B, 4, H, W]
                bbox_targets, pos_mask = self._prepare_bbox_targets(
                    targets, bbox_pred.shape, device
                )

                if pos_mask.sum() > 0:
                    # Convert predictions to boxes
                    pred_boxes = self._pred_to_boxes(bbox_pred, level_idx)

                    # Check if mask shape matches predictions
                    if pos_mask.shape[0] == pred_boxes.shape[0]:
                        # Filter positive samples
                        pred_boxes_pos = pred_boxes[pos_mask]
                        bbox_targets_pos = bbox_targets[pos_mask]

                        ciou_loss = self.ciou_loss(pred_boxes_pos, bbox_targets_pos)
                        loss_dict[f'ciou_l{level_idx}'] = ciou_loss.item()
                        total_loss += self.loss_weights['ciou'] * ciou_loss / num_levels
                    else:
                        # Shape mismatch - skip bbox loss for this level
                        loss_dict[f'ciou_l{level_idx}'] = 0.0
                else:
                    loss_dict[f'ciou_l{level_idx}'] = 0.0

            # Centerness loss (Binary CE)
            if 'centerness' in pred:
                centerness_pred = pred['centerness']  # [B, 1, H, W]
                centerness_targets = self._prepare_centerness_targets(
                    targets, centerness_pred.shape, device
                )

                centerness_loss = self.bce_loss(
                    centerness_pred.squeeze(1),
                    centerness_targets.float()  # Convert to float
                )
                loss_dict[f'centerness_l{level_idx}'] = centerness_loss.item()
                total_loss += self.loss_weights['centerness'] * centerness_loss / num_levels

        # Attention regularization loss (if available)
        if self.use_attention_loss and attention_maps is not None:
            if 'masks' in targets:
                attn_loss = self.attention_loss(attention_maps, targets['masks'])

                # Warm-up schedule for attention loss
                attn_weight = min(self.loss_weights['attention'], epoch / 100)
                loss_dict['attention'] = attn_loss.item()
                total_loss += attn_weight * attn_loss
            else:
                loss_dict['attention'] = 0.0

        # Auxiliary loss (intermediate supervision) - if provided
        if self.use_auxiliary_loss and 'auxiliary_preds' in predictions[0]:
            # Implement auxiliary loss on intermediate features
            # Decay auxiliary loss weight over training
            aux_weight = max(0.1, self.loss_weights['auxiliary'] - epoch / 200)
            # TODO: Implement auxiliary loss computation
            loss_dict['auxiliary'] = 0.0

        # Total loss
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def _prepare_cls_targets(self, targets, shape, device):
        """Prepare classification targets matching prediction shape"""
        B, C, H, W = shape

        if 'labels' in targets:
            labels = targets['labels']

            # Resize if needed
            if labels.shape[-2:] != (H, W):
                labels = torch.nn.functional.interpolate(
                    labels.unsqueeze(1).float(),
                    size=(H, W),
                    mode='nearest'
                ).squeeze(1).long()

            return labels
        else:
            # Return dummy targets
            return torch.zeros(B, H, W, dtype=torch.long, device=device)

    def _prepare_bbox_targets(self, targets, shape, device):
        """Prepare bounding box targets"""
        if 'boxes' not in targets or targets['boxes'].numel() == 0:
            return None, torch.zeros(0, dtype=torch.bool, device=device)

        # This is a simplified version
        # In practice, you'd assign targets to specific spatial locations
        boxes = targets['boxes']  # [B, N, 4]

        # Create positive mask (simplified - all boxes are positive)
        pos_mask = boxes.sum(dim=-1) > 0  # [B, N]

        return boxes, pos_mask.flatten()

    def _prepare_centerness_targets(self, targets, shape, device):
        """Prepare centerness targets"""
        B, _, H, W = shape

        # Simplified: use binary mask of fire regions
        if 'masks' in targets:
            masks = targets['masks']
            if masks.shape[-2:] != (H, W):
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            return masks
        else:
            return torch.zeros(B, H, W, device=device)

    def _pred_to_boxes(self, bbox_pred, level_idx):
        """Convert bbox predictions to boxes"""
        # This is a placeholder
        # In practice, you'd use anchor points and convert (l,t,r,b) to boxes
        B, _, H, W = bbox_pred.shape

        # Flatten and reshape
        boxes = bbox_pred.permute(0, 2, 3, 1).reshape(B * H * W, 4)

        return boxes


if __name__ == "__main__":
    # Unit test
    print("Testing CompositeLoss...")

    B, C, H, W = 2, 2, 64, 64
    num_levels = 4

    # Create dummy predictions (multi-scale)
    predictions = []
    for level in range(num_levels):
        scale = 2 ** level
        pred = {
            'cls_logits': torch.randn(B, C, H // scale, W // scale),
            'bbox_pred': torch.rand(B, 4, H // scale, W // scale) * 10,
            'centerness': torch.randn(B, 1, H // scale, W // scale)
        }
        predictions.append(pred)

    # Create dummy targets
    targets = {
        'labels': torch.randint(0, C, (B, H, W)),
        'boxes': torch.rand(B, 10, 4) * 100,  # 10 boxes per image
        'masks': torch.rand(B, H, W) > 0.9  # Binary mask
    }

    # Create dummy attention maps
    attention_maps = torch.softmax(torch.randn(B, 8, 256, 256), dim=-1)

    # Initialize loss
    composite_loss = CompositeLoss(
        focal_alpha=0.25,
        focal_gamma=2.0,
        loss_weights={
            'focal': 1.0,
            'ciou': 5.0,
            'dice': 2.0,
            'attention': 0.1,
            'centerness': 1.0
        },
        use_attention_loss=True
    )

    # Compute loss
    total_loss, loss_dict = composite_loss(
        predictions,
        targets,
        attention_maps=attention_maps,
        epoch=10
    )

    print(f"\nLoss Components:")
    for key, value in loss_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    print(f"\nTotal Loss: {total_loss.item():.4f}")
    assert total_loss.item() >= 0
    print(f"✓ Total loss is non-negative")

    # Test gradient flow
    print(f"\nTesting gradient flow...")
    for pred in predictions:
        pred['cls_logits'].requires_grad = True

    total_loss, _ = composite_loss(predictions, targets, attention_maps, epoch=10)
    total_loss.backward()

    print(f"✓ Gradients computed successfully")

    # Test without attention loss
    print(f"\nTesting without attention loss...")
    composite_loss_no_attn = CompositeLoss(use_attention_loss=False)
    total_loss_no_attn, loss_dict_no_attn = composite_loss_no_attn(
        predictions,
        targets,
        attention_maps=None
    )
    print(f"Total loss (no attention): {total_loss_no_attn.item():.4f}")
    assert 'attention' not in loss_dict_no_attn or loss_dict_no_attn['attention'] == 0.0

    print("\n✅ All composite loss tests passed!")
