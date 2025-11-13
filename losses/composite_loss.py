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
        use_auxiliary_loss=False,
        strides=None
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

        # Strides for each FPN level (for bbox normalization)
        self.strides = strides or [8, 16, 32, 64]

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
            targets: Encoded dict from FCOSTargetEncoder with:
                - 'cls_targets': list of [B, H, W]
                - 'box_targets': list of [B, 4, H, W]
                - 'centerness': list of [B, 1, H, W]
                - 'pos_mask': list of [B, H, W] booleans
                - 'anchor_points': list of [H, W, 2]
                - 'attention_masks': optional [B, H_img, W_img]
            attention_maps: Optional transformer attention maps
            epoch: Current training epoch (for dynamic weighting)

        Returns:
            loss_dict: Dict with total loss and individual components
        """
        device = predictions[0]['cls_logits'].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # Accumulate losses from all pyramid levels
        num_levels = len(predictions)

        attention_masks = targets.get('attention_masks')

        for level_idx, pred in enumerate(predictions):
            # Classification loss (Focal)
            cls_logits = pred['cls_logits']  # [B, C, H, W]
            cls_targets = targets['cls_targets'][level_idx]  # [B, H, W]

            focal_loss = self.focal_loss(cls_logits, cls_targets)
            loss_dict[f'focal_l{level_idx}'] = focal_loss.item()
            total_loss += self.loss_weights['focal'] * focal_loss / num_levels

            # Dice loss (region-based)
            dice_loss = self.dice_loss(cls_logits, cls_targets)
            loss_dict[f'dice_l{level_idx}'] = dice_loss.item()
            total_loss += self.loss_weights['dice'] * dice_loss / num_levels

            # Bounding box loss (CIoU) - only on positive samples
            bbox_pred = pred['bbox_pred']  # [B, 4, H, W]
            bbox_targets = targets['box_targets'][level_idx]
            pos_mask = targets['pos_mask'][level_idx]  # [B, H, W]
            anchor_points = targets['anchor_points'][level_idx].to(device)  # [H, W, 2]

            pos_mask_flat = pos_mask.reshape(-1)

            if pos_mask_flat.any():
                pred_ltrb = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                target_ltrb = bbox_targets.permute(0, 2, 3, 1).reshape(-1, 4)

                # Cast to fp32 for numerical stability
                pred_ltrb = pred_ltrb.float()
                target_ltrb = target_ltrb.float()

                anchor = anchor_points.reshape(-1, 2)
                B = bbox_pred.shape[0]
                anchor = anchor.unsqueeze(0).repeat(B, 1, 1).reshape(-1, 2)

                pred_boxes = self._ltrb_to_xyxy(pred_ltrb[pos_mask_flat], anchor[pos_mask_flat])
                target_boxes = self._ltrb_to_xyxy(target_ltrb[pos_mask_flat], anchor[pos_mask_flat])

                ciou_loss = self.ciou_loss(pred_boxes, target_boxes)
                loss_dict[f'ciou_l{level_idx}'] = ciou_loss.item()
                total_loss += self.loss_weights['ciou'] * ciou_loss / num_levels
            else:
                loss_dict[f'ciou_l{level_idx}'] = 0.0

            # Centerness loss (Binary CE)
            centerness_pred = pred.get('centerness')
            if centerness_pred is not None:
                centerness_targets = targets['centerness'][level_idx].squeeze(1)
                centerness_logits = centerness_pred.squeeze(1)

                if pos_mask_flat.any():
                    centerness_logits_flat = centerness_logits.reshape(-1)
                    centerness_targets_flat = centerness_targets.reshape(-1)
                    centerness_loss = self.bce_loss(
                        centerness_logits_flat[pos_mask_flat],
                        centerness_targets_flat[pos_mask_flat]
                    )
                else:
                    centerness_loss = torch.tensor(0.0, device=device)

                loss_dict[f'centerness_l{level_idx}'] = centerness_loss.item()
                total_loss += self.loss_weights['centerness'] * centerness_loss / num_levels

        # Attention regularization loss (if available)
        if self.use_attention_loss and attention_maps is not None:
            if attention_masks is not None:
                attn_loss = self.attention_loss(attention_maps, attention_masks)

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

    def _ltrb_to_xyxy(self, ltrb, anchor_points):
        x = anchor_points[:, 0]
        y = anchor_points[:, 1]

        x1 = x - ltrb[:, 0]
        y1 = y - ltrb[:, 1]
        x2 = x + ltrb[:, 2]
        y2 = y + ltrb[:, 3]

        return torch.stack([x1, y1, x2, y2], dim=1)


if __name__ == "__main__":
    # Unit test
    print("Testing CompositeLoss...")

    from utils.target_encoder import FCOSTargetEncoder

    B, C, H, W = 2, 3, 64, 64
    num_levels = 4

    # Create dummy predictions (multi-scale)
    predictions = []
    for level in range(num_levels):
        scale = 2 ** level
        pred = {
            'cls_logits': torch.randn(B, C, H // scale, W // scale),
            'bbox_pred': torch.rand(B, 4, H // scale, W // scale) * 5,
            'centerness': torch.randn(B, 1, H // scale, W // scale)
        }
        predictions.append(pred)

    # Build dummy raw targets (COCO format boxes)
    raw_targets = []
    for b in range(B):
        num_boxes = 5
        boxes = torch.rand(num_boxes, 4) * 50
        boxes[:, 2:] = boxes[:, 2:].abs() + 5  # width/height >=5
        labels = torch.randint(0, 2, (num_boxes,))
        masks = torch.zeros(H, W)
        masks[10:30, 10:30] = 1.0
        raw_targets.append({
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        })

    encoder = FCOSTargetEncoder(num_classes=C)
    encoded_targets = encoder.encode(raw_targets, predictions)

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
        encoded_targets,
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

    total_loss, _ = composite_loss(predictions, encoded_targets, attention_maps, epoch=10)
    total_loss.backward()

    print(f"✓ Gradients computed successfully")

    # Test without attention loss
    print(f"\nTesting without attention loss...")
    composite_loss_no_attn = CompositeLoss(use_attention_loss=False)
    total_loss_no_attn, loss_dict_no_attn = composite_loss_no_attn(
        predictions,
        encoded_targets,
        attention_maps=None
    )
    print(f"Total loss (no attention): {total_loss_no_attn.item():.4f}")
    assert 'attention' not in loss_dict_no_attn or loss_dict_no_attn['attention'] == 0.0

    print("\n✅ All composite loss tests passed!")
