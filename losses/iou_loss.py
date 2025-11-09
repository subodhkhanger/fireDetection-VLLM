"""
IoU-based Losses for Bounding Box Regression

Implements IoU, GIoU, DIoU, and CIoU losses for better localization
compared to L1/L2 losses.

References:
- GIoU: Generalized Intersection over Union (Rezatofighi et al., CVPR 2019)
- DIoU/CIoU: Distance-IoU Loss (Zheng et al., AAAI 2020)
"""

import torch
import torch.nn as nn
import math


def box_iou(boxes1, boxes2, mode='iou'):
    """
    Compute IoU between two sets of boxes

    Args:
        boxes1: [N, 4] boxes in (x1, y1, x2, y2) format
        boxes2: [N, 4] boxes in (x1, y1, x2, y2) format
        mode: 'iou', 'giou', 'diou', or 'ciou'

    Returns:
        iou: [N] IoU values
    """
    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # left-top
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # right-bottom

    wh = (rb - lt).clamp(min=0)  # width-height
    inter = wh[:, 0] * wh[:, 1]  # intersection area

    # Compute union
    union = area1 + area2 - inter

    # IoU
    iou = inter / (union + 1e-6)

    if mode == 'iou':
        return iou

    # GIoU: consider smallest enclosing box
    if mode == 'giou':
        # Enclosing box
        enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

        # GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
        return giou

    # DIoU and CIoU: consider center distance
    if mode in ['diou', 'ciou']:
        # Center points
        center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2
        center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2

        # Center distance
        center_distance = torch.sum((center1 - center2) ** 2, dim=1)

        # Diagonal of enclosing box
        enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        enclose_diag = torch.sum((enclose_rb - enclose_lt) ** 2, dim=1)

        # DIoU
        diou = iou - center_distance / (enclose_diag + 1e-6)

        if mode == 'diou':
            return diou

        # CIoU: add aspect ratio consistency
        if mode == 'ciou':
            # Aspect ratio
            w1, h1 = boxes1[:, 2] - boxes1[:, 0], boxes1[:, 3] - boxes1[:, 1]
            w2, h2 = boxes2[:, 2] - boxes2[:, 0], boxes2[:, 3] - boxes2[:, 1]

            v = (4 / (math.pi ** 2)) * torch.pow(
                torch.atan(w1 / (h1 + 1e-6)) - torch.atan(w2 / (h2 + 1e-6)), 2
            )

            with torch.no_grad():
                alpha = v / (1 - iou + v + 1e-6)

            ciou = diou - alpha * v
            return ciou

    raise ValueError(f"Unknown mode: {mode}")


class IoULoss(nn.Module):
    """
    IoU Loss: L = 1 - IoU

    Simple and effective for bounding box regression.
    """

    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2)
            target_boxes: [N, 4] target boxes (x1, y1, x2, y2)

        Returns:
            loss: Scalar or [N] depending on reduction
        """
        iou = box_iou(pred_boxes, target_boxes, mode='iou')
        loss = 1 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss

    Considers the smallest enclosing box.
    Better gradient flow than IoU loss, especially for non-overlapping boxes.

    L_GIoU = 1 - GIoU
    where GIoU = IoU - (C - A ∪ B) / C
    and C is the smallest enclosing box area
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2)
            target_boxes: [N, 4] target boxes (x1, y1, x2, y2)

        Returns:
            loss: Scalar or [N]
        """
        giou = box_iou(pred_boxes, target_boxes, mode='giou')
        loss = 1 - giou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DIoULoss(nn.Module):
    """
    Distance IoU Loss

    Considers the distance between box centers.
    Faster convergence than GIoU.

    L_DIoU = 1 - IoU + ρ²(b, b_gt) / c²
    where:
    - ρ²: squared Euclidean distance between centers
    - c²: squared diagonal length of smallest enclosing box
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2)
            target_boxes: [N, 4] target boxes (x1, y1, x2, y2)

        Returns:
            loss: Scalar or [N]
        """
        diou = box_iou(pred_boxes, target_boxes, mode='diou')
        loss = 1 - diou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CIoULoss(nn.Module):
    """
    Complete IoU Loss

    Best overall performance. Considers:
    1. IoU (overlap)
    2. Distance between centers
    3. Aspect ratio consistency

    L_CIoU = 1 - IoU + ρ²(b, b_gt) / c² + α * v

    where:
    - ρ²: squared distance between centers
    - c²: squared diagonal of enclosing box
    - v: aspect ratio consistency
    - α: trade-off parameter

    This is our primary bbox loss for Fire-ViT.
    """

    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2)
            target_boxes: [N, 4] target boxes (x1, y1, x2, y2)

        Returns:
            loss: Scalar or [N]
        """
        ciou = box_iou(pred_boxes, target_boxes, mode='ciou')
        loss = 1 - ciou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    # Unit test
    print("Testing IoU Losses...")

    N = 100

    # Create random boxes
    boxes1 = torch.rand(N, 4) * 100
    boxes1[:, 2:] += boxes1[:, :2]  # Ensure x2 > x1, y2 > y1

    boxes2 = torch.rand(N, 4) * 100
    boxes2[:, 2:] += boxes2[:, :2]

    print(f"\nBox 1 shape: {boxes1.shape}")
    print(f"Box 2 shape: {boxes2.shape}")

    # Test all IoU variants
    iou = box_iou(boxes1, boxes2, mode='iou')
    giou = box_iou(boxes1, boxes2, mode='giou')
    diou = box_iou(boxes1, boxes2, mode='diou')
    ciou = box_iou(boxes1, boxes2, mode='ciou')

    print(f"\nIoU range: [{iou.min():.3f}, {iou.max():.3f}], mean: {iou.mean():.3f}")
    print(f"GIoU range: [{giou.min():.3f}, {giou.max():.3f}], mean: {giou.mean():.3f}")
    print(f"DIoU range: [{diou.min():.3f}, {diou.max():.3f}], mean: {diou.mean():.3f}")
    print(f"CIoU range: [{ciou.min():.3f}, {ciou.max():.3f}], mean: {ciou.mean():.3f}")

    # Test loss modules
    print(f"\nTesting Loss Modules:")

    iou_loss = IoULoss()
    giou_loss = GIoULoss()
    diou_loss = DIoULoss()
    ciou_loss = CIoULoss()

    print(f"  IoU Loss: {iou_loss(boxes1, boxes2).item():.4f}")
    print(f"  GIoU Loss: {giou_loss(boxes1, boxes2).item():.4f}")
    print(f"  DIoU Loss: {diou_loss(boxes1, boxes2).item():.4f}")
    print(f"  CIoU Loss: {ciou_loss(boxes1, boxes2).item():.4f}")

    # Test perfect overlap
    print(f"\nPerfect overlap (same boxes):")
    perfect_loss = ciou_loss(boxes1, boxes1)
    print(f"  CIoU Loss: {perfect_loss.item():.6f}")
    assert perfect_loss.item() < 1e-4, "Perfect overlap should have ~0 loss"

    # Test gradient flow
    print(f"\nTesting gradient flow:")
    pred_boxes = boxes1.clone().requires_grad_(True)
    loss = ciou_loss(pred_boxes, boxes2)
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient shape: {pred_boxes.grad.shape}")
    print(f"  Gradient norm: {pred_boxes.grad.norm().item():.4f}")
    assert pred_boxes.grad is not None
    print(f"  ✓ Gradients computed successfully")

    print("\n✅ All IoU loss tests passed!")
