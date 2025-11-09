"""
Custom loss functions for Fire-ViT
"""

from .focal_loss import FocalLoss
from .iou_loss import CIoULoss, GIoULoss, IoULoss
from .dice_loss import DiceLoss
from .attention_loss import AttentionRegularizationLoss
from .composite_loss import CompositeLoss

__all__ = [
    'FocalLoss',
    'CIoULoss',
    'GIoULoss',
    'IoULoss',
    'DiceLoss',
    'AttentionRegularizationLoss',
    'CompositeLoss'
]
