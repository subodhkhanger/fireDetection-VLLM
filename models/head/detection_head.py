"""
Anchor-Free Detection Head for Fire-ViT

Implements FCOS-style anchor-free detection with:
- Per-pixel classification (fire vs smoke vs background)
- Bounding box regression (center-based)
- Centerness/objectness prediction

References:
- FCOS: Fully Convolutional One-Stage Object Detection (Tian et al., ICCV 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FireDetectionHead(nn.Module):
    """
    Anchor-free detection head for fire and smoke detection

    Predicts per-pixel:
    - Classification: fire (0) vs smoke (1) probabilities
    - Bounding box: (l, t, r, b) distances to box edges OR (cx, cy, w, h)
    - Centerness: quality score for the prediction

    Args:
        in_channels (int): Input feature channels from FPN
        num_classes (int): Number of classes (2 for fire, smoke)
        num_convs (int): Number of conv layers in head
        prior_prob (float): Prior probability for focal loss initialization
    """

    def __init__(
        self,
        in_channels=256,
        num_classes=3,
        num_convs=4,
        prior_prob=0.01
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Shared feature extraction
        self.shared_conv = self._make_conv_layers(in_channels, in_channels, num_convs)

        # Classification head (predict class probabilities)
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

        # Bounding box regression head (predict 4 offsets)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

        # Centerness head (predict objectness/quality)
        self.centerness_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        # Scales for bbox regression (learnable per pyramid level)
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, 1),
                nn.ReLU()
            )
            for _ in range(4)  # 4 pyramid levels
        ])

        self._init_weights(prior_prob)

    def _make_conv_layers(self, in_ch, out_ch, num_convs):
        """Create stacked conv layers with GroupNorm and ReLU"""
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.GroupNorm(32, out_ch),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def _init_weights(self, prior_prob):
        """Initialize weights"""
        # Initialize shared conv
        for layer in self.shared_conv.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)

        # Initialize heads
        for modules in [self.cls_head, self.bbox_head, self.centerness_head]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Focal loss initialization for classification head
        # Initialize bias to achieve prior_prob at start
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head[-1].bias, bias_value)

        # CRITICAL: Initialize bbox head bias to negative values
        # This ensures exp(bbox_pred) starts small (around 1-10 pixels)
        # exp(-2.3) ≈ 0.1, exp(-1.0) ≈ 0.37, exp(0) = 1
        nn.init.constant_(self.bbox_head[-1].bias, -2.0)

    def forward(self, features, level_idx=0):
        """
        Forward pass for a single feature level

        Args:
            features: [B, C, H, W] feature map from FPN
            level_idx: Index of pyramid level (for scale parameter)

        Returns:
            predictions: dict with:
                - 'cls_logits': [B, num_classes, H, W] classification logits
                - 'bbox_pred': [B, 4, H, W] bbox predictions (l, t, r, b)
                - 'centerness': [B, 1, H, W] centerness predictions
        """
        # Shared features
        x = self.shared_conv(features)

        # Classification predictions
        cls_logits = self.cls_head(x)  # [B, num_classes, H, W]

        # Bounding box predictions
        bbox_pred = self.bbox_head(x)  # [B, 4, H, W]

        # CRITICAL: Cast to fp32 for numerical stability with exp()
        # fp16 exp() is highly unstable and can produce inf/nan
        original_dtype = bbox_pred.dtype
        bbox_pred = bbox_pred.float()

        # Clamp before exp to prevent extreme values
        # Allows bbox sizes from ~0.05 to ~1096 pixels
        bbox_pred = torch.clamp(bbox_pred, min=-3.0, max=7.0)

        # Apply exponential to ensure positive box dimensions
        bbox_pred = torch.exp(bbox_pred)

        # Cast back to original dtype for mixed precision training
        bbox_pred = bbox_pred.to(original_dtype)

        # Centerness predictions
        centerness = self.centerness_head(x)  # [B, 1, H, W]

        return {
            'cls_logits': cls_logits,
            'bbox_pred': bbox_pred,
            'centerness': centerness
        }


class MultiScaleDetectionHead(nn.Module):
    """
    Multi-scale detection head that applies detection to all FPN levels

    Args:
        in_channels (int): Input feature channels from FPN
        num_classes (int): Number of classes
        num_convs (int): Number of conv layers in each head
    """

    def __init__(
        self,
        in_channels=256,
        num_classes=3,
        num_convs=4
    ):
        super().__init__()

        # Single shared detection head for all levels
        self.detection_head = FireDetectionHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_convs=num_convs
        )

    def forward(self, pyramid_features):
        """
        Apply detection head to all pyramid levels

        Args:
            pyramid_features: List of [B, C, H_i, W_i] feature maps

        Returns:
            predictions: List of dicts, one per pyramid level
        """
        predictions = []

        for level_idx, features in enumerate(pyramid_features):
            pred = self.detection_head(features, level_idx=level_idx)
            predictions.append(pred)

        return predictions

    def get_predictions_at_points(self, pyramid_features, points):
        """
        Get predictions at specific points (for targeted evaluation)

        Args:
            pyramid_features: List of feature maps
            points: [N, 2] (x, y) coordinates

        Returns:
            predictions at those points
        """
        # This can be useful for debugging or visualization
        pass


def generate_anchor_points(feature_map_size, stride, device='cpu'):
    """
    Generate anchor points (center locations) for a feature map

    Args:
        feature_map_size: (H, W) size of feature map
        stride: Stride relative to input image
        device: torch device

    Returns:
        anchor_points: [H*W, 2] (x, y) coordinates in input image space
    """
    H, W = feature_map_size

    # Generate grid coordinates
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Convert to input image coordinates
    x_coords = (x_coords + 0.5) * stride
    y_coords = (y_coords + 0.5) * stride

    # Stack and flatten
    anchor_points = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]
    anchor_points = anchor_points.reshape(-1, 2)  # [H*W, 2]

    return anchor_points


def bbox_pred_to_boxes(bbox_pred, anchor_points):
    """
    Convert bbox predictions to actual boxes

    Args:
        bbox_pred: [B, 4, H, W] predicted (l, t, r, b) distances
        anchor_points: [H*W, 2] anchor point coordinates

    Returns:
        boxes: [B, H*W, 4] predicted boxes in (x1, y1, x2, y2) format
    """
    B, _, H, W = bbox_pred.shape

    # Reshape predictions: [B, 4, H, W] -> [B, H*W, 4]
    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)

    # Extract l, t, r, b
    l = bbox_pred[:, :, 0]
    t = bbox_pred[:, :, 1]
    r = bbox_pred[:, :, 2]
    b = bbox_pred[:, :, 3]

    # Convert to x1, y1, x2, y2
    x1 = anchor_points[:, 0] - l
    y1 = anchor_points[:, 1] - t
    x2 = anchor_points[:, 0] + r
    y2 = anchor_points[:, 1] + b

    # Stack
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, H*W, 4]

    return boxes


if __name__ == "__main__":
    # Unit test
    print("Testing FireDetectionHead...")

    batch_size = 2
    in_channels = 256
    num_classes = 2

    # Test single-scale head
    print("\n1. Testing Single-Scale Detection Head:")
    detection_head = FireDetectionHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_convs=4
    )

    # Create dummy feature map
    features = torch.randn(batch_size, in_channels, 64, 64)
    print(f"   Input features: {features.shape}")

    # Forward pass
    predictions = detection_head(features, level_idx=0)

    print(f"\n   Predictions:")
    print(f"   - Classification logits: {predictions['cls_logits'].shape}")
    print(f"   - Bbox predictions: {predictions['bbox_pred'].shape}")
    print(f"   - Centerness: {predictions['centerness'].shape}")

    assert predictions['cls_logits'].shape == (batch_size, num_classes, 64, 64)
    assert predictions['bbox_pred'].shape == (batch_size, 4, 64, 64)
    assert predictions['centerness'].shape == (batch_size, 1, 64, 64)
    print(f"   ✓ All shapes correct")

    # Test multi-scale head
    print("\n2. Testing Multi-Scale Detection Head:")
    multi_scale_head = MultiScaleDetectionHead(
        in_channels=in_channels,
        num_classes=num_classes
    )

    # Create multi-scale features (simulating FPN output)
    pyramid_features = [
        torch.randn(batch_size, in_channels, 64, 64),  # P3: /8
        torch.randn(batch_size, in_channels, 32, 32),  # P4: /16
        torch.randn(batch_size, in_channels, 16, 16),  # P5: /32
        torch.randn(batch_size, in_channels, 8, 8),    # P6: /64
    ]

    print(f"   Input pyramid features:")
    for i, feat in enumerate(pyramid_features):
        print(f"   - P{i+3}: {feat.shape}")

    # Forward pass
    multi_predictions = multi_scale_head(pyramid_features)

    print(f"\n   Multi-scale predictions:")
    for i, pred in enumerate(multi_predictions):
        print(f"   - Level {i+3}:")
        print(f"     Classification: {pred['cls_logits'].shape}")
        print(f"     Bbox: {pred['bbox_pred'].shape}")
        print(f"     Centerness: {pred['centerness'].shape}")

    # Test anchor point generation
    print("\n3. Testing Anchor Point Generation:")
    anchor_points = generate_anchor_points((64, 64), stride=8)
    print(f"   Anchor points shape: {anchor_points.shape}")
    print(f"   First few points: {anchor_points[:5]}")
    print(f"   Last few points: {anchor_points[-5:]}")

    # Test bbox conversion
    print("\n4. Testing Bbox Prediction to Boxes:")
    bbox_pred = predictions['bbox_pred']
    boxes = bbox_pred_to_boxes(bbox_pred, anchor_points)
    print(f"   Converted boxes shape: {boxes.shape}")
    print(f"   Sample boxes: {boxes[0, :3]}")

    # Count parameters
    params = sum(p.numel() for p in detection_head.parameters())
    multi_params = sum(p.numel() for p in multi_scale_head.parameters())

    print(f"\n{'='*50}")
    print(f"Single-scale head parameters: {params:,}")
    print(f"Multi-scale head parameters: {multi_params:,}")
    print(f"{'='*50}")

    print("\n✅ All tests passed!")
