"""
Feature Pyramid Network (FPN) for Fire-ViT

Implements FPN-style multi-scale feature fusion with:
- Top-down pathway with lateral connections
- Bottom-up pathway for low-level to high-level features
- Smooth feature maps with 3x3 convolutions

References:
- Feature Pyramid Networks for Object Detection (Lin et al., CVPR 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale fusion

    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                   Top-Down Pathway                  │
    │                                                     │
    │  P4 (highest) ──┐                                  │
    │                 │ upsample                          │
    │  P3 ───────────┼───> + ──> Conv3x3 ──> Out_P3     │
    │      lateral   │                                    │
    │                │ upsample                           │
    │  P2 ───────────┼───> + ──> Conv3x3 ──> Out_P2     │
    │      lateral   │                                    │
    │                │ upsample                           │
    │  P1 ───────────┴───> + ──> Conv3x3 ──> Out_P1     │
    │      lateral                                        │
    └─────────────────────────────────────────────────────┘

    Args:
        in_channels_list (list): List of input channels for each level
        out_channels (int): Number of output channels (same for all levels)
        use_extra_levels (bool): Whether to add extra pyramid levels
    """

    def __init__(
        self,
        in_channels_list,
        out_channels=256,
        use_extra_levels=False
    ):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_levels = len(in_channels_list)

        # Lateral connections (1x1 conv to match dimensions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Output convolutions (3x3 conv for smoothing)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

        # Extra pyramid levels (for even larger receptive fields)
        self.use_extra_levels = use_extra_levels
        if use_extra_levels:
            # P5: stride 64 (downsample from P4)
            self.extra_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        """
        Forward pass through FPN

        Args:
            features: List of [B, D_i, H_i, W_i] from encoder stages
                     [P1, P2, P3, P4] where P1 is highest resolution

        Returns:
            pyramid_features: List of [B, out_channels, H_i, W_i]
                            Multi-scale fused features
        """
        assert len(features) == self.num_levels

        # Build lateral connections
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down pathway with fusion
        # Start from the highest level (smallest resolution, P4)
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher-level feature
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i-1].shape[-2:],
                mode='nearest'
            )

            # Add to lower-level feature (element-wise)
            laterals[i-1] = laterals[i-1] + upsampled

        # Apply output convolutions (smoothing)
        pyramid_features = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

        # Add extra pyramid level if needed
        if self.use_extra_levels:
            # P5: downsample from P4
            p5 = self.extra_conv(pyramid_features[-1])
            pyramid_features.append(p5)

        return pyramid_features


class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network (BiFPN)

    Enhanced version of FPN with:
    - Bidirectional cross-scale connections
    - Weighted feature fusion
    - Faster and more accurate than standard FPN

    References:
    - EfficientDet: Scalable and Efficient Object Detection (Tan et al., CVPR 2020)

    Args:
        in_channels_list (list): List of input channels for each level
        out_channels (int): Number of output channels
        num_layers (int): Number of BiFPN layers to stack
    """

    def __init__(
        self,
        in_channels_list,
        out_channels=256,
        num_layers=2
    ):
        super().__init__()

        self.num_levels = len(in_channels_list)
        self.num_layers = num_layers

        # Input convolutions to match dimensions
        self.input_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, self.num_levels)
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        """
        Args:
            features: List of [B, D_i, H_i, W_i]

        Returns:
            pyramid_features: List of [B, out_channels, H_i, W_i]
        """
        # Match input dimensions
        features = [
            conv(feat)
            for conv, feat in zip(self.input_convs, features)
        ]

        # Apply BiFPN layers
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)

        return features


class BiFPNLayer(nn.Module):
    """
    Single BiFPN layer with bidirectional connections

    Args:
        channels (int): Number of channels
        num_levels (int): Number of pyramid levels
    """

    def __init__(self, channels, num_levels):
        super().__init__()

        self.num_levels = num_levels
        self.epsilon = 1e-4

        # Top-down pathway convs
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_levels - 1)
        ])

        # Bottom-up pathway convs
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_levels - 1)
        ])

        # Learnable fusion weights
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2))
            for _ in range(num_levels - 1)
        ])

        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3))
            for _ in range(num_levels - 1)
        ])

    def forward(self, features):
        """
        Args:
            features: List of [B, C, H_i, W_i]

        Returns:
            out_features: List of [B, C, H_i, W_i]
        """
        # Top-down pathway
        td_features = [features[-1]]  # Start from highest level

        for i in range(self.num_levels - 2, -1, -1):
            # Upsample from higher level
            up_feat = F.interpolate(
                td_features[0],
                size=features[i].shape[-2:],
                mode='nearest'
            )

            # Weighted fusion
            w = F.relu(self.td_weights[i])
            w = w / (w.sum() + self.epsilon)

            fused = w[0] * features[i] + w[1] * up_feat
            fused = self.td_convs[i](fused)

            td_features.insert(0, fused)

        # Bottom-up pathway
        bu_features = [td_features[0]]  # Start from lowest level

        for i in range(1, self.num_levels):
            # Downsample from lower level
            down_feat = F.max_pool2d(bu_features[-1], kernel_size=2)

            # Weighted fusion (3 inputs: original, top-down, bottom-up)
            w = F.relu(self.bu_weights[i - 1])
            w = w / (w.sum() + self.epsilon)

            fused = w[0] * features[i] + w[1] * td_features[i] + w[2] * down_feat
            fused = self.bu_convs[i - 1](fused)

            bu_features.append(fused)

        return bu_features


if __name__ == "__main__":
    # Unit test
    print("Testing FeaturePyramidNetwork...")

    batch_size = 2
    in_channels_list = [192, 384, 768, 768]
    out_channels = 256

    # Create multi-scale features (simulating encoder output)
    features = [
        torch.randn(batch_size, in_channels_list[0], 64, 64),  # P1: /8
        torch.randn(batch_size, in_channels_list[1], 32, 32),  # P2: /16
        torch.randn(batch_size, in_channels_list[2], 16, 16),  # P3: /32
        torch.randn(batch_size, in_channels_list[3], 8, 8),    # P4: /64
    ]

    print("\nInput features:")
    for i, feat in enumerate(features):
        print(f"  P{i+1}: {feat.shape}")

    # Test standard FPN
    print("\n1. Testing Standard FPN:")
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=out_channels
    )

    pyramid_features = fpn(features)

    print(f"\nOutput features:")
    for i, feat in enumerate(pyramid_features):
        print(f"  P{i+1}: {feat.shape}")
        assert feat.shape[1] == out_channels
        print(f"  ✓ Channels: {feat.shape[1]} (expected {out_channels})")

    # Test FPN with extra levels
    print("\n2. Testing FPN with Extra Levels:")
    fpn_extra = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        use_extra_levels=True
    )

    pyramid_features_extra = fpn_extra(features)
    print(f"\nOutput features with extra level:")
    for i, feat in enumerate(pyramid_features_extra):
        print(f"  P{i+1}: {feat.shape}")

    # Test BiFPN
    print("\n3. Testing BiFPN:")
    bifpn = BiFPN(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        num_layers=2
    )

    bifpn_features = bifpn(features)
    print(f"\nBiFPN output features:")
    for i, feat in enumerate(bifpn_features):
        print(f"  P{i+1}: {feat.shape}")
        assert feat.shape[1] == out_channels

    # Count parameters
    fpn_params = sum(p.numel() for p in fpn.parameters())
    bifpn_params = sum(p.numel() for p in bifpn.parameters())

    print(f"\n{'='*50}")
    print(f"FPN parameters: {fpn_params:,}")
    print(f"BiFPN parameters: {bifpn_params:,}")
    print(f"Difference: +{bifpn_params - fpn_params:,}")
    print(f"{'='*50}")

    print("\n✅ All tests passed!")
