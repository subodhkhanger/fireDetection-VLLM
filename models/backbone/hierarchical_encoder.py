"""
Hierarchical Transformer Encoder for Fire-ViT

Implements a 4-stage hierarchical vision transformer that progressively:
- Reduces spatial resolution (downsampling)
- Increases embedding dimension
- Extracts multi-scale features

Stage 1: [H/8, W/8], D=192, 2 blocks
Stage 2: [H/16, W/16], D=384, 2 blocks
Stage 3: [H/32, W/32], D=768, 6 blocks
Stage 4: [H/64, W/64], D=768, 2 blocks
"""

import torch
import torch.nn as nn
from .patch_embed import OverlappingPatchEmbed
from .transformer_block import TransformerBlock


class PatchMerging(nn.Module):
    """
    Patch Merging Layer for downsampling

    Reduces spatial resolution by 2x and increases channels.
    Similar to pooling but learnable.

    Args:
        dim (int): Input dimension
        out_dim (int): Output dimension
        downsample_ratio (int): Downsampling ratio (default: 2)
    """

    def __init__(self, dim, out_dim, downsample_ratio=2):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.downsample_ratio = downsample_ratio

        # Use strided convolution for downsampling
        self.reduction = nn.Conv2d(
            dim,
            out_dim,
            kernel_size=3,
            stride=downsample_ratio,
            padding=1
        )
        self.norm = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.reduction.weight)
        if self.reduction.bias is not None:
            nn.init.zeros_(self.reduction.bias)

    def forward(self, x, H, W):
        """
        Args:
            x: [B, H*W, C] input features
            H, W: Spatial dimensions

        Returns:
            x: [B, (H/2)*(W/2), out_dim] downsampled features
            H_out, W_out: New spatial dimensions
        """
        B, N, C = x.shape

        # Reshape to spatial: [B, H*W, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Downsample
        x = self.reduction(x)  # [B, out_dim, H/2, W/2]

        # Get new dimensions
        _, _, H_out, W_out = x.shape

        # Flatten: [B, out_dim, H/2, W/2] -> [B, (H/2)*(W/2), out_dim]
        x = x.flatten(2).transpose(1, 2)

        # Normalize
        x = self.norm(x)

        return x, H_out, W_out


class HierarchicalTransformerEncoder(nn.Module):
    """
    Hierarchical Vision Transformer Encoder

    4 stages with progressively:
    - Reduced spatial resolution (downsampling)
    - Increased embedding dimension
    - Different number of transformer blocks

    Args:
        img_size (int): Input image size
        patch_size (int): Initial patch size
        stride (int): Patch extraction stride
        in_chans (int): Number of input channels
        embed_dims (list): Embedding dimensions for each stage
        num_heads (list): Number of attention heads for each stage
        num_blocks (list): Number of transformer blocks for each stage
        mlp_ratios (list): MLP expansion ratios for each stage
        num_points (int): Number of sampling points for deformable attention
        drop_rate (float): Dropout rate
        attn_drop_rate (float): Attention dropout rate
        drop_path_rate (float): Stochastic depth rate
        use_deformable (bool): Whether to use deformable attention
    """

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        stride=8,
        in_chans=3,
        embed_dims=[192, 384, 768, 768],
        num_heads=[8, 12, 16, 16],
        num_blocks=[2, 2, 6, 2],
        mlp_ratios=[4, 4, 4, 4],
        num_points=4,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_deformable=True
    ):
        super().__init__()

        self.num_stages = len(embed_dims)
        self.embed_dims = embed_dims

        # Stochastic depth decay rule (linearly increase drop path rate)
        total_blocks = sum(num_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Stage 1: Initial patch embedding
        self.patch_embed = OverlappingPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dims[0]
        )

        # Calculate initial spatial dimensions
        self.init_H = (img_size - patch_size) // stride + 1
        self.init_W = (img_size - patch_size) // stride + 1

        # Build stages
        self.stages = nn.ModuleList()
        cur_block_idx = 0

        for stage_idx in range(self.num_stages):
            # Patch merging (downsampling) - skip for first stage
            if stage_idx > 0:
                patch_merging = PatchMerging(
                    dim=embed_dims[stage_idx - 1],
                    out_dim=embed_dims[stage_idx],
                    downsample_ratio=2
                )
            else:
                patch_merging = nn.Identity()

            # Transformer blocks for this stage
            blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    num_points=num_points,
                    mlp_ratio=mlp_ratios[stage_idx],
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur_block_idx + i],
                    use_deformable=use_deformable
                )
                for i in range(num_blocks[stage_idx])
            ])

            cur_block_idx += num_blocks[stage_idx]

            # Add stage
            self.stages.append(nn.ModuleDict({
                'patch_merging': patch_merging,
                'blocks': blocks
            }))

    def forward(self, x):
        """
        Forward pass through hierarchical encoder

        Args:
            x: [B, 3, H, W] input images

        Returns:
            features: List of [B, D_i, H_i, W_i] for each stage (4 feature maps)
        """
        B = x.shape[0]

        # Initial patch embedding
        x, (H, W) = self.patch_embed(x)  # [B, N, D_0]

        # Multi-scale features from each stage
        features = []

        for stage_idx, stage in enumerate(self.stages):
            # Downsampling (patch merging)
            if stage_idx > 0:
                x, H, W = stage['patch_merging'](x, H, W)

            # Transformer blocks
            for block in stage['blocks']:
                x = block(x, spatial_shape=(H, W))

            # Save features (reshape to spatial format)
            # [B, N, D] -> [B, D, H, W]
            feat = x.transpose(1, 2).reshape(B, -1, H, W)
            features.append(feat)

        return features

    def get_feature_dims(self):
        """Return feature dimensions for each stage"""
        return self.embed_dims

    def get_num_stages(self):
        """Return number of stages"""
        return self.num_stages


if __name__ == "__main__":
    # Unit test
    print("Testing HierarchicalTransformerEncoder...")

    batch_size = 2
    img_size = 512
    in_chans = 3

    # Create random input
    x = torch.randn(batch_size, in_chans, img_size, img_size)

    # Initialize encoder
    encoder = HierarchicalTransformerEncoder(
        img_size=img_size,
        patch_size=16,
        stride=8,
        in_chans=in_chans,
        embed_dims=[192, 384, 768, 768],
        num_heads=[8, 12, 16, 16],
        num_blocks=[2, 2, 6, 2],
        mlp_ratios=[4, 4, 4, 4],
        num_points=4,
        drop_path_rate=0.3,
        use_deformable=True
    )

    # Forward pass
    print(f"\nInput shape: {x.shape}")
    features = encoder(x)

    # Verify output
    print(f"\nNumber of output feature maps: {len(features)}")
    expected_dims = [192, 384, 768, 768]
    expected_spatial = [
        (img_size // 8, img_size // 8),    # Stage 1: /8
        (img_size // 16, img_size // 16),  # Stage 2: /16
        (img_size // 32, img_size // 32),  # Stage 3: /32
        (img_size // 64, img_size // 64)   # Stage 4: /64
    ]

    for i, (feat, exp_dim, exp_spatial) in enumerate(zip(features, expected_dims, expected_spatial)):
        print(f"\nStage {i+1}:")
        print(f"  Output shape: {feat.shape}")
        print(f"  Expected: [B={batch_size}, D={exp_dim}, H={exp_spatial[0]}, W={exp_spatial[1]}]")
        assert feat.shape == (batch_size, exp_dim, exp_spatial[0], exp_spatial[1])
        print(f"  ✓ Shape correct")

    # Count total parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"\n{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*50}")

    # Test with standard attention
    print("\n\nTesting with Standard Attention (non-deformable)...")
    encoder_std = HierarchicalTransformerEncoder(
        img_size=img_size,
        embed_dims=[192, 384, 768, 768],
        num_heads=[8, 12, 16, 16],
        num_blocks=[2, 2, 6, 2],
        use_deformable=False
    )

    features_std = encoder_std(x)
    print(f"Number of output feature maps: {len(features_std)}")
    for i, feat in enumerate(features_std):
        print(f"  Stage {i+1} shape: {feat.shape}")

    std_params = sum(p.numel() for p in encoder_std.parameters())
    print(f"\nStandard attention parameters: {std_params:,}")
    print(f"Deformable attention parameters: {total_params:,}")
    print(f"Difference: +{total_params - std_params:,} ({(total_params/std_params - 1)*100:.1f}%)")

    print("\n✅ All tests passed!")
