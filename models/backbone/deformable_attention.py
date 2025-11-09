"""
Deformable Multi-Head Self-Attention for Fire-ViT

Implements deformable attention mechanism that learns adaptive sampling
locations instead of fixed grid patterns. This is crucial for detecting
irregular fire and smoke shapes.

References:
- Deformable DETR (Zhu et al., ICLR 2021)
- Deformable Attention Transformer (Xia et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeformableMultiHeadAttention(nn.Module):
    """
    Deformable Multi-Head Self-Attention (DMHSA)

    Key Innovation:
    - Learns adaptive sampling offsets instead of using fixed attention patterns
    - Better for irregular fire/smoke shapes that don't follow grid patterns
    - Reduces computation compared to full self-attention

    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads
        num_points (int): Number of sampling points per head
        qkv_bias (bool): Whether to use bias in QKV projection
        attn_drop (float): Attention dropout rate
        proj_drop (float): Output projection dropout rate
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        num_points=4,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Offset network: predicts 2D offsets for each sampling point
        # Output: [num_heads * num_points * 2] (x, y offsets for each point)
        self.offset_network = nn.Linear(dim, num_heads * num_points * 2)

        # Attention weights for sampled points
        self.attention_weights = nn.Linear(dim, num_heads * num_points)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Initialize offset network to produce small offsets initially
        nn.init.zeros_(self.offset_network.weight)
        nn.init.zeros_(self.offset_network.bias)

        # Initialize attention weights
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

        # Initialize QKV projection
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)

        # Initialize output projection
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _get_reference_points(self, H, W, device):
        """
        Generate reference points (grid centers)

        Args:
            H (int): Height of feature map
            W (int): Width of feature map
            device: torch device

        Returns:
            reference_points: [H*W, 2] normalized coordinates in [0, H-1] x [0, W-1]
        """
        y_coords = torch.arange(H, dtype=torch.float32, device=device)
        x_coords = torch.arange(W, dtype=torch.float32, device=device)
        y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing='ij')

        reference_points = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]
        reference_points = reference_points.reshape(-1, 2)  # [H*W, 2]

        return reference_points

    def forward(self, x, spatial_shape):
        """
        Forward pass with deformable attention

        Args:
            x: [B, N, D] input features (N = H*W patches)
            spatial_shape: (H, W) spatial dimensions

        Returns:
            out: [B, N, D] attended features
        """
        B, N, D = x.shape
        H, W = spatial_shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Predict sampling offsets
        offsets = self.offset_network(x)  # [B, N, num_heads*num_points*2]
        offsets = offsets.reshape(B, N, self.num_heads, self.num_points, 2)
        offsets = torch.tanh(offsets) * 2.0  # Constrain to [-2, 2] grid units

        # Generate reference points (grid centers)
        reference_points = self._get_reference_points(H, W, x.device)  # [H*W, 2]
        reference_points = reference_points.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]

        # Apply offsets to reference points
        # [B, N, 2] -> [B, N, 1, 1, 2] + [B, N, num_heads, num_points, 2]
        sampling_locations = reference_points[:, :, None, None, :] + offsets  # [B, N, num_heads, num_points, 2]

        # Normalize to [-1, 1] for grid_sample
        sampling_locations_normalized = sampling_locations.clone()
        sampling_locations_normalized[..., 0] = sampling_locations[..., 0] / (W - 1) * 2 - 1  # x
        sampling_locations_normalized[..., 1] = sampling_locations[..., 1] / (H - 1) * 2 - 1  # y

        # Reshape value for spatial sampling
        # [B, num_heads, N, head_dim] -> [B, num_heads, head_dim, H, W]
        v_spatial = v.permute(0, 1, 3, 2).reshape(B, self.num_heads, self.head_dim, H, W)

        # Sample features at deformable locations for each head
        sampled_features_list = []

        for head_idx in range(self.num_heads):
            # Get sampling locations for this head [B, N, num_points, 2]
            locs = sampling_locations_normalized[:, :, head_idx, :, :]

            # Reshape for grid_sample: [B, N*num_points, 1, 2]
            locs_flat = locs.reshape(B, N * self.num_points, 1, 2)

            # Sample from value features for this head
            # Input: [B, head_dim, H, W], Grid: [B, N*num_points, 1, 2]
            # Output: [B, head_dim, N*num_points, 1]
            sampled = F.grid_sample(
                v_spatial[:, head_idx, :, :, :],
                locs_flat,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )

            # Reshape: [B, head_dim, N*num_points, 1] -> [B, head_dim, N, num_points]
            sampled = sampled.squeeze(-1).reshape(B, self.head_dim, N, self.num_points)

            # Permute: [B, head_dim, N, num_points] -> [B, N, num_points, head_dim]
            sampled = sampled.permute(0, 2, 3, 1)

            sampled_features_list.append(sampled)

        # Stack all heads: [B, N, num_heads, num_points, head_dim]
        sampled_features = torch.stack(sampled_features_list, dim=2)

        # Compute attention weights for sampled points
        attn_weights = self.attention_weights(x)  # [B, N, num_heads*num_points]
        attn_weights = attn_weights.reshape(B, N, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)  # Softmax over sampling points
        attn_weights = self.attn_drop(attn_weights)

        # Weighted aggregation
        # [B, N, num_heads, num_points, 1] * [B, N, num_heads, num_points, head_dim]
        attn_weights = attn_weights.unsqueeze(-1)
        out = (sampled_features * attn_weights).sum(dim=3)  # [B, N, num_heads, head_dim]

        # Concatenate heads
        out = out.reshape(B, N, D)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class StandardMultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention for comparison

    This is the baseline attention mechanism used in standard ViT.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, spatial_shape=None):
        """
        Args:
            x: [B, N, D]
            spatial_shape: unused, for API compatibility

        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention: Q @ K^T
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum: Attn @ V
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


if __name__ == "__main__":
    # Unit test
    print("Testing DeformableMultiHeadAttention...")

    batch_size = 2
    H, W = 64, 64
    N = H * W
    dim = 768
    num_heads = 8
    num_points = 4

    # Create random input
    x = torch.randn(batch_size, N, dim)

    # Initialize deformable attention
    deform_attn = DeformableMultiHeadAttention(
        dim=dim,
        num_heads=num_heads,
        num_points=num_points
    )

    # Forward pass
    out = deform_attn(x, spatial_shape=(H, W))

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    assert out.shape == (batch_size, N, dim)
    print(f"✓ Spatial shape: ({H}, {W})")

    # Test standard attention for comparison
    print("\nTesting StandardMultiHeadAttention...")
    std_attn = StandardMultiHeadAttention(dim=dim, num_heads=num_heads)
    out_std = std_attn(x)

    print(f"✓ Standard attention output shape: {out_std.shape}")
    assert out_std.shape == (batch_size, N, dim)

    print("\n✅ All tests passed!")

    # Count parameters
    deform_params = sum(p.numel() for p in deform_attn.parameters())
    std_params = sum(p.numel() for p in std_attn.parameters())
    print(f"\nDeformable attention parameters: {deform_params:,}")
    print(f"Standard attention parameters: {std_params:,}")
    print(f"Difference: {deform_params - std_params:,} (+{(deform_params/std_params - 1)*100:.1f}%)")
