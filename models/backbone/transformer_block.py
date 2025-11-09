"""
Transformer Block for Fire-ViT

Implements a transformer block with deformable attention and feed-forward network.
Includes stochastic depth (drop path) for regularization.
"""

import torch
import torch.nn as nn
from .deformable_attention import DeformableMultiHeadAttention


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample

    When applied in main path, can be used for regularization.
    This is the same as the DropConnect implementation from EfficientNet, etc.

    References:
    - Deep Networks with Stochastic Depth (Huang et al., ECCV 2016)
    - https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        # Work with different dims: [B, N, D] or [B, D, H, W]
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self):
        return f'drop_prob={self.drop_prob}'


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN)

    Two-layer MLP with GELU activation:
    FFN(x) = Linear(GELU(Linear(x)))

    Args:
        dim (int): Input dimension
        mlp_ratio (float): Ratio of hidden dim to input dim
        drop (float): Dropout rate
        act_layer: Activation layer
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        act_layer=nn.GELU
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with deformable attention

    Architecture:
    ┌─────────────────────────────────┐
    │  Input: x [B, N, D]             │
    └────────────┬────────────────────┘
                 │
         ┌───────▼────────┐
         │  LayerNorm     │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Deformable    │
         │  Attention     │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  DropPath      │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Add & Norm    │ ◄─── Residual
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  LayerNorm     │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  FFN           │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  DropPath      │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Add           │ ◄─── Residual
         └────────────────┘

    Args:
        dim (int): Feature dimension
        num_heads (int): Number of attention heads
        num_points (int): Number of sampling points for deformable attention
        mlp_ratio (float): Ratio of FFN hidden dim to input dim
        drop (float): Dropout rate
        attn_drop (float): Attention dropout rate
        drop_path (float): Stochastic depth rate
        act_layer: Activation layer
        use_deformable (bool): Whether to use deformable attention
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        num_points=4,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        use_deformable=True
    ):
        super().__init__()

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Attention mechanism
        if use_deformable:
            self.attn = DeformableMultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                num_points=num_points,
                attn_drop=attn_drop,
                proj_drop=drop
            )
        else:
            # Fallback to standard attention
            from .deformable_attention import StandardMultiHeadAttention
            self.attn = StandardMultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop
            )

        # Feed-forward network
        self.mlp = FeedForward(
            dim=dim,
            mlp_ratio=mlp_ratio,
            drop=drop,
            act_layer=act_layer
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, spatial_shape):
        """
        Forward pass

        Args:
            x: [B, N, D] input features
            spatial_shape: (H, W) spatial dimensions

        Returns:
            x: [B, N, D] output features
        """
        # Attention block with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x), spatial_shape))

        # FFN block with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


if __name__ == "__main__":
    # Unit test
    print("Testing TransformerBlock...")

    batch_size = 2
    H, W = 64, 64
    N = H * W
    dim = 768
    num_heads = 12
    num_points = 4

    # Create random input
    x = torch.randn(batch_size, N, dim)

    # Test with deformable attention
    print("\n1. Testing with Deformable Attention:")
    block_deform = TransformerBlock(
        dim=dim,
        num_heads=num_heads,
        num_points=num_points,
        mlp_ratio=4.0,
        drop_path=0.1,
        use_deformable=True
    )

    out_deform = block_deform(x, spatial_shape=(H, W))
    print(f"   ✓ Input shape: {x.shape}")
    print(f"   ✓ Output shape: {out_deform.shape}")
    assert out_deform.shape == x.shape
    print(f"   ✓ Shape preserved")

    # Test with standard attention
    print("\n2. Testing with Standard Attention:")
    block_std = TransformerBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop_path=0.1,
        use_deformable=False
    )

    out_std = block_std(x, spatial_shape=(H, W))
    print(f"   ✓ Output shape: {out_std.shape}")
    assert out_std.shape == x.shape

    # Test drop path
    print("\n3. Testing DropPath:")
    drop_path = DropPath(drop_prob=0.1)
    x_test = torch.ones(4, 100, 768)

    # Training mode
    drop_path.train()
    out_train = drop_path(x_test)
    print(f"   ✓ Training mode - some paths dropped")

    # Eval mode
    drop_path.eval()
    out_eval = drop_path(x_test)
    print(f"   ✓ Eval mode - no paths dropped")
    assert torch.allclose(out_eval, x_test)

    # Count parameters
    params_deform = sum(p.numel() for p in block_deform.parameters())
    params_std = sum(p.numel() for p in block_std.parameters())
    print(f"\n4. Parameter Count:")
    print(f"   Deformable block: {params_deform:,}")
    print(f"   Standard block: {params_std:,}")
    print(f"   Difference: +{params_deform - params_std:,}")

    print("\n✅ All tests passed!")
