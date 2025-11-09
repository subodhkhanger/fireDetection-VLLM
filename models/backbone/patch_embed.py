"""
Overlapping Patch Embedding for Fire-ViT

Implements patch embedding with overlapping patches for better
fine-grained feature extraction compared to standard ViT.
"""

import torch
import torch.nn as nn
import math


class OverlappingPatchEmbed(nn.Module):
    """
    Overlapping patch embedding with learned position encoding

    Standard ViT: 16x16 patches, stride=16 (non-overlapping)
    Our approach: 16x16 patches, stride=8 (50% overlap)

    Benefit: Better preservation of fine-grained details for small fires/smoke

    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        stride (int): Stride for patch extraction
        in_chans (int): Number of input channels
        embed_dim (int): Embedding dimension
    """

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        stride=8,
        in_chans=3,
        embed_dim=768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim

        # Calculate number of patches
        self.num_patches_h = (img_size - patch_size) // stride + 1
        self.num_patches_w = (img_size - patch_size) // stride + 1
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch projection using convolution
        # kernel_size=patch_size, stride=stride for overlapping patches
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride
        )

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using standard techniques"""
        # Xavier initialization for conv
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        # Truncated normal for position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input images

        Returns:
            patches: [B, N, D] embedded patches with position encoding
            spatial_shape: (H', W') spatial dimensions after patching
        """
        B, C, H, W = x.shape

        # Project patches: [B, C, H, W] -> [B, D, H', W']
        x = self.proj(x)

        # Flatten spatial dimensions: [B, D, H', W'] -> [B, D, N]
        x = x.flatten(2)

        # Transpose: [B, D, N] -> [B, N, D]
        x = x.transpose(1, 2)

        # Add position embeddings
        x = x + self.pos_embed

        # Apply layer normalization
        x = self.norm(x)

        # Return embeddings and spatial shape for later use
        spatial_shape = (self.num_patches_h, self.num_patches_w)

        return x, spatial_shape

    def get_num_patches(self):
        """Return total number of patches"""
        return self.num_patches


class SinusoidalPositionEncoding2D(nn.Module):
    """
    2D sinusoidal position encoding (fixed, not learned)
    Alternative to learned position embeddings

    Based on "Attention is All You Need" (Vaswani et al., 2017)
    Extended to 2D as in "End-to-End Object Detection with Transformers" (Carion et al., 2020)

    Args:
        embed_dim (int): Embedding dimension (must be divisible by 4)
        height (int): Height of the feature map
        width (int): Width of the feature map
        temperature (float): Temperature for frequency scaling
    """

    def __init__(self, embed_dim, height, width, temperature=10000):
        super().__init__()
        assert embed_dim % 4 == 0, "Embed dim must be divisible by 4 for 2D encoding"

        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        self.temperature = temperature

        # Generate position encoding
        pos_embed = self._generate_2d_sincos_embed()
        self.register_buffer('pos_embed', pos_embed)

    def _generate_2d_sincos_embed(self):
        """
        Generate 2D sinusoidal position encoding

        Returns:
            pos_embed: [1, H*W, D] position embeddings
        """
        # Generate grid coordinates
        y_pos = torch.arange(self.height, dtype=torch.float32)
        x_pos = torch.arange(self.width, dtype=torch.float32)
        y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing='ij')

        # Normalize to [0, 1]
        y_pos = y_pos / self.height
        x_pos = x_pos / self.width

        # Embedding dimension per coordinate (D/4 each for sin/cos of x and y)
        dim_per_coord = self.embed_dim // 4

        # Frequency bands
        omega = torch.arange(dim_per_coord, dtype=torch.float32)
        omega = 1.0 / (self.temperature ** (omega / dim_per_coord))

        # Apply sinusoidal encoding
        y_embed = y_pos[:, :, None] * omega[None, None, :]  # [H, W, D/4]
        x_embed = x_pos[:, :, None] * omega[None, None, :]  # [H, W, D/4]

        # Concatenate sin and cos for both coordinates
        pos_embed = torch.cat([
            torch.sin(y_embed),  # [H, W, D/4]
            torch.cos(y_embed),  # [H, W, D/4]
            torch.sin(x_embed),  # [H, W, D/4]
            torch.cos(x_embed)   # [H, W, D/4]
        ], dim=-1)  # [H, W, D]

        # Flatten spatial dimensions: [H, W, D] -> [H*W, D]
        pos_embed = pos_embed.reshape(-1, self.embed_dim)

        # Add batch dimension: [H*W, D] -> [1, H*W, D]
        pos_embed = pos_embed.unsqueeze(0)

        return pos_embed

    def forward(self, x):
        """
        Add position encoding to input

        Args:
            x: [B, N, D] input features

        Returns:
            x: [B, N, D] features with position encoding added
        """
        return x + self.pos_embed


if __name__ == "__main__":
    # Unit test
    print("Testing OverlappingPatchEmbed...")

    batch_size = 4
    img_size = 512
    patch_size = 16
    stride = 8
    embed_dim = 768

    # Create random input
    x = torch.randn(batch_size, 3, img_size, img_size)

    # Initialize patch embedding
    patch_embed = OverlappingPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        stride=stride,
        embed_dim=embed_dim
    )

    # Forward pass
    patches, spatial_shape = patch_embed(x)

    # Verify output shape
    expected_patches = ((img_size - patch_size) // stride + 1) ** 2
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {patches.shape}")
    print(f"✓ Expected patches: {expected_patches}, Got: {patches.shape[1]}")
    print(f"✓ Spatial shape: {spatial_shape}")
    print(f"✓ Position embedding shape: {patch_embed.pos_embed.shape}")

    assert patches.shape == (batch_size, expected_patches, embed_dim)
    print("\n✅ All tests passed!")

    # Test sinusoidal position encoding
    print("\nTesting SinusoidalPositionEncoding2D...")
    h, w = spatial_shape
    sin_pos_enc = SinusoidalPositionEncoding2D(embed_dim, h, w)

    patches_with_sin_pos = sin_pos_enc(patches)
    print(f"✓ Sinusoidal position encoding shape: {sin_pos_enc.pos_embed.shape}")
    print(f"✓ Output with sin pos encoding: {patches_with_sin_pos.shape}")
    print("\n✅ Sinusoidal position encoding test passed!")
