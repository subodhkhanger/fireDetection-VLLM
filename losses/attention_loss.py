"""
Attention Regularization Loss for Fire-ViT

Encourages attention heads to focus on fire/smoke regions.
Improves interpretability and potentially accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionRegularizationLoss(nn.Module):
    """
    Attention Regularization Loss

    Penalizes attention heads that don't focus on ground-truth fire regions.
    Encourages the model to learn interpretable attention patterns.

    L_attn = -Σ [A_ij * M_ij] / Σ M_ij

    where:
    - A_ij: attention weights for position (i,j)
    - M_ij: binary mask (1 if fire/smoke, 0 otherwise)

    This loss encourages high attention weights on fire/smoke regions.

    Args:
        reduction (str): 'mean' or 'sum'
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, attention_maps, gt_masks):
        """
        Args:
            attention_maps: [B, H, N, N] or [B, H, L] attention weights
                          where H is num heads, N is sequence length
            gt_masks: [B, H_img, W_img] binary ground truth masks

        Returns:
            loss: Scalar
        """
        if attention_maps is None:
            # If no attention maps available, return zero loss
            return torch.tensor(0.0, device=gt_masks.device)

        B = attention_maps.shape[0]
        num_heads = attention_maps.shape[1]

        # Resize masks to match attention resolution
        if attention_maps.dim() == 4:  # [B, H, N, N]
            N = attention_maps.shape[2]
            size = int(N ** 0.5)
            if size * size != N:
                # Non-square attention, skip
                return torch.tensor(0.0, device=gt_masks.device)

            # Average attention across queries (use mean attention per token)
            attn_avg = attention_maps.mean(dim=2)  # [B, H, N]
            attn_avg = attn_avg.reshape(B, num_heads, size, size)

        elif attention_maps.dim() == 3:  # [B, H, N]
            N = attention_maps.shape[2]
            size = int(N ** 0.5)
            if size * size != N:
                return torch.tensor(0.0, device=gt_masks.device)

            attn_avg = attention_maps.reshape(B, num_heads, size, size)
        else:
            return torch.tensor(0.0, device=gt_masks.device)

        # Resize ground truth masks
        masks_resized = F.interpolate(
            gt_masks.unsqueeze(1).float(),  # [B, 1, H, W]
            size=(size, size),
            mode='nearest'
        )  # [B, 1, size, size]

        # Expand for all heads
        masks_resized = masks_resized.expand(-1, num_heads, -1, -1)  # [B, H, size, size]

        # Compute overlap between attention and masks
        overlap = (attn_avg * masks_resized).sum(dim=(-2, -1))  # [B, H]

        # Normalize by mask area
        mask_area = masks_resized.sum(dim=(-2, -1)).clamp(min=1e-6)  # [B, H]
        normalized_overlap = overlap / mask_area

        # Clamp to [0, 1] to prevent negative loss when overlap > mask_area
        # This can happen when attention is highly concentrated on fire regions
        normalized_overlap = normalized_overlap.clamp(max=1.0)

        # Negative log to maximize overlap
        # Add small epsilon to avoid log(0)
        loss = -(normalized_overlap + 1e-6).log()

        # Average across heads and batch
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class AttentionDiversityLoss(nn.Module):
    """
    Attention Diversity Loss

    Encourages different attention heads to learn diverse patterns.
    Prevents all heads from collapsing to similar attention.

    L_div = Σ_i Σ_j (A_i · A_j) where i ≠ j

    Args:
        reduction (str): 'mean' or 'sum'
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, attention_maps):
        """
        Args:
            attention_maps: [B, H, N, N] attention weights

        Returns:
            loss: Scalar (should be minimized)
        """
        if attention_maps is None or attention_maps.dim() != 4:
            return torch.tensor(0.0)

        B, H, N, _ = attention_maps.shape

        # Average attention across sequence
        attn_avg = attention_maps.mean(dim=-1)  # [B, H, N]

        # Flatten
        attn_flat = attn_avg.reshape(B, H, -1)  # [B, H, N]

        # Normalize
        attn_norm = F.normalize(attn_flat, p=2, dim=-1)

        # Compute pairwise similarity
        similarity = torch.bmm(attn_norm, attn_norm.transpose(1, 2))  # [B, H, H]

        # Mask diagonal (self-similarity)
        mask = torch.eye(H, device=similarity.device).unsqueeze(0)
        similarity = similarity * (1 - mask)

        # Sum of similarities (minimize)
        loss = similarity.abs().sum(dim=(-2, -1))

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


if __name__ == "__main__":
    # Unit test
    print("Testing AttentionRegularizationLoss...")

    B, H, N = 4, 8, 64  # batch, heads, sequence_length
    H_img, W_img = 256, 256

    # Create dummy attention maps [B, H, N, N]
    attention_maps = torch.softmax(torch.randn(B, H, N, N), dim=-1)

    # Create dummy ground truth masks [B, H_img, W_img]
    gt_masks = torch.zeros(B, H_img, W_img)
    # Add some fire regions
    gt_masks[:, 50:150, 50:150] = 1.0

    # Test attention regularization loss
    attn_loss = AttentionRegularizationLoss()
    loss = attn_loss(attention_maps, gt_masks)

    print(f"  Attention maps shape: {attention_maps.shape}")
    print(f"  GT masks shape: {gt_masks.shape}")
    print(f"  Attention Regularization Loss: {loss.item():.4f}")
    assert loss.item() >= 0
    print(f"  ✓ Loss is non-negative")

    # Test with None attention maps
    loss_none = attn_loss(None, gt_masks)
    print(f"  Loss with None attention: {loss_none.item():.4f}")
    assert loss_none.item() == 0.0

    # Test diversity loss
    print(f"\nTesting AttentionDiversityLoss...")
    div_loss = AttentionDiversityLoss()
    loss_div = div_loss(attention_maps)

    print(f"  Attention Diversity Loss: {loss_div.item():.4f}")

    # Test with identical heads (should have high diversity loss)
    identical_attn = attention_maps[:, :1, :, :].expand(-1, H, -1, -1)
    loss_div_identical = div_loss(identical_attn)
    print(f"  Diversity loss (identical heads): {loss_div_identical.item():.4f}")
    print(f"  Diversity loss (diverse heads): {loss_div.item():.4f}")
    assert loss_div_identical.item() > loss_div.item()
    print(f"  ✓ Identical heads have higher diversity loss")

    print("\n✅ All attention loss tests passed!")
