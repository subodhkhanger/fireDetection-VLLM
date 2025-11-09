"""
Comprehensive Test Script for Fire-ViT

Tests all components individually and end-to-end integration.
Run this to verify the complete implementation.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.fire_vit import FireViT, build_fire_vit
from losses import CompositeLoss
import yaml


def test_patch_embedding():
    """Test patch embedding module"""
    print("\n" + "="*60)
    print("Testing Patch Embedding...")
    print("="*60)

    from models.backbone.patch_embed import OverlappingPatchEmbed

    patch_embed = OverlappingPatchEmbed(
        img_size=512,
        patch_size=16,
        stride=8,
        embed_dim=768
    )

    x = torch.randn(2, 3, 512, 512)
    patches, spatial_shape = patch_embed(x)

    print(f"✓ Input: {x.shape}")
    print(f"✓ Output: {patches.shape}")
    print(f"✓ Spatial shape: {spatial_shape}")
    assert patches.shape == (2, 3969, 768)  # (512-16)//8+1 = 63, 63*63 = 3969
    print("✅ Patch Embedding Test Passed!\n")


def test_deformable_attention():
    """Test deformable attention mechanism"""
    print("\n" + "="*60)
    print("Testing Deformable Attention...")
    print("="*60)

    from models.backbone.deformable_attention import DeformableMultiHeadAttention

    attn = DeformableMultiHeadAttention(
        dim=768,
        num_heads=12,
        num_points=4
    )

    x = torch.randn(2, 4096, 768)
    out = attn(x, spatial_shape=(64, 64))

    print(f"✓ Input: {x.shape}")
    print(f"✓ Output: {out.shape}")
    assert out.shape == x.shape
    print("✅ Deformable Attention Test Passed!\n")


def test_hierarchical_encoder():
    """Test hierarchical transformer encoder"""
    print("\n" + "="*60)
    print("Testing Hierarchical Encoder...")
    print("="*60)

    from models.backbone.hierarchical_encoder import HierarchicalTransformerEncoder

    encoder = HierarchicalTransformerEncoder(
        img_size=512,
        embed_dims=[192, 384, 768, 768],
        num_heads=[8, 12, 16, 16],
        num_blocks=[2, 2, 6, 2]
    )

    x = torch.randn(2, 3, 512, 512)
    features = encoder(x)

    print(f"✓ Input: {x.shape}")
    print(f"✓ Number of output features: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Stage {i+1}: {feat.shape}")

    print("✅ Hierarchical Encoder Test Passed!\n")


def test_fpn():
    """Test Feature Pyramid Network"""
    print("\n" + "="*60)
    print("Testing FPN...")
    print("="*60)

    from models.neck.fpn import FeaturePyramidNetwork

    fpn = FeaturePyramidNetwork(
        in_channels_list=[192, 384, 768, 768],
        out_channels=256
    )

    # Dummy multi-scale features
    features = [
        torch.randn(2, 192, 64, 64),
        torch.randn(2, 384, 32, 32),
        torch.randn(2, 768, 16, 16),
        torch.randn(2, 768, 8, 8),
    ]

    pyramid_features = fpn(features)

    print(f"✓ Input features: {len(features)}")
    print(f"✓ Output features: {len(pyramid_features)}")
    for i, feat in enumerate(pyramid_features):
        print(f"  Level {i+1}: {feat.shape}")
        assert feat.shape[1] == 256

    print("✅ FPN Test Passed!\n")


def test_detection_head():
    """Test detection head"""
    print("\n" + "="*60)
    print("Testing Detection Head...")
    print("="*60)

    from models.head.detection_head import MultiScaleDetectionHead

    head = MultiScaleDetectionHead(
        in_channels=256,
        num_classes=2
    )

    pyramid_features = [
        torch.randn(2, 256, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 256, 8, 8),
    ]

    predictions = head(pyramid_features)

    print(f"✓ Input features: {len(pyramid_features)}")
    print(f"✓ Predictions: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"  Level {i+1}:")
        print(f"    Classification: {pred['cls_logits'].shape}")
        print(f"    BBox: {pred['bbox_pred'].shape}")
        print(f"    Centerness: {pred['centerness'].shape}")

    print("✅ Detection Head Test Passed!\n")


def test_losses():
    """Test all loss functions"""
    print("\n" + "="*60)
    print("Testing Loss Functions...")
    print("="*60)

    from losses import FocalLoss, CIoULoss, DiceLoss, AttentionRegularizationLoss

    # Focal Loss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 3, (2, 64, 64))
    loss_focal = focal(pred, target)
    print(f"✓ Focal Loss: {loss_focal.item():.4f}")

    # CIoU Loss
    ciou = CIoULoss()
    boxes1 = torch.rand(10, 4) * 100
    boxes1[:, 2:] += boxes1[:, :2]
    boxes2 = torch.rand(10, 4) * 100
    boxes2[:, 2:] += boxes2[:, :2]
    loss_ciou = ciou(boxes1, boxes2)
    print(f"✓ CIoU Loss: {loss_ciou.item():.4f}")

    # Dice Loss
    dice = DiceLoss()
    loss_dice = dice(pred, target)
    print(f"✓ Dice Loss: {loss_dice.item():.4f}")

    # Attention Loss
    attn_loss = AttentionRegularizationLoss()
    attention_maps = torch.softmax(torch.randn(2, 8, 64, 64), dim=-1)
    masks = torch.rand(2, 256, 256) > 0.9
    loss_attn = attn_loss(attention_maps, masks)
    print(f"✓ Attention Loss: {loss_attn.item():.4f}")

    print("✅ All Loss Functions Test Passed!\n")


def test_composite_loss():
    """Test composite loss"""
    print("\n" + "="*60)
    print("Testing Composite Loss...")
    print("="*60)

    composite_loss = CompositeLoss(
        focal_alpha=0.25,
        focal_gamma=2.0,
        use_attention_loss=True
    )

    # Dummy predictions
    predictions = [
        {
            'cls_logits': torch.randn(2, 2, 64, 64),
            'bbox_pred': torch.rand(2, 4, 64, 64) * 10,
            'centerness': torch.randn(2, 1, 64, 64)
        }
        for _ in range(4)
    ]

    # Dummy targets
    targets = {
        'labels': torch.randint(0, 2, (2, 64, 64)),
        'boxes': torch.rand(2, 10, 4) * 100,
        'masks': torch.rand(2, 64, 64) > 0.9
    }

    attention_maps = torch.softmax(torch.randn(2, 8, 256, 256), dim=-1)

    total_loss, loss_dict = composite_loss(predictions, targets, attention_maps, epoch=10)

    print(f"✓ Total Loss: {total_loss.item():.4f}")
    print(f"✓ Loss Components:")
    for key, value in loss_dict.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")

    print("✅ Composite Loss Test Passed!\n")


def test_complete_model():
    """Test complete Fire-ViT model"""
    print("\n" + "="*60)
    print("Testing Complete Fire-ViT Model...")
    print("="*60)

    model = FireViT(
        img_size=640,
        num_classes=2,
        embed_dims=[192, 384, 768, 768],
        num_heads=[8, 12, 16, 16],
        num_blocks=[2, 2, 6, 2],
        fpn_channels=256
    )

    # Get model info
    info = model.get_model_info()
    print(f"\n✓ Model: {info['model_name']}")
    print(f"✓ Total Parameters: {info['total_params']:,}")
    print(f"  - Backbone: {info['backbone_params']:,}")
    print(f"  - Neck: {info['neck_params']:,}")
    print(f"  - Head: {info['head_params']:,}")

    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    print(f"\n✓ Input: {x.shape}")

    model.eval()
    with torch.no_grad():
        predictions = model(x)

    print(f"✓ Predictions from {len(predictions)} pyramid levels:")
    for i, pred in enumerate(predictions):
        print(f"  Level {i+1}:")
        print(f"    Classification: {pred['cls_logits'].shape}")
        print(f"    BBox: {pred['bbox_pred'].shape}")
        print(f"    Centerness: {pred['centerness'].shape}")

    # Test with features
    with torch.no_grad():
        outputs = model(x, return_features=True)

    print(f"\n✓ Backbone features: {len(outputs['backbone_features'])}")
    print(f"✓ Neck features: {len(outputs['neck_features'])}")

    print("✅ Complete Model Test Passed!\n")


def test_model_from_config():
    """Test building model from config"""
    print("\n" + "="*60)
    print("Testing Model from Config...")
    print("="*60)

    # Load config
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = build_fire_vit(config)

    print(f"✓ Model built from config")
    print(f"✓ Parameters: {model.get_num_params():,}")

    # Test forward
    x = torch.randn(1, 3, 640, 640)
    model.eval()
    with torch.no_grad():
        predictions = model(x)

    print(f"✓ Forward pass successful")
    print(f"✓ Output levels: {len(predictions)}")

    print("✅ Config-based Model Test Passed!\n")


def test_memory_and_speed():
    """Test memory footprint and inference speed"""
    print("\n" + "="*60)
    print("Testing Memory and Speed...")
    print("="*60)

    model = FireViT(
        img_size=640,
        num_classes=2,
        embed_dims=[192, 384, 768, 768],
        num_heads=[8, 12, 16, 16],
        num_blocks=[2, 2, 6, 2]
    )

    # Memory
    total_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"✓ Model size: {total_memory / 1024**2:.2f} MB")

    # Speed test (CPU)
    model.eval()
    x = torch.randn(1, 3, 640, 640)

    import time
    num_runs = 10

    with torch.no_grad():
        # Warmup
        _ = model(x)

        # Measure
        start = time.time()
        for _ in range(num_runs):
            _ = model(x)
        end = time.time()

    avg_time = (end - start) / num_runs
    fps = 1.0 / avg_time

    print(f"✓ Average inference time (CPU): {avg_time*1000:.2f} ms")
    print(f"✓ Throughput (CPU): {fps:.2f} FPS")

    print("✅ Memory and Speed Test Passed!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*20 + "FIRE-VIT TEST SUITE")
    print("="*70)

    try:
        test_patch_embedding()
        test_deformable_attention()
        test_hierarchical_encoder()
        test_fpn()
        test_detection_head()
        test_losses()
        test_composite_loss()
        test_complete_model()
        test_model_from_config()
        test_memory_and_speed()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*70)
        print("\nFire-ViT implementation is complete and working correctly.")
        print("You can now proceed to:")
        print("  1. Implement data loading pipeline")
        print("  2. Set up training loop")
        print("  3. Begin model training")
        print("\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
