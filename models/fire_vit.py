"""
Fire-ViT: Custom Transformer-Based Fire Detection Model

Complete end-to-end model integrating:
- Hierarchical transformer encoder (backbone)
- Feature Pyramid Network (neck)
- Multi-scale detection head

Architecture:
Input Image -> Patch Embedding -> Hierarchical Encoder -> FPN -> Detection Heads -> Predictions
"""

import torch
import torch.nn as nn
from .backbone import HierarchicalTransformerEncoder
from .neck import FeaturePyramidNetwork
from .head import FireDetectionHead, MultiScaleDetectionHead


class FireViT(nn.Module):
    """
    Fire-ViT: Vision Transformer for Fire and Smoke Detection

    Complete model with:
    - Deformable attention for irregular fire shapes
    - Multi-scale feature extraction
    - Anchor-free detection

    Args:
        img_size (int): Input image size
        num_classes (int): Number of classes (fire, smoke)
        **kwargs: Additional model configuration
    """

    def __init__(
        self,
        img_size=640,
        num_classes=3,
        # Backbone config
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
        use_deformable=True,
        # Neck config
        fpn_channels=256,
        use_bifpn=False,
        bifpn_layers=2,
        # Head config
        num_head_convs=4,
        **kwargs
    ):
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        # Backbone: Hierarchical Transformer Encoder
        self.backbone = HierarchicalTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_ratios=mlp_ratios,
            num_points=num_points,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_deformable=use_deformable
        )

        # Neck: Feature Pyramid Network
        if use_bifpn:
            from .neck.fpn import BiFPN
            self.neck = BiFPN(
                in_channels_list=embed_dims,
                out_channels=fpn_channels,
                num_layers=bifpn_layers
            )
        else:
            self.neck = FeaturePyramidNetwork(
                in_channels_list=embed_dims,
                out_channels=fpn_channels,
                use_extra_levels=False
            )

        # Head: Multi-Scale Detection Head
        self.head = MultiScaleDetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=num_head_convs
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for modules without custom initialization"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False, return_attention=False):
        """
        Forward pass

        Args:
            x: [B, 3, H, W] input images
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention maps

        Returns:
            predictions: List of prediction dicts for each pyramid level
            (optional) features: Multi-scale features from backbone
            (optional) attention_maps: Attention maps from transformer
        """
        # Backbone: Extract multi-scale features
        backbone_features = self.backbone(x)  # List of [B, D_i, H_i, W_i]

        # Neck: Fuse multi-scale features
        neck_features = self.neck(backbone_features)  # List of [B, fpn_channels, H_i, W_i]

        # Head: Detect objects at each scale
        predictions = self.head(neck_features)  # List of prediction dicts

        # Prepare outputs
        outputs = {'predictions': predictions}

        if return_features:
            outputs['backbone_features'] = backbone_features
            outputs['neck_features'] = neck_features

        if return_attention:
            # TODO: Extract attention maps from backbone
            outputs['attention_maps'] = None

        if return_features or return_attention:
            return outputs
        else:
            return predictions

    def get_num_params(self):
        """Return number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        """Return model information"""
        info = {
            'model_name': 'Fire-ViT',
            'total_params': self.get_num_params(),
            'backbone_params': sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            'neck_params': sum(p.numel() for p in self.neck.parameters() if p.requires_grad),
            'head_params': sum(p.numel() for p in self.head.parameters() if p.requires_grad),
            'input_size': self.img_size,
            'num_classes': self.num_classes,
        }
        return info


def build_fire_vit(config):
    """
    Build Fire-ViT model from configuration

    Args:
        config: Configuration dict or object

    Returns:
        model: FireViT model
    """
    model_config = config.get('model', {}) if isinstance(config, dict) else config.model

    model = FireViT(
        img_size=model_config.get('input_size', [640, 640])[0],
        num_classes=model_config.get('num_classes', 3),
        patch_size=model_config.get('patch_size', 16),
        stride=model_config.get('stride', 8),
        embed_dims=model_config.get('embed_dims', [192, 384, 768, 768]),
        num_heads=model_config.get('num_heads', [8, 12, 16, 16]),
        num_blocks=model_config.get('num_blocks', [2, 2, 6, 2]),
        mlp_ratios=model_config.get('mlp_ratios', [4, 4, 4, 4]),
        num_points=model_config.get('num_sampling_points', 4),
        drop_rate=model_config.get('drop_rate', 0.0),
        attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
        drop_path_rate=model_config.get('drop_path_rate', 0.3),
        fpn_channels=model_config.get('fpn_channels', 256),
    )

    return model


if __name__ == "__main__":
    # Unit test
    print("Testing FireViT...")

    batch_size = 2
    img_size = 640
    num_classes = 2

    # Create random input
    x = torch.randn(batch_size, 3, img_size, img_size)
    print(f"\nInput shape: {x.shape}")

    # Initialize model
    print("\nInitializing Fire-ViT...")
    model = FireViT(
        img_size=img_size,
        num_classes=num_classes,
        embed_dims=[192, 384, 768, 768],
        num_heads=[8, 12, 16, 16],
        num_blocks=[2, 2, 6, 2],
        fpn_channels=256,
        use_deformable=True
    )

    # Get model info
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Model: {info['model_name']}")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  - Backbone: {info['backbone_params']:,}")
    print(f"  - Neck: {info['neck_params']:,}")
    print(f"  - Head: {info['head_params']:,}")
    print(f"  Input size: {info['input_size']}")
    print(f"  Num classes: {info['num_classes']}")

    # Forward pass
    print(f"\nForward pass...")
    model.eval()
    with torch.no_grad():
        predictions = model(x)

    print(f"\nPredictions from {len(predictions)} pyramid levels:")
    for i, pred in enumerate(predictions):
        print(f"\n  Level {i+1} (P{i+3}):")
        print(f"    Classification: {pred['cls_logits'].shape}")
        print(f"    Bbox: {pred['bbox_pred'].shape}")
        print(f"    Centerness: {pred['centerness'].shape}")

    # Test with return_features
    print(f"\nForward pass with features...")
    with torch.no_grad():
        outputs = model(x, return_features=True)

    print(f"\nBackbone features:")
    for i, feat in enumerate(outputs['backbone_features']):
        print(f"  Stage {i+1}: {feat.shape}")

    print(f"\nNeck features:")
    for i, feat in enumerate(outputs['neck_features']):
        print(f"  Level {i+1}: {feat.shape}")

    # Test model building from config
    print(f"\nTesting model building from config...")
    config = {
        'model': {
            'input_size': [640, 640],
            'num_classes': 2,
            'patch_size': 16,
            'stride': 8,
            'embed_dims': [192, 384, 768, 768],
            'num_heads': [8, 12, 16, 16],
            'num_blocks': [2, 2, 6, 2],
            'fpn_channels': 256
        }
    }

    model_from_config = build_fire_vit(config)
    print(f"  ✓ Model built from config")
    print(f"  Parameters: {model_from_config.get_num_params():,}")

    # Memory test
    print(f"\nMemory footprint:")
    total_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Model size: {total_memory / 1024**2:.2f} MB")

    print("\n✅ All tests passed!")
