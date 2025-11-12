"""
Debug script to understand model output format
"""

import torch
import yaml
from pathlib import Path
from models.fire_vit import build_fire_vit
from utils.checkpoint import load_checkpoint
from data.fire_dataset import FireDetectionDataset
from data.augmentations import get_val_transforms

# Load config
with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build model
print("Building model...")
model = build_fire_vit(config)

# Load checkpoint
checkpoint_path = 'experiments/checkpoints/best_model.pth'
print(f"Loading checkpoint: {checkpoint_path}")
epoch, metrics = load_checkpoint(checkpoint_path, model, device='cpu')
print(f"Loaded epoch {epoch}, metrics: {metrics}")

model.eval()

# Load a test image
test_annotation = config['data']['test_annotation']
test_img_dir = Path(test_annotation).parent
img_size = config['model']['input_size'][0]

test_dataset = FireDetectionDataset(
    image_dir=test_img_dir,
    annotation_file=test_annotation,
    transform=get_val_transforms(img_size),
    mode='test',
    img_size=img_size
)

print(f"\nTest dataset size: {len(test_dataset)}")

# Get a sample with objects
print("\nLooking for sample with ground truth boxes...")
for idx in range(min(20, len(test_dataset))):
    image, target = test_dataset[idx]
    if len(target['boxes']) > 0:
        print(f"\nFound sample at index {idx}")
        print(f"Image shape: {image.shape}")
        print(f"Ground truth boxes: {target['boxes'].shape}")
        print(f"Ground truth labels: {target['labels']}")
        print(f"Unique labels: {torch.unique(target['labels']).tolist()}")
        print(f"Original size: {target['orig_size']}")

        # Run inference
        print("\n" + "="*60)
        print("Running inference...")
        print("="*60)

        with torch.no_grad():
            image_batch = image.unsqueeze(0)  # Add batch dimension
            predictions = model(image_batch)

        print(f"\nNumber of prediction levels: {len(predictions)}")

        for level_idx, pred in enumerate(predictions):
            print(f"\n--- Level {level_idx} ---")
            print(f"Keys: {pred.keys()}")

            for key, value in pred.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                    print(f"  min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}")

                    if key == 'cls_logits':
                        # Check classification scores
                        cls_probs = torch.sigmoid(value)
                        print(f"  After sigmoid - min={cls_probs.min().item():.4f}, max={cls_probs.max().item():.4f}")
                        print(f"  Number of predictions > 0.01: {(cls_probs > 0.01).sum().item()}")
                        print(f"  Number of predictions > 0.1: {(cls_probs > 0.1).sum().item()}")
                        print(f"  Number of predictions > 0.25: {(cls_probs > 0.25).sum().item()}")
                        print(f"  Number of predictions > 0.5: {(cls_probs > 0.5).sum().item()}")

                    elif key == 'centerness':
                        centerness_probs = torch.sigmoid(value)
                        print(f"  After sigmoid - min={centerness_probs.min().item():.4f}, max={centerness_probs.max().item():.4f}")

        # Try to extract predictions with low threshold
        print("\n" + "="*60)
        print("Extracting predictions with conf_threshold=0.01...")
        print("="*60)

        pred = predictions[0]
        cls_logits = pred['cls_logits']
        bbox_pred = pred['bbox_pred']
        centerness = pred.get('centerness', None)

        B, C, H, W = cls_logits.shape
        print(f"\nBatch={B}, Classes={C}, Height={H}, Width={W}")

        cls_probs = torch.sigmoid(cls_logits)

        if centerness is not None:
            obj_scores = torch.sigmoid(centerness)
        else:
            obj_scores = torch.ones(B, 1, H, W)

        # Flatten
        cls_probs_flat = cls_probs.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        obj_scores_flat = obj_scores.reshape(B, 1, -1).permute(0, 2, 1)  # [B, H*W, 1]
        bbox_pred_flat = bbox_pred.reshape(B, 4, -1).permute(0, 2, 1)  # [B, H*W, 4]

        scores = cls_probs_flat * obj_scores_flat
        max_scores, max_classes = scores.max(dim=2)

        print(f"\nMax scores shape: {max_scores.shape}")
        print(f"Score statistics:")
        print(f"  min={max_scores.min().item():.6f}, max={max_scores.max().item():.6f}, mean={max_scores.mean().item():.6f}")

        for threshold in [0.001, 0.01, 0.05, 0.1, 0.25]:
            num_above = (max_scores[0] > threshold).sum().item()
            print(f"  Predictions > {threshold}: {num_above}")

        # Get top 10 predictions
        top_k = min(10, max_scores.shape[1])
        top_scores, top_indices = torch.topk(max_scores[0], top_k)
        top_classes = max_classes[0][top_indices]
        top_boxes = bbox_pred_flat[0][top_indices]

        print(f"\nTop {top_k} predictions:")
        for i in range(top_k):
            print(f"  {i+1}. score={top_scores[i].item():.6f}, class={top_classes[i].item()}, box={top_boxes[i].tolist()}")

        break
else:
    print("No samples with ground truth boxes found in first 20 images!")
