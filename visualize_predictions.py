"""Visualize predictions vs ground truth to debug bbox format"""

import torch
import yaml
import cv2
import numpy as np
from pathlib import Path
from models.fire_vit import build_fire_vit
from utils.checkpoint import load_checkpoint
from data.fire_dataset import FireDetectionDataset
from data.augmentations import get_val_transforms

# Load config
with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build model
model = build_fire_vit(config)
checkpoint_path = 'experiments/checkpoints/best_model.pth'
load_checkpoint(checkpoint_path, model, device='cpu')
model.eval()

# Load test dataset
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

# Find a sample with GT boxes
for idx in range(min(50, len(test_dataset))):
    image, target = test_dataset[idx]
    if len(target['boxes']) > 0:
        print(f"\nSample {idx}:")
        print(f"Image shape: {image.shape}")
        print(f"Original size: {target['orig_size']}")
        print(f"Ground truth boxes: {target['boxes']}")
        print(f"Ground truth labels: {target['labels']}")

        # Run inference
        with torch.no_grad():
            predictions = model(image.unsqueeze(0))

        pred = predictions[0]
        cls_logits = pred['cls_logits']
        bbox_pred = pred['bbox_pred']

        B, C, H, W = cls_logits.shape
        print(f"\nPrediction shape: C={C}, H={H}, W={W}")

        cls_probs = torch.sigmoid(cls_logits)
        cls_probs_flat = cls_probs.reshape(B, C, -1).permute(0, 2, 1)
        bbox_pred_flat = bbox_pred.reshape(B, 4, -1).permute(0, 2, 1)

        max_scores, max_classes = cls_probs_flat.max(dim=2)

        # Get top predictions
        top_k = 5
        top_scores, top_indices = torch.topk(max_scores[0], top_k)
        top_classes = max_classes[0][top_indices]
        top_boxes = bbox_pred_flat[0][top_indices]

        print(f"\nTop {top_k} predictions:")
        for i in range(top_k):
            y_idx = (top_indices[i] // W).item()
            x_idx = (top_indices[i] % W).item()
            print(f"  {i+1}. score={top_scores[i].item():.4f}, class={top_classes[i].item()}, ")
            print(f"      grid_pos=({x_idx}, {y_idx}), box={top_boxes[i].tolist()}")

        # Visualize
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean) * 255
        img_np = img_np.astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Resize to original size
        orig_h, orig_w = target['orig_size'].tolist()
        img_orig = cv2.resize(img_np, (orig_w, orig_h))

        # Draw ground truth (green)
        for box, label in zip(target['boxes'], target['labels']):
            x, y, w, h = box.tolist()
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img_orig, f'GT:{label}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw top predictions (red)
        scale_x = orig_w / W
        scale_y = orig_h / H

        for i in range(min(3, top_k)):
            box = top_boxes[i].tolist()
            score = top_scores[i].item()
            cls = top_classes[i].item() + 1  # Remap

            # Try different box interpretations
            x, y, w, h = box

            # Interpretation 1: Direct scaling
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + w) * scale_x)
            y2 = int((y + h) * scale_y)

            cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_orig, f'P{i+1}:{cls}:{score:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        output_path = f'debug_sample_{idx}.jpg'
        cv2.imwrite(output_path, img_orig)
        print(f"\n✓ Saved visualization to {output_path}")

        break
