"""
Comprehensive debug script to diagnose evaluation issues

Tests:
1. Dataset loading and label format
2. Model architecture and output format
3. Prediction decoding
4. Ground truth vs prediction matching
"""

import torch
import yaml
import numpy as np
from pathlib import Path

from models.fire_vit import build_fire_vit
from utils.checkpoint import load_checkpoint
from data.fire_dataset import FireDetectionDataset, collate_fn
from data.augmentations import get_val_transforms
from torch.utils.data import DataLoader
from utils.postprocess import decode_predictions


def debug_dataset(config, annotation_file, image_dir=None):
    """Debug dataset loading"""
    print("\n" + "="*80)
    print("1. DATASET LOADING DEBUG")
    print("="*80)

    img_size = config['model']['input_size'][0]

    # If image_dir not specified, try to infer from annotation path
    if image_dir is None:
        ann_path = Path(annotation_file)
        # Check if it's FireTiny format
        if 'FireTiny' in str(ann_path):
            img_dir = ann_path.parent.parent / 'images' / 'val'
        else:
            img_dir = ann_path.parent
    else:
        img_dir = Path(image_dir)

    print(f"Annotation file: {annotation_file}")
    print(f"Image directory: {img_dir}")

    dataset = FireDetectionDataset(
        image_dir=img_dir,
        annotation_file=annotation_file,
        transform=get_val_transforms(img_size),
        mode='val',
        img_size=img_size
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("❌ ERROR: Dataset is empty!")
        return None

    # Test first sample
    image, target = dataset[0]

    print(f"\nFirst sample:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image min/max: {image.min():.3f} / {image.max():.3f}")
    print(f"  Boxes shape: {target['boxes'].shape}")
    print(f"  Boxes: {target['boxes']}")
    print(f"  Labels: {target['labels']}")
    print(f"  Unique labels: {torch.unique(target['labels'])}")
    print(f"  Orig size: {target['orig_size']}")
    print(f"  Image ID: {target['image_id']}")

    # Check multiple samples
    total_boxes = 0
    label_counts = {0: 0, 1: 0}

    for i in range(min(10, len(dataset))):
        _, target = dataset[i]
        total_boxes += len(target['boxes'])
        for label in target['labels']:
            label_val = label.item()
            if label_val in label_counts:
                label_counts[label_val] += 1

    print(f"\nFirst 10 samples stats:")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Fire boxes (label=0): {label_counts[0]}")
    print(f"  Smoke boxes (label=1): {label_counts[1]}")

    if total_boxes == 0:
        print("❌ ERROR: No boxes found in dataset!")
    else:
        print("✓ Dataset loading OK")

    return dataset


def debug_model(config, checkpoint_path, device):
    """Debug model architecture and output"""
    print("\n" + "="*80)
    print("2. MODEL ARCHITECTURE DEBUG")
    print("="*80)

    model = build_fire_vit(config)
    epoch, metrics = load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    model.eval()

    print(f"Model loaded from epoch {epoch}")
    if metrics:
        print(f"Checkpoint metrics: {metrics}")

    # Check model config
    print(f"\nModel config:")
    print(f"  Input size: {config['model']['input_size']}")
    print(f"  Num classes: {config['model']['num_classes']}")
    if 'backbone' in config['model']:
        print(f"  Backbone: {config['model']['backbone']}")

    # Test forward pass
    img_size = config['model']['input_size'][0]
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"\nModel outputs (FPN levels: {len(outputs)}):")
    for i, out in enumerate(outputs):
        print(f"  Level {i}:")
        print(f"    cls_logits shape: {out['cls_logits'].shape}")
        print(f"    bbox_pred shape: {out['bbox_pred'].shape}")
        if 'centerness' in out:
            print(f"    centerness shape: {out['centerness'].shape}")

        # Check output values
        cls_logits = out['cls_logits']
        print(f"    cls_logits min/max: {cls_logits.min():.3f} / {cls_logits.max():.3f}")
        print(f"    cls_logits mean: {cls_logits.mean():.3f}")

        # Check if all zeros or all same value
        if torch.allclose(cls_logits, torch.zeros_like(cls_logits)):
            print(f"    ⚠️  WARNING: cls_logits are all zeros!")
        elif torch.std(cls_logits) < 0.01:
            print(f"    ⚠️  WARNING: cls_logits have very low variance!")

    print("✓ Model forward pass OK")
    return model


def debug_predictions(model, dataset, device, config):
    """Debug prediction decoding"""
    print("\n" + "="*80)
    print("3. PREDICTION DECODING DEBUG")
    print("="*80)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    img_size = config['model']['input_size'][0]
    conf_threshold = 0.001  # Very low to catch any predictions

    print(f"Testing with confidence threshold: {conf_threshold}")

    total_predictions = 0
    images_with_predictions = 0

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            if idx >= 10:  # Test first 10 images
                break

            images = images.to(device)

            # Forward pass
            predictions = model(images)

            # Decode predictions
            decoded = decode_predictions(
                predictions,
                img_size=img_size,
                conf_threshold=conf_threshold,
                nms_threshold=0.5,
                topk=300,
                max_detections=300
            )

            pred = decoded[0]
            num_preds = len(pred['boxes'])
            total_predictions += num_preds

            if num_preds > 0:
                images_with_predictions += 1

            target = targets[0]
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            print(f"\nImage {idx}:")
            print(f"  GT boxes: {len(gt_boxes)}, labels: {gt_labels.tolist() if len(gt_labels) > 0 else []}")
            print(f"  Predictions: {num_preds}")

            if num_preds > 0:
                print(f"  Pred labels: {pred['labels'].tolist()[:5]}")
                print(f"  Pred scores: {pred['scores'].tolist()[:5]}")
                print(f"  Pred boxes (first 3): {pred['boxes'][:3].tolist()}")

            # Check raw model outputs before decoding
            level0 = predictions[0]
            cls_probs = torch.softmax(level0['cls_logits'], dim=1)
            if cls_probs.shape[1] > 1:
                cls_probs_no_bg = cls_probs[:, 1:, :, :]  # Remove background
            else:
                cls_probs_no_bg = cls_probs

            max_prob = cls_probs_no_bg.max().item()
            print(f"  Max class probability (before threshold): {max_prob:.4f}")

            if max_prob < conf_threshold:
                print(f"  ⚠️  All predictions below threshold!")

    print(f"\n" + "-"*80)
    print(f"Summary (first 10 images):")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Images with predictions: {images_with_predictions}/10")

    if total_predictions == 0:
        print("❌ ERROR: No predictions generated at all!")
        print("   Possible causes:")
        print("   - Model not trained properly")
        print("   - All outputs below confidence threshold")
        print("   - Model weights are random/not loaded")
    else:
        print("✓ Predictions are being generated")

    return total_predictions > 0


def debug_matching(model, dataset, device, config):
    """Debug prediction vs ground truth matching"""
    print("\n" + "="*80)
    print("4. PREDICTION-GT MATCHING DEBUG")
    print("="*80)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    img_size = config['model']['input_size'][0]
    conf_threshold = 0.3  # Reasonable threshold

    print(f"Using confidence threshold: {conf_threshold}")

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            if idx >= 3:  # Test first 3 images with predictions
                break

            images = images.to(device)
            predictions = model(images)

            decoded = decode_predictions(
                predictions,
                img_size=img_size,
                conf_threshold=conf_threshold,
                nms_threshold=0.5,
                topk=300,
                max_detections=300
            )

            pred = decoded[0]
            target = targets[0]

            if len(pred['boxes']) == 0:
                continue

            print(f"\nImage {idx} (detailed):")
            print(f"  Original size: {target['orig_size'].tolist()}")

            # Ground truth
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()

            print(f"\n  Ground Truth ({len(gt_boxes)} boxes):")
            for i, (box, label) in enumerate(zip(gt_boxes[:3], gt_labels[:3])):
                label_name = 'fire' if label == 0 else 'smoke'
                print(f"    Box {i}: {box} label={label}({label_name})")

            # Predictions
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()

            print(f"\n  Predictions ({len(pred_boxes)} boxes):")
            for i, (box, label, score) in enumerate(zip(pred_boxes[:3], pred_labels[:3], pred_scores[:3])):
                label_name = 'fire' if label == 0 else 'smoke'
                print(f"    Box {i}: {box} label={label}({label_name}) score={score:.3f}")

            # Check label overlap
            gt_label_set = set(gt_labels.tolist())
            pred_label_set = set(pred_labels.tolist())

            print(f"\n  Label comparison:")
            print(f"    GT labels: {gt_label_set}")
            print(f"    Pred labels: {pred_label_set}")
            print(f"    Overlap: {gt_label_set & pred_label_set}")

            if not (gt_label_set & pred_label_set):
                print(f"    ❌ WARNING: No label overlap!")

            break  # Just show first image with predictions


def main():
    """Run all debug tests"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mac_m1_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--annotation', type=str, required=True)
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Run debug tests
    dataset = debug_dataset(config, args.annotation)
    if dataset is None:
        return

    model = debug_model(config, args.checkpoint, device)

    has_predictions = debug_predictions(model, dataset, device, config)

    if has_predictions:
        debug_matching(model, dataset, device, config)

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
