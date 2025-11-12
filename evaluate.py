"""
Evaluation script for Fire-ViT on test dataset

Computes mAP, precision, recall, F1-score and other metrics
"""

import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

from models.fire_vit import build_fire_vit
from utils.checkpoint import load_checkpoint
from data.fire_dataset import FireDetectionDataset, collate_fn
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fire-ViT Evaluation')

    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-annotation', type=str, default=None,
                       help='Path to test annotations (overrides config)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for mAP calculation')
    parser.add_argument('--save-results', action='store_true',
                       help='Save detailed results to JSON')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')

    return parser.parse_args()


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in [x, y, w, h] format

    Args:
        box1: [x, y, w, h]
        box2: [x, y, w, h]

    Returns:
        iou: Intersection over Union
    """
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # Union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def match_predictions_to_ground_truth(pred_boxes, pred_labels, pred_scores,
                                     gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes using IoU

    Returns:
        matches: List of (pred_idx, gt_idx, iou) tuples
        unmatched_preds: List of unmatched prediction indices
        unmatched_gts: List of unmatched ground truth indices
    """
    matches = []
    matched_gt = set()
    matched_pred = set()

    # Sort predictions by score (highest first)
    if len(pred_scores) > 0:
        sorted_indices = np.argsort(-pred_scores)
    else:
        sorted_indices = []

    # Match predictions to ground truth
    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_label = pred_labels[pred_idx]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_idx in matched_gt:
                continue

            # Only match same class
            if pred_label != gt_label:
                continue

            iou = compute_iou(pred_box, gt_box)

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx, best_iou))
            matched_pred.add(pred_idx)
            matched_gt.add(best_gt_idx)

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gt]

    return matches, unmatched_preds, unmatched_gts


def compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation"""
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0

    return ap


def extract_predictions_from_model_output(predictions, conf_threshold, img_size, orig_size):
    """
    Extract bounding boxes from model predictions

    Args:
        predictions: List of prediction dicts from model
        conf_threshold: Confidence threshold
        img_size: Model input size (H, W)
        orig_size: Original image size (H, W)

    Returns:
        boxes: [N, 4] in [x, y, w, h] format (original image coords)
        labels: [N]
        scores: [N]
    """
    # Use finest pyramid level
    pred = predictions[0]

    cls_logits = pred['cls_logits']  # [B, C, H, W]
    bbox_pred = pred['bbox_pred']  # [B, 4, H, W]
    centerness = pred.get('centerness', None)

    B, C, H, W = cls_logits.shape

    # Apply sigmoid
    cls_probs = torch.sigmoid(cls_logits)

    if centerness is not None:
        obj_scores = torch.sigmoid(centerness)
    else:
        obj_scores = torch.ones(B, 1, H, W, device=cls_logits.device)

    # Flatten
    cls_probs = cls_probs.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
    obj_scores = obj_scores.reshape(B, 1, -1).permute(0, 2, 1)  # [B, H*W, 1]
    bbox_pred = bbox_pred.reshape(B, 4, -1).permute(0, 2, 1)  # [B, H*W, 4]

    # Get scores
    scores = cls_probs * obj_scores  # [B, H*W, C]
    max_scores, max_classes = scores.max(dim=2)

    # Filter by confidence
    mask = max_scores[0] > conf_threshold
    filtered_scores = max_scores[0][mask].cpu().numpy()
    filtered_classes = max_classes[0][mask].cpu().numpy()
    filtered_boxes = bbox_pred[0][mask].cpu().numpy()

    # Scale to original image size
    orig_h, orig_w = orig_size
    scale_x = orig_w / img_size[1]
    scale_y = orig_h / img_size[0]

    if len(filtered_boxes) > 0:
        # Assuming boxes are in format relative to feature map
        # This is a simplified version - may need adjustment based on actual output format
        scaled_boxes = filtered_boxes.copy()
        scaled_boxes[:, 0] *= scale_x
        scaled_boxes[:, 1] *= scale_y
        scaled_boxes[:, 2] *= scale_x
        scaled_boxes[:, 3] *= scale_y

        # Clip to image bounds
        scaled_boxes[:, 0] = np.clip(scaled_boxes[:, 0], 0, orig_w)
        scaled_boxes[:, 1] = np.clip(scaled_boxes[:, 1], 0, orig_h)
        scaled_boxes[:, 2] = np.clip(scaled_boxes[:, 2], 0, orig_w - scaled_boxes[:, 0])
        scaled_boxes[:, 3] = np.clip(scaled_boxes[:, 3], 0, orig_h - scaled_boxes[:, 1])
    else:
        scaled_boxes = np.zeros((0, 4))

    return scaled_boxes, filtered_classes, filtered_scores


def evaluate_model(model, dataloader, device, conf_threshold, iou_threshold, img_size):
    """
    Evaluate model on test dataset

    Returns:
        metrics: Dictionary containing evaluation metrics
        all_results: List of per-image results
    """
    model.eval()

    all_predictions = defaultdict(list)  # class_id -> list of (score, matched)
    all_ground_truths = defaultdict(int)  # class_id -> count

    all_results = []

    print("\nRunning evaluation...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)

            # Forward pass
            predictions = model(images)

            # Process each image in batch
            batch_size = images.size(0)
            for i in range(batch_size):
                target = targets[i]

                # Get ground truth
                gt_boxes = target['boxes'].cpu().numpy()  # [N, 4] in COCO format
                gt_labels = target['labels'].cpu().numpy()  # [N]
                orig_size = target['orig_size'].cpu().numpy()  # [H, W]

                # Extract predictions
                pred_boxes, pred_labels, pred_scores = extract_predictions_from_model_output(
                    predictions, conf_threshold, (img_size, img_size), tuple(orig_size)
                )

                # Match predictions to ground truth
                matches, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
                    pred_boxes, pred_labels, pred_scores,
                    gt_boxes, gt_labels, iou_threshold
                )

                # Record per-class statistics
                for label in gt_labels:
                    all_ground_truths[label] += 1

                for pred_idx, score in enumerate(pred_scores):
                    label = pred_labels[pred_idx]
                    matched = pred_idx in [m[0] for m in matches]
                    all_predictions[label].append((score, matched))

                # Store per-image results
                result = {
                    'image_id': target['image_id'].item(),
                    'num_gt': len(gt_boxes),
                    'num_pred': len(pred_boxes),
                    'num_matches': len(matches),
                    'num_false_positives': len(unmatched_preds),
                    'num_false_negatives': len(unmatched_gts),
                }
                all_results.append(result)

    # Compute metrics per class
    print("\nComputing metrics...")

    class_metrics = {}

    for class_id in sorted(all_ground_truths.keys()):
        class_name = f"class_{class_id}"

        if class_id not in all_predictions or len(all_predictions[class_id]) == 0:
            # No predictions for this class
            class_metrics[class_name] = {
                'ap': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_gt': all_ground_truths[class_id],
                'num_pred': 0
            }
            continue

        # Sort predictions by score
        predictions = sorted(all_predictions[class_id], key=lambda x: x[0], reverse=True)

        # Compute precision-recall curve
        tp = 0
        fp = 0
        precisions = []
        recalls = []

        for score, matched in predictions:
            if matched:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / all_ground_truths[class_id] if all_ground_truths[class_id] > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        # Compute AP
        ap = compute_ap(recalls, precisions)

        # Best F1 score
        f1_scores = []
        for p, r in zip(precisions, recalls):
            if p + r > 0:
                f1 = 2 * (p * r) / (p + r)
            else:
                f1 = 0.0
            f1_scores.append(f1)

        best_f1 = max(f1_scores) if f1_scores else 0.0
        best_f1_idx = np.argmax(f1_scores) if f1_scores else 0

        class_metrics[class_name] = {
            'ap': ap,
            'precision': precisions[best_f1_idx] if precisions else 0.0,
            'recall': recalls[best_f1_idx] if recalls else 0.0,
            'f1': best_f1,
            'num_gt': all_ground_truths[class_id],
            'num_pred': len(predictions)
        }

    # Compute overall metrics
    if len(class_metrics) > 0:
        mAP = np.mean([m['ap'] for m in class_metrics.values()])
        avg_precision = np.mean([m['precision'] for m in class_metrics.values()])
        avg_recall = np.mean([m['recall'] for m in class_metrics.values()])
        avg_f1 = np.mean([m['f1'] for m in class_metrics.values()])
    else:
        mAP = avg_precision = avg_recall = avg_f1 = 0.0

    metrics = {
        'mAP': mAP,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'per_class': class_metrics,
        'total_images': len(all_results),
        'total_gt_boxes': sum(r['num_gt'] for r in all_results),
        'total_pred_boxes': sum(r['num_pred'] for r in all_results),
        'total_matches': sum(r['num_matches'] for r in all_results),
    }

    return metrics, all_results


def main():
    """Main evaluation function"""
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    print("\nBuilding model...")
    model = build_fire_vit(config)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    epoch, ckpt_metrics = load_checkpoint(args.checkpoint, model, device=device)
    print(f"Checkpoint from epoch {epoch}")
    if ckpt_metrics:
        print(f"Checkpoint metrics: {ckpt_metrics}")

    model.to(device)
    model.eval()

    # Setup test dataset
    test_annotation = args.test_annotation or config['data']['test_annotation']
    test_img_dir = Path(test_annotation).parent

    print(f"\nLoading test dataset from: {test_annotation}")

    from data.augmentations import get_val_transforms
    img_size = config['model']['input_size'][0]

    test_dataset = FireDetectionDataset(
        image_dir=test_img_dir,
        annotation_file=test_annotation,
        transform=get_val_transforms(img_size),
        mode='test',
        img_size=img_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Evaluate
    metrics, results = evaluate_model(
        model, test_loader, device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        img_size=img_size
    )

    # Print results
    print("\n" + "="*70)
    print(" "*25 + "EVALUATION RESULTS")
    print("="*70)
    print(f"\nOverall Metrics (IoU threshold: {args.iou_threshold}):")
    print(f"  mAP@{args.iou_threshold:.2f}:  {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")

    print(f"\nDataset Statistics:")
    print(f"  Total images:      {metrics['total_images']}")
    print(f"  Total GT boxes:    {metrics['total_gt_boxes']}")
    print(f"  Total predictions: {metrics['total_pred_boxes']}")
    print(f"  Total matches:     {metrics['total_matches']}")

    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"\n  {class_name}:")
        print(f"    AP:        {class_metrics['ap']:.4f}")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1']:.4f}")
        print(f"    GT boxes:  {class_metrics['num_gt']}")
        print(f"    Pred boxes: {class_metrics['num_pred']}")

    print("\n" + "="*70)

    # Save results
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'config': {
                    'checkpoint': args.checkpoint,
                    'conf_threshold': args.conf_threshold,
                    'iou_threshold': args.iou_threshold,
                },
                'per_image_results': results
            }, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")


if __name__ == '__main__':
    main()
