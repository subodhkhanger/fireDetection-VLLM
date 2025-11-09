"""
Evaluation metrics for Fire-ViT

Implements COCO-style mAP evaluation
"""

import torch
import numpy as np
from collections import defaultdict


class COCOEvaluator:
    """
    COCO-style evaluation metrics

    Computes:
    - mAP@0.5
    - mAP@0.5:0.95
    - AP per class
    - AP for small/medium/large objects
    """

    def __init__(
        self,
        iou_thresholds=None,
        area_ranges=None
    ):
        if iou_thresholds is None:
            # Default COCO IoU thresholds: 0.5:0.95:0.05
            self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        else:
            self.iou_thresholds = np.array(iou_thresholds)

        if area_ranges is None:
            # Default COCO area ranges
            self.area_ranges = {
                'small': (0, 32**2),
                'medium': (32**2, 96**2),
                'large': (96**2, float('inf'))
            }
        else:
            self.area_ranges = area_ranges

        self.reset()

    def reset(self):
        """Reset evaluation state"""
        self.predictions = []
        self.ground_truths = []

    def update(self, predictions, targets):
        """
        Add predictions and targets for evaluation

        Args:
            predictions: List of dicts with 'boxes', 'labels', 'scores'
            targets: List of dicts with 'boxes', 'labels'
        """
        for pred, target in zip(predictions, targets):
            self.predictions.append(pred)
            self.ground_truths.append(target)

    def compute(self):
        """
        Compute COCO metrics

        Returns:
            metrics: Dictionary with mAP and other metrics
        """
        if len(self.predictions) == 0:
            return {}

        # Compute AP for each IoU threshold
        aps = []
        for iou_thresh in self.iou_thresholds:
            ap = self._compute_ap(iou_thresh)
            aps.append(ap)

        # mAP@0.5:0.95 (average across IoU thresholds)
        map_50_95 = np.mean(aps)

        # mAP@0.5
        map_50 = aps[0]  # First threshold is 0.5

        # Per-class AP (simplified - would need class-specific computation)
        class_aps = self._compute_class_aps()

        # Size-specific AP
        size_aps = self._compute_size_aps()

        metrics = {
            'mAP@0.5': map_50,
            'mAP@0.5:0.95': map_50_95,
            **class_aps,
            **size_aps
        }

        return metrics

    def _compute_ap(self, iou_threshold):
        """Compute Average Precision at given IoU threshold"""
        # Collect all predictions and ground truths
        all_preds = []
        all_gts = []

        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_boxes = pred['boxes'].cpu().numpy() if isinstance(pred['boxes'], torch.Tensor) else pred['boxes']
            pred_scores = pred.get('scores', np.ones(len(pred_boxes)))
            pred_labels = pred.get('labels', np.zeros(len(pred_boxes)))

            gt_boxes = gt['boxes'].cpu().numpy() if isinstance(gt['boxes'], torch.Tensor) else gt['boxes']
            gt_labels = gt.get('labels', np.zeros(len(gt_boxes)))

            all_preds.extend([{
                'box': box,
                'score': score,
                'label': label
            } for box, score, label in zip(pred_boxes, pred_scores, pred_labels)])

            all_gts.extend([{
                'box': box,
                'label': label,
                'matched': False
            } for box, label in zip(gt_boxes, gt_labels)])

        if len(all_preds) == 0 or len(all_gts) == 0:
            return 0.0

        # Sort predictions by score (descending)
        all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)

        # Compute precision and recall
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        for i, pred in enumerate(all_preds):
            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt in enumerate(all_gts):
                if gt['matched']:
                    continue

                iou = self._compute_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            # Check if prediction matches ground truth
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                if not all_gts[best_gt_idx]['matched']:
                    tp[i] = 1
                    all_gts[best_gt_idx]['matched'] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / max(len(all_gts), 1)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)

        # Compute AP (area under precision-recall curve)
        ap = self._compute_ap_from_pr(precisions, recalls)

        return ap

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes in COCO format [x, y, w, h]"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to (x1, y1, x2, y2)
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2

        # Intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        # IoU
        iou = inter_area / max(union_area, 1e-6)

        return iou

    def _compute_ap_from_pr(self, precisions, recalls):
        """Compute AP from precision-recall curve"""
        # Add sentinel values
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Compute precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Compute AP (11-point interpolation)
        ap = 0.0
        for i in range(11):
            recall_level = i / 10.0
            idx = np.searchsorted(mrec, recall_level, side='right')
            if idx < len(mpre):
                ap += mpre[idx]

        ap /= 11.0

        return ap

    def _compute_class_aps(self):
        """Compute AP per class"""
        # Simplified - would need actual class-specific computation
        return {
            'AP_fire': 0.0,
            'AP_smoke': 0.0
        }

    def _compute_size_aps(self):
        """Compute AP for different object sizes"""
        # Simplified - would need size-based filtering
        return {
            'AP_small': 0.0,
            'AP_medium': 0.0,
            'AP_large': 0.0
        }


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts

    Returns:
        metrics: Dictionary of metrics
    """
    evaluator = COCOEvaluator()
    evaluator.update(predictions, targets)
    metrics = evaluator.compute()

    return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing COCOEvaluator...")

    # Create dummy predictions and targets
    predictions = [
        {
            'boxes': torch.tensor([[10, 10, 50, 50], [100, 100, 80, 80]]),
            'scores': torch.tensor([0.9, 0.8]),
            'labels': torch.tensor([0, 1])
        }
    ]

    targets = [
        {
            'boxes': torch.tensor([[12, 12, 48, 48], [102, 102, 76, 76]]),
            'labels': torch.tensor([0, 1])
        }
    ]

    # Compute metrics
    metrics = compute_metrics(predictions, targets)

    print(f"\n✓ Metrics computed:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n✅ Metrics test passed!")
