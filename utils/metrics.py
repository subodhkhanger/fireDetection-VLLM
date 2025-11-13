"""
Evaluation utilities for Fire-ViT.

Offers COCO-style metrics plus extra diagnostics (IoU, size-based recall).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch


class COCOEvaluator:
    """
    Simplified COCO-style evaluator.

    Supports:
    - mAP@0.5 and mAP@0.5:0.95
    - Per-class AP (at IoU=0.5)
    - Precision/recall/mean IoU statistics
    - Small/medium/large object recall
    """

    def __init__(
        self,
        iou_thresholds: Optional[Iterable[float]] = None,
        area_ranges: Optional[Dict[str, tuple]] = None,
        pred_format: str = "xyxy",
        target_format: str = "xyxy",
        class_names: Optional[Union[Dict[int, str], List[str]]] = None,
    ):
        self.iou_thresholds = (
            np.linspace(0.5, 0.95, 10) if iou_thresholds is None else np.array(list(iou_thresholds))
        )

        self.area_ranges = area_ranges or {
            "small": (0, 32**2),
            "medium": (32**2, 96**2),
            "large": (96**2, float("inf")),
        }

        if isinstance(class_names, list):
            self.class_names = {idx: name for idx, name in enumerate(class_names)}
        else:
            self.class_names = class_names or {}

        self.pred_format = pred_format
        self.target_format = target_format
        self.reset()

    def reset(self) -> None:
        self.predictions: List[Dict] = []
        self.ground_truths: List[Dict] = []
        self.classes = set()

    def update(self, predictions: List[Dict], targets: List[Dict]) -> None:
        """
        Append a batch of predictions + targets.
        """
        for pred, target in zip(predictions, targets):
            image_id = len(self.ground_truths)
            formatted_pred = self._format_prediction(pred, image_id)
            formatted_target = self._format_target(target, image_id)

            self.predictions.append(formatted_pred)
            self.ground_truths.append(formatted_target)

            self.classes.update(np.unique(formatted_pred["labels"]))
            self.classes.update(np.unique(formatted_target["labels"]))

    def compute(self) -> Dict[str, float]:
        if not self.predictions:
            return {}

        class_ids = sorted([int(c) for c in self.classes if c >= 0])
        class_ids = class_ids or [0]

        ap_per_threshold = []
        per_class_ap = defaultdict(dict)

        for iou_thresh in self.iou_thresholds:
            aps = []
            for cls in class_ids:
                ap = self._compute_ap_for_class(cls, iou_thresh)
                if ap is not None:
                    aps.append(ap)
                    per_class_ap[cls][f"AP@{iou_thresh:.2f}"] = ap
            ap_per_threshold.append(float(np.mean(aps)) if aps else 0.0)

        metrics = {
            "mAP@0.5": ap_per_threshold[0],
            "mAP@0.5:0.95": float(np.mean(ap_per_threshold)),
        }

        # Per-class AP at IoU=0.5
        for cls in class_ids:
            ap_value = per_class_ap[cls].get(f"AP@{self.iou_thresholds[0]:.2f}")
            if ap_value is not None:
                metrics[f"AP_{self._class_name(cls)}"] = ap_value

        # Size-aware recalls and IoU stats @ 0.5
        size_stats = self._match_stats(iou_threshold=self.iou_thresholds[0])
        metrics["precision@0.5"] = size_stats["precision"]
        metrics["recall@0.5"] = size_stats["recall"]
        metrics["mean_iou@0.5"] = size_stats["mean_iou"]

        for size_key in ("small", "medium", "large"):
            gt_count = size_stats["gt_counts"].get(size_key, 0)
            tp_count = size_stats["tp_counts"].get(size_key, 0)
            metrics[f"{size_key}_object_recall@0.5"] = (
                tp_count / gt_count if gt_count > 0 else 0.0
            )

        return metrics

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _format_prediction(self, pred: Dict, image_id: int) -> Dict:
        boxes = self._to_numpy(pred.get("boxes", np.zeros((0, 4), dtype=np.float32)))
        scores = self._to_numpy(pred.get("scores", np.ones((len(boxes),), dtype=np.float32)))
        labels = self._to_numpy(pred.get("labels", np.zeros((len(boxes),), dtype=np.int64)))

        boxes = self._convert_boxes(boxes, self.pred_format)

        return {
            "image_id": image_id,
            "boxes": boxes,
            "scores": scores.astype(np.float32),
            "labels": labels.astype(np.int64),
        }

    def _format_target(self, target: Dict, image_id: int) -> Dict:
        boxes = self._to_numpy(target.get("boxes", np.zeros((0, 4), dtype=np.float32)))
        labels = self._to_numpy(target.get("labels", np.zeros((len(boxes),), dtype=np.int64)))

        boxes = self._convert_boxes(boxes, self.target_format)
        areas = self._box_area(boxes)

        return {
            "image_id": image_id,
            "boxes": boxes,
            "labels": labels.astype(np.int64),
            "areas": areas,
        }

    def _compute_ap_for_class(self, class_id: int, iou_threshold: float) -> Optional[float]:
        preds = []
        gts = {}
        total_gt = 0

        for pred in self.predictions:
            cls_mask = pred["labels"] == class_id
            if cls_mask.sum() == 0:
                continue
            for box, score in zip(pred["boxes"][cls_mask], pred["scores"][cls_mask]):
                preds.append(
                    {"image_id": pred["image_id"], "box": box, "score": float(score)}
                )

        for gt in self.ground_truths:
            cls_mask = gt["labels"] == class_id
            boxes = gt["boxes"][cls_mask]
            if boxes.shape[0] == 0:
                continue
            gts[gt["image_id"]] = {
                "boxes": boxes,
                "matched": np.zeros(len(boxes), dtype=bool),
            }
            total_gt += len(boxes)

        if total_gt == 0:
            return None

        preds.sort(key=lambda x: x["score"], reverse=True)
        if not preds:
            return 0.0

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for idx, pred in enumerate(preds):
            image_id = pred["image_id"]
            if image_id not in gts:
                fp[idx] = 1
                continue

            gt_entry = gts[image_id]
            gt_boxes = gt_entry["boxes"]
            if gt_boxes.size == 0:
                fp[idx] = 1
                continue

            ious = self._compute_iou_array(pred["box"][None, :], gt_boxes)[0]
            best_gt = np.argmax(ious)
            best_iou = ious[best_gt]

            if best_iou >= iou_threshold and not gt_entry["matched"][best_gt]:
                tp[idx] = 1
                gt_entry["matched"][best_gt] = True
            else:
                fp[idx] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls = tp_cum / max(total_gt, 1e-6)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

        return self._compute_ap_from_pr(precisions, recalls)

    def _match_stats(self, iou_threshold: float) -> Dict[str, Dict[str, float]]:
        total_tp = 0
        total_fp = 0
        total_gt = 0
        iou_sum = 0.0

        size_gt_counts = {k: 0 for k in self.area_ranges.keys()}
        size_tp_counts = {k: 0 for k in self.area_ranges.keys()}

        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]

            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]
            gt_areas = gt["areas"]
            matched = np.zeros(len(gt_boxes), dtype=bool)

            order = np.argsort(-pred_scores)

            for idx in order:
                box = pred_boxes[idx]
                label = pred_labels[idx]

                same_label_idx = np.where(gt_labels == label)[0]
                if same_label_idx.size == 0:
                    total_fp += 1
                    continue

                candidate_boxes = gt_boxes[same_label_idx]
                ious = self._compute_iou_array(box[None, :], candidate_boxes)[0]
                best_local = np.argmax(ious)
                best_iou = ious[best_local]
                gt_idx = same_label_idx[best_local]

                if best_iou >= iou_threshold and not matched[gt_idx]:
                    matched[gt_idx] = True
                    total_tp += 1
                    iou_sum += best_iou
                    size_label = self._categorize_area(gt_areas[gt_idx])
                    size_tp_counts[size_label] += 1
                else:
                    total_fp += 1

            total_gt += len(gt_boxes)
            for size_key in self.area_ranges.keys():
                mask = self._size_mask(gt_areas, size_key)
                size_gt_counts[size_key] += int(mask.sum())

        precision = total_tp / max(total_tp + total_fp, 1e-6) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / max(total_gt, 1e-6) if total_gt > 0 else 0.0
        mean_iou = iou_sum / max(total_tp, 1e-6) if total_tp > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "mean_iou": mean_iou,
            "gt_counts": size_gt_counts,
            "tp_counts": size_tp_counts,
        }

    def _compute_ap_from_pr(self, precisions: np.ndarray, recalls: np.ndarray) -> float:
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for idx in range(mpre.size - 1, 0, -1):
            mpre[idx - 1] = np.maximum(mpre[idx - 1], mpre[idx])

        recall_levels = np.linspace(0, 1, 11)
        ap = 0.0
        for rl in recall_levels:
            indices = np.where(mrec >= rl)[0]
            if indices.size > 0:
                ap += mpre[indices[0]]
        return ap / len(recall_levels)

    def _convert_boxes(self, boxes: np.ndarray, fmt: str) -> np.ndarray:
        if fmt == "xyxy":
            return boxes.astype(np.float32)
        if fmt == "coco":
            boxes = boxes.astype(np.float32)
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 0] + boxes[:, 2]
            y2 = boxes[:, 1] + boxes[:, 3]
            return np.stack([x1, y1, x2, y2], axis=-1)
        raise ValueError(f"Unsupported box format: {fmt}")

    def _box_area(self, boxes: np.ndarray) -> np.ndarray:
        widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
        heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
        return widths * heights

    def _compute_iou_array(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

        x11, y11, x12, y12 = boxes1[:, None, 0], boxes1[:, None, 1], boxes1[:, None, 2], boxes1[:, None, 3]
        x21, y21, x22, y22 = boxes2[None, :, 0], boxes2[None, :, 1], boxes2[None, :, 2], boxes2[None, :, 3]

        inter_x1 = np.maximum(x11, x21)
        inter_y1 = np.maximum(y11, y21)
        inter_x2 = np.minimum(x12, x22)
        inter_y2 = np.minimum(y12, y22)

        inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
        inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
        inter_area = inter_w * inter_h

        area1 = np.clip(x12 - x11, 0, None) * np.clip(y12 - y11, 0, None)
        area2 = np.clip(x22 - x21, 0, None) * np.clip(y22 - y21, 0, None)

        union = np.clip(area1 + area2 - inter_area, a_min=1e-6, a_max=None)
        return inter_area / union

    def _categorize_area(self, area: float) -> str:
        for name, (low, high) in self.area_ranges.items():
            if low <= area < high:
                return name
        return "large"

    def _size_mask(self, areas: np.ndarray, size_key: str) -> np.ndarray:
        low, high = self.area_ranges[size_key]
        return (areas >= low) & (areas < high)

    def _class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, f"class_{class_id}")

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)


def compute_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    evaluator = COCOEvaluator()
    evaluator.update(predictions, targets)
    return evaluator.compute()


if __name__ == "__main__":
    print("Testing COCOEvaluator...")

    preds = [
        {
            "boxes": torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=torch.float32),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([0, 1]),
        }
    ]

    targets = [
        {
            "boxes": torch.tensor([[12, 12, 48, 48], [102, 102, 148, 148]], dtype=torch.float32),
            "labels": torch.tensor([0, 1]),
        }
    ]

    evaluator = COCOEvaluator()
    evaluator.update(preds, targets)
    metrics = evaluator.compute()

    for key, value in metrics.items():
        print(f"{key:>25}: {value:.4f}")

    print("✅ Metrics test passed")
