"""
Post-processing utilities for Fire-ViT detections.

Provides helpers to:
- Decode multi-scale predictions into bounding boxes
- Apply non-maximum suppression
- Prepare detections for evaluation/inference
"""

from typing import List, Sequence, Union

import torch

from models.head.detection_head import (
    generate_anchor_points,
    bbox_pred_to_boxes,
)

DEFAULT_STRIDES: Sequence[int] = (8, 16, 32, 64)


@torch.no_grad()
def decode_predictions(
    predictions: List[dict],
    img_size: Union[Sequence[int], int],
    conf_threshold: float = 0.4,
    nms_threshold: float = 0.5,
    topk: Union[int, None] = 200,  # Reduced from 300 to limit candidates
    max_detections: Union[int, None] = 100,  # Reduced from 300 for cleaner output
    strides: Union[Sequence[int], None] = None,
) -> List[dict]:
    """
    Convert raw head outputs into final detections.

    Args:
        predictions: List of pyramid-level outputs from Fire-ViT head.
        img_size: Target input resolution (int or (H, W)).
        conf_threshold: Minimum confidence for keeping a point.
        nms_threshold: IoU threshold for NMS.
        topk: Optional per-level cap before NMS.
        max_detections: Optional cap after NMS.
        strides: Optional list of strides per pyramid level.

    Returns:
        List of detections per image with keys: boxes, scores, labels.
    """
    if isinstance(img_size, (list, tuple)):
        img_h, img_w = img_size[:2]
    else:
        img_h = img_w = int(img_size)

    if strides is None:
        strides = DEFAULT_STRIDES[: len(predictions)]

    batch_size = predictions[0]["cls_logits"].shape[0]
    anchor_cache: List[torch.Tensor] = []

    for level_idx, pred in enumerate(predictions):
        device = pred["cls_logits"].device
        _, _, H, W = pred["cls_logits"].shape
        anchor_points = generate_anchor_points(
            (H, W),
            stride=strides[level_idx],
            device=device,
        )
        anchor_cache.append(anchor_points)

    batch_results: List[dict] = []

    for batch_idx in range(batch_size):
        boxes_all: List[torch.Tensor] = []
        scores_all: List[torch.Tensor] = []
        labels_all: List[torch.Tensor] = []

        for level_idx, pred in enumerate(predictions):
            cls_logits = pred["cls_logits"][batch_idx : batch_idx + 1]
            bbox_pred = pred["bbox_pred"][batch_idx : batch_idx + 1]
            centerness = pred.get("centerness")

            if centerness is None:
                centerness = torch.zeros(
                    (1, 1, cls_logits.shape[2], cls_logits.shape[3]),
                    device=cls_logits.device,
                )
            else:
                centerness = centerness[batch_idx : batch_idx + 1]

            cls_probs = torch.softmax(cls_logits, dim=1)
            if cls_probs.shape[1] > 1:
                cls_probs = cls_probs[:, 1:, :, :]
            cls_probs = cls_probs.permute(0, 2, 3, 1).reshape(1, -1, cls_probs.shape[1])

            centerness_probs = torch.sigmoid(centerness)
            centerness_probs = centerness_probs.permute(0, 2, 3, 1).reshape(1, -1, 1)

            scores = cls_probs * centerness_probs  # [1, HW, C]

            # CRITICAL: Pass stride for denormalization (targets are normalized by stride)
            boxes = bbox_pred_to_boxes(bbox_pred, anchor_cache[level_idx], stride=strides[level_idx])[0]
            boxes[:, 0].clamp_(0, img_w)
            boxes[:, 2].clamp_(0, img_w)
            boxes[:, 1].clamp_(0, img_h)
            boxes[:, 3].clamp_(0, img_h)

            max_scores, labels = scores[0].max(dim=1)
            keep = max_scores >= conf_threshold
            if keep.sum() == 0:
                continue

            boxes = boxes[keep]
            labels = labels[keep]
            max_scores = max_scores[keep]

            if topk is not None and boxes.shape[0] > topk:
                top_scores, top_idx = torch.topk(max_scores, topk)
                boxes = boxes[top_idx]
                labels = labels[top_idx]
                max_scores = top_scores

            boxes_all.append(boxes)
            scores_all.append(max_scores)
            labels_all.append(labels)

        if boxes_all:
            boxes = torch.cat(boxes_all, dim=0)
            scores = torch.cat(scores_all, dim=0)
            labels = torch.cat(labels_all, dim=0)
            keep_idx = batched_nms(
                boxes,
                scores,
                labels,
                iou_threshold=nms_threshold,
                max_detections=max_detections,
            )
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]
        else:
            device = predictions[0]["cls_logits"].device
            boxes = torch.zeros((0, 4), device=device)
            scores = torch.zeros((0,), device=device)
            labels = torch.zeros((0,), dtype=torch.long, device=device)

        batch_results.append(
            {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        )

    return batch_results


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    max_detections: Union[int, None] = None,
) -> torch.Tensor:
    """
    Apply per-class NMS and return indices to keep.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    keep_indices: List[torch.Tensor] = []
    order_template = torch.arange(boxes.shape[0], device=boxes.device)

    for cls in labels.unique():
        cls_mask = labels == cls
        if cls_mask.sum() == 0:
            continue

        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = order_template[cls_mask]

        cls_keep = _nms_single(cls_boxes, cls_scores, iou_threshold)
        keep_indices.append(cls_indices[cls_keep])

    if not keep_indices:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    keep_indices = torch.cat(keep_indices)
    keep_scores = scores[keep_indices]
    sorted_scores, order = torch.sort(keep_scores, descending=True)
    keep_indices = keep_indices[order]

    if max_detections is not None and keep_indices.numel() > max_detections:
        keep_indices = keep_indices[:max_detections]

    return keep_indices


def _nms_single(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Simple NMS for a single class.
    """
    keep: List[int] = []
    order = scores.argsort(descending=True)

    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break

        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        remaining = ious <= iou_threshold
        order = order[1:][remaining]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU matrix between two sets of boxes in xyxy format.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            device=boxes1.device,
        )

    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = torch.max(x11[:, None], x21[None, :])
    inter_y1 = torch.max(y11[:, None], y21[None, :])
    inter_x2 = torch.min(x12[:, None], x22[None, :])
    inter_y2 = torch.min(y12[:, None], y22[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter_area
    union = union.clamp(min=1e-6)

    return inter_area / union


__all__ = ["decode_predictions", "batched_nms", "box_iou", "DEFAULT_STRIDES"]
