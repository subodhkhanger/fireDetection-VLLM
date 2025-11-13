"""
FCOS-style target encoder for Fire-ViT.

Converts per-image annotations into dense per-level tensors aligned with the
anchor-free detection head (classification labels, l/t/r/b offsets, centerness).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from models.head.detection_head import generate_anchor_points


class FCOSTargetEncoder:
    """
    Builds training targets for each FPN level.

    Args:
        num_classes: Number of foreground classes + background.
        strides: Spatial stride per FPN level.
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int] | None = None,
    ):
        self.num_classes = num_classes
        self.strides = strides or (8, 16, 32, 64)
        self.eps = 1e-6

    @torch.no_grad()
    def encode(
        self,
        raw_targets: List[Dict],
        predictions: List[Dict],
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            raw_targets: List (len = batch) of dicts from dataset.
            predictions: Model outputs for each FPN level.

        Returns:
            Dict with per-level tensors (cls_targets, box_targets, centerness,
            positive mask, and anchor points).
        """
        device = predictions[0]["cls_logits"].device

        encoded = {
            "cls_targets": [],
            "box_targets": [],
            "centerness": [],
            "pos_mask": [],
            "anchor_points": [],
        }

        for level_idx, pred in enumerate(predictions):
            stride = self.strides[level_idx] if level_idx < len(self.strides) else self.strides[-1]
            B, _, H, W = pred["cls_logits"].shape

            cls_level = torch.zeros((B, H, W), dtype=torch.long, device=device)
            box_level = torch.zeros((B, 4, H, W), dtype=torch.float32, device=device)
            centerness_level = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
            pos_mask_level = torch.zeros((B, H, W), dtype=torch.bool, device=device)

            anchor_points = generate_anchor_points((H, W), stride=stride, device=device)
            anchor_points_hw = anchor_points.view(H, W, 2)

            points = anchor_points  # [H*W, 2]
            num_points = points.shape[0]

            for b_idx, target in enumerate(raw_targets):
                boxes = target["boxes"].to(device)
                labels = target["labels"].to(device)

                if boxes.numel() == 0:
                    continue

                boxes_xyxy = self._coco_to_xyxy(boxes)
                areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
                # Foreground labels start at 1, background=0
                cls_ids = torch.clamp(labels + 1, max=self.num_classes - 1)

                assigned_area = torch.full((num_points,), float("inf"), device=device)
                cls_assign = torch.zeros((num_points,), dtype=torch.long, device=device)
                box_assign = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
                centerness_assign = torch.zeros((num_points,), dtype=torch.float32, device=device)

                for gt_idx in range(boxes_xyxy.shape[0]):
                    box = boxes_xyxy[gt_idx]
                    area = areas[gt_idx]
                    cls_target = cls_ids[gt_idx]

                    l = points[:, 0] - box[0]
                    t = points[:, 1] - box[1]
                    r = box[2] - points[:, 0]
                    b = box[3] - points[:, 1]
                    ltrb = torch.stack([l, t, r, b], dim=1)

                    inside = (ltrb.min(dim=1).values > 0)
                    if not inside.any():
                        continue

                    update_mask = inside & (area < assigned_area)
                    if not update_mask.any():
                        continue

                    assigned_area[update_mask] = area
                    cls_assign[update_mask] = cls_target
                    box_assign[update_mask] = ltrb[update_mask]
                    centerness_assign[update_mask] = self._compute_centerness(ltrb[update_mask])

                cls_map = cls_assign.view(H, W)
                box_map = box_assign.view(H, W, 4).permute(2, 0, 1)
                centerness_map = centerness_assign.view(H, W)
                pos_mask = cls_map > 0

                cls_level[b_idx] = cls_map
                box_level[b_idx] = box_map
                centerness_level[b_idx, 0] = centerness_map
                pos_mask_level[b_idx] = pos_mask

            encoded["cls_targets"].append(cls_level)
            encoded["box_targets"].append(box_level)
            encoded["centerness"].append(centerness_level)
            encoded["pos_mask"].append(pos_mask_level)
            encoded["anchor_points"].append(anchor_points_hw)  # shared across batch

        if raw_targets and "masks" in raw_targets[0]:
            attention_masks = torch.stack([
                t["masks"].to(device).float() for t in raw_targets
            ])
        else:
            attention_masks = None

        encoded["attention_masks"] = attention_masks

        return encoded

    def _compute_centerness(self, ltrb: torch.Tensor) -> torch.Tensor:
        left_right = ltrb[:, [0, 2]]
        top_bottom = ltrb[:, [1, 3]]

        lr_min = left_right.min(dim=1).values
        lr_max = left_right.max(dim=1).values.clamp(min=self.eps)
        tb_min = top_bottom.min(dim=1).values
        tb_max = top_bottom.max(dim=1).values.clamp(min=self.eps)

        centerness = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))
        return centerness

    def _coco_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)
