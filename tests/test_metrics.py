import torch
import pytest

from utils.metrics import COCOEvaluator


def test_coco_evaluator_perfect_match():
    evaluator = COCOEvaluator(
        pred_format="xyxy",
        target_format="xyxy",
        class_names={0: "fire"},
    )

    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([0]),
        }
    ]

    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            "labels": torch.tensor([0]),
        }
    ]

    evaluator.update(predictions, targets)
    metrics = evaluator.compute()

    assert pytest.approx(metrics["mAP@0.5"], rel=1e-3) == 1.0
    assert pytest.approx(metrics["precision@0.5"], rel=1e-3) == 1.0
    assert pytest.approx(metrics["recall@0.5"], rel=1e-3) == 1.0
    assert pytest.approx(metrics["small_object_recall@0.5"], rel=1e-3) == 1.0
    assert pytest.approx(metrics["mean_iou@0.5"], rel=1e-3) == 1.0


def test_coco_evaluator_partial_detections():
    evaluator = COCOEvaluator(pred_format="xyxy", target_format="xyxy")

    predictions = [
        {
            "boxes": torch.tensor(
                [
                    [0.0, 0.0, 10.0, 10.0],  # good match
                    [50.0, 50.0, 80.0, 80.0],  # no gt
                ],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([0, 0]),
        }
    ]

    targets = [
        {
            "boxes": torch.tensor([[1.0, 1.0, 11.0, 11.0]], dtype=torch.float32),
            "labels": torch.tensor([0]),
        }
    ]

    evaluator.update(predictions, targets)
    metrics = evaluator.compute()

    assert 0.4 <= metrics["precision@0.5"] <= 1.0
    assert 0.4 <= metrics["recall@0.5"] <= 1.0
    assert metrics["small_object_recall@0.5"] <= 1.0
