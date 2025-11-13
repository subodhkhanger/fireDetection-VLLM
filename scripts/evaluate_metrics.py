#!/usr/bin/env python
"""
Evaluate a Fire-ViT checkpoint on a dataset and report detection metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fire_dataset import FireDetectionDataset, collate_fn, _resolve_split_dir
from data import get_val_transforms
from models.fire_vit import build_fire_vit
from utils.checkpoint import load_checkpoint
from utils.metrics import COCOEvaluator
from utils.postprocess import decode_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fire-ViT checkpoint")

    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to model config")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing split folders")
    parser.add_argument("--ann", type=str, required=True,
                        help="COCO annotation file to evaluate against")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Split name to resolve image directory")

    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to run evaluation on")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers")

    parser.add_argument("--conf-threshold", type=float, default=0.4,
                        help="Confidence threshold before NMS")
    parser.add_argument("--nms-threshold", type=float, default=0.5,
                        help="IoU threshold for NMS")
    parser.add_argument("--max-detections", type=int, default=300,
                        help="Maximum detections per image")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optionally limit number of samples for quick tests")

    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON file to store metrics")

    return parser.parse_args()


def build_dataloader(args, img_size: int):
    split_dir = _resolve_split_dir(args.data_dir, args.ann, args.split)
    transforms = get_val_transforms(img_size)

    dataset = FireDetectionDataset(
        image_dir=split_dir,
        annotation_file=args.ann,
        transform=transforms,
        mode=args.split,
        img_size=img_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    return loader, dataset.categories


def evaluate_checkpoint():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    img_size = config["model"]["input_size"][0]

    dataloader, categories = build_dataloader(args, img_size)

    model = build_fire_vit(config)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)
    model.eval()

    class_names = {cat_id: name for cat_id, name in categories.items()} if isinstance(categories, dict) else None
    evaluator = COCOEvaluator(
        pred_format="xyxy",
        target_format="coco",
        class_names=class_names,
    )

    seen = 0
    progress = tqdm(dataloader, desc="Evaluating", total=len(dataloader))

    with torch.no_grad():
        for images, targets in progress:
            images = images.to(device)
            outputs = model(images)

            decoded = decode_predictions(
                outputs,
                img_size=config["model"]["input_size"],
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold,
                max_detections=args.max_detections,
            )

            predictions_batch = []
            targets_batch = []

            for idx, target in enumerate(targets):
                predictions_batch.append(
                    {
                        "boxes": decoded[idx]["boxes"].detach().cpu(),
                        "scores": decoded[idx]["scores"].detach().cpu(),
                        "labels": decoded[idx]["labels"].detach().cpu(),
                    }
                )
                targets_batch.append(
                    {
                        "boxes": target["boxes"].cpu(),
                        "labels": target["labels"].cpu(),
                    }
                )

            evaluator.update(predictions_batch, targets_batch)
            seen += len(targets)

            if args.max_samples and seen >= args.max_samples:
                break

    metrics = evaluator.compute()
    if not metrics:
        print("No predictions to evaluate. Did the model produce any detections?")
        return

    print("\n========== Evaluation Metrics ==========")
    for key, value in metrics.items():
        print(f"{key:>30}: {value:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Saved metrics to {output_path}")


if __name__ == "__main__":
    evaluate_checkpoint()
