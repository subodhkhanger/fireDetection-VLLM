#!/usr/bin/env python3
"""
Convert Kaggle fire detection datasets to COCO format

Usage:
    python scripts/convert_to_coco.py \
        --input-dir data/FireKaggle \
        --output-dir data/FireKaggle_COCO \
        --train-ratio 0.7 \
        --val-ratio 0.15
"""

import os
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_coco_dataset(image_files, output_dir, split_name):
    """Create COCO format annotations for a list of images"""

    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 0, 'name': 'fire'},
            {'id': 1, 'name': 'smoke'}
        ]
    }

    ann_id = 0

    for img_id, img_file in enumerate(tqdm(image_files, desc=f"Processing {split_name}")):
        try:
            # Read image to get dimensions
            img = Image.open(img_file)
            width, height = img.size

            # Copy image to output directory
            output_img_dir = output_dir / 'images' / split_name
            output_img_dir.mkdir(parents=True, exist_ok=True)

            img_name = f"{split_name}_{img_id:04d}.jpg"

            # Convert to RGB if needed and save as JPG
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(output_img_dir / img_name, 'JPEG')

            # Add image info
            coco_data['images'].append({
                'id': img_id,
                'file_name': img_name,
                'height': height,
                'width': width
            })

            # Create dummy bounding box (entire image or center region)
            # NOTE: These are placeholder annotations!
            # For real training, you need actual fire/smoke bounding boxes
            box_width = width * 0.5
            box_height = height * 0.5
            x = (width - box_width) / 2
            y = (height - box_height) / 2

            coco_data['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 0,  # fire
                'bbox': [float(x), float(y), float(box_width), float(box_height)],
                'area': float(box_width * box_height),
                'iscrowd': 0
            })
            ann_id += 1

        except Exception as e:
            print(f"  Warning: Failed to process {img_file}: {e}")
            continue

    # Save annotations
    ann_dir = output_dir / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)

    ann_file = ann_dir / f'instances_{split_name}.json'
    with open(ann_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"✓ Created {len(coco_data['images'])} images for {split_name}")
    print(f"✓ Created {len(coco_data['annotations'])} annotations")
    print(f"✓ Saved to {ann_file}")

    return len(coco_data['images'])


def main():
    parser = argparse.ArgumentParser(
        description='Convert Kaggle fire dataset to COCO format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-dir', required=True, help='Input directory with images')
    parser.add_argument('--output-dir', required=True, help='Output directory for COCO format')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not 0.99 <= total_ratio <= 1.01:
        print(f"Error: Ratios sum to {total_ratio}, should be 1.0")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    print("="*80)
    print("Kaggle to COCO Converter")
    print("="*80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Find all images
    print("Searching for images...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []

    for ext in image_extensions:
        all_images.extend(list(input_dir.rglob(f'*{ext}')))
        all_images.extend(list(input_dir.rglob(f'*{ext.upper()}')))

    # Remove duplicates
    all_images = list(set(all_images))

    print(f"Found {len(all_images)} images")

    if len(all_images) == 0:
        print("Error: No images found!")
        print(f"Searched in: {input_dir}")
        print(f"Extensions: {image_extensions}")
        return

    # Shuffle and split
    np.random.seed(args.seed)
    np.random.shuffle(all_images)

    n_train = int(len(all_images) * args.train_ratio)
    n_val = int(len(all_images) * args.val_ratio)

    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]

    print()
    print("Split:")
    print(f"  Train: {len(train_images)} ({args.train_ratio*100:.0f}%)")
    print(f"  Val: {len(val_images)} ({args.val_ratio*100:.0f}%)")
    print(f"  Test: {len(test_images)} ({args.test_ratio*100:.0f}%)")
    print()

    # Create COCO datasets
    n_train_created = create_coco_dataset(train_images, output_dir, 'train')
    print()
    n_val_created = create_coco_dataset(val_images, output_dir, 'val')
    print()
    n_test_created = create_coco_dataset(test_images, output_dir, 'test')

    print()
    print("="*80)
    print("✅ Conversion complete!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Total images created: {n_train_created + n_val_created + n_test_created}")
    print()
    print("⚠️  IMPORTANT: These are placeholder annotations!")
    print("For real fire detection, you need actual bounding box annotations.")
    print("The current boxes cover the center 50% of each image.")
    print()
    print("Next steps:")
    print(f"  1. Train model:")
    print(f"     python train.py \\")
    print(f"         --config configs/mac_m1_config.yaml \\")
    print(f"         --data-dir {output_dir}/images \\")
    print(f"         --train-ann {output_dir}/annotations/instances_train.json \\")
    print(f"         --val-ann {output_dir}/annotations/instances_val.json \\")
    print(f"         --output-dir experiments/kaggle_dataset \\")
    print(f"         --device mps \\")
    print(f"         --epochs 50")
    print()


if __name__ == '__main__':
    main()
