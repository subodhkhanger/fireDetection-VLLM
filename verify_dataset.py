"""
Verify training and test datasets match
Check for data consistency issues
"""

import json
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load config
with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("="*70)
print(" "*20 + "DATASET VERIFICATION")
print("="*70)

def analyze_annotations(annotation_file, split_name):
    """Analyze annotation file"""
    print(f"\n{'='*70}")
    print(f"Analyzing {split_name.upper()} set: {annotation_file}")
    print(f"{'='*70}")

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Categories
    print("\nCategories:")
    for cat in data['categories']:
        print(f"  ID {cat['id']}: {cat['name']}")

    # Images
    print(f"\nImages: {len(data['images'])}")
    if len(data['images']) > 0:
        sample_img = data['images'][0]
        print(f"  Sample: {sample_img['file_name']}")
        print(f"  Size: {sample_img.get('width', '?')}x{sample_img.get('height', '?')}")

    # Annotations
    print(f"\nAnnotations: {len(data['annotations'])}")

    # Class distribution
    class_counts = {}
    for ann in data['annotations']:
        cid = ann['category_id']
        class_counts[cid] = class_counts.get(cid, 0) + 1

    print("\nClass Distribution:")
    for cid, count in sorted(class_counts.items()):
        cat_name = next((c['name'] for c in data['categories'] if c['id'] == cid), 'unknown')
        print(f"  Class {cid} ({cat_name}): {count} instances")

    # Bounding box analysis
    print("\nBounding Box Analysis:")
    all_boxes = []
    for ann in data['annotations']:
        bbox = ann['bbox']
        all_boxes.append(bbox)

    if len(all_boxes) > 0:
        all_boxes = np.array(all_boxes)

        print(f"  Format: [x, y, w, h]")
        print(f"  X range: [{all_boxes[:, 0].min():.4f}, {all_boxes[:, 0].max():.4f}]")
        print(f"  Y range: [{all_boxes[:, 1].min():.4f}, {all_boxes[:, 1].max():.4f}]")
        print(f"  W range: [{all_boxes[:, 2].min():.4f}, {all_boxes[:, 2].max():.4f}]")
        print(f"  H range: [{all_boxes[:, 3].min():.4f}, {all_boxes[:, 3].max():.4f}]")

        # Check if normalized
        is_normalized = (all_boxes.max() <= 1.0)
        print(f"  Normalized (0-1): {'YES' if is_normalized else 'NO'}")

        # Sample boxes
        print(f"\n  Sample boxes (first 5):")
        for i in range(min(5, len(all_boxes))):
            ann = data['annotations'][i]
            print(f"    {i+1}. class={ann['category_id']}, bbox={ann['bbox']}")

    return {
        'num_images': len(data['images']),
        'num_annotations': len(data['annotations']),
        'class_counts': class_counts,
        'categories': data['categories']
    }

# Analyze each split
train_stats = analyze_annotations(config['data']['train_annotation'], 'train')
val_stats = analyze_annotations(config['data']['val_annotation'], 'val')
test_stats = analyze_annotations(config['data']['test_annotation'], 'test')

# Compare splits
print("\n" + "="*70)
print(" "*20 + "COMPARISON SUMMARY")
print("="*70)

print("\nDataset Sizes:")
print(f"  Train: {train_stats['num_images']} images, {train_stats['num_annotations']} annotations")
print(f"  Val:   {val_stats['num_images']} images, {val_stats['num_annotations']} annotations")
print(f"  Test:  {test_stats['num_images']} images, {test_stats['num_annotations']} annotations")

print("\nClass Distribution Consistency:")
all_classes = set()
for stats in [train_stats, val_stats, test_stats]:
    all_classes.update(stats['class_counts'].keys())

for cls_id in sorted(all_classes):
    train_count = train_stats['class_counts'].get(cls_id, 0)
    val_count = val_stats['class_counts'].get(cls_id, 0)
    test_count = test_stats['class_counts'].get(cls_id, 0)

    cat_name = next((c['name'] for c in train_stats['categories'] if c['id'] == cls_id), 'unknown')

    print(f"\n  Class {cls_id} ({cat_name}):")
    print(f"    Train: {train_count}")
    print(f"    Val:   {val_count}")
    print(f"    Test:  {test_count}")

    if train_count == 0:
        print(f"    ⚠️  WARNING: No training examples for class {cls_id}!")

# Model config check
print("\n" + "="*70)
print(" "*20 + "MODEL CONFIGURATION CHECK")
print("="*70)

print(f"\nModel expects:")
print(f"  num_classes: {config['model']['num_classes']}")
print(f"  This means model predicts classes: 0 to {config['model']['num_classes']-1}")

print(f"\nDataset has:")
print(f"  Class IDs in use: {sorted(all_classes)}")

if len(all_classes) != config['model']['num_classes']:
    print(f"\n⚠️  MISMATCH: Dataset has {len(all_classes)} classes but model expects {config['model']['num_classes']}")

if min(all_classes) != 0:
    print(f"\n⚠️  WARNING: Dataset classes start at {min(all_classes)}, not 0!")
    print(f"   Model will predict 0-{config['model']['num_classes']-1}")
    print(f"   Dataset has classes {sorted(all_classes)}")
    print(f"   This causes a mismatch!")

# Check images per class ratio
print("\n" + "="*70)
print(" "*20 + "POTENTIAL ISSUES")
print("="*70)

total_train_anns = sum(train_stats['class_counts'].values())
print("\nClass Imbalance in Training:")
for cls_id in sorted(train_stats['class_counts'].keys()):
    count = train_stats['class_counts'][cls_id]
    percentage = (count / total_train_anns) * 100
    cat_name = next((c['name'] for c in train_stats['categories'] if c['id'] == cls_id), 'unknown')
    print(f"  Class {cls_id} ({cat_name}): {count} ({percentage:.1f}%)")

    if percentage < 10:
        print(f"    ⚠️  WARNING: Class {cls_id} has < 10% of training data!")

# Check test set representativeness
print("\nTest Set Representativeness:")
total_test_anns = sum(test_stats['class_counts'].values())
if total_test_anns > 0:
    for cls_id in sorted(test_stats['class_counts'].keys()):
        count = test_stats['class_counts'][cls_id]
        percentage = (count / total_test_anns) * 100
        cat_name = next((c['name'] for c in test_stats['categories'] if c['id'] == cls_id), 'unknown')
        print(f"  Class {cls_id} ({cat_name}): {count} ({percentage:.1f}%)")
else:
    print("  ⚠️  No annotations in test set!")

# Test actual data loading
print("\n" + "="*70)
print(" "*20 + "DATA LOADING TEST")
print("="*70)

print("\nTesting actual data loading...")
try:
    from data.fire_dataset import FireDetectionDataset
    from data.augmentations import get_train_transforms, get_val_transforms

    # Test train loader
    train_img_dir = Path(config['data']['train_annotation']).parent
    train_dataset = FireDetectionDataset(
        image_dir=train_img_dir,
        annotation_file=config['data']['train_annotation'],
        transform=get_train_transforms(640),
        mode='train',
        img_size=640
    )

    print(f"\n✓ Train dataset loaded: {len(train_dataset)} samples")

    # Load a sample
    if len(train_dataset) > 0:
        img, target = train_dataset[0]
        print(f"  Sample shape: {img.shape}")
        print(f"  Target keys: {target.keys()}")
        print(f"  Boxes shape: {target['boxes'].shape}")
        print(f"  Labels: {target['labels'].tolist()}")
        print(f"  Label range: [{target['labels'].min()}, {target['labels'].max()}]")

        if len(target['boxes']) > 0:
            boxes = target['boxes'].numpy()
            print(f"  Box ranges:")
            print(f"    X: [{boxes[:, 0].min():.4f}, {boxes[:, 0].max():.4f}]")
            print(f"    Y: [{boxes[:, 1].min():.4f}, {boxes[:, 1].max():.4f}]")
            print(f"    W: [{boxes[:, 2].min():.4f}, {boxes[:, 2].max():.4f}]")
            print(f"    H: [{boxes[:, 3].min():.4f}, {boxes[:, 3].max():.4f}]")

    # Check what classes actually appear in loaded data
    print("\n  Checking class distribution in loaded dataset...")
    class_counts_loaded = {}
    for i in range(min(100, len(train_dataset))):
        _, target = train_dataset[i]
        for label in target['labels'].tolist():
            class_counts_loaded[label] = class_counts_loaded.get(label, 0) + 1

    print(f"  Classes found in first {min(100, len(train_dataset))} samples:")
    for cls_id, count in sorted(class_counts_loaded.items()):
        print(f"    Class {cls_id}: {count} instances")

except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print(" "*20 + "VERIFICATION COMPLETE")
print("="*70)
