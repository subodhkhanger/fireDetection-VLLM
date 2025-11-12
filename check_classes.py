"""Check class distribution in predictions vs ground truth"""

import json
import yaml
from pathlib import Path

# Load config
with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check test annotations
test_annotation = config['data']['test_annotation']
print(f"Loading: {test_annotation}")

with open(test_annotation, 'r') as f:
    test_data = json.load(f)

# Check categories
print("\nCategories in test set:")
for cat in test_data['categories']:
    print(f"  {cat}")

# Check annotations
ann_classes = set()
for ann in test_data['annotations']:
    ann_classes.add(ann['category_id'])

print(f"\nUnique annotation class IDs: {sorted(ann_classes)}")

# Check train annotations for comparison
train_annotation = config['data']['train_annotation']
print(f"\nLoading: {train_annotation}")

with open(train_annotation, 'r') as f:
    train_data = json.load(f)

print("\nCategories in train set:")
for cat in train_data['categories']:
    print(f"  {cat}")

train_classes = set()
for ann in train_data['annotations']:
    train_classes.add(ann['category_id'])

print(f"\nUnique train class IDs: {sorted(train_classes)}")

# Check model config
print(f"\nModel config:")
print(f"  num_classes: {config['model']['num_classes']}")
print(f"\n⚠️  Issue: Model predicts classes 0-{config['model']['num_classes']-1}")
print(f"  But test data has classes: {sorted(ann_classes)}")
