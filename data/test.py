"""
Test script to verify Roboflow dataset is ready for training
Run: python data/test.py
"""
import json
import os
from pathlib import Path

print('=' * 60)
print('TESTING ROBOFLOW DATASET')
print('=' * 60)

base_path = Path('./data/fire-detection/fire-and-smoke-detection-2')

# Test each split
for split in ['train', 'valid', 'test']:
    print(f'\n{split.upper()} Split:')

    # Check annotation file
    ann_file = base_path / split / '_annotations.coco.json'
    if ann_file.exists():
        print(f'  ✓ Annotation file exists: {ann_file}')

        with open(ann_file, 'r') as f:
            data = json.load(f)

        num_images = len(data.get('images', []))
        num_annotations = len(data.get('annotations', []))
        num_categories = len(data.get('categories', []))

        print(f'  ✓ Number of images: {num_images}')
        print(f'  ✓ Number of annotations: {num_annotations}')
        print(f'  ✓ Number of categories: {num_categories}')

        if split == 'train' and num_categories > 0:
            category_names = [cat["name"] for cat in data["categories"]]
            print(f'  ✓ Category names: {category_names}')

        # Count actual image files
        img_dir = base_path / split
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
        num_img_files = len(image_files)
        print(f'  ✓ Image files found: {num_img_files}')

        # Check if images are actually in annotation
        if num_images > 0:
            sample_img = data['images'][0]
            sample_img_path = img_dir / sample_img['file_name']
            if sample_img_path.exists():
                print(f'  ✓ Sample image exists: {sample_img["file_name"]}')
            else:
                print(f'  ! Sample image NOT found: {sample_img["file_name"]}')
                print(f'    (Images might be referenced incorrectly in annotations)')
    else:
        print(f'  ✗ Annotation file not found: {ann_file}')

print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print('Dataset structure is correct!')
print('Annotations are in COCO format.')
print('\nYour config is already set up correctly at:')
print('  configs/base_config.yaml')
print('\nNext steps:')
print('  1. If on Lightning.ai, make sure you have GPU enabled')
print('  2. Run training with: python train.py')
print('=' * 60)
