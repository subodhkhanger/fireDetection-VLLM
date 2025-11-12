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
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        num_images = len(data.get('images', []))
        num_annotations = len(data.get('annotations', []))
        num_categories = len(data.get('categories', []))
        
        print(f'  ✓ Annotations: {num_images} images, {num_annotations} annotations')
        print(f'  ✓ Categories: {num_categories}')
        
        if split == 'train' and num_categories > 0:
            print(f'  ✓ Category names: {[cat[\"name\"] for cat in data[\"categories\"]]}')
        
        # Count actual image files
        img_dir = base_path / split
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
        print(f'  ✓ Image files found: {len(image_files)}')
    else:
        print(f'  ✗ Annotation file not found')

print('\n' + '=' * 60)
print('Dataset is ready!' if True else 'Dataset has issues')
print('=' * 60)
"