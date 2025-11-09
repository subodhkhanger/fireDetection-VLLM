#!/bin/bash
set -e

echo "=================================================="
echo "🍎 Fire-ViT Setup for Mac M1"
echo "=================================================="
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p data/FireTiny/images/{train,val,test}
mkdir -p data/FireTiny/annotations
mkdir -p scripts
mkdir -p experiments

# Check if virtual environment exists
if [ ! -d "fire_vit_env" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv fire_vit_env
else
    echo "✓ Virtual environment already exists"
fi

# Activate environment
echo "🔄 Activating virtual environment..."
source fire_vit_env/bin/activate

# Install dependencies
echo "📦 Installing dependencies for Mac M1..."
pip install --upgrade pip --quiet

echo "  Installing PyTorch for Mac M1..."
pip install torch torchvision torchaudio --quiet

echo "  Installing other dependencies..."
pip install albumentations opencv-python pillow --quiet
pip install numpy pandas matplotlib seaborn --quiet
pip install tqdm pyyaml einops scipy scikit-learn --quiet

echo "✓ Dependencies installed"

# Create tiny dataset
echo ""
echo "🔥 Creating tiny fire detection dataset..."
python3 - << 'PYTHON_SCRIPT'
from pathlib import Path
import json
import numpy as np
import cv2

np.random.seed(42)

print("  Creating COCO annotations...")
for split, num_images in [('train', 30), ('val', 15), ('test', 15)]:
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 0, 'name': 'fire'},
            {'id': 1, 'name': 'smoke'}
        ]
    }

    ann_id = 0

    for img_id in range(num_images):
        filename = f'{split}_{img_id:04d}.jpg'
        coco_data['images'].append({
            'id': img_id,
            'file_name': filename,
            'height': 256,
            'width': 256
        })

        # Add 1-2 random boxes per image
        num_boxes = np.random.randint(1, 3)
        for _ in range(num_boxes):
            x = np.random.randint(10, 150)
            y = np.random.randint(10, 150)
            w = np.random.randint(40, 100)
            h = np.random.randint(40, 100)

            # Ensure box is within image
            x = min(x, 256 - w)
            y = min(y, 256 - h)

            coco_data['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': np.random.randint(0, 2),
                'bbox': [float(x), float(y), float(w), float(h)],
                'area': float(w * h),
                'iscrowd': 0
            })
            ann_id += 1

    # Save annotations
    ann_file = f'data/FireTiny/annotations/instances_{split}.json'
    with open(ann_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"    ✓ {split}: {num_images} images, {ann_id} annotations -> {ann_file}")

print("\n  Creating dummy images with fire-like patterns...")
for split, num_images in [('train', 30), ('val', 15), ('test', 15)]:
    img_dir = Path(f'data/FireTiny/images/{split}')

    for i in range(num_images):
        # Create base image
        img = np.random.randint(20, 100, (256, 256, 3), dtype=np.uint8)

        # Add fire-like regions (orange/red)
        num_fire_regions = np.random.randint(1, 3)
        for _ in range(num_fire_regions):
            x = np.random.randint(0, 200)
            y = np.random.randint(0, 200)
            w = np.random.randint(40, 80)
            h = np.random.randint(40, 80)

            # Create fire-like gradient
            for dy in range(h):
                for dx in range(w):
                    if y + dy < 256 and x + dx < 256:
                        # Red channel (fire)
                        img[y + dy, x + dx, 2] = min(255, 200 + np.random.randint(0, 55))
                        # Green channel
                        img[y + dy, x + dx, 1] = np.random.randint(100, 180)
                        # Blue channel (low for fire)
                        img[y + dy, x + dx, 0] = np.random.randint(0, 50)

        # Add some smoke-like regions (gray)
        num_smoke_regions = np.random.randint(0, 2)
        for _ in range(num_smoke_regions):
            x = np.random.randint(0, 200)
            y = np.random.randint(0, 200)
            w = np.random.randint(50, 100)
            h = np.random.randint(50, 100)

            gray_val = np.random.randint(100, 180)
            for dy in range(h):
                for dx in range(w):
                    if y + dy < 256 and x + dx < 256:
                        img[y + dy, x + dx, :] = gray_val + np.random.randint(-20, 20)

        # Save image
        filename = img_dir / f'{split}_{i:04d}.jpg'
        cv2.imwrite(str(filename), img)

    print(f"    ✓ {split}: {num_images} images created")

print("\n✅ Tiny dataset created successfully!")
print(f"   Total: 60 images (30 train, 15 val, 15 test)")
PYTHON_SCRIPT

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "📊 Dataset Summary:"
echo "  Location: data/FireTiny/"
echo "  Train: 30 images"
echo "  Val: 15 images"
echo "  Test: 15 images"
echo "  Image size: 256x256"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Activate environment:"
echo "   source fire_vit_env/bin/activate"
echo ""
echo "2. Test the model (2 minutes):"
echo "   python test_model.py"
echo ""
echo "3. Quick training test (5 minutes):"
echo "   python train.py \\"
echo "     --config configs/mac_m1_config.yaml \\"
echo "     --data-dir data/FireTiny/images/train \\"
echo "     --train-ann data/FireTiny/annotations/instances_train.json \\"
echo "     --val-ann data/FireTiny/annotations/instances_val.json \\"
echo "     --output-dir experiments/mac_test \\"
echo "     --device mps \\"
echo "     --epochs 10 \\"
echo "     --batch-size 4"
echo ""
echo "4. See DATASET_SETUP_MAC.md for more options"
echo ""
