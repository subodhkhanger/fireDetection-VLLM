"""
Fire Detection Dataset

Implements dataset loading for D-Fire and FASDD datasets with:
- COCO format annotation parsing
- Advanced augmentations
- Multi-scale training support
"""

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

# Suppress specific numpy warnings from albumentations during augmentation
# This warning occurs when CoarseDropout calculates visibility ratios for tiny boxes
warnings.filterwarnings('ignore', category=RuntimeWarning,
                       message='invalid value encountered in divide')


class FireDetectionDataset(Dataset):
    """
    D-Fire dataset with COCO format annotations

    Args:
        image_dir (str): Path to images directory
        annotation_file (str): Path to COCO format JSON annotations
        transform: Albumentations transforms
        mode (str): 'train', 'val', or 'test'
        img_size (int): Target image size
    """

    def __init__(
        self,
        image_dir,
        annotation_file,
        transform=None,
        mode='train',
        img_size=640
    ):
        self.image_dir = Path(image_dir)
        self.mode = mode
        self.img_size = img_size

        # Load COCO format annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data['images']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        # Group annotations by image_id
        self.annotations = self._group_annotations()

        # Setup transforms
        self.transform = transform if transform else self._get_default_transform()

    def _group_annotations(self):
        """Group annotations by image_id"""
        grouped = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in grouped:
                grouped[img_id] = []
            grouped[img_id].append(ann)
        return grouped

    def _get_default_transform(self):
        """Get default augmentation pipeline"""
        if self.mode == 'train':
            return A.Compose([
                # Resize
                A.Resize(height=self.img_size, width=self.img_size),
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),

                # Color augmentations
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),

                # Noise and blur
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=7, p=0.2),

                # Cutout/Erase
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    fill_value=0,
                    p=0.3
                ),

                # Weather augmentations (fire detection specific)
                A.RandomSnow(p=0.1),
                A.RandomFog(p=0.1),
                A.RandomSunFlare(p=0.05),

                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='coco',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='coco',
                label_fields=['class_labels']
            ))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_info = self.images[idx]
        img_path = self.image_dir / img_info['file_name']

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations
        img_id = img_info['id']
        anns = self.annotations.get(img_id, [])

        # Get image dimensions for normalization
        img_height, img_width = image.shape[:2]

        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h] in COCO format (PIXEL coordinates)
            category_id = ann['category_id']

            # Normalize bounding box coordinates to [0, 1] range
            # Annotations are in PIXEL coordinates, need to normalize them
            x, y, w, h = bbox
            x = x / img_width
            y = y / img_height
            w = w / img_width
            h = h / img_height

            # Clip to valid range [0, 1] to handle floating point errors
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            # Ensure x+w and y+h don't exceed 1.0
            w = min(w, 1.0 - x)
            h = min(h, 1.0 - y)

            # Skip invalid/degenerate bounding boxes
            # Require minimum size of 0.001 (0.1% of image dimension) for both width and height
            # and minimum area of 0.00001 (0.001% of image area) to filter out annotation errors
            min_size = 0.001
            min_area = 0.00001
            area = w * h

            if w < min_size or h < min_size or area < min_area:
                continue

            boxes.append([x, y, w, h])
            # Remap class IDs from 1,2 to 0,1 for model compatibility
            # Dataset uses: 1=fire, 2=smoke
            # Model expects: 0=fire, 1=smoke
            labels.append(category_id - 1)

        # Handle empty annotations
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        # Apply transforms
        try:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
        except Exception as e:
            # If transform fails, return without augmentation
            print(f"Transform failed for {img_path}: {e}")
            basic_transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            transformed = basic_transform(image=image)
            transformed['bboxes'] = []
            transformed['class_labels'] = []

        # Prepare output
        image_tensor = transformed['image']
        boxes_transformed = transformed['bboxes']
        labels_transformed = transformed['class_labels']

        # Convert to tensors
        if len(boxes_transformed) > 0:
            boxes_tensor = torch.as_tensor(boxes_transformed, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels_transformed, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        # Create binary mask for attention loss
        mask = self._create_mask(image_tensor.shape[-2:], boxes_transformed)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'masks': mask,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([img_info['height'], img_info['width']])
        }

        return image_tensor, target

    def _create_mask(self, size, boxes):
        """Create binary mask from bounding boxes"""
        H, W = size
        mask = torch.zeros((H, W), dtype=torch.float32)

        for box in boxes:
            x, y, w, h = box
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0

        return mask


def collate_fn(batch):
    """
    Custom collate function for DataLoader

    Handles variable number of boxes per image
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    # Stack images
    images = torch.stack(images, dim=0)

    return images, targets


def create_dataloaders(
    data_dir,
    train_annotation,
    val_annotation,
    test_annotation=None,
    batch_size=32,
    num_workers=8,
    img_size=640,
    pin_memory=True
):
    """
    Create train, validation, and test dataloaders

    Args:
        data_dir (str): Root directory containing images (can be overridden by annotation file location)
        train_annotation (str): Path to train annotations
        val_annotation (str): Path to val annotations
        test_annotation (str): Path to test annotations (optional)
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        img_size (int): Target image size
        pin_memory (bool): Pin memory for faster GPU transfer

    Returns:
        dict: Dictionary with 'train', 'val', and optionally 'test' dataloaders
    """
    from .augmentations import get_train_transforms, get_val_transforms

    # Infer image directories from annotation file paths
    # If annotation is at path/to/train/_annotations.coco.json, images are at path/to/train/
    train_img_dir = Path(train_annotation).parent
    val_img_dir = Path(val_annotation).parent

    # Create datasets
    train_dataset = FireDetectionDataset(
        image_dir=train_img_dir,
        annotation_file=train_annotation,
        transform=get_train_transforms(img_size),
        mode='train',
        img_size=img_size
    )

    val_dataset = FireDetectionDataset(
        image_dir=val_img_dir,
        annotation_file=val_annotation,
        transform=get_val_transforms(img_size),
        mode='val',
        img_size=img_size
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Optional test loader
    if test_annotation is not None:
        test_img_dir = Path(test_annotation).parent
        test_dataset = FireDetectionDataset(
            image_dir=test_img_dir,
            annotation_file=test_annotation,
            transform=get_val_transforms(img_size),
            mode='test',
            img_size=img_size
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

        dataloaders['test'] = test_loader

    return dataloaders


if __name__ == "__main__":
    # Test dataset
    print("Testing FireDetectionDataset...")

    # Note: This requires actual D-Fire data to run
    # Placeholder paths - update with actual data location
    image_dir = "./data/D-Fire/images"
    annotation_file = "./data/D-Fire/annotations/instances_train.json"

    if os.path.exists(annotation_file):
        dataset = FireDetectionDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            mode='train',
            img_size=640
        )

        print(f"✓ Dataset size: {len(dataset)}")

        # Test loading a sample
        img, target = dataset[0]
        print(f"✓ Image shape: {img.shape}")
        print(f"✓ Boxes: {target['boxes'].shape}")
        print(f"✓ Labels: {target['labels'].shape}")
        print(f"✓ Mask: {target['masks'].shape}")

        print("\n✅ Dataset test passed!")
    else:
        print(f"⚠️  Dataset not found at {annotation_file}")
        print("   Please download D-Fire dataset first")
        print("   See SKILLS.md for instructions")
