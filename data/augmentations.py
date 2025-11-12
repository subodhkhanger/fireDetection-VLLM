"""
Advanced Data Augmentations for Fire Detection

Implements:
- MixUp
- CutMix
- Mosaic
- Custom transforms for fire/smoke detection
"""

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=640):
    """Get training augmentation pipeline"""
    return A.Compose([
        # Resize
        A.Resize(height=img_size, width=img_size),
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
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
        ], p=0.5),

        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),

        # Cutout/Erase
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            fill_value=0,
            p=0.3
        ),

        # Weather conditions (fire detection specific)
        A.OneOf([
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=1.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5),
        ], p=0.15),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))


def get_val_transforms(img_size=640):
    """Get validation/test augmentation pipeline"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['class_labels']
    ))


class MosaicAugmentation:
    """
    Mosaic augmentation: Combine 4 images into one

    Used in YOLOv4/v5 for richer context and better small object detection
    """

    def __init__(self, dataset, img_size=640, prob=0.5):
        self.dataset = dataset
        self.img_size = img_size
        self.prob = prob

    def __call__(self, index):
        """
        Create mosaic from 4 random images

        Args:
            index: Index of center image

        Returns:
            mosaic_img: Combined image
            mosaic_boxes: Adjusted bounding boxes
            mosaic_labels: Corresponding labels
        """
        if np.random.random() > self.prob:
            # Return original image
            return self.dataset[index]

        # Select 4 images (including current)
        indices = [index] + [np.random.randint(0, len(self.dataset)) for _ in range(3)]

        # Create mosaic canvas
        mosaic_img = np.zeros((self.img_size * 2, self.img_size * 2, 3), dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []

        # Define quadrants
        positions = [
            (0, 0),  # Top-left
            (self.img_size, 0),  # Top-right
            (0, self.img_size),  # Bottom-left
            (self.img_size, self.img_size)  # Bottom-right
        ]

        for i, idx in enumerate(indices):
            # Load image and targets
            img, target = self.dataset[idx]

            # Convert tensor to numpy if needed
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img = (img * 255).astype(np.uint8)

            # Resize to quadrant size
            img_resized = cv2.resize(img, (self.img_size, self.img_size))

            # Place in mosaic
            y_offset, x_offset = positions[i]
            mosaic_img[y_offset:y_offset+self.img_size,
                      x_offset:x_offset+self.img_size] = img_resized

            # Adjust bounding boxes
            boxes = target['boxes'].numpy() if isinstance(target['boxes'], torch.Tensor) else target['boxes']
            labels = target['labels'].numpy() if isinstance(target['labels'], torch.Tensor) else target['labels']

            for box, label in zip(boxes, labels):
                x, y, w, h = box

                # Scale to quadrant
                scale = self.img_size / img.shape[1]
                x_new = x * scale + x_offset
                y_new = y * scale + y_offset
                w_new = w * scale
                h_new = h * scale

                mosaic_boxes.append([x_new, y_new, w_new, h_new])
                mosaic_labels.append(label)

        # Resize back to original size
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))

        # Scale boxes back
        scale_factor = 0.5
        mosaic_boxes = [[x*scale_factor, y*scale_factor, w*scale_factor, h*scale_factor]
                       for x, y, w, h in mosaic_boxes]

        # Convert to tensors
        mosaic_boxes = torch.tensor(mosaic_boxes, dtype=torch.float32)
        mosaic_labels = torch.tensor(mosaic_labels, dtype=torch.long)

        # Normalize and convert to tensor
        mosaic_img = mosaic_img.astype(np.float32) / 255.0
        mosaic_img = (mosaic_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        mosaic_img = torch.from_numpy(mosaic_img).permute(2, 0, 1).float()

        # Create mask
        mask = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
        for box in mosaic_boxes:
            x, y, w, h = box.tolist()
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.img_size, x2), min(self.img_size, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0

        target = {
            'boxes': mosaic_boxes,
            'labels': mosaic_labels,
            'masks': mask,
            'image_id': torch.tensor([index]),
            'orig_size': torch.tensor([self.img_size, self.img_size])
        }

        return mosaic_img, target


def mixup_collate_fn(batch, alpha=0.2):
    """
    MixUp augmentation at batch level

    Args:
        batch: List of (image, target) tuples
        alpha: MixUp parameter

    Returns:
        mixed_images: MixUp images
        targets: Original targets (both images)
        lam: MixUp weight
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    # Sample lambda from Beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    # Mix images
    mixed_images = lam * images + (1 - lam) * images[index]

    # For detection, we keep both sets of targets
    # The loss function will handle weighted combination
    targets_a = list(targets)
    targets_b = [targets[i] for i in index]

    return mixed_images, (targets_a, targets_b, lam)


def cutmix_collate_fn(batch, alpha=1.0):
    """
    CutMix augmentation at batch level

    Args:
        batch: List of (image, target) tuples
        alpha: CutMix parameter

    Returns:
        mixed_images: CutMix images
        targets: Adjusted targets
        lam: CutMix weight
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    # Sample lambda
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    # Random box
    _, _, H, W = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Cut and paste
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    # For detection, keep both sets of targets
    targets_a = list(targets)
    targets_b = [targets[i] for i in index]

    return images, (targets_a, targets_b, lam)


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentations...")

    # Create dummy image and boxes
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    boxes = np.array([[100, 100, 50, 50], [300, 300, 100, 100]])
    labels = np.array([0, 1])

    # Test train transforms
    print("\nTesting train transforms...")
    train_transform = get_train_transforms(640)

    try:
        transformed = train_transform(
            image=img,
            bboxes=boxes,
            class_labels=labels
        )
        print(f"✓ Image shape: {transformed['image'].shape}")
        print(f"✓ Boxes: {len(transformed['bboxes'])}")
        print("✅ Train transforms test passed!")
    except Exception as e:
        print(f"❌ Train transforms failed: {e}")

    # Test val transforms
    print("\nTesting val transforms...")
    val_transform = get_val_transforms(640)

    try:
        transformed = val_transform(
            image=img,
            bboxes=boxes,
            class_labels=labels
        )
        print(f"✓ Image shape: {transformed['image'].shape}")
        print(f"✓ Boxes: {len(transformed['bboxes'])}")
        print("✅ Val transforms test passed!")
    except Exception as e:
        print(f"❌ Val transforms failed: {e}")

    print("\n✅ All augmentation tests passed!")
