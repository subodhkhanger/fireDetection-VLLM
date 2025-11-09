"""
Visualization utilities for Fire-ViT

Functions for visualizing:
- Predictions
- Attention maps
- Training curves
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def visualize_predictions(
    image,
    predictions,
    class_names=None,
    conf_threshold=0.5,
    save_path=None
):
    """
    Visualize detection predictions on image

    Args:
        image: numpy array (H, W, 3) or torch.Tensor (3, H, W)
        predictions: dict with 'boxes', 'labels', 'scores'
        class_names: List of class names
        conf_threshold: Confidence threshold for display
        save_path: Path to save visualization

    Returns:
        vis_image: Visualized image
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image * std + mean) * 255
        image = image.astype(np.uint8)

    # Create copy for visualization
    vis_image = image.copy()

    # Default class names
    if class_names is None:
        class_names = ['fire', 'smoke']

    # Colors for each class
    colors = [
        (0, 0, 255),  # Red for fire
        (128, 128, 128)  # Gray for smoke
    ]

    # Get predictions
    boxes = predictions.get('boxes', torch.tensor([]))
    labels = predictions.get('labels', torch.tensor([]))
    scores = predictions.get('scores', torch.ones(len(boxes)))

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue

        x, y, w, h = box
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        # Get class info
        class_idx = int(label)
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        color = colors[class_idx] if class_idx < len(colors) else (255, 255, 255)

        # Draw rectangle
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = f"{class_name}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1
        )

        cv2.rectangle(
            vis_image,
            (x1, y1 - text_h - 10),
            (x1 + text_w, y1),
            color,
            -1
        )

        cv2.putText(
            vis_image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    # Save if path provided
    if save_path is not None:
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    return vis_image


def visualize_attention_maps(
    image,
    attention_maps,
    save_path=None,
    num_heads=4
):
    """
    Visualize attention maps

    Args:
        image: Input image
        attention_maps: Attention tensor [B, H, N, N] or [B, H, L]
        save_path: Path to save visualization
        num_heads: Number of attention heads to visualize

    Returns:
        fig: Matplotlib figure
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image * std + mean).clip(0, 1)

    if isinstance(attention_maps, torch.Tensor):
        attention_maps = attention_maps.cpu().numpy()

    # Create subplot grid
    fig, axes = plt.subplots(2, num_heads, figsize=(4*num_heads, 8))

    for i in range(min(num_heads, attention_maps.shape[1])):
        # Original image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Head {i+1}")
        axes[0, i].axis('off')

        # Attention map
        attn = attention_maps[0, i]  # [N, N] or [N]

        if attn.ndim == 2:
            # Average over sequence
            attn = attn.mean(axis=0)

        # Reshape to spatial
        size = int(np.sqrt(len(attn)))
        if size * size == len(attn):
            attn = attn.reshape(size, size)

            # Upsample to image size
            attn = cv2.resize(attn, (image.shape[1], image.shape[0]))

            # Visualize
            axes[1, i].imshow(image)
            axes[1, i].imshow(attn, alpha=0.5, cmap='jet')
            axes[1, i].axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_training_curves(
    metrics_history,
    save_path=None
):
    """
    Plot training curves

    Args:
        metrics_history: Dict with lists of metrics over epochs
        save_path: Path to save plot

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        axes[0, 0].plot(metrics_history['train_loss'], label='Train')
        axes[0, 0].plot(metrics_history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    # Learning rate
    if 'lr' in metrics_history:
        axes[0, 1].plot(metrics_history['lr'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True)
        axes[0, 1].set_yscale('log')

    # mAP curves (if available)
    if 'val_mAP_50' in metrics_history:
        axes[1, 0].plot(metrics_history['val_mAP_50'], label='mAP@0.5')
        if 'val_mAP_50_95' in metrics_history:
            axes[1, 0].plot(metrics_history['val_mAP_50_95'], label='mAP@0.5:0.95')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].set_title('Validation mAP')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Loss components
    loss_components = [k for k in metrics_history.keys() if 'focal' in k or 'ciou' in k or 'dice' in k]
    if loss_components:
        for comp in loss_components[:5]:  # Max 5 components
            axes[1, 1].plot(metrics_history[comp], label=comp)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Component')
        axes[1, 1].set_title('Loss Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def create_comparison_grid(
    images,
    predictions_list,
    titles=None,
    save_path=None
):
    """
    Create grid of images with predictions for comparison

    Args:
        images: List of images
        predictions_list: List of prediction dicts
        titles: List of titles for each image
        save_path: Path to save grid

    Returns:
        fig: Matplotlib figure
    """
    num_images = len(images)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = np.array(axes).reshape(-1)

    for i, (image, preds) in enumerate(zip(images, predictions_list)):
        if i >= len(axes):
            break

        # Visualize predictions
        vis_image = visualize_predictions(image, preds)

        axes[i].imshow(vis_image)
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis('off')

    # Hide empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization utilities...")

    # Create dummy image and predictions
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    predictions = {
        'boxes': torch.tensor([[100, 100, 150, 150], [300, 300, 200, 200]]),
        'labels': torch.tensor([0, 1]),
        'scores': torch.tensor([0.9, 0.8])
    }

    # Test prediction visualization
    print("\nTesting prediction visualization...")
    vis_image = visualize_predictions(image, predictions)
    print(f"✓ Visualization shape: {vis_image.shape}")

    # Test attention visualization
    print("\nTesting attention visualization...")
    attention_maps = torch.randn(1, 8, 64, 64)
    fig = visualize_attention_maps(
        torch.from_numpy(image).permute(2, 0, 1) / 255.0,
        attention_maps
    )
    plt.close(fig)
    print("✓ Attention visualization created")

    # Test training curves
    print("\nTesting training curves...")
    metrics_history = {
        'train_loss': np.random.rand(50) + 0.5,
        'val_loss': np.random.rand(50) + 0.6,
        'lr': [1e-4 * (0.99 ** i) for i in range(50)]
    }
    fig = plot_training_curves(metrics_history)
    plt.close(fig)
    print("✓ Training curves created")

    print("\n✅ All visualization tests passed!")
