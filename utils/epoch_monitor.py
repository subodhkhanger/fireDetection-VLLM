"""
Epoch-by-epoch monitoring to ensure model is learning correctly

Tracks:
- Class prediction distribution (fire vs smoke)
- Confidence scores per class
- Sample predictions visualization
- Loss components per class
"""

import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
import json


class EpochMonitor:
    """
    Monitor model predictions at each epoch to detect training issues early
    """

    def __init__(self, num_classes=2, save_dir='./monitoring', class_names=None):
        """
        Args:
            num_classes: Number of classes (excluding background)
            save_dir: Directory to save monitoring results
            class_names: Dict mapping class IDs to names {0: 'fire', 1: 'smoke'}
        """
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names or {i: f'class_{i}' for i in range(num_classes)}

        # History tracking
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'class_distribution': [],  # Predictions per class
            'class_confidence': [],    # Average confidence per class
            'gt_distribution': [],      # Ground truth per class
        }

    def analyze_predictions(self, predictions, targets, conf_threshold=0.3):
        """
        Analyze model predictions to check for issues

        Args:
            predictions: List of decoded predictions from model
            targets: List of ground truth targets
            conf_threshold: Confidence threshold for counting predictions

        Returns:
            dict with analysis results
        """
        pred_counts = defaultdict(int)
        pred_scores = defaultdict(list)
        gt_counts = defaultdict(int)

        total_preds = 0
        total_gt = 0

        for pred, target in zip(predictions, targets):
            # Analyze predictions
            pred_labels = pred['labels'].cpu().numpy()
            pred_confs = pred['scores'].cpu().numpy()

            # Filter by confidence
            mask = pred_confs >= conf_threshold
            filtered_labels = pred_labels[mask]
            filtered_scores = pred_confs[mask]

            for label, score in zip(filtered_labels, filtered_scores):
                pred_counts[int(label)] += 1
                pred_scores[int(label)].append(float(score))
                total_preds += 1

            # Analyze ground truth
            gt_labels = target['labels'].cpu().numpy()
            for label in gt_labels:
                if label >= 0:  # Skip invalid labels
                    gt_counts[int(label)] += 1
                    total_gt += 1

        # Calculate statistics
        pred_distribution = {}
        avg_confidence = {}

        for cls_id in range(self.num_classes):
            cls_name = self.class_names[cls_id]
            count = pred_counts.get(cls_id, 0)
            scores = pred_scores.get(cls_id, [])

            pred_distribution[cls_name] = count
            avg_confidence[cls_name] = float(np.mean(scores)) if scores else 0.0

        gt_distribution = {}
        for cls_id in range(self.num_classes):
            cls_name = self.class_names[cls_id]
            gt_distribution[cls_name] = gt_counts.get(cls_id, 0)

        return {
            'total_predictions': total_preds,
            'total_ground_truth': total_gt,
            'pred_distribution': pred_distribution,
            'avg_confidence': avg_confidence,
            'gt_distribution': gt_distribution,
            'pred_counts_raw': dict(pred_counts),
            'gt_counts_raw': dict(gt_counts),
        }

    def check_health(self, analysis):
        """
        Check for common training issues

        Returns:
            dict with warnings and status
        """
        warnings = []
        status = 'healthy'

        pred_dist = analysis['pred_distribution']
        gt_dist = analysis['gt_distribution']
        avg_conf = analysis['avg_confidence']

        # Check 1: Are we making predictions?
        if analysis['total_predictions'] == 0:
            warnings.append("❌ CRITICAL: No predictions above threshold!")
            status = 'critical'

        # Check 2: Class imbalance in predictions
        pred_values = list(pred_dist.values())
        if len(pred_values) > 1 and max(pred_values) > 0:
            max_pred = max(pred_values)
            min_pred = min(pred_values)
            ratio = max_pred / max(min_pred, 1)

            if ratio > 100:
                warnings.append(f"⚠️  WARNING: Severe class imbalance in predictions! Ratio: {ratio:.1f}:1")
                status = 'warning'
            elif ratio > 10:
                warnings.append(f"⚠️  Class imbalance in predictions. Ratio: {ratio:.1f}:1")
                if status != 'warning':
                    status = 'caution'

        # Check 3: Missing classes in predictions
        for cls_name, count in pred_dist.items():
            if count == 0 and gt_dist.get(cls_name, 0) > 0:
                warnings.append(f"⚠️  WARNING: No predictions for '{cls_name}' but {gt_dist[cls_name]} GT boxes exist!")
                status = 'warning'

        # Check 4: Low confidence
        for cls_name, conf in avg_conf.items():
            if conf > 0 and conf < 0.3:
                warnings.append(f"⚠️  Low confidence for '{cls_name}': {conf:.3f}")
                if status == 'healthy':
                    status = 'caution'

        # Check 5: Predicting non-existent classes
        for cls_name, count in pred_dist.items():
            if count > 0 and gt_dist.get(cls_name, 0) == 0:
                warnings.append(f"⚠️  Predicting '{cls_name}' but no GT boxes exist in validation set")

        return {
            'status': status,
            'warnings': warnings
        }

    def log_epoch(self, epoch, train_loss, val_loss, val_predictions, val_targets,
                  conf_threshold=0.3, verbose=True):
        """
        Log monitoring info for an epoch

        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
            val_predictions: Decoded predictions from validation
            val_targets: Validation ground truth targets
            conf_threshold: Confidence threshold for analysis
            verbose: Whether to print detailed info
        """
        # Analyze predictions
        analysis = self.analyze_predictions(val_predictions, val_targets, conf_threshold)
        health = self.check_health(analysis)

        # Update history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['class_distribution'].append(analysis['pred_distribution'])
        self.history['class_confidence'].append(analysis['avg_confidence'])
        self.history['gt_distribution'].append(analysis['gt_distribution'])

        # Print summary
        if verbose:
            self._print_epoch_summary(epoch, train_loss, val_loss, analysis, health)

        # Save to file
        self._save_epoch_log(epoch, analysis, health)

        return {
            'analysis': analysis,
            'health': health
        }

    def _print_epoch_summary(self, epoch, train_loss, val_loss, analysis, health):
        """Print formatted epoch summary"""
        print("\n" + "="*80)
        print(f"EPOCH {epoch} MONITORING")
        print("="*80)

        # Loss
        print(f"\nLoss:")
        print(f"  Train: {train_loss:.4f}")
        print(f"  Val:   {val_loss:.4f}")

        # Predictions
        print(f"\nPredictions: {analysis['total_predictions']} total")
        print(f"Ground Truth: {analysis['total_ground_truth']} total")

        # Per-class breakdown
        print(f"\nPer-Class Analysis:")
        print(f"{'Class':<15} {'Predictions':<15} {'Avg Conf':<15} {'Ground Truth':<15}")
        print("-" * 60)

        for cls_name in self.class_names.values():
            pred_count = analysis['pred_distribution'].get(cls_name, 0)
            avg_conf = analysis['avg_confidence'].get(cls_name, 0.0)
            gt_count = analysis['gt_distribution'].get(cls_name, 0)

            conf_str = f"{avg_conf:.3f}" if avg_conf > 0 else "-"
            print(f"{cls_name:<15} {pred_count:<15} {conf_str:<15} {gt_count:<15}")

        # Health status
        print(f"\nHealth Status: {health['status'].upper()}")
        if health['warnings']:
            for warning in health['warnings']:
                print(f"  {warning}")
        else:
            print("  ✓ No issues detected")

        print("="*80)

    def _save_epoch_log(self, epoch, analysis, health):
        """Save epoch log to JSON file"""
        log_file = self.save_dir / f'epoch_{epoch:03d}.json'

        log_data = {
            'epoch': epoch,
            'analysis': analysis,
            'health': health,
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def save_history(self):
        """Save full training history"""
        history_file = self.save_dir / 'training_history.json'

        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Also create a summary
        self._create_summary()

    def _create_summary(self):
        """Create training summary report"""
        summary_file = self.save_dir / 'training_summary.txt'

        with open(summary_file, 'w') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total Epochs: {len(self.history['epoch'])}\n")
            f.write(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}\n")
            f.write(f"Best Val Loss: {min(self.history['val_loss']):.4f}\n\n")

            # Class distribution evolution
            f.write("Final Class Prediction Distribution:\n")
            final_dist = self.history['class_distribution'][-1]
            for cls_name, count in final_dist.items():
                f.write(f"  {cls_name}: {count}\n")

            f.write("\nFinal Average Confidence:\n")
            final_conf = self.history['class_confidence'][-1]
            for cls_name, conf in final_conf.items():
                f.write(f"  {cls_name}: {conf:.3f}\n")

    def plot_history(self, save_path=None):
        """
        Create plots of training history
        Requires matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plots")
            return

        if len(self.history['epoch']) == 0:
            print("No history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = self.history['epoch']

        # Plot 1: Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        ax.plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)

        # Plot 2: Class Distribution
        ax = axes[0, 1]
        for cls_name in self.class_names.values():
            counts = [dist.get(cls_name, 0) for dist in self.history['class_distribution']]
            ax.plot(epochs, counts, label=f'{cls_name} predictions', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Number of Predictions')
        ax.set_title('Prediction Distribution by Class')
        ax.legend()
        ax.grid(True)

        # Plot 3: Average Confidence
        ax = axes[1, 0]
        for cls_name in self.class_names.values():
            confs = [conf.get(cls_name, 0) for conf in self.history['class_confidence']]
            ax.plot(epochs, confs, label=f'{cls_name} confidence', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Average Confidence by Class')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 1])

        # Plot 4: Ground Truth Distribution (should be constant)
        ax = axes[1, 1]
        for cls_name in self.class_names.values():
            counts = [dist.get(cls_name, 0) for dist in self.history['gt_distribution']]
            ax.plot(epochs, counts, label=f'{cls_name} GT', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Number of GT Boxes')
        ax.set_title('Ground Truth Distribution (Validation Set)')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / 'training_plots.png'

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plots saved to: {save_path}")
        plt.close()


def quick_validation_check(model, dataloader, device, conf_threshold=0.3, max_batches=5):
    """
    Quick validation check during training

    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: Device to run on
        conf_threshold: Confidence threshold
        max_batches: Number of batches to check

    Returns:
        predictions, targets lists
    """
    from utils.postprocess import decode_predictions

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            images = images.to(device)
            outputs = model(images)

            # Decode predictions
            img_size = images.shape[-1]
            decoded = decode_predictions(
                outputs,
                img_size=img_size,
                conf_threshold=conf_threshold,
                nms_threshold=0.5,
                topk=100,
                max_detections=100
            )

            all_predictions.extend(decoded)
            all_targets.extend(targets)

    model.train()
    return all_predictions, all_targets
