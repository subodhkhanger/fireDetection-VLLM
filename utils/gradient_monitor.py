"""
Gradient monitoring utilities

Tracks gradient norms to detect training issues:
- Vanishing gradients (too small)
- Exploding gradients (too large)
- NaN/Inf gradients
- Layer-wise gradient flow
"""

import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
import json


class GradientMonitor:
    """
    Monitor gradient norms during training

    Tracks:
    - Global gradient norm
    - Per-layer gradient norms
    - Gradient statistics (mean, std, max, min)
    - Issues like vanishing/exploding gradients
    """

    def __init__(self, model, save_dir='./gradient_monitoring'):
        """
        Args:
            model: PyTorch model to monitor
            save_dir: Directory to save gradient logs
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # History tracking
        self.history = {
            'step': [],
            'global_grad_norm': [],
            'grad_mean': [],
            'grad_std': [],
            'grad_max': [],
            'grad_min': [],
            'clipped': [],
            'layer_norms': defaultdict(list),
        }

        # Get layer names for detailed tracking
        self.layer_names = [name for name, param in model.named_parameters()
                           if param.requires_grad]

    def compute_grad_norm(self, norm_type=2.0):
        """
        Compute global gradient norm

        Args:
            norm_type: Type of norm (default: L2 norm)

        Returns:
            dict with gradient statistics
        """
        parameters = [p for p in self.model.parameters()
                     if p.grad is not None and p.requires_grad]

        if len(parameters) == 0:
            return {
                'global_norm': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0,
                'has_nan': False,
                'has_inf': False,
                'num_params': 0
            }

        # Collect all gradients
        all_grads = []
        for p in parameters:
            all_grads.append(p.grad.detach().flatten())

        all_grads = torch.cat(all_grads)

        # Check for NaN/Inf
        has_nan = torch.isnan(all_grads).any().item()
        has_inf = torch.isinf(all_grads).any().item()

        if has_nan or has_inf:
            return {
                'global_norm': float('nan'),
                'mean': float('nan'),
                'std': float('nan'),
                'max': float('nan'),
                'min': float('nan'),
                'has_nan': has_nan,
                'has_inf': has_inf,
                'num_params': len(parameters)
            }

        # Compute statistics
        device = parameters[0].grad.device
        grad_norms = torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device)
            for p in parameters
        ])

        global_norm = torch.norm(grad_norms, norm_type).item()

        return {
            'global_norm': global_norm,
            'mean': all_grads.mean().item(),
            'std': all_grads.std().item(),
            'max': all_grads.max().item(),
            'min': all_grads.min().item(),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'num_params': len(parameters)
        }

    def compute_layer_grad_norms(self):
        """
        Compute gradient norms per layer

        Returns:
            dict mapping layer names to gradient norms
        """
        layer_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_norm = torch.norm(param.grad.detach(), 2.0).item()
                layer_norms[name] = grad_norm

        return layer_norms

    def check_gradient_health(self, grad_stats, clip_norm=None):
        """
        Check for gradient issues

        Args:
            grad_stats: Dict from compute_grad_norm()
            clip_norm: Gradient clipping norm (if used)

        Returns:
            dict with warnings and status
        """
        warnings = []
        status = 'healthy'

        global_norm = grad_stats['global_norm']

        # Check for NaN/Inf
        if grad_stats['has_nan']:
            warnings.append("❌ CRITICAL: Gradients contain NaN!")
            status = 'critical'
            return {'status': status, 'warnings': warnings}

        if grad_stats['has_inf']:
            warnings.append("❌ CRITICAL: Gradients contain Inf!")
            status = 'critical'
            return {'status': status, 'warnings': warnings}

        # Check for vanishing gradients
        if global_norm < 1e-7:
            warnings.append(f"❌ CRITICAL: Vanishing gradients! Norm: {global_norm:.2e}")
            status = 'critical'
        elif global_norm < 1e-5:
            warnings.append(f"⚠️  WARNING: Very small gradients. Norm: {global_norm:.2e}")
            status = 'warning'
        elif global_norm < 1e-3:
            warnings.append(f"⚠️  Small gradients. Norm: {global_norm:.2e}")
            if status == 'healthy':
                status = 'caution'

        # Check for exploding gradients
        if global_norm > 100:
            warnings.append(f"❌ CRITICAL: Exploding gradients! Norm: {global_norm:.2f}")
            status = 'critical'
        elif global_norm > 10:
            warnings.append(f"⚠️  WARNING: Large gradients. Norm: {global_norm:.2f}")
            if status != 'critical':
                status = 'warning'

        # Check if clipping is happening frequently
        if clip_norm is not None and global_norm > clip_norm:
            warnings.append(f"⚠️  Gradients were clipped ({global_norm:.2f} -> {clip_norm:.2f})")

        # Check gradient distribution
        if abs(grad_stats['mean']) > 0.1:
            warnings.append(f"⚠️  High gradient mean: {grad_stats['mean']:.2e} (should be ~0)")

        return {
            'status': status,
            'warnings': warnings
        }

    def log_gradients(self, step, clip_norm=None, verbose=False):
        """
        Log gradient information for a training step

        Args:
            step: Current training step/iteration
            clip_norm: Gradient clipping norm (if used)
            verbose: Print detailed info

        Returns:
            dict with gradient stats and health
        """
        # Compute gradient statistics
        grad_stats = self.compute_grad_norm()
        layer_norms = self.compute_layer_grad_norms()

        # Check health
        was_clipped = clip_norm is not None and grad_stats['global_norm'] > clip_norm
        health = self.check_gradient_health(grad_stats, clip_norm)

        # Update history
        self.history['step'].append(step)
        self.history['global_grad_norm'].append(grad_stats['global_norm'])
        self.history['grad_mean'].append(grad_stats['mean'])
        self.history['grad_std'].append(grad_stats['std'])
        self.history['grad_max'].append(grad_stats['max'])
        self.history['grad_min'].append(grad_stats['min'])
        self.history['clipped'].append(was_clipped)

        # Track layer-wise norms
        for name, norm in layer_norms.items():
            self.history['layer_norms'][name].append(norm)

        # Print if verbose
        if verbose:
            self._print_gradient_summary(step, grad_stats, health, clip_norm)

        return {
            'grad_stats': grad_stats,
            'layer_norms': layer_norms,
            'health': health,
            'clipped': was_clipped
        }

    def _print_gradient_summary(self, step, grad_stats, health, clip_norm):
        """Print formatted gradient summary"""
        print("\n" + "="*80)
        print(f"GRADIENT CHECK - Step {step}")
        print("="*80)

        print(f"\nGlobal Gradient Norm: {grad_stats['global_norm']:.6f}")
        print(f"  Mean:  {grad_stats['mean']:.2e}")
        print(f"  Std:   {grad_stats['std']:.2e}")
        print(f"  Max:   {grad_stats['max']:.2e}")
        print(f"  Min:   {grad_stats['min']:.2e}")

        if clip_norm is not None:
            print(f"  Clip Norm: {clip_norm:.2f}")
            if grad_stats['global_norm'] > clip_norm:
                print(f"  ⚠️  CLIPPED: {grad_stats['global_norm']:.2f} -> {clip_norm:.2f}")

        print(f"\nHealth Status: {health['status'].upper()}")
        if health['warnings']:
            for warning in health['warnings']:
                print(f"  {warning}")
        else:
            print("  ✓ Gradients look healthy")

        print("="*80)

    def get_layer_summary(self, top_n=10, bottom_n=10):
        """
        Get summary of layer gradient norms

        Args:
            top_n: Number of layers with largest gradients to show
            bottom_n: Number of layers with smallest gradients to show

        Returns:
            dict with layer summaries
        """
        if len(self.history['step']) == 0:
            return {}

        # Get latest layer norms
        latest_norms = {}
        for name in self.layer_names:
            if name in self.history['layer_norms'] and len(self.history['layer_norms'][name]) > 0:
                latest_norms[name] = self.history['layer_norms'][name][-1]

        # Sort by norm
        sorted_layers = sorted(latest_norms.items(), key=lambda x: x[1], reverse=True)

        return {
            'largest_gradients': sorted_layers[:top_n],
            'smallest_gradients': sorted_layers[-bottom_n:] if len(sorted_layers) > bottom_n else [],
            'total_layers': len(latest_norms)
        }

    def save_history(self):
        """Save gradient history to file"""
        history_file = self.save_dir / 'gradient_history.json'

        # Convert defaultdict to regular dict for JSON
        history_to_save = dict(self.history)
        history_to_save['layer_norms'] = dict(history_to_save['layer_norms'])

        with open(history_file, 'w') as f:
            json.dump(history_to_save, f, indent=2)

        print(f"✓ Gradient history saved to: {history_file}")

    def plot_gradients(self, save_path=None):
        """
        Plot gradient history
        Requires matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plots")
            return

        if len(self.history['step']) == 0:
            print("No gradient history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        steps = self.history['step']

        # Plot 1: Global gradient norm
        ax = axes[0, 0]
        ax.plot(steps, self.history['global_grad_norm'], marker='o', markersize=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Global Gradient Norm')
        ax.set_title('Global Gradient Norm Over Training')
        ax.grid(True)
        ax.set_yscale('log')

        # Plot 2: Gradient statistics
        ax = axes[0, 1]
        ax.plot(steps, self.history['grad_mean'], label='Mean', marker='o', markersize=2)
        ax.plot(steps, self.history['grad_std'], label='Std', marker='s', markersize=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.set_title('Gradient Mean and Std')
        ax.legend()
        ax.grid(True)

        # Plot 3: Gradient range (max/min)
        ax = axes[1, 0]
        ax.plot(steps, self.history['grad_max'], label='Max', marker='o', markersize=2)
        ax.plot(steps, np.abs(self.history['grad_min']), label='|Min|', marker='s', markersize=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Absolute Value')
        ax.set_title('Gradient Range (Max and |Min|)')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')

        # Plot 4: Clipping frequency
        ax = axes[1, 1]
        window_size = 100
        if len(self.history['clipped']) >= window_size:
            # Moving average of clipping
            clipped_array = np.array(self.history['clipped'], dtype=float)
            clipping_rate = np.convolve(clipped_array,
                                       np.ones(window_size)/window_size,
                                       mode='valid')
            ax.plot(steps[window_size-1:], clipping_rate * 100)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Clipping Rate (%)')
            ax.set_title(f'Gradient Clipping Rate (Moving Avg, window={window_size})')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / 'gradient_plots.png'

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gradient plots saved to: {save_path}")
        plt.close()


def compute_gradient_norm(model, norm_type=2.0):
    """
    Standalone function to compute gradient norm

    Args:
        model: PyTorch model
        norm_type: Type of norm (default: L2)

    Returns:
        Global gradient norm value
    """
    parameters = [p for p in model.parameters()
                 if p.grad is not None and p.requires_grad]

    if len(parameters) == 0:
        return 0.0

    device = parameters[0].grad.device
    grad_norms = torch.stack([
        torch.norm(p.grad.detach(), norm_type).to(device)
        for p in parameters
    ])

    total_norm = torch.norm(grad_norms, norm_type)
    return total_norm.item()


def check_gradients_quick(model, threshold_low=1e-6, threshold_high=10.0):
    """
    Quick gradient check

    Args:
        model: PyTorch model
        threshold_low: Threshold for vanishing gradients
        threshold_high: Threshold for exploding gradients

    Returns:
        str: 'ok', 'vanishing', 'exploding', 'nan', or 'none'
    """
    parameters = [p for p in model.parameters()
                 if p.grad is not None and p.requires_grad]

    if len(parameters) == 0:
        return 'none'

    # Check for NaN/Inf
    for p in parameters:
        if torch.isnan(p.grad).any():
            return 'nan'
        if torch.isinf(p.grad).any():
            return 'nan'

    # Compute norm
    norm = compute_gradient_norm(model)

    if norm < threshold_low:
        return 'vanishing'
    elif norm > threshold_high:
        return 'exploding'
    else:
        return 'ok'
