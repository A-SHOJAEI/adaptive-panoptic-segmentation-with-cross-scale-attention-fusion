"""Results analysis and visualization utilities."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PerClassAnalysis:
    """Per-class performance analysis for segmentation results."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """Initialize per-class analysis.

        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        if class_names is None:
            self.class_names = [f"Class_{i}" for i in range(num_classes)]
        else:
            self.class_names = class_names

        self.reset()

    def reset(self) -> None:
        """Reset analysis state."""
        self.class_tp = np.zeros(self.num_classes)
        self.class_fp = np.zeros(self.num_classes)
        self.class_fn = np.zeros(self.num_classes)
        self.class_tn = np.zeros(self.num_classes)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update with predictions and targets.

        Args:
            pred: Predicted labels of shape (B, H, W) or (N,)
            target: Ground truth labels of shape (B, H, W) or (N,)
        """
        pred = pred.flatten().cpu().numpy()
        target = target.flatten().cpu().numpy()

        for cls in range(self.num_classes):
            pred_pos = (pred == cls)
            target_pos = (target == cls)

            self.class_tp[cls] += np.sum(pred_pos & target_pos)
            self.class_fp[cls] += np.sum(pred_pos & ~target_pos)
            self.class_fn[cls] += np.sum(~pred_pos & target_pos)
            self.class_tn[cls] += np.sum(~pred_pos & ~target_pos)

    def compute_metrics(self) -> Dict[str, np.ndarray]:
        """Compute per-class metrics.

        Returns:
            Dictionary with 'precision', 'recall', 'f1', 'iou' arrays
        """
        precision = self.class_tp / (self.class_tp + self.class_fp + 1e-8)
        recall = self.class_tp / (self.class_tp + self.class_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = self.class_tp / (self.class_tp + self.class_fp + self.class_fn + 1e-8)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou
        }

    def generate_report(self) -> str:
        """Generate text report of per-class performance.

        Returns:
            Formatted report string
        """
        metrics = self.compute_metrics()

        report = "Per-Class Performance Report\n"
        report += "=" * 80 + "\n"
        report += f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}\n"
        report += "-" * 80 + "\n"

        for i in range(self.num_classes):
            if self.class_tp[i] + self.class_fp[i] + self.class_fn[i] == 0:
                continue

            report += (
                f"{self.class_names[i]:<20} "
                f"{metrics['precision'][i]:>10.4f} "
                f"{metrics['recall'][i]:>10.4f} "
                f"{metrics['f1'][i]:>10.4f} "
                f"{metrics['iou'][i]:>10.4f}\n"
            )

        report += "-" * 80 + "\n"
        report += (
            f"{'Mean':<20} "
            f"{np.nanmean(metrics['precision']):>10.4f} "
            f"{np.nanmean(metrics['recall']):>10.4f} "
            f"{np.nanmean(metrics['f1']):>10.4f} "
            f"{np.nanmean(metrics['iou']):>10.4f}\n"
        )

        return report

    def plot_metrics(self, save_path: Optional[Path] = None) -> None:
        """Plot per-class metrics as bar charts.

        Args:
            save_path: Optional path to save plot
        """
        metrics = self.compute_metrics()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Per-Class Performance Metrics', fontsize=16)

        metric_names = ['precision', 'recall', 'f1', 'iou']
        titles = ['Precision', 'Recall', 'F1 Score', 'IoU']

        for idx, (metric_name, title) in enumerate(zip(metric_names, titles)):
            ax = axes[idx // 2, idx % 2]
            values = metrics[metric_name]

            # Filter out classes with no data
            valid_indices = [
                i for i in range(self.num_classes)
                if self.class_tp[i] + self.class_fp[i] + self.class_fn[i] > 0
            ]
            valid_values = values[valid_indices]
            valid_names = [self.class_names[i] for i in valid_indices]

            bars = ax.bar(range(len(valid_values)), valid_values)
            ax.set_xlabel('Class')
            ax.set_ylabel(title)
            ax.set_title(f'{title} by Class')
            ax.set_xticks(range(len(valid_values)))
            ax.set_xticklabels(valid_names, rotation=45, ha='right')
            ax.set_ylim([0, 1])
            ax.axhline(y=np.nanmean(values), color='r', linestyle='--', label='Mean')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved metrics plot to {save_path}")
        else:
            plt.show()

        plt.close()


def visualize_predictions(
    image: np.ndarray,
    pred_semantic: np.ndarray,
    target_semantic: np.ndarray,
    pred_instance: Optional[np.ndarray] = None,
    target_instance: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    class_colors: Optional[np.ndarray] = None
) -> None:
    """Visualize predictions compared to ground truth.

    Args:
        image: RGB image array of shape (H, W, 3)
        pred_semantic: Predicted semantic labels (H, W)
        target_semantic: Ground truth semantic labels (H, W)
        pred_instance: Optional predicted instance labels (H, W)
        target_instance: Optional ground truth instance labels (H, W)
        save_path: Optional path to save visualization
        class_colors: Optional color map for classes (num_classes, 3)
    """
    # Generate color map if not provided
    if class_colors is None:
        num_classes = max(pred_semantic.max(), target_semantic.max()) + 1
        class_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_classes))[:, :3]
        class_colors = (class_colors * 255).astype(np.uint8)

    # Create colored segmentation maps
    pred_colored = class_colors[pred_semantic]
    target_colored = class_colors[target_semantic]

    # Determine number of rows
    num_rows = 2 if pred_instance is None else 3

    # Create figure
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # Row 1: Image and semantic segmentation
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target_colored)
    axes[0, 1].set_title('Ground Truth Semantic')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_colored)
    axes[0, 2].set_title('Predicted Semantic')
    axes[0, 2].axis('off')

    # Row 2: Overlay comparisons
    overlay_target = 0.5 * image + 0.5 * target_colored
    overlay_target = np.clip(overlay_target, 0, 255).astype(np.uint8)

    overlay_pred = 0.5 * image + 0.5 * pred_colored
    overlay_pred = np.clip(overlay_pred, 0, 255).astype(np.uint8)

    # Compute error map
    error_map = (pred_semantic != target_semantic).astype(np.uint8)
    error_colored = np.zeros_like(image)
    error_colored[error_map == 1] = [255, 0, 0]  # Red for errors

    axes[1, 0].imshow(overlay_target)
    axes[1, 0].set_title('GT Overlay')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay_pred)
    axes[1, 1].set_title('Prediction Overlay')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(error_colored)
    axes[1, 2].set_title('Error Map (Red = Wrong)')
    axes[1, 2].axis('off')

    # Row 3: Instance segmentation (if provided)
    if pred_instance is not None and target_instance is not None:
        # Create colored instance maps
        instance_colors = plt.cm.get_cmap('tab20')(
            np.linspace(0, 1, max(pred_instance.max(), target_instance.max()) + 1)
        )[:, :3]
        instance_colors = (instance_colors * 255).astype(np.uint8)

        target_inst_colored = instance_colors[target_instance % len(instance_colors)]
        pred_inst_colored = instance_colors[pred_instance % len(instance_colors)]

        axes[2, 0].imshow(image)
        axes[2, 0].set_title('Input Image')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(target_inst_colored)
        axes[2, 1].set_title('Ground Truth Instances')
        axes[2, 1].axis('off')

        axes[2, 2].imshow(pred_inst_colored)
        axes[2, 2].set_title('Predicted Instances')
        axes[2, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_dir: Optional[Path] = None
) -> None:
    """Plot training history curves.

    Args:
        history: Dictionary with training metrics over epochs
        save_dir: Optional directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)

    # Loss curves
    if 'train_loss' in history and 'val_loss' in history:
        ax = axes[0, 0]
        ax.plot(history['train_loss'], label='Train Loss')
        ax.plot(history['val_loss'], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(alpha=0.3)

    # Semantic loss
    if 'train_semantic_loss' in history and 'val_semantic_loss' in history:
        ax = axes[0, 1]
        ax.plot(history['train_semantic_loss'], label='Train Semantic Loss')
        ax.plot(history['val_semantic_loss'], label='Val Semantic Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Semantic Loss')
        ax.legend()
        ax.grid(alpha=0.3)

    # PQ metric
    if 'val_pq' in history:
        ax = axes[1, 0]
        ax.plot(history['val_pq'], label='Validation PQ', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Panoptic Quality')
        ax.set_title('Validation Panoptic Quality')
        ax.legend()
        ax.grid(alpha=0.3)

    # Learning rate
    if 'learning_rates' in history:
        ax = axes[1, 1]
        ax.plot(history['learning_rates'], label='Learning Rate', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    else:
        plt.show()

    plt.close()
