#!/usr/bin/env python
"""Evaluation script for adaptive panoptic segmentation.

Loads a trained model and evaluates it on test set with comprehensive metrics:
- Panoptic Quality (PQ)
- Segmentation Quality (SQ)
- Recognition Quality (RQ)
- Boundary F1 score
- Per-class performance analysis
"""

import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.models.model import (
    AdaptivePanopticSegmentationModel,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.data.loader import (
    get_cityscapes_dataloaders,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.evaluation.metrics import (
    PanopticQualityMetric,
    compute_boundary_f1,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.evaluation.analysis import (
    PerClassAnalysis,
    visualize_predictions,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.utils.config import (
    load_config,
    setup_logging,
    get_device,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Evaluate adaptive panoptic segmentation model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (auto-detect if None)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization images'
    )
    parser.add_argument(
        '--num-visualize',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    config: dict,
    device: torch.device
) -> AdaptivePanopticSegmentationModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device to load model on

    Returns:
        Loaded model
    """
    import logging
    logger = logging.getLogger(__name__)

    # Create model
    model = AdaptivePanopticSegmentationModel(
        num_classes=config.get('model', {}).get('num_classes', 19),
        embed_dim=config.get('model', {}).get('embed_dim', 64),
        fpn_channels=config.get('model', {}).get('fpn_channels', 256),
        use_complexity_conditioning=config.get('model', {}).get('use_complexity_conditioning', True),
        pretrained_backbone=False  # Don't load pretrained weights when loading checkpoint
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded model state dict")

    model = model.to(device)
    model.eval()

    return model


def evaluate_model(
    model: AdaptivePanopticSegmentationModel,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 19
) -> dict:
    """Evaluate model on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device for computation
        num_classes: Number of classes

    Returns:
        Dictionary of evaluation metrics
    """
    import logging
    logger = logging.getLogger(__name__)

    # Initialize metrics
    pq_metric = PanopticQualityMetric(num_classes=num_classes)
    per_class_analysis = PerClassAnalysis(num_classes=num_classes)

    boundary_f1_scores = []

    logger.info("Evaluating model...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            images = batch['image'].to(device)
            semantic_labels = batch['semantic_mask'].to(device)
            instance_labels = batch['instance_mask'].to(device)

            # Forward pass
            predictions = model(images, return_aux=False)

            # Get predictions
            semantic_preds = predictions['semantic_logits'].argmax(dim=1)

            # Update metrics
            pq_metric.update(
                semantic_preds,
                semantic_preds,  # Placeholder for instance preds
                semantic_labels,
                instance_labels
            )

            per_class_analysis.update(semantic_preds, semantic_labels)

            # Compute boundary F1 for batch
            for i in range(images.size(0)):
                bf1 = compute_boundary_f1(
                    semantic_preds[i],
                    semantic_labels[i],
                    boundary_width=2
                )
                boundary_f1_scores.append(bf1)

    # Compute final metrics
    pq_results = pq_metric.compute()
    per_class_metrics = per_class_analysis.compute_metrics()

    results = {
        'panoptic_quality': pq_results['panoptic_quality'],
        'segmentation_quality': pq_results['segmentation_quality'],
        'recognition_quality': pq_results['recognition_quality'],
        'boundary_f1': np.mean(boundary_f1_scores),
        'mean_iou': np.nanmean(per_class_metrics['iou']),
        'mean_precision': np.nanmean(per_class_metrics['precision']),
        'mean_recall': np.nanmean(per_class_metrics['recall']),
        'mean_f1': np.nanmean(per_class_metrics['f1']),
        'per_class_iou': per_class_metrics['iou'].tolist(),
        'per_class_f1': per_class_metrics['f1'].tolist(),
    }

    logger.info("Evaluation complete!")
    logger.info(f"Panoptic Quality (PQ): {results['panoptic_quality']:.4f}")
    logger.info(f"Segmentation Quality (SQ): {results['segmentation_quality']:.4f}")
    logger.info(f"Recognition Quality (RQ): {results['recognition_quality']:.4f}")
    logger.info(f"Boundary F1: {results['boundary_f1']:.4f}")
    logger.info(f"Mean IoU: {results['mean_iou']:.4f}")

    return results, per_class_analysis


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(log_level='INFO')

    import logging
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Adaptive Panoptic Segmentation Evaluation")
    logger.info("=" * 80)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load or auto-detect config
    if args.config:
        config = load_config(args.config)
    else:
        # Try to find config in checkpoint directory
        config_path = checkpoint_path.parent / 'config.yaml'
        if config_path.exists():
            config = load_config(str(config_path))
            logger.info(f"Auto-detected config: {config_path}")
        else:
            # Use default config
            config = load_config('configs/default.yaml')
            logger.warning("Using default config")

    # Get device
    device = get_device(config.get('device'))

    # Create dataloaders
    logger.info("Creating dataloaders...")
    _, _, test_loader = get_cityscapes_dataloaders(
        root_dir=config.get('data', {}).get('root_dir'),
        batch_size=config.get('data', {}).get('batch_size', 4),
        num_workers=config.get('data', {}).get('num_workers', 4),
        image_size=(
            config.get('data', {}).get('image_height', 512),
            config.get('data', {}).get('image_width', 1024)
        ),
        use_synthetic=config.get('data', {}).get('use_synthetic', False),
        pin_memory=True
    )

    # Load model
    logger.info(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, config, device)

    # Evaluate model
    results, per_class_analysis = evaluate_model(
        model,
        test_loader,
        device,
        num_classes=config.get('model', {}).get('num_classes', 19)
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    results_file = output_dir / f'evaluation_results_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_file}")

    # Save per-class report
    report = per_class_analysis.generate_report()
    report_file = output_dir / f'per_class_report_{datetime.now():%Y%m%d_%H%M%S}.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Saved per-class report to {report_file}")
    print("\n" + report)

    # Plot per-class metrics
    per_class_analysis.plot_metrics(save_path=output_dir / 'per_class_metrics.png')

    # Generate visualizations if requested
    if args.visualize:
        logger.info(f"Generating {args.num_visualize} visualizations...")
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        model.eval()
        num_visualized = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if num_visualized >= args.num_visualize:
                    break

                images = batch['image'].to(device)
                semantic_labels = batch['semantic_mask'].to(device)
                instance_labels = batch['instance_mask'].to(device)

                predictions = model(images, return_aux=False)
                semantic_preds = predictions['semantic_logits'].argmax(dim=1)

                # Visualize each image in batch
                for i in range(min(images.size(0), args.num_visualize - num_visualized)):
                    # Denormalize image
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = (img * std + mean) * 255
                    img = np.clip(img, 0, 255).astype(np.uint8)

                    visualize_predictions(
                        image=img,
                        pred_semantic=semantic_preds[i].cpu().numpy(),
                        target_semantic=semantic_labels[i].cpu().numpy(),
                        save_path=viz_dir / f'sample_{num_visualized:03d}.png'
                    )

                    num_visualized += 1

        logger.info(f"Saved visualizations to {viz_dir}")

    logger.info("Evaluation complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
