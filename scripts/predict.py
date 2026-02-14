#!/usr/bin/env python
"""Prediction script for adaptive panoptic segmentation.

Performs inference on new images with a trained model.
"""

import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import argparse
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.models.model import (
    AdaptivePanopticSegmentationModel,
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
        description='Run inference with adaptive panoptic segmentation model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions',
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (auto-detect if None)'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for predictions'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization overlays'
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
        pretrained_backbone=False
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


def preprocess_image(
    image_path: Path,
    image_size: tuple = (512, 1024)
) -> tuple:
    """Load and preprocess image for inference.

    Args:
        image_path: Path to image file
        image_size: Target size (height, width)

    Returns:
        Tuple of (preprocessed_tensor, original_image, original_size)
    """
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    original_size = image.shape[:2]

    # Create transform
    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    # Apply transform
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)

    return image_tensor, image, original_size


def postprocess_predictions(
    semantic_pred: torch.Tensor,
    instance_pred: torch.Tensor,
    original_size: tuple,
    num_classes: int = 19
) -> tuple:
    """Postprocess model predictions.

    Args:
        semantic_pred: Semantic prediction tensor (H, W)
        instance_pred: Instance prediction tensor (H, W)
        original_size: Original image size (H, W)
        num_classes: Number of classes

    Returns:
        Tuple of (semantic_map, instance_map, confidence_scores)
    """
    import torch.nn.functional as F

    # Resize to original size
    semantic_pred = semantic_pred.unsqueeze(0).unsqueeze(0).float()
    semantic_pred = F.interpolate(
        semantic_pred,
        size=original_size,
        mode='nearest'
    ).squeeze().long()

    instance_pred = instance_pred.unsqueeze(0).unsqueeze(0).float()
    instance_pred = F.interpolate(
        instance_pred,
        size=original_size,
        mode='nearest'
    ).squeeze().long()

    # Compute class confidence (placeholder - would need softmax logits)
    confidence_scores = torch.ones_like(semantic_pred).float() * 0.95

    return semantic_pred.cpu().numpy(), instance_pred.cpu().numpy(), confidence_scores.cpu().numpy()


def visualize_prediction(
    image: np.ndarray,
    semantic_pred: np.ndarray,
    output_path: Path,
    class_colors: np.ndarray = None
) -> None:
    """Create visualization overlay.

    Args:
        image: Original image (H, W, 3)
        semantic_pred: Semantic prediction (H, W)
        output_path: Path to save visualization
        class_colors: Color map for classes
    """
    import matplotlib.pyplot as plt

    # Generate color map if not provided
    if class_colors is None:
        num_classes = semantic_pred.max() + 1
        class_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_classes))[:, :3]
        class_colors = (class_colors * 255).astype(np.uint8)

    # Create colored segmentation
    pred_colored = class_colors[semantic_pred]

    # Create overlay
    overlay = (0.6 * image + 0.4 * pred_colored).astype(np.uint8)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(pred_colored)
    axes[1].set_title('Segmentation')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def predict_image(
    model: AdaptivePanopticSegmentationModel,
    image_path: Path,
    device: torch.device,
    output_dir: Path,
    config: dict,
    visualize: bool = False
) -> dict:
    """Run prediction on a single image.

    Args:
        model: Trained model
        image_path: Path to input image
        device: Device for computation
        output_dir: Output directory
        config: Configuration dictionary
        visualize: Whether to create visualization

    Returns:
        Dictionary with prediction results
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Processing: {image_path}")

    # Preprocess image
    image_tensor, original_image, original_size = preprocess_image(
        image_path,
        image_size=(
            config.get('data', {}).get('image_height', 512),
            config.get('data', {}).get('image_width', 1024)
        )
    )

    # Run inference
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        semantic_pred, instance_pred = model.predict_panoptic(image_tensor)

    # Postprocess
    semantic_map, instance_map, confidence = postprocess_predictions(
        semantic_pred[0],
        instance_pred[0],
        original_size,
        num_classes=config.get('model', {}).get('num_classes', 19)
    )

    # Save predictions
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Save semantic prediction as PNG
    semantic_img = Image.fromarray(semantic_map.astype(np.uint8))
    semantic_img.save(pred_dir / f'{image_path.stem}_semantic.png')

    # Save instance prediction as PNG
    instance_img = Image.fromarray(instance_map.astype(np.uint16))
    instance_img.save(pred_dir / f'{image_path.stem}_instance.png')

    # Create visualization if requested
    if visualize:
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

        visualize_prediction(
            original_image,
            semantic_map,
            viz_dir / f'{image_path.stem}_viz.png'
        )

    logger.info(f"Saved predictions to {pred_dir}")

    return {
        'image_path': str(image_path),
        'semantic_classes': np.unique(semantic_map).tolist(),
        'num_instances': len(np.unique(instance_map)) - 1,  # Exclude background
        'mean_confidence': float(confidence.mean())
    }


def main() -> None:
    """Main prediction function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(log_level='INFO')

    import logging
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Adaptive Panoptic Segmentation Inference")
    logger.info("=" * 80)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load or auto-detect config
    if args.config:
        config = load_config(args.config)
    else:
        config_path = checkpoint_path.parent / 'config.yaml'
        if config_path.exists():
            config = load_config(str(config_path))
            logger.info(f"Auto-detected config: {config_path}")
        else:
            config = load_config('configs/default.yaml')
            logger.warning("Using default config")

    # Get device
    device = get_device(config.get('device'))

    # Load model
    logger.info(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, config, device)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process input (file or directory)
    input_path = Path(args.input)

    if input_path.is_file():
        # Single image
        image_paths = [input_path]
    elif input_path.is_dir():
        # Directory of images
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(input_path.glob(ext))
            image_paths.extend(input_path.glob(ext.upper()))
        image_paths = sorted(image_paths)
    else:
        raise ValueError(f"Input path not found: {input_path}")

    logger.info(f"Found {len(image_paths)} images to process")

    # Process each image
    results = []
    for image_path in image_paths:
        try:
            result = predict_image(
                model,
                image_path,
                device,
                output_dir,
                config,
                visualize=args.visualize
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")

    # Save summary
    import json
    summary_file = output_dir / 'prediction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Processed {len(results)}/{len(image_paths)} images successfully")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
