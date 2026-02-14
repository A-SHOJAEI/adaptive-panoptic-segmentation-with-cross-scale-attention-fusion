#!/usr/bin/env python
"""Training script for adaptive panoptic segmentation with curriculum learning.

This script implements the complete training pipeline including:
- Configuration management via YAML
- Data loading with synthetic/real Cityscapes data
- Model training with curriculum learning
- Learning rate scheduling and early stopping
- Checkpoint saving and MLflow tracking
- Mixed precision training
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
from datetime import datetime

from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.models.model import (
    AdaptivePanopticSegmentationModel,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.data.loader import (
    get_cityscapes_dataloaders,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.training.trainer import (
    PanopticSegmentationTrainer,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.utils.config import (
    load_config,
    save_config,
    setup_logging,
    set_random_seeds,
    get_device,
    count_parameters,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.evaluation.analysis import (
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train adaptive panoptic segmentation model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Skip training (for testing)'
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        log_level=config.get('logging', {}).get('log_level', 'INFO'),
        log_file=config.get('logging', {}).get('log_file')
    )

    import logging
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Adaptive Panoptic Segmentation Training")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")

    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    set_random_seeds(seed)

    # Get device
    device = get_device(config.get('device'))

    # Initialize MLflow (wrapped in try/except)
    use_mlflow = config.get('logging', {}).get('use_mlflow', False)
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(
                config.get('logging', {}).get('mlflow_experiment_name', 'adaptive_panoptic')
            )
            mlflow.start_run()
            mlflow.log_params(config)
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            use_mlflow = False

    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = get_cityscapes_dataloaders(
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

        logger.info(
            f"Data: train={len(train_loader)} batches, "
            f"val={len(val_loader)} batches, "
            f"test={len(test_loader)} batches"
        )

        # Create model
        logger.info("Creating model...")
        model = AdaptivePanopticSegmentationModel(
            num_classes=config.get('model', {}).get('num_classes', 19),
            embed_dim=config.get('model', {}).get('embed_dim', 64),
            fpn_channels=config.get('model', {}).get('fpn_channels', 256),
            use_complexity_conditioning=config.get('model', {}).get('use_complexity_conditioning', True),
            pretrained_backbone=config.get('model', {}).get('pretrained_backbone', True)
        )

        num_params = count_parameters(model)
        logger.info(f"Model created with {num_params:,} trainable parameters")

        # Create checkpoint directory
        checkpoint_dir = Path(config.get('checkpoint', {}).get('save_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration to checkpoint directory
        save_config(config, checkpoint_dir / 'config.yaml')

        # Flatten config for trainer
        flat_config = {
            'num_epochs': config.get('training', {}).get('num_epochs', 100),
            'learning_rate': config.get('training', {}).get('learning_rate', 0.0001),
            'weight_decay': config.get('training', {}).get('weight_decay', 0.0001),
            'optimizer': config.get('training', {}).get('optimizer', 'adamw'),
            'lr_scheduler': config.get('training', {}).get('lr_scheduler', 'cosine'),
            'lr_step_size': config.get('training', {}).get('lr_step_size', 30),
            'lr_gamma': config.get('training', {}).get('lr_gamma', 0.1),
            'gradient_clip': config.get('training', {}).get('gradient_clip', 5.0),
            'use_amp': config.get('training', {}).get('use_amp', True),
            'early_stop_patience': config.get('early_stopping', {}).get('patience', 15),
            'num_classes': config.get('model', {}).get('num_classes', 19),
            'boundary_weight': config.get('loss', {}).get('boundary_weight', 2.0),
            'semantic_weight': config.get('loss', {}).get('semantic_weight', 1.0),
            'instance_weight': config.get('loss', {}).get('instance_weight', 0.5),
            'use_curriculum': config.get('curriculum', {}).get('use_curriculum', True),
            'curriculum_stages': config.get('curriculum', {}).get('curriculum_stages', 3),
            'curriculum_epochs_per_stage': config.get('curriculum', {}).get('curriculum_epochs_per_stage', 10),
        }

        # Create trainer
        logger.info("Creating trainer...")
        trainer = PanopticSegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=flat_config,
            device=device,
            checkpoint_dir=checkpoint_dir
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(Path(args.resume))

        # Train model
        if not args.no_train:
            logger.info("Starting training...")
            history = trainer.train()

            # Save training history
            results_dir = Path('results')
            results_dir.mkdir(parents=True, exist_ok=True)

            history_file = results_dir / f'training_history_{datetime.now():%Y%m%d_%H%M%S}.json'
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Saved training history to {history_file}")

            # Plot training history
            plot_training_history(history, save_dir=results_dir)

            # Log final metrics to MLflow
            if use_mlflow:
                try:
                    mlflow.log_metrics({
                        'final_train_loss': history['train_loss'][-1],
                        'final_val_loss': history['val_loss'][-1],
                        'best_val_loss': trainer.best_val_loss,
                        'final_val_pq': history['val_pq'][-1],
                    })
                    mlflow.log_artifact(str(history_file))
                except Exception as e:
                    logger.warning(f"Failed to log to MLflow: {e}")

            logger.info("Training complete!")
            logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
            logger.info(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")

        else:
            logger.info("Skipping training (--no-train flag)")

    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        raise

    finally:
        # End MLflow run
        if use_mlflow:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()
