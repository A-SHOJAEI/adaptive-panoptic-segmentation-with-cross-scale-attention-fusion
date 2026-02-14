"""Tests for training pipeline."""

import pytest
import torch
from pathlib import Path

from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.models.model import (
    AdaptivePanopticSegmentationModel,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.training.trainer import (
    PanopticSegmentationTrainer,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.data.loader import (
    get_cityscapes_dataloaders,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.evaluation.metrics import (
    PanopticQualityMetric,
    compute_iou,
    compute_boundary_f1,
)


class TestTrainer:
    """Test training pipeline."""

    @pytest.fixture
    def small_dataloaders(self):
        """Create small dataloaders for testing."""
        return get_cityscapes_dataloaders(
            root_dir=None,
            batch_size=2,
            num_workers=0,
            image_size=(128, 256),
            use_synthetic=True,
            pin_memory=False
        )

    def test_trainer_creation(
        self,
        num_classes: int,
        sample_config: dict,
        device: torch.device,
        small_dataloaders: tuple,
        temp_checkpoint_dir: Path
    ):
        """Test trainer instantiation."""
        train_loader, val_loader, _ = small_dataloaders

        model = AdaptivePanopticSegmentationModel(
            num_classes=num_classes,
            pretrained_backbone=False
        )

        trainer = PanopticSegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sample_config,
            device=device,
            checkpoint_dir=temp_checkpoint_dir
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_train_epoch(
        self,
        num_classes: int,
        sample_config: dict,
        device: torch.device,
        small_dataloaders: tuple,
        temp_checkpoint_dir: Path
    ):
        """Test training for one epoch."""
        train_loader, val_loader, _ = small_dataloaders

        model = AdaptivePanopticSegmentationModel(
            num_classes=num_classes,
            pretrained_backbone=False
        )

        trainer = PanopticSegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sample_config,
            device=device,
            checkpoint_dir=temp_checkpoint_dir
        )

        metrics = trainer.train_epoch(epoch=1)

        assert 'train_loss' in metrics
        assert 'train_semantic_loss' in metrics
        assert metrics['train_loss'] >= 0

    def test_validate(
        self,
        num_classes: int,
        sample_config: dict,
        device: torch.device,
        small_dataloaders: tuple,
        temp_checkpoint_dir: Path
    ):
        """Test validation."""
        train_loader, val_loader, _ = small_dataloaders

        model = AdaptivePanopticSegmentationModel(
            num_classes=num_classes,
            pretrained_backbone=False
        )

        trainer = PanopticSegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sample_config,
            device=device,
            checkpoint_dir=temp_checkpoint_dir
        )

        metrics = trainer.validate(epoch=1)

        assert 'val_loss' in metrics
        assert 'val_semantic_loss' in metrics
        assert 'val_pq' in metrics
        assert metrics['val_loss'] >= 0

    def test_checkpoint_save_load(
        self,
        num_classes: int,
        sample_config: dict,
        device: torch.device,
        small_dataloaders: tuple,
        temp_checkpoint_dir: Path
    ):
        """Test checkpoint saving and loading."""
        train_loader, val_loader, _ = small_dataloaders

        model = AdaptivePanopticSegmentationModel(
            num_classes=num_classes,
            pretrained_backbone=False
        )

        trainer = PanopticSegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sample_config,
            device=device,
            checkpoint_dir=temp_checkpoint_dir
        )

        # Save checkpoint
        trainer.save_checkpoint(epoch=1, is_best=True)

        checkpoint_path = temp_checkpoint_dir / 'best_model.pt'
        assert checkpoint_path.exists()

        # Load checkpoint
        epoch = trainer.load_checkpoint(checkpoint_path)
        assert epoch == 1

    def test_full_training_loop(
        self,
        num_classes: int,
        device: torch.device,
        small_dataloaders: tuple,
        temp_checkpoint_dir: Path
    ):
        """Test full training loop for a few epochs."""
        train_loader, val_loader, _ = small_dataloaders

        model = AdaptivePanopticSegmentationModel(
            num_classes=num_classes,
            pretrained_backbone=False
        )

        # Short training config
        config = {
            'num_epochs': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'adamw',
            'lr_scheduler': 'cosine',
            'gradient_clip': 5.0,
            'use_amp': False,
            'early_stop_patience': 10,
            'num_classes': num_classes,
            'boundary_weight': 2.0,
            'semantic_weight': 1.0,
            'instance_weight': 0.5,
            'use_curriculum': False,
            'curriculum_stages': 1,
            'curriculum_epochs_per_stage': 1,
        }

        trainer = PanopticSegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=temp_checkpoint_dir
        )

        history = trainer.train()

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 2


class TestMetrics:
    """Test evaluation metrics."""

    def test_compute_iou(self, num_classes: int, device: torch.device):
        """Test IoU computation."""
        pred = torch.randint(0, num_classes, (100,)).to(device)
        target = torch.randint(0, num_classes, (100,)).to(device)

        ious = compute_iou(pred, target, num_classes)

        assert ious.shape == (num_classes,)
        assert (ious[~torch.isnan(ious)] >= 0).all()
        assert (ious[~torch.isnan(ious)] <= 1).all()

    def test_boundary_f1(self, device: torch.device):
        """Test boundary F1 score."""
        # Create simple segmentation with clear boundary
        pred = torch.zeros(64, 64, dtype=torch.long).to(device)
        pred[:, 32:] = 1

        target = torch.zeros(64, 64, dtype=torch.long).to(device)
        target[:, 30:] = 1

        f1 = compute_boundary_f1(pred, target, boundary_width=2)

        assert 0 <= f1 <= 1

    def test_panoptic_quality_metric(self, num_classes: int, device: torch.device):
        """Test PQ metric computation."""
        pq_metric = PanopticQualityMetric(num_classes=num_classes)

        # Create dummy predictions and targets
        batch_size = 2
        height, width = 64, 128

        pred_semantic = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
        pred_instance = torch.randint(0, 5, (batch_size, height, width)).to(device)
        target_semantic = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
        target_instance = torch.randint(0, 5, (batch_size, height, width)).to(device)

        pq_metric.update(pred_semantic, pred_instance, target_semantic, target_instance)

        results = pq_metric.compute()

        assert 'panoptic_quality' in results
        assert 'segmentation_quality' in results
        assert 'recognition_quality' in results
        assert 0 <= results['panoptic_quality'] <= 1

    def test_pq_metric_reset(self, num_classes: int):
        """Test PQ metric reset."""
        pq_metric = PanopticQualityMetric(num_classes=num_classes)

        # Add some data
        pred = torch.randint(0, num_classes, (2, 64, 64))
        target = torch.randint(0, num_classes, (2, 64, 64))
        pq_metric.update(pred, pred, target, target)

        # Reset
        pq_metric.reset()

        # Should have empty state
        assert len(pq_metric.tp) == 0
        assert len(pq_metric.fp) == 0
        assert len(pq_metric.fn) == 0
