"""Tests for model components and architecture."""

import pytest
import torch

from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.models.model import (
    AdaptivePanopticSegmentationModel,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.models.components import (
    CrossScaleAttentionFusion,
    SceneComplexityEstimator,
    BoundaryAwareLoss,
    FeaturePyramidNetwork,
)


class TestSceneComplexityEstimator:
    """Test SceneComplexityEstimator."""

    def test_forward_pass(self, batch_size: int, device: torch.device):
        """Test forward pass through complexity estimator."""
        estimator = SceneComplexityEstimator(in_channels=256, hidden_dim=128).to(device)

        features = torch.randn(batch_size, 256, 32, 64).to(device)
        complexity = estimator(features)

        assert complexity.shape == (batch_size, 2)
        assert (complexity >= 0).all() and (complexity <= 1).all()


class TestCrossScaleAttentionFusion:
    """Test CrossScaleAttentionFusion module."""

    def test_forward_pass(self, batch_size: int, device: torch.device):
        """Test forward pass through attention fusion."""
        fusion = CrossScaleAttentionFusion(
            in_channels_list=[256, 256, 256],
            out_channels=256,
            num_scales=3,
            use_complexity_conditioning=True
        ).to(device)

        # Create multi-scale features
        features = [
            torch.randn(batch_size, 256, 64, 128).to(device),
            torch.randn(batch_size, 256, 32, 64).to(device),
            torch.randn(batch_size, 256, 16, 32).to(device),
        ]

        fused, aux = fusion(features, target_size=(64, 128))

        assert fused.shape == (batch_size, 256, 64, 128)
        assert 'attention_weights' in aux
        assert aux['attention_weights'].shape == (batch_size, 3)
        assert 'complexity' in aux

    def test_without_complexity_conditioning(self, batch_size: int, device: torch.device):
        """Test fusion without complexity conditioning."""
        fusion = CrossScaleAttentionFusion(
            in_channels_list=[256, 256, 256],
            out_channels=256,
            num_scales=3,
            use_complexity_conditioning=False
        ).to(device)

        features = [
            torch.randn(batch_size, 256, 64, 128).to(device),
            torch.randn(batch_size, 256, 32, 64).to(device),
            torch.randn(batch_size, 256, 16, 32).to(device),
        ]

        fused, aux = fusion(features)

        assert fused.shape == (batch_size, 256, 64, 128)
        assert 'complexity' not in aux


class TestBoundaryAwareLoss:
    """Test BoundaryAwareLoss."""

    def test_loss_computation(self, batch_size: int, num_classes: int, device: torch.device):
        """Test loss computation."""
        loss_fn = BoundaryAwareLoss(
            num_classes=num_classes,
            boundary_weight=2.0,
            semantic_weight=1.0,
            instance_weight=0.5
        ).to(device)

        semantic_logits = torch.randn(batch_size, num_classes, 128, 256).to(device)
        instance_logits = torch.randn(batch_size, 64, 128, 256).to(device)
        semantic_labels = torch.randint(0, num_classes, (batch_size, 128, 256)).to(device)
        instance_labels = torch.randint(0, 10, (batch_size, 128, 256)).to(device)

        losses = loss_fn(
            semantic_logits,
            instance_logits,
            semantic_labels,
            instance_labels
        )

        assert 'total_loss' in losses
        assert 'semantic_loss' in losses
        assert 'instance_loss' in losses
        assert losses['total_loss'].item() >= 0

    def test_boundary_detection(self, num_classes: int, device: torch.device):
        """Test boundary detection."""
        loss_fn = BoundaryAwareLoss(num_classes=num_classes).to(device)

        # Create simple label map with clear boundary
        labels = torch.zeros(2, 64, 64, dtype=torch.long).to(device)
        labels[:, :, 32:] = 1

        boundaries = loss_fn.detect_boundaries(labels)

        assert boundaries.shape == (2, 64, 64)
        assert boundaries.max() <= 1.0
        # Should detect boundary at column 32
        assert boundaries[:, :, 31:34].sum() > 0


class TestFeaturePyramidNetwork:
    """Test FeaturePyramidNetwork."""

    def test_fpn_forward(self, batch_size: int, device: torch.device):
        """Test FPN forward pass."""
        fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        ).to(device)

        features = [
            torch.randn(batch_size, 256, 64, 128).to(device),
            torch.randn(batch_size, 512, 32, 64).to(device),
            torch.randn(batch_size, 1024, 16, 32).to(device),
            torch.randn(batch_size, 2048, 8, 16).to(device),
        ]

        outputs = fpn(features)

        assert len(outputs) == 4
        for i, output in enumerate(outputs):
            assert output.shape[1] == 256  # All outputs have same channels


class TestAdaptivePanopticSegmentationModel:
    """Test main model."""

    def test_model_creation(self, num_classes: int):
        """Test model instantiation."""
        model = AdaptivePanopticSegmentationModel(
            num_classes=num_classes,
            embed_dim=64,
            fpn_channels=256,
            use_complexity_conditioning=True,
            pretrained_backbone=False
        )

        assert model.num_classes == num_classes
        assert model.embed_dim == 64

    def test_forward_pass(
        self,
        sample_image: torch.Tensor,
        num_classes: int,
        device: torch.device
    ):
        """Test forward pass through model."""
        model = AdaptivePanopticSegmentationModel(
            num_classes=num_classes,
            embed_dim=64,
            fpn_channels=256,
            pretrained_backbone=False
        ).to(device)

        sample_image = sample_image.to(device)
        outputs = model(sample_image, return_aux=True)

        batch_size, _, height, width = sample_image.shape

        assert 'semantic_logits' in outputs
        assert 'instance_embeddings' in outputs
        assert outputs['semantic_logits'].shape == (batch_size, num_classes, height, width)
        assert outputs['instance_embeddings'].shape == (batch_size, 64, height, width)

    def test_loss_computation(
        self,
        sample_image: torch.Tensor,
        sample_semantic_mask: torch.Tensor,
        sample_instance_mask: torch.Tensor,
        num_classes: int,
        device: torch.device
    ):
        """Test loss computation."""
        model = AdaptivePanopticSegmentationModel(
            num_classes=num_classes,
            pretrained_backbone=False
        ).to(device)

        sample_image = sample_image.to(device)
        sample_semantic_mask = sample_semantic_mask.to(device)
        sample_instance_mask = sample_instance_mask.to(device)

        predictions = model(sample_image)
        losses = model.get_loss(predictions, sample_semantic_mask, sample_instance_mask)

        assert 'total_loss' in losses
        assert losses['total_loss'].item() >= 0

    def test_predict_panoptic(
        self,
        sample_image: torch.Tensor,
        num_classes: int,
        device: torch.device
    ):
        """Test panoptic prediction."""
        model = AdaptivePanopticSegmentationModel(
            num_classes=num_classes,
            pretrained_backbone=False
        ).to(device)

        sample_image = sample_image.to(device)
        semantic_preds, instance_preds = model.predict_panoptic(sample_image)

        batch_size, _, height, width = sample_image.shape

        assert semantic_preds.shape == (batch_size, height, width)
        assert instance_preds.shape == (batch_size, height, width)
        assert semantic_preds.max() < num_classes

    def test_model_with_different_configs(self, device: torch.device):
        """Test model with different configurations."""
        configs = [
            {'use_complexity_conditioning': True},
            {'use_complexity_conditioning': False},
            {'embed_dim': 32},
            {'embed_dim': 128},
        ]

        for config in configs:
            model = AdaptivePanopticSegmentationModel(
                num_classes=19,
                pretrained_backbone=False,
                **config
            ).to(device)

            x = torch.randn(1, 3, 256, 512).to(device)
            outputs = model(x)

            assert 'semantic_logits' in outputs
            assert 'instance_embeddings' in outputs
