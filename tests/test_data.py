"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np
from pathlib import Path

from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.data.loader import (
    CityscapesDataset,
    get_cityscapes_dataloaders,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    CurriculumAugmentation,
    denormalize_image,
)


class TestCityscapesDataset:
    """Test CityscapesDataset class."""

    def test_synthetic_dataset_creation(self):
        """Test creating dataset with synthetic data."""
        dataset = CityscapesDataset(
            root_dir=None,
            split='train',
            transform=None,
            use_synthetic=True,
            synthetic_samples=100
        )

        assert len(dataset) == 100
        assert not dataset.has_real_data

    def test_synthetic_sample_generation(self):
        """Test generating synthetic samples."""
        dataset = CityscapesDataset(
            root_dir=None,
            split='train',
            transform=None,
            use_synthetic=True,
            synthetic_samples=10
        )

        sample = dataset[0]

        assert 'image' in sample
        assert 'semantic_mask' in sample
        assert 'instance_mask' in sample

        assert sample['image'].shape[0] == 3  # RGB
        assert sample['semantic_mask'].dim() == 2
        assert sample['instance_mask'].dim() == 2

    def test_dataset_with_transforms(self):
        """Test dataset with augmentation transforms."""
        transform = get_train_transforms(image_size=(256, 512))
        dataset = CityscapesDataset(
            root_dir=None,
            split='train',
            transform=transform,
            use_synthetic=True,
            synthetic_samples=10
        )

        sample = dataset[0]

        # Check output shapes after transforms
        assert sample['image'].shape == (3, 256, 512)
        assert sample['semantic_mask'].shape == (256, 512)
        assert sample['instance_mask'].shape == (256, 512)

    def test_dataset_reproducibility(self):
        """Test that synthetic samples are reproducible with same index."""
        dataset1 = CityscapesDataset(
            root_dir=None,
            split='train',
            transform=None,
            use_synthetic=True,
            synthetic_samples=10
        )

        dataset2 = CityscapesDataset(
            root_dir=None,
            split='train',
            transform=None,
            use_synthetic=True,
            synthetic_samples=10
        )

        sample1 = dataset1[0]
        sample2 = dataset2[0]

        # Samples with same index should be identical
        assert torch.allclose(sample1['image'], sample2['image'])
        assert torch.equal(sample1['semantic_mask'], sample2['semantic_mask'])


class TestDataLoaders:
    """Test dataloader creation."""

    def test_get_dataloaders(self):
        """Test creating train/val/test dataloaders."""
        train_loader, val_loader, test_loader = get_cityscapes_dataloaders(
            root_dir=None,
            batch_size=2,
            num_workers=0,
            image_size=(256, 512),
            use_synthetic=True,
            pin_memory=False
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_dataloader_batch(self):
        """Test loading a batch from dataloader."""
        train_loader, _, _ = get_cityscapes_dataloaders(
            root_dir=None,
            batch_size=4,
            num_workers=0,
            image_size=(256, 512),
            use_synthetic=True,
            pin_memory=False
        )

        batch = next(iter(train_loader))

        assert batch['image'].shape == (4, 3, 256, 512)
        assert batch['semantic_mask'].shape == (4, 256, 512)
        assert batch['instance_mask'].shape == (4, 256, 512)


class TestTransforms:
    """Test augmentation transforms."""

    def test_train_transforms(self):
        """Test training transforms."""
        transform = get_train_transforms(image_size=(256, 512), augmentation_strength='medium')

        image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        semantic_mask = np.random.randint(0, 19, (512, 1024), dtype=np.int64)
        instance_mask = np.random.randint(0, 10, (512, 1024), dtype=np.int64)

        transformed = transform(
            image=image,
            semantic_mask=semantic_mask,
            instance_mask=instance_mask
        )

        assert transformed['image'].shape == (3, 256, 512)
        assert transformed['semantic_mask'].shape == (256, 512)
        assert transformed['instance_mask'].shape == (256, 512)

    def test_val_transforms(self):
        """Test validation transforms."""
        transform = get_val_transforms(image_size=(256, 512))

        image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        semantic_mask = np.random.randint(0, 19, (512, 1024), dtype=np.int64)
        instance_mask = np.random.randint(0, 10, (512, 1024), dtype=np.int64)

        transformed = transform(
            image=image,
            semantic_mask=semantic_mask,
            instance_mask=instance_mask
        )

        assert transformed['image'].shape == (3, 256, 512)

    def test_curriculum_augmentation(self):
        """Test curriculum augmentation."""
        curriculum = CurriculumAugmentation(
            image_size=(256, 512),
            num_stages=3,
            stage_epochs=5
        )

        assert curriculum.current_stage == 0

        # Advance to stage 1
        curriculum.update_epoch(5)
        assert curriculum.current_stage == 1

        # Advance to stage 2
        curriculum.update_epoch(10)
        assert curriculum.current_stage == 2

        # Stay at stage 2 (max stage)
        curriculum.update_epoch(20)
        assert curriculum.current_stage == 2

    def test_denormalize_image(self):
        """Test image denormalization."""
        # Create normalized tensor
        normalized = torch.randn(3, 256, 512)

        # Denormalize
        denorm = denormalize_image(normalized.numpy())

        assert denorm.shape == (256, 512, 3)
        assert denorm.dtype == np.uint8
        assert denorm.min() >= 0
        assert denorm.max() <= 255
