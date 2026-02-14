"""Pytest configuration and fixtures for testing."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def device() -> torch.device:
    """Get device for testing (CPU to avoid CUDA issues in CI)."""
    return torch.device('cpu')


@pytest.fixture
def batch_size() -> int:
    """Batch size for testing."""
    return 2


@pytest.fixture
def num_classes() -> int:
    """Number of classes for testing."""
    return 19


@pytest.fixture
def image_size() -> tuple:
    """Image size for testing (H, W)."""
    return (256, 512)


@pytest.fixture
def sample_image(batch_size: int, image_size: tuple) -> torch.Tensor:
    """Generate sample image tensor.

    Args:
        batch_size: Batch size
        image_size: Image size (H, W)

    Returns:
        Random image tensor of shape (B, 3, H, W)
    """
    return torch.randn(batch_size, 3, image_size[0], image_size[1])


@pytest.fixture
def sample_semantic_mask(batch_size: int, num_classes: int, image_size: tuple) -> torch.Tensor:
    """Generate sample semantic mask.

    Args:
        batch_size: Batch size
        num_classes: Number of classes
        image_size: Image size (H, W)

    Returns:
        Random semantic mask of shape (B, H, W)
    """
    return torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))


@pytest.fixture
def sample_instance_mask(batch_size: int, image_size: tuple) -> torch.Tensor:
    """Generate sample instance mask.

    Args:
        batch_size: Batch size
        image_size: Image size (H, W)

    Returns:
        Random instance mask of shape (B, H, W)
    """
    return torch.randint(0, 10, (batch_size, image_size[0], image_size[1]))


@pytest.fixture
def sample_config() -> dict:
    """Generate sample configuration dictionary."""
    return {
        'num_epochs': 2,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'gradient_clip': 5.0,
        'use_amp': False,  # Disable AMP for testing
        'early_stop_patience': 5,
        'num_classes': 19,
        'boundary_weight': 2.0,
        'semantic_weight': 1.0,
        'instance_weight': 0.5,
        'use_curriculum': False,
        'curriculum_stages': 1,
        'curriculum_epochs_per_stage': 1,
    }


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary checkpoint directory.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to temporary checkpoint directory
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir
