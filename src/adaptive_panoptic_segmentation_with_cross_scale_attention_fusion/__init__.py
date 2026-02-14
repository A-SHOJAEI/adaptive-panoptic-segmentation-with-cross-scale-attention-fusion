"""Adaptive Panoptic Segmentation with Cross-Scale Attention Fusion.

A panoptic segmentation framework that dynamically fuses multi-scale features
using learned attention weights conditioned on scene complexity metrics.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.models.model import (
    AdaptivePanopticSegmentationModel,
)
from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.models.components import (
    BoundaryAwareLoss,
    CrossScaleAttentionFusion,
)

__all__ = [
    "AdaptivePanopticSegmentationModel",
    "BoundaryAwareLoss",
    "CrossScaleAttentionFusion",
]
