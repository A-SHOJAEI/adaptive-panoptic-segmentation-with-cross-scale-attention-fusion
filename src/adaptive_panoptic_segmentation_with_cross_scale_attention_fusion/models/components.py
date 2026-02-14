"""Custom components for adaptive panoptic segmentation.

This module implements novel components including:
- CrossScaleAttentionFusion: Adaptive attention-based feature fusion
- SceneComplexityEstimator: Estimates object density and scale variance
- BoundaryAwareLoss: Custom loss that over-weights boundary regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SceneComplexityEstimator(nn.Module):
    """Estimates scene complexity metrics from feature maps.

    Computes object density and scale variance as indicators of scene complexity
    to guide adaptive feature fusion.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128):
        """Initialize scene complexity estimator.

        Args:
            in_channels: Number of input channels from feature map
            hidden_dim: Hidden dimension for complexity prediction network
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Global pooling followed by MLP to predict complexity scores
        self.complexity_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # [object_density, scale_variance]
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate scene complexity from features.

        Args:
            features: Feature tensor of shape (B, C, H, W)

        Returns:
            Complexity scores of shape (B, 2) with [object_density, scale_variance]
        """
        complexity = self.complexity_net(features)
        return complexity


class CrossScaleAttentionFusion(nn.Module):
    """Adaptive cross-scale attention fusion module.

    Dynamically fuses multi-scale features using learned attention weights
    conditioned on scene complexity. This is the core novel component that
    selectively emphasizes fine-grained vs contextual features based on
    local scene statistics.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        num_scales: int = 3,
        use_complexity_conditioning: bool = True
    ):
        """Initialize cross-scale attention fusion module.

        Args:
            in_channels_list: List of input channels for each scale
            out_channels: Output channel dimension
            num_scales: Number of scales to fuse
            use_complexity_conditioning: Whether to condition on scene complexity
        """
        super().__init__()
        self.num_scales = num_scales
        self.out_channels = out_channels
        self.use_complexity_conditioning = use_complexity_conditioning

        # Project each scale to common dimension
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])

        # Scene complexity estimator (operates on finest scale)
        if use_complexity_conditioning:
            self.complexity_estimator = SceneComplexityEstimator(
                in_channels_list[0], hidden_dim=128
            )

        # Attention weight generator conditioned on complexity
        attention_input_dim = out_channels
        if use_complexity_conditioning:
            attention_input_dim += 2  # Add complexity features

        self.attention_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(attention_input_dim, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, num_scales),
            nn.Softmax(dim=1)
        )

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fuse multi-scale features with adaptive attention.

        Args:
            multi_scale_features: List of feature tensors at different scales
                Each tensor has shape (B, C_i, H_i, W_i)
            target_size: Target spatial size for upsampling, uses largest if None

        Returns:
            Tuple of (fused_features, aux_outputs) where:
                - fused_features: Fused output of shape (B, out_channels, H, W)
                - aux_outputs: Dictionary with 'attention_weights' and optionally 'complexity'
        """
        assert len(multi_scale_features) == self.num_scales

        batch_size = multi_scale_features[0].size(0)
        if target_size is None:
            target_size = multi_scale_features[0].shape[2:]

        # Project all scales to common dimension and resize
        projected_features = []
        for feat, proj in zip(multi_scale_features, self.scale_projections):
            proj_feat = proj(feat)
            if proj_feat.shape[2:] != target_size:
                proj_feat = F.interpolate(
                    proj_feat, size=target_size,
                    mode='bilinear', align_corners=False
                )
            projected_features.append(proj_feat)

        # Stack features: (B, num_scales, C, H, W)
        stacked_features = torch.stack(projected_features, dim=1)

        # Estimate scene complexity from finest scale
        aux_outputs = {}
        if self.use_complexity_conditioning:
            complexity_scores = self.complexity_estimator(multi_scale_features[0])
            aux_outputs['complexity'] = complexity_scores

            # Generate attention weights conditioned on complexity
            # Use average of projected features for attention generation
            pooled_features = F.adaptive_avg_pool2d(
                projected_features[0], 1
            ).flatten(1)
            attention_input = torch.cat([pooled_features, complexity_scores], dim=1)
        else:
            pooled_features = F.adaptive_avg_pool2d(
                projected_features[0], 1
            ).flatten(1)
            attention_input = pooled_features

        # Generate attention weights: (B, num_scales)
        attention_weights = self.attention_gen(
            attention_input.unsqueeze(-1).unsqueeze(-1)
        )
        aux_outputs['attention_weights'] = attention_weights

        # Apply attention weights to fuse features
        # Reshape attention: (B, num_scales, 1, 1, 1)
        attention_weights = attention_weights.view(
            batch_size, self.num_scales, 1, 1, 1
        )

        # Weighted sum: (B, C, H, W)
        fused_features = (stacked_features * attention_weights).sum(dim=1)

        return fused_features, aux_outputs


class BoundaryAwareLoss(nn.Module):
    """Custom boundary-aware loss for panoptic segmentation.

    Over-weights semantically ambiguous regions (object boundaries) to improve
    segmentation quality at edges. Combines semantic segmentation loss with
    instance segmentation loss and boundary enhancement.
    """

    def __init__(
        self,
        num_classes: int,
        boundary_weight: float = 2.0,
        semantic_weight: float = 1.0,
        instance_weight: float = 1.0,
        ignore_index: int = 255
    ):
        """Initialize boundary-aware loss.

        Args:
            num_classes: Number of semantic classes
            boundary_weight: Weight multiplier for boundary pixels
            semantic_weight: Weight for semantic segmentation loss
            instance_weight: Weight for instance segmentation loss
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.num_classes = num_classes
        self.boundary_weight = boundary_weight
        self.semantic_weight = semantic_weight
        self.instance_weight = instance_weight
        self.ignore_index = ignore_index

        # Sobel filters for boundary detection
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )

    def detect_boundaries(self, semantic_labels: torch.Tensor) -> torch.Tensor:
        """Detect boundary pixels from semantic labels using gradient.

        Args:
            semantic_labels: Semantic label map of shape (B, H, W)

        Returns:
            Binary boundary mask of shape (B, H, W)
        """
        # Convert to float for gradient computation
        labels_float = semantic_labels.float().unsqueeze(1)

        # Apply Sobel filters
        grad_x = F.conv2d(labels_float, self.sobel_x, padding=1)
        grad_y = F.conv2d(labels_float, self.sobel_y, padding=1)

        # Compute gradient magnitude
        boundary_mask = (grad_x ** 2 + grad_y ** 2).sqrt().squeeze(1)

        # Threshold to create binary mask
        boundary_mask = (boundary_mask > 0.1).float()

        return boundary_mask

    def forward(
        self,
        semantic_logits: torch.Tensor,
        instance_logits: Optional[torch.Tensor],
        semantic_labels: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute boundary-aware panoptic segmentation loss.

        Args:
            semantic_logits: Predicted semantic logits of shape (B, num_classes, H, W)
            instance_logits: Predicted instance embeddings of shape (B, embed_dim, H, W)
            semantic_labels: Ground truth semantic labels of shape (B, H, W)
            instance_labels: Ground truth instance labels of shape (B, H, W)

        Returns:
            Dictionary with 'total_loss', 'semantic_loss', 'instance_loss', 'boundary_loss'
        """
        batch_size = semantic_logits.size(0)

        # Detect boundaries from ground truth labels
        boundary_mask = self.detect_boundaries(semantic_labels)

        # Create pixel weights (higher for boundaries)
        pixel_weights = torch.ones_like(semantic_labels, dtype=torch.float32)
        pixel_weights = pixel_weights + boundary_mask * (self.boundary_weight - 1.0)

        # Ignore invalid pixels
        valid_mask = (semantic_labels != self.ignore_index).float()
        pixel_weights = pixel_weights * valid_mask

        # Semantic segmentation loss with boundary weighting
        semantic_loss = F.cross_entropy(
            semantic_logits,
            semantic_labels,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        semantic_loss = (semantic_loss * pixel_weights).sum() / (pixel_weights.sum() + 1e-8)

        # Instance segmentation loss (discriminative loss for embeddings)
        instance_loss = torch.tensor(0.0, device=semantic_logits.device)
        if instance_logits is not None and instance_labels is not None:
            instance_loss = self._discriminative_loss(
                instance_logits, instance_labels, boundary_mask
            )

        # Total loss
        total_loss = (
            self.semantic_weight * semantic_loss +
            self.instance_weight * instance_loss
        )

        return {
            'total_loss': total_loss,
            'semantic_loss': semantic_loss,
            'instance_loss': instance_loss,
            'boundary_weight_mean': pixel_weights.mean()
        }

    def _discriminative_loss(
        self,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor,
        boundary_mask: torch.Tensor,
        delta_v: float = 0.5,
        delta_d: float = 1.5
    ) -> torch.Tensor:
        """Discriminative loss for instance embeddings.

        Encourages embeddings of same instance to be close (variance loss)
        and embeddings of different instances to be far (distance loss).

        Args:
            embeddings: Instance embeddings of shape (B, embed_dim, H, W)
            instance_labels: Instance labels of shape (B, H, W)
            boundary_mask: Boundary mask of shape (B, H, W)
            delta_v: Variance margin
            delta_d: Distance margin

        Returns:
            Discriminative loss value
        """
        batch_size, embed_dim, height, width = embeddings.shape
        embeddings = embeddings.permute(0, 2, 3, 1).contiguous()
        embeddings = embeddings.view(batch_size, height * width, embed_dim)
        instance_labels_flat = instance_labels.view(batch_size, height * width)

        loss_var_list = []
        loss_dist_list = []

        for b in range(batch_size):
            embed = embeddings[b]
            labels = instance_labels_flat[b]

            unique_labels = labels.unique()
            unique_labels = unique_labels[unique_labels != 0]  # Exclude background

            if len(unique_labels) == 0:
                continue

            # Compute cluster centers
            centers = []
            for label in unique_labels:
                mask = (labels == label)
                if mask.sum() == 0:
                    continue
                center = embed[mask].mean(dim=0)
                centers.append(center)

                # Variance loss: pull embeddings toward center
                var_loss = torch.clamp(
                    (embed[mask] - center).norm(dim=1) - delta_v, min=0.0
                ).pow(2).mean()
                loss_var_list.append(var_loss)

            # Distance loss: push centers apart
            if len(centers) > 1:
                centers_tensor = torch.stack(centers)
                num_centers = centers_tensor.size(0)

                # Pairwise distances
                centers_expanded_1 = centers_tensor.unsqueeze(1).expand(
                    num_centers, num_centers, embed_dim
                )
                centers_expanded_2 = centers_tensor.unsqueeze(0).expand(
                    num_centers, num_centers, embed_dim
                )
                dist = (centers_expanded_1 - centers_expanded_2).norm(dim=2)

                # Distance loss (exclude diagonal)
                mask = torch.eye(num_centers, device=dist.device).bool()
                dist_loss = torch.clamp(
                    2 * delta_d - dist[~mask], min=0.0
                ).pow(2).mean()
                loss_dist_list.append(dist_loss)

        loss_var = torch.stack(loss_var_list).mean() if loss_var_list else torch.tensor(0.0, device=embeddings.device)
        loss_dist = torch.stack(loss_dist_list).mean() if loss_dist_list else torch.tensor(0.0, device=embeddings.device)

        return loss_var + loss_dist


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction."""

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256
    ):
        """Initialize FPN.

        Args:
            in_channels_list: List of input channels for each level
            out_channels: Output channels for all levels
        """
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through FPN.

        Args:
            features: List of feature tensors from backbone, from low to high level

        Returns:
            List of FPN feature tensors at multiple scales
        """
        # Apply lateral connections
        laterals = [
            lateral_conv(feat)
            for feat, lateral_conv in zip(features, self.lateral_convs)
        ]

        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='nearest'
            )

        # Apply output convolutions
        outputs = [
            output_conv(lateral)
            for lateral, output_conv in zip(laterals, self.output_convs)
        ]

        return outputs
