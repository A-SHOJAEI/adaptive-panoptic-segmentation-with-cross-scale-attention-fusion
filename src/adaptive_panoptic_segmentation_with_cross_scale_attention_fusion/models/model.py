"""Core adaptive panoptic segmentation model implementation.

Combines semantic and instance segmentation with adaptive cross-scale attention
fusion conditioned on scene complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, List, Tuple, Optional
import logging

from adaptive_panoptic_segmentation_with_cross_scale_attention_fusion.models.components import (
    CrossScaleAttentionFusion,
    BoundaryAwareLoss,
    FeaturePyramidNetwork,
)

logger = logging.getLogger(__name__)


class AdaptivePanopticSegmentationModel(nn.Module):
    """Adaptive panoptic segmentation model with cross-scale attention fusion.

    Novel features:
    1. Adaptive cross-scale attention fusion conditioned on scene complexity
    2. Boundary-aware loss for improved edge segmentation
    3. Curriculum learning compatible architecture
    """

    def __init__(
        self,
        num_classes: int = 19,
        embed_dim: int = 64,
        fpn_channels: int = 256,
        use_complexity_conditioning: bool = True,
        pretrained_backbone: bool = True
    ):
        """Initialize adaptive panoptic segmentation model.

        Args:
            num_classes: Number of semantic classes
            embed_dim: Embedding dimension for instance segmentation
            fpn_channels: Number of channels in FPN outputs
            use_complexity_conditioning: Whether to use scene complexity conditioning
            pretrained_backbone: Whether to use pretrained ResNet backbone
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.fpn_channels = fpn_channels

        # ResNet50 backbone
        if pretrained_backbone:
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        backbone = resnet50(weights=weights)

        # Extract feature layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 256 channels, stride 4
        self.layer2 = backbone.layer2  # 512 channels, stride 8
        self.layer3 = backbone.layer3  # 1024 channels, stride 16
        self.layer4 = backbone.layer4  # 2048 channels, stride 32

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_channels
        )

        # Adaptive cross-scale attention fusion (novel component)
        self.cross_scale_fusion = CrossScaleAttentionFusion(
            in_channels_list=[fpn_channels] * 3,  # Use top 3 FPN levels
            out_channels=fpn_channels,
            num_scales=3,
            use_complexity_conditioning=use_complexity_conditioning
        )

        # Semantic segmentation head
        self.semantic_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
        )

        # Instance segmentation head (embedding prediction)
        self.instance_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels // 2, embed_dim, kernel_size=1),
            nn.Tanh()  # Normalize embeddings to [-1, 1]
        )

        # Initialize weights
        self._initialize_weights()

        logger.info(f"Initialized AdaptivePanopticSegmentationModel with {num_classes} classes")

    def _initialize_weights(self) -> None:
        """Initialize model weights using kaiming initialization."""
        for m in [self.semantic_head, self.instance_head]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input images of shape (B, 3, H, W)
            return_aux: Whether to return auxiliary outputs (attention weights, complexity)

        Returns:
            Dictionary containing:
                - semantic_logits: Semantic segmentation logits (B, num_classes, H, W)
                - instance_embeddings: Instance embeddings (B, embed_dim, H, W)
                - attention_weights: Attention weights if return_aux=True
                - complexity: Scene complexity scores if return_aux=True
        """
        batch_size, _, input_h, input_w = x.shape

        # Backbone forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)  # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        # FPN forward pass
        fpn_features = self.fpn([c2, c3, c4, c5])

        # Adaptive cross-scale attention fusion (novel component)
        # Use top 3 FPN levels (highest resolution)
        multi_scale_features = fpn_features[:3]
        fused_features, aux_outputs = self.cross_scale_fusion(
            multi_scale_features,
            target_size=(input_h // 4, input_w // 4)
        )

        # Semantic segmentation head
        semantic_logits = self.semantic_head(fused_features)
        semantic_logits = F.interpolate(
            semantic_logits,
            size=(input_h, input_w),
            mode='bilinear',
            align_corners=False
        )

        # Instance segmentation head
        instance_embeddings = self.instance_head(fused_features)
        instance_embeddings = F.interpolate(
            instance_embeddings,
            size=(input_h, input_w),
            mode='bilinear',
            align_corners=False
        )

        outputs = {
            'semantic_logits': semantic_logits,
            'instance_embeddings': instance_embeddings,
        }

        if return_aux:
            outputs.update(aux_outputs)

        return outputs

    def get_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        semantic_labels: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None,
        loss_fn: Optional[BoundaryAwareLoss] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for training.

        Args:
            predictions: Model predictions from forward()
            semantic_labels: Ground truth semantic labels (B, H, W)
            instance_labels: Ground truth instance labels (B, H, W)
            loss_fn: Custom loss function, creates default if None

        Returns:
            Dictionary with loss values
        """
        if loss_fn is None:
            loss_fn = BoundaryAwareLoss(
                num_classes=self.num_classes,
                boundary_weight=2.0,
                semantic_weight=1.0,
                instance_weight=0.5
            )
            loss_fn = loss_fn.to(predictions['semantic_logits'].device)

        losses = loss_fn(
            semantic_logits=predictions['semantic_logits'],
            instance_logits=predictions['instance_embeddings'],
            semantic_labels=semantic_labels,
            instance_labels=instance_labels
        )

        return losses

    def predict_panoptic(
        self,
        x: torch.Tensor,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate panoptic segmentation predictions.

        Args:
            x: Input images of shape (B, 3, H, W)
            score_threshold: Confidence threshold for predictions
            nms_threshold: IoU threshold for instance NMS

        Returns:
            Tuple of (semantic_predictions, instance_predictions)
                - semantic_predictions: (B, H, W) with class indices
                - instance_predictions: (B, H, W) with instance IDs
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, return_aux=False)

            # Semantic predictions
            semantic_probs = F.softmax(outputs['semantic_logits'], dim=1)
            semantic_preds = semantic_probs.argmax(dim=1)

            # Instance predictions (cluster embeddings)
            instance_embeddings = outputs['instance_embeddings']
            instance_preds = self._cluster_embeddings(
                instance_embeddings,
                semantic_preds,
                score_threshold=score_threshold
            )

        return semantic_preds, instance_preds

    def _cluster_embeddings(
        self,
        embeddings: torch.Tensor,
        semantic_preds: torch.Tensor,
        score_threshold: float = 0.5,
        bandwidth: float = 0.5
    ) -> torch.Tensor:
        """Cluster instance embeddings using simple heuristic.

        Args:
            embeddings: Instance embeddings (B, embed_dim, H, W)
            semantic_preds: Semantic predictions (B, H, W)
            score_threshold: Unused in this simple version
            bandwidth: Distance threshold for clustering

        Returns:
            Instance IDs of shape (B, H, W)
        """
        batch_size, embed_dim, height, width = embeddings.shape
        embeddings = embeddings.permute(0, 2, 3, 1).contiguous()
        embeddings = embeddings.view(batch_size, height, width, embed_dim)

        instance_preds = torch.zeros(
            batch_size, height, width,
            dtype=torch.long,
            device=embeddings.device
        )

        # Simple per-image clustering
        for b in range(batch_size):
            embed = embeddings[b].view(-1, embed_dim)
            sem = semantic_preds[b].view(-1)

            # Get thing classes (classes with instances)
            # Assume classes 11-18 are thing classes for Cityscapes
            thing_mask = (sem >= 11) & (sem <= 18)

            if thing_mask.sum() == 0:
                continue

            # Simple distance-based clustering for thing pixels
            thing_embed = embed[thing_mask]
            instance_ids = torch.zeros(thing_mask.sum(), dtype=torch.long, device=embed.device)

            current_id = 1
            remaining = torch.ones(thing_embed.size(0), dtype=torch.bool, device=embed.device)

            while remaining.any():
                # Pick a seed
                seed_idx = remaining.nonzero()[0]
                seed_embed = thing_embed[seed_idx]

                # Find nearby embeddings
                distances = (thing_embed - seed_embed).norm(dim=1)
                cluster_mask = (distances < bandwidth) & remaining

                # Assign instance ID
                instance_ids[cluster_mask] = current_id
                remaining[cluster_mask] = False
                current_id += 1

            # Map back to full image
            full_instance_ids = torch.zeros_like(sem)
            full_instance_ids[thing_mask] = instance_ids
            instance_preds[b] = full_instance_ids.view(height, width)

        return instance_preds
