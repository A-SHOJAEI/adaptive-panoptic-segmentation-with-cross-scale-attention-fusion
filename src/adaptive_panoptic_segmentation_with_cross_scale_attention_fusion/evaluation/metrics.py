"""Evaluation metrics for panoptic segmentation.

Implements panoptic quality (PQ), segmentation quality (SQ), recognition quality (RQ),
and boundary F1 score for comprehensive evaluation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute Intersection over Union (IoU) for each class.

    Args:
        pred: Predicted segmentation of shape (N,)
        target: Ground truth segmentation of shape (N,)
        num_classes: Number of classes

    Returns:
        IoU tensor of shape (num_classes,)
    """
    ious = torch.zeros(num_classes, device=pred.device)

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            ious[cls] = intersection / union
        else:
            ious[cls] = float('nan')

    return ious


def compute_boundary_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundary_width: int = 2
) -> float:
    """Compute boundary F1 score.

    Measures segmentation quality at object boundaries by checking if predicted
    boundaries align with ground truth boundaries within a certain width.

    Args:
        pred: Predicted segmentation of shape (H, W)
        target: Ground truth segmentation of shape (H, W)
        boundary_width: Width threshold for boundary matching

    Returns:
        Boundary F1 score
    """
    import torch.nn.functional as F

    # Convert to float for gradient computation
    pred_float = pred.float().unsqueeze(0).unsqueeze(0)
    target_float = target.float().unsqueeze(0).unsqueeze(0)

    # Sobel filters for edge detection
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1, 1, 3, 3)

    # Compute boundaries
    pred_grad_x = F.conv2d(pred_float, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred_float, sobel_y, padding=1)
    pred_boundary = ((pred_grad_x ** 2 + pred_grad_y ** 2).sqrt() > 0.1).squeeze()

    target_grad_x = F.conv2d(target_float, sobel_x, padding=1)
    target_grad_y = F.conv2d(target_float, sobel_y, padding=1)
    target_boundary = ((target_grad_x ** 2 + target_grad_y ** 2).sqrt() > 0.1).squeeze()

    # Dilate boundaries to allow for small misalignments
    if boundary_width > 1:
        kernel = torch.ones(
            1, 1, boundary_width * 2 + 1, boundary_width * 2 + 1,
            device=pred.device
        )
        pred_boundary_dilated = F.conv2d(
            pred_boundary.float().unsqueeze(0).unsqueeze(0),
            kernel,
            padding=boundary_width
        ).squeeze() > 0
        target_boundary_dilated = F.conv2d(
            target_boundary.float().unsqueeze(0).unsqueeze(0),
            kernel,
            padding=boundary_width
        ).squeeze() > 0
    else:
        pred_boundary_dilated = pred_boundary
        target_boundary_dilated = target_boundary

    # Compute precision and recall
    true_positive = (pred_boundary & target_boundary_dilated).sum().float()
    false_positive = (pred_boundary & ~target_boundary_dilated).sum().float()
    false_negative = (target_boundary & ~pred_boundary_dilated).sum().float()

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1.item()


class PanopticQualityMetric:
    """Panoptic Quality (PQ) metric for panoptic segmentation evaluation.

    PQ = SQ × RQ where:
    - SQ (Segmentation Quality): average IoU of matched segments
    - RQ (Recognition Quality): F1 score of segment detection
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        thing_classes: Optional[list] = None
    ):
        """Initialize PQ metric.

        Args:
            num_classes: Number of semantic classes
            ignore_index: Label index to ignore
            thing_classes: List of class indices that are thing classes
                          (have instances). If None, assumes 11-18 for Cityscapes.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Default thing classes for Cityscapes (11-18: person, rider, car, truck, bus, train, motorcycle, bicycle)
        if thing_classes is None:
            self.thing_classes = set(range(11, 19))
        else:
            self.thing_classes = set(thing_classes)

        self.stuff_classes = set(range(num_classes)) - self.thing_classes

        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.tp = defaultdict(int)  # True positives per class
        self.fp = defaultdict(int)  # False positives per class
        self.fn = defaultdict(int)  # False negatives per class
        self.iou_sum = defaultdict(float)  # Sum of IoUs for TP per class

    def update(
        self,
        pred_semantic: torch.Tensor,
        pred_instance: torch.Tensor,
        target_semantic: torch.Tensor,
        target_instance: torch.Tensor
    ) -> None:
        """Update metric with batch predictions.

        Args:
            pred_semantic: Predicted semantic labels (B, H, W)
            pred_instance: Predicted instance IDs (B, H, W)
            target_semantic: Ground truth semantic labels (B, H, W)
            target_instance: Ground truth instance IDs (B, H, W)
        """
        batch_size = pred_semantic.size(0)

        for b in range(batch_size):
            self._update_single(
                pred_semantic[b],
                pred_instance[b],
                target_semantic[b],
                target_instance[b]
            )

    def _update_single(
        self,
        pred_sem: torch.Tensor,
        pred_inst: torch.Tensor,
        target_sem: torch.Tensor,
        target_inst: torch.Tensor
    ) -> None:
        """Update metric for a single image.

        Args:
            pred_sem: Predicted semantic labels (H, W)
            pred_inst: Predicted instance IDs (H, W)
            target_sem: Ground truth semantic labels (H, W)
            target_inst: Ground truth instance IDs (H, W)
        """
        # Filter out ignore pixels
        valid_mask = target_sem != self.ignore_index
        pred_sem = pred_sem[valid_mask]
        pred_inst = pred_inst[valid_mask]
        target_sem = target_sem[valid_mask]
        target_inst = target_inst[valid_mask]

        # Process stuff classes (semantic only)
        for cls in self.stuff_classes:
            pred_mask = (pred_sem == cls)
            target_mask = (target_sem == cls)

            if target_mask.sum() == 0 and pred_mask.sum() == 0:
                continue

            if target_mask.sum() == 0:
                self.fp[cls] += 1
                continue

            if pred_mask.sum() == 0:
                self.fn[cls] += 1
                continue

            # Compute IoU
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            iou = (intersection / union).item()

            if iou > 0.5:
                self.tp[cls] += 1
                self.iou_sum[cls] += iou
            else:
                self.fp[cls] += 1
                self.fn[cls] += 1

        # Process thing classes (semantic + instance)
        for cls in self.thing_classes:
            pred_cls_mask = (pred_sem == cls)
            target_cls_mask = (target_sem == cls)

            if target_cls_mask.sum() == 0 and pred_cls_mask.sum() == 0:
                continue

            # Get instance IDs for this class
            pred_instances = pred_inst[pred_cls_mask].unique()
            target_instances = target_inst[target_cls_mask].unique()

            # Remove background (0)
            pred_instances = pred_instances[pred_instances != 0]
            target_instances = target_instances[target_instances != 0]

            # Match instances using IoU
            matched_target = set()

            for pred_id in pred_instances:
                pred_inst_mask = pred_cls_mask & (pred_inst == pred_id)
                best_iou = 0.0
                best_target_id = None

                for target_id in target_instances:
                    if target_id in matched_target:
                        continue

                    target_inst_mask = target_cls_mask & (target_inst == target_id)

                    intersection = (pred_inst_mask & target_inst_mask).sum().float()
                    union = (pred_inst_mask | target_inst_mask).sum().float()
                    iou = (intersection / union).item()

                    if iou > best_iou:
                        best_iou = iou
                        best_target_id = target_id

                # Check if match is valid (IoU > 0.5)
                if best_iou > 0.5:
                    self.tp[cls] += 1
                    self.iou_sum[cls] += best_iou
                    matched_target.add(best_target_id)
                else:
                    self.fp[cls] += 1

            # Unmatched ground truth instances are false negatives
            self.fn[cls] += len(target_instances) - len(matched_target)

    def compute(self) -> Dict[str, float]:
        """Compute final PQ metrics.

        Returns:
            Dictionary with 'panoptic_quality', 'segmentation_quality',
            'recognition_quality', and per-category metrics
        """
        pq_per_class = {}
        sq_per_class = {}
        rq_per_class = {}

        pq_sum = 0.0
        sq_sum = 0.0
        rq_sum = 0.0
        num_classes_computed = 0

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            tp = self.tp.get(cls, 0)
            fp = self.fp.get(cls, 0)
            fn = self.fn.get(cls, 0)
            iou_sum = self.iou_sum.get(cls, 0.0)

            # Skip classes with no predictions or ground truth
            if tp + fp + fn == 0:
                continue

            # Compute SQ (average IoU of matched segments)
            sq = iou_sum / tp if tp > 0 else 0.0

            # Compute RQ (F1 of segment detection)
            rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0

            # Compute PQ
            pq = sq * rq

            pq_per_class[cls] = pq
            sq_per_class[cls] = sq
            rq_per_class[cls] = rq

            pq_sum += pq
            sq_sum += sq
            rq_sum += rq
            num_classes_computed += 1

        # Compute averages
        if num_classes_computed > 0:
            avg_pq = pq_sum / num_classes_computed
            avg_sq = sq_sum / num_classes_computed
            avg_rq = rq_sum / num_classes_computed
        else:
            avg_pq = 0.0
            avg_sq = 0.0
            avg_rq = 0.0

        return {
            'panoptic_quality': avg_pq,
            'segmentation_quality': avg_sq,
            'recognition_quality': avg_rq,
            'pq_per_class': pq_per_class,
            'sq_per_class': sq_per_class,
            'rq_per_class': rq_per_class,
            'num_classes': num_classes_computed
        }
