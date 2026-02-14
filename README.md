# Adaptive Panoptic Segmentation with Cross-Scale Attention Fusion

A panoptic segmentation framework that dynamically fuses multi-scale features using learned attention weights conditioned on scene complexity metrics. Combines semantic and instance segmentation with an adaptive cross-scale attention module that selectively emphasizes fine-grained vs. contextual features based on local scene statistics (object density, scale variance).

## Key Features

- Adaptive cross-scale attention fusion conditioned on scene complexity
- Boundary-aware loss function for improved edge segmentation
- Curriculum learning with progressive scene complexity
- ResNet50 backbone with Feature Pyramid Network
- Comprehensive evaluation metrics (PQ, SQ, RQ, Boundary F1)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train the full model with adaptive attention fusion:

```bash
python scripts/train.py --config configs/default.yaml
```

Train baseline model without adaptive components (ablation study):

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate a trained model on test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --visualize
```

### Inference

Run inference on new images:

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pt --input path/to/image.jpg --visualize
```

## Architecture

The model consists of:

1. **ResNet50 Backbone**: Pre-trained on ImageNet for feature extraction
2. **Feature Pyramid Network (FPN)**: Multi-scale feature representation
3. **Scene Complexity Estimator**: Predicts object density and scale variance
4. **Cross-Scale Attention Fusion**: Adaptively fuses features based on complexity
5. **Dual Segmentation Heads**: Semantic and instance segmentation branches

### Novel Components

- **Adaptive Attention Fusion**: Learns to weight multi-scale features based on scene complexity rather than using fixed fusion
- **Boundary-Aware Loss**: Over-weights semantically ambiguous boundary regions by 2x to improve edge quality
- **Curriculum Learning**: Progressively introduces augmentation complexity over 3 stages

## Methodology

### Problem Statement

Traditional panoptic segmentation methods use fixed fusion strategies (e.g., element-wise addition, concatenation) to combine multi-scale features. This approach fails to adapt to varying scene characteristics: simple scenes with few large objects benefit from contextual features, while complex crowded scenes require fine-grained details.

### Our Approach

We introduce a **complexity-conditioned attention mechanism** that dynamically determines how to fuse multi-scale features:

1. **Scene Complexity Estimation**: A lightweight CNN predicts two complexity metrics from high-resolution features:
   - **Object Density**: Estimated number of distinct objects in the scene
   - **Scale Variance**: Diversity of object sizes (high variance = multi-scale scene)

2. **Adaptive Fusion**: Instead of fixed weights, we generate attention weights α₁, α₂, α₃ for three feature scales based on complexity scores:
   ```
   F_fused = α₁·F_high + α₂·F_mid + α₃·F_low
   where α = softmax(MLP([complexity_features, pooled_features]))
   ```

   This allows the model to:
   - Emphasize fine-grained features (F_high) when complexity is high
   - Prioritize contextual features (F_low) in simple scenes
   - Balance both in mixed scenes

3. **Boundary-Aware Loss**: We detect semantic boundaries using gradient magnitude on segmentation maps and apply 2× loss weighting to these regions, forcing the network to focus on challenging edge cases.

4. **Curriculum Learning**: Training progresses through 3 stages with increasing augmentation strength, allowing the model to first learn basic patterns before tackling complex variations.

### Key Innovation

Unlike fixed fusion (e.g., FPN's lateral connections), our attention weights are **learned and adaptive**. The ablation study (`configs/ablation.yaml`) disables this by setting `use_complexity_conditioning: false`, reverting to uniform fusion weights. This allows quantifying the contribution of adaptive fusion to overall performance.

## Configuration

Key configuration parameters in `configs/default.yaml`:

```yaml
model:
  num_classes: 19
  use_complexity_conditioning: true  # Enable adaptive fusion

training:
  num_epochs: 100
  learning_rate: 0.0001
  lr_scheduler: cosine

loss:
  boundary_weight: 2.0  # Boundary loss weighting

curriculum:
  use_curriculum: true
  curriculum_stages: 3
```

## Results

Run training to reproduce results:

```bash
python scripts/train.py
```

Expected performance on Cityscapes validation set:

| Metric | Target |
|--------|--------|
| Panoptic Quality (PQ) | 0.62 |
| Segmentation Quality (SQ) | 0.81 |
| Recognition Quality (RQ) | 0.76 |
| Boundary F1 | 0.73 |

## Dataset

The model is designed for Cityscapes dataset. Set `data.root_dir` in config to your Cityscapes path, or use synthetic data for testing by setting `data.use_synthetic: true`.

## Project Structure

```
adaptive-panoptic-segmentation-with-cross-scale-attention-fusion/
├── src/
│   └── adaptive_panoptic_segmentation_with_cross_scale_attention_fusion/
│       ├── models/          # Model architecture and components
│       ├── data/            # Data loading and preprocessing
│       ├── training/        # Training pipeline
│       ├── evaluation/      # Metrics and analysis
│       └── utils/           # Configuration and utilities
├── configs/                 # YAML configuration files
├── scripts/                 # Training, evaluation, and inference scripts
├── tests/                   # Unit tests
└── results/                 # Saved results and visualizations
```

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
