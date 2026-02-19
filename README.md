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

## Training Results

> **Note on data**: The results below were obtained using **synthetic data** (800 training / 150 validation / 150 test samples) because real Cityscapes data was not available at training time. Performance on real Cityscapes data may differ significantly. The synthetic data is procedurally generated and does not capture the full complexity of real-world street scenes.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 3090 (25.30 GB) |
| Total Parameters | 28,049,432 |
| Optimizer | AdamW (lr=0.0001, weight\_decay=0.0001) |
| LR Scheduler | Cosine annealing |
| Batch Size | 4 |
| Image Size | 512 x 1024 |
| Mixed Precision | Yes (AMP) |
| Curriculum Learning | 3 stages, 10 epochs each |
| Early Stopping Patience | 15 epochs |
| Epochs Completed | 83 / 100 (early stopping triggered) |
| Total Training Time | 0.98 hours |

### Final Metrics

| Metric | Value |
|--------|-------|
| Best Validation Loss | 0.0094 (epoch 68) |
| Best Panoptic Quality (PQ) | 0.7475 (epoch 78) |
| Final Validation Loss (epoch 83) | 0.0096 |
| Final Val PQ (epoch 83) | 0.7470 |
| Final Train Loss (epoch 83) | 0.0143 |

### Training Progression

| Epoch | Train Loss | Val Loss | Val PQ | Learning Rate |
|-------|-----------|----------|--------|---------------|
| 1 | 0.8164 | 0.1284 | 0.7328 | 1.00e-04 |
| 10 | 0.0435 | 0.0294 | 0.7445 | 9.80e-05 |
| 20 | 0.0341 | 0.0204 | 0.7445 | 9.14e-05 |
| 30 | 0.0277 | 0.0186 | 0.7466 | 8.08e-05 |
| 40 | 0.0242 | 0.0130 | 0.7466 | 6.73e-05 |
| 50 | 0.0188 | 0.0118 | 0.7468 | 5.21e-05 |
| 60 | 0.0171 | 0.0145 | 0.7468 | 3.52e-05 |
| 70 | 0.0161 | 0.0103 | 0.7471 | 2.12e-05 |
| 80 | 0.0150 | 0.0108 | 0.7471 | 1.00e-05 |
| 83 | 0.0143 | 0.0096 | 0.7470 | 8.71e-06 |

Training was stopped early at epoch 83 (patience 15) as validation loss did not improve beyond the best value of 0.0094 achieved at epoch 68. The model showed rapid initial convergence (train loss dropped from 0.8164 to 0.0435 in the first 10 epochs) followed by gradual refinement. PQ scores on synthetic data plateaued around 0.747, with the best PQ of 0.7475 observed at epoch 78.

### Observations

- **Rapid early convergence**: The bulk of loss reduction occurred in the first 10 epochs, with train loss dropping ~95% from 0.8164 to 0.0435.
- **PQ plateau on synthetic data**: Panoptic Quality stabilized in a narrow band (0.744--0.748) after epoch 10, suggesting the synthetic data's limited complexity was largely captured early.
- **Occasional PQ dips**: Epochs 4 and 8 showed PQ drops to ~0.59, likely due to curriculum stage transitions introducing harder augmentations.
- **Cosine LR schedule**: The learning rate decayed smoothly from 1.0e-4 to 8.7e-6 over 83 epochs, which helped fine-tune in later stages without overshooting.

## Dataset

The model is designed for the Cityscapes dataset. Set `data.root_dir` in config to your Cityscapes path. If real data is not available, synthetic data is generated automatically (`data.use_synthetic: true`).

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
