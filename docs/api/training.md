# mlx_primitives.training

Training utilities and infrastructure.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| `checkpoint` | Gradient checkpointing for memory efficiency |
| `checkpoint_sequential` | Checkpoint sequential modules |
| `Trainer` | Full training loop |
| `EMA` | Exponential moving average |
| `CosineAnnealingLR` | Cosine learning rate scheduler |

## Learning Rate Schedulers

- `CosineAnnealingLR` - Cosine annealing with optional warmup
- `WarmupCosineScheduler` - Linear warmup + cosine decay
- `OneCycleLR` - One-cycle policy
- `PolynomialDecayLR` - Polynomial decay

## Callbacks

- `EarlyStopping` - Stop training on plateau
- `ModelCheckpoint` - Save best/periodic checkpoints
- `LRMonitor` - Log learning rate
- `GradientMonitor` - Track gradient statistics

## Module Contents

::: mlx_primitives.training
    options:
      show_root_heading: false
      members_order: source
      show_source: true
