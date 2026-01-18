# Getting Started with MLX Primitives

This guide covers the basics of using MLX Primitives for building ML models on Apple Silicon.

## Installation

```bash
pip install mlx-primitives
```

Requirements:
- Python >= 3.10
- MLX >= 0.20.0
- Apple Silicon Mac (M1/M2/M3/M4)

## Basic Usage

### Flash Attention

The most common use case is memory-efficient attention:

```python
import mlx.core as mx
from mlx_primitives.attention import FlashAttention

# Create attention layer
attn = FlashAttention(
    dims=768,        # Model dimension
    num_heads=12,    # Number of attention heads
    causal=True,     # Use causal (autoregressive) masking
)

# Forward pass
x = mx.random.normal((2, 128, 768))  # (batch, seq, dims)
output = attn(x)
print(output.shape)  # (2, 128, 768)
```

### Grouped Query Attention (GQA)

For LLaMA-style attention with fewer KV heads:

```python
from mlx_primitives.attention import GroupedQueryAttention

# 12 query heads, 4 KV heads (3 Q heads share each KV head)
gqa = GroupedQueryAttention(
    dims=768,
    num_heads=12,
    num_kv_heads=4,
    causal=True,
)

x = mx.random.normal((2, 128, 768))
output, cache = gqa(x)  # Returns (output, kv_cache)
```

### Rotary Position Embeddings (RoPE)

Apply rotary embeddings to query and key tensors:

```python
from mlx_primitives.attention import RoPE

rope = RoPE(
    dims=64,           # Head dimension
    max_seq_len=2048,  # Maximum sequence length
)

# Apply to Q and K
q = mx.random.normal((2, 128, 12, 64))  # (batch, seq, heads, head_dim)
k = mx.random.normal((2, 128, 12, 64))
q_rot, k_rot = rope(q, k)
```

## Layers

### Normalization

```python
from mlx_primitives.layers import RMSNorm, GroupNorm

# RMSNorm (used in LLaMA, Mistral)
norm = RMSNorm(dims=768)
x = mx.random.normal((2, 128, 768))
y = norm(x)

# GroupNorm
gnorm = GroupNorm(num_groups=8, num_channels=256)
x = mx.random.normal((2, 32, 32, 256))  # NHWC format
y = gnorm(x)
```

### Activations

```python
from mlx_primitives.layers import SwiGLU, GeGLU, Mish

# SwiGLU (used in LLaMA, Mistral)
swiglu = SwiGLU(in_features=768, hidden_features=2048)
x = mx.random.normal((2, 128, 768))
y = swiglu(x)  # (2, 128, 768)

# Mish activation
mish = Mish()
y = mish(x)
```

## Training

### Basic Training Loop

```python
from mlx_primitives.training import (
    Trainer,
    TrainingConfig,
    CosineAnnealingLR,
    EarlyStopping,
    ModelCheckpoint,
)

# Create config
config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10,
)

# Create trainer with callbacks
trainer = Trainer(
    model=my_model,
    config=config,
    scheduler=CosineAnnealingLR(lr=1e-4, T_max=1000),
    callbacks=[
        EarlyStopping(patience=3),
        ModelCheckpoint(save_dir="checkpoints"),
    ],
)

# Train
trainer.fit(train_loader, val_loader)
```

### EMA (Exponential Moving Average)

```python
from mlx_primitives.training import EMA

ema = EMA(model, decay=0.999)

# During training
for batch in loader:
    loss = train_step(batch)
    ema.update()

# For evaluation, use EMA weights
with ema.apply_to_model():
    eval_loss = eval_step(val_batch)
```

## Advanced Features

### Mixture of Experts (MoE)

```python
from mlx_primitives.advanced import MoELayer, TopKRouter, Expert

# Create experts
experts = [Expert(dims=768, hidden_dims=2048) for _ in range(8)]

# Create router (top-2 routing)
router = TopKRouter(dims=768, num_experts=8, top_k=2)

# Create MoE layer
moe = MoELayer(router=router, experts=experts)

x = mx.random.normal((2, 128, 768))
output, aux_loss = moe(x)  # aux_loss for load balancing
```

### State Space Models (Mamba)

```python
from mlx_primitives.advanced import MambaBlock, Mamba

# Single Mamba block
block = MambaBlock(dims=768, d_state=16)
x = mx.random.normal((2, 128, 768))
y = block(x)

# Full Mamba model
mamba = Mamba(
    dims=768,
    depth=12,
    d_state=16,
)
```

### KV Cache for Inference

```python
from mlx_primitives.advanced import SlidingWindowCache

# Create sliding window cache
cache = SlidingWindowCache(
    max_seq_len=4096,
    window_size=512,
    batch_size=1,
    num_heads=12,
    head_dim=64,
)

# Use with attention
for token in tokens:
    # ... compute q, k, v
    k, v = cache.update(k, v)
    # ... compute attention
```

## Configuration

### Precision Control

Control numerical precision for performance vs accuracy:

```python
from mlx_primitives.config import (
    PrecisionMode,
    set_precision_mode,
    precision_context,
)

# Force float16 for maximum performance
set_precision_mode(PrecisionMode.FLOAT16)

# Or use context manager for temporary override
with precision_context(mode=PrecisionMode.FLOAT32):
    # This will use float32
    output = attention(q, k, v)
```

## Next Steps

- Check out the [examples](../examples/) for complete working code
- Read the [API reference](api/attention.md) for detailed documentation
- See the [precision guide](guides/precision.md) for performance tuning
