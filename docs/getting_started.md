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
from mlx_primitives import FlashAttention, flash_attention

# Create attention layer
attn = FlashAttention(
    num_heads=12,    # Number of attention heads
    head_dim=64,     # Dimension per head
    causal=True,     # Use causal (autoregressive) masking
)

# Input tensors: (batch, seq, heads, head_dim)
q = mx.random.normal((2, 128, 12, 64))
k = mx.random.normal((2, 128, 12, 64))
v = mx.random.normal((2, 128, 12, 64))

# Forward pass - O(n) memory complexity
output = attn(q, k, v)
print(output.shape)  # (2, 128, 12, 64)

# Or use functional API
output = flash_attention(q, k, v, causal=True)
```

### Sliding Window Attention

For fixed context window attention (Mistral-style):

```python
from mlx_primitives import SlidingWindowAttention

# 512-token sliding window
attn = SlidingWindowAttention(
    num_heads=12,
    head_dim=64,
    window_size=512,
    causal=True,
)

q = mx.random.normal((2, 2048, 12, 64))
k = mx.random.normal((2, 2048, 12, 64))
v = mx.random.normal((2, 2048, 12, 64))

output = attn(q, k, v)
```

### Chunked Cross-Attention

Memory-efficient cross-attention for long KV sequences:

```python
from mlx_primitives import ChunkedCrossAttention

attn = ChunkedCrossAttention(
    num_heads=12,
    head_dim=64,
    chunk_size=256,  # Process KV in chunks
)

# Query sequence (shorter)
q = mx.random.normal((2, 128, 12, 64))
# Key/Value sequence (longer)
k = mx.random.normal((2, 4096, 12, 64))
v = mx.random.normal((2, 4096, 12, 64))

output = attn(q, k, v)
```

## Fused Kernels

### SwiGLU / GeGLU Activations

```python
from mlx_primitives.kernels import SwiGLU, GeGLU

# SwiGLU (used in LLaMA, Mistral)
swiglu = SwiGLU(in_features=768, hidden_features=2048)
x = mx.random.normal((2, 128, 768))
y = swiglu(x)  # (2, 128, 768)

# GeGLU
geglu = GeGLU(in_features=768, hidden_features=2048)
y = geglu(x)
```

### INT8/INT4 Quantization

```python
from mlx_primitives.kernels import quantize_int8, quantize_int4, dequantize_int8

# Quantize weights
weights = mx.random.normal((1024, 768))
q_weights, scales = quantize_int8(weights)

# Use in forward pass (dequantize on the fly)
x = mx.random.normal((2, 128, 768))
output = dequantize_int8(q_weights, scales) @ x.T
```

## Training Utilities

### Gradient Checkpointing

Reduce memory usage during training by recomputing activations:

```python
from mlx_primitives import checkpoint, checkpoint_sequential

def transformer_block(x):
    # Your transformer block implementation
    return x

# Checkpoint a single function
x = mx.random.normal((2, 128, 768))
output = checkpoint(transformer_block, x)

# Checkpoint a sequence of layers
layers = [layer1, layer2, layer3, layer4]
output = checkpoint_sequential(layers, x, segments=2)
```

## Core Primitives

### Parallel Scan (for SSMs)

```python
from mlx_primitives import associative_scan, selective_scan

# Simple cumulative sum
x = mx.random.normal((2, 128, 64))
cumsum = associative_scan(x, operator="add", axis=1)

# SSM-style scan: h[t] = A[t] * h[t-1] + x[t]
A = mx.random.uniform(shape=(2, 128, 64))  # Decay factors
x = mx.random.normal((2, 128, 64))
hidden_states = associative_scan(x, operator="ssm", A=A, axis=1)
```

### Mixture of Experts (MoE)

```python
from mlx_primitives import SparseMoELayer, build_expert_dispatch

# Create MoE layer
moe = SparseMoELayer(
    dims=768,
    num_experts=8,
    top_k=2,
    hidden_dims=2048,
)

x = mx.random.normal((2, 128, 768))
output, aux_loss = moe(x)  # aux_loss for load balancing
```

## KV Cache (Submodule)

```python
from mlx_primitives.cache import SimpleKVCache

# Create cache
cache = SimpleKVCache(
    batch_size=1,
    num_heads=12,
    head_dim=64,
    max_seq_len=2048,
)

# Use during generation
for step in range(max_tokens):
    # ... compute q, k, v for current token
    k_cached, v_cached = cache.update(k, v)
    # ... compute attention with cached KV
```

## Generation Engine (Submodule)

```python
from mlx_primitives.generation import GenerationEngine, EngineConfig

# Define your model's forward function
def model_forward(input_ids, attention_mask):
    return model(input_ids, attention_mask)

# Create generation engine
engine = GenerationEngine(
    model_forward_fn=model_forward,
    config=EngineConfig(vocab_size=32000, eos_token_id=2),
)

# Submit generation requests
req = engine.submit(input_ids, max_new_tokens=100)

# Generate tokens with streaming
for step_output in engine.generate_stream():
    print(step_output)
```

## Hardware Detection (Submodule)

```python
from mlx_primitives.hardware import get_chip_info

# Get current chip information
info = get_chip_info()
print(f"Chip: {info.name}")
print(f"GPU Cores: {info.gpu_cores}")
print(f"L2 Cache: {info.l2_cache_mb} MB")
```

## Next Steps

- Check out the [examples](../examples/) for complete working code
- Read the API source code for detailed documentation
- Run benchmarks with `python -m benchmarks.runner`
