# mlx_primitives.advanced

Advanced components: MoE, SSM, and quantization.

## Mixture of Experts (MoE)

| Class | Description |
|-------|-------------|
| `MoELayer` | Full MoE layer with routing |
| `TopKRouter` | Top-K expert routing |
| `ExpertChoiceRouter` | Expert-choice routing |
| `load_balancing_loss` | Auxiliary load balancing loss |

### MoE Example

```python
from mlx_primitives.advanced.moe import MoELayer

moe = MoELayer(
    dims=768,
    num_experts=8,
    top_k=2,
    hidden_dims=3072,
)

result = moe(x)
output = result.output
aux_loss = result.aux_loss  # Add to training loss
```

## State Space Models (SSM)

| Class | Description |
|-------|-------------|
| `MambaBlock` | Single Mamba block |
| `Mamba` | Full Mamba model |
| `S4Layer` | S4 layer |
| `H3Block` | H3 block |

### Mamba Example

```python
from mlx_primitives.advanced.ssm import MambaBlock

block = MambaBlock(
    dims=768,
    state_dims=16,
    expand=2,
)

out = block(x)
```

## Quantization

| Class | Description |
|-------|-------------|
| `QuantizedLinear` | Weight-only quantized linear |
| `Int4Linear` | INT4 quantization |
| `AWQLinear` | AWQ quantization |
| `GPTQLinear` | GPTQ quantization |

## Module Contents

::: mlx_primitives.advanced
    options:
      show_root_heading: false
      members_order: source
      show_source: true
