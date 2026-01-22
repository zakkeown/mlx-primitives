# mlx_primitives.generation

Batched text generation engine.

## Quick Reference

| Class | Description |
|-------|-------------|
| `GenerationEngine` | Full generation pipeline |
| `GenerationRequest` | Request object with metadata |
| `SamplingConfig` | Temperature, top-k, top-p settings |
| `TokenSampler` | Configurable token sampler |
| `SequenceBatcher` | Variable-length sequence batching |

## Usage Example

```python
from mlx_primitives.generation import GenerationEngine, SamplingConfig

config = SamplingConfig(
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)

engine = GenerationEngine(model, tokenizer)
output = engine.generate(prompt, config=config, max_tokens=100)
```

## Module Contents

::: mlx_primitives.generation
    options:
      show_root_heading: false
      members_order: source
      show_source: true
