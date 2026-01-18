"""Batched generation utilities for MLX.

This module provides generation infrastructure:
- Variable-length batched generation
- Request scheduling with priority support
- Continuous batching
- Efficient token sampling

Example:
    >>> from mlx_primitives.generation import GenerationEngine, SamplingConfig
    >>>
    >>> engine = GenerationEngine(
    ...     model_forward_fn=model,
    ...     config=EngineConfig(vocab_size=32000, eos_token_id=2),
    ... )
    >>>
    >>> req = engine.submit(
    ...     input_ids,
    ...     SamplingConfig(temperature=0.7, top_p=0.9),
    ...     max_new_tokens=100,
    ... )
    >>>
    >>> for request_id, token in engine.generate_stream():
    ...     print(f"{request_id}: {token}")
"""

from mlx_primitives.generation.requests import (
    GenerationRequest,
    RequestStatus,
    SamplingConfig,
    StopCondition,
    StopConditionType,
    create_request,
)
from mlx_primitives.generation.batch_manager import (
    BatchedSequences,
    BatchingStrategy,
    DynamicBatcher,
    PaddingSide,
    SequenceBatcher,
    create_attention_mask,
    create_combined_mask,
    unbatch_outputs,
)
from mlx_primitives.generation.samplers import (
    TokenSampler,
    apply_frequency_penalty,
    apply_presence_penalty,
    apply_repetition_penalty,
    apply_repetition_penalty_batch,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_beam,
    sample_greedy,
    sample_multinomial,
)
from mlx_primitives.generation.scheduler import (
    PriorityScheduler,
    RequestScheduler,
    SchedulerConfig,
)
from mlx_primitives.generation.engine import (
    EngineConfig,
    GenerationEngine,
    StreamingOutput,
    create_generation_engine,
)

__all__ = [
    # Requests
    "GenerationRequest",
    "RequestStatus",
    "SamplingConfig",
    "StopCondition",
    "StopConditionType",
    "create_request",
    # Batch management
    "SequenceBatcher",
    "BatchedSequences",
    "BatchingStrategy",
    "PaddingSide",
    "DynamicBatcher",
    "create_attention_mask",
    "create_combined_mask",
    "unbatch_outputs",
    # Sampling
    "TokenSampler",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "apply_repetition_penalty",
    "apply_repetition_penalty_batch",
    "apply_presence_penalty",
    "apply_frequency_penalty",
    "sample_greedy",
    "sample_multinomial",
    "sample_beam",
    # Scheduling
    "RequestScheduler",
    "PriorityScheduler",
    "SchedulerConfig",
    # Engine
    "GenerationEngine",
    "EngineConfig",
    "StreamingOutput",
    "create_generation_engine",
]
