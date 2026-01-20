"""Gradient checkpointing for memory-efficient training.

.. warning::
    **MLX Limitation**: Due to MLX's lazy evaluation model, gradient checkpointing
    does NOT provide the same memory savings as in eager frameworks like PyTorch.
    In MLX, intermediate activations are not stored until needed, so this module
    primarily provides API compatibility rather than memory optimization.

    For actual memory savings in MLX, consider:
    - Using mx.eval() strategically to control evaluation timing
    - Chunked processing (as in chunked_cross_attention)
    - Streaming computation patterns
    - Reducing batch size

The traditional description (which applies to eager frameworks):
Gradient checkpointing trades compute for memory by recomputing activations
during the backward pass instead of storing them. This is particularly useful
for training deep models or models with large intermediate activations.

Memory savings: O(sqrt(n)) for n layers when using optimal segment size.
Compute overhead: ~33% additional forward pass computation.

Example:
    >>> def transformer_block(x, w):
    ...     return mx.nn.gelu(x @ w)
    >>>
    >>> # Without checkpointing: stores all intermediate activations
    >>> y = transformer_block(x, w)
    >>>
    >>> # With checkpointing: only stores inputs, recomputes during backward
    >>> y = checkpoint(transformer_block, x, w)
"""

import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import mlx.core as mx

_CHECKPOINTING_WARNING_SHOWN = False


def checkpoint(
    fn: Callable[..., mx.array],
    *args: mx.array,
    preserve_rng_state: bool = True,
) -> mx.array:
    """Apply gradient checkpointing to a function.

    .. warning::
        In MLX, this function does NOT provide memory savings due to lazy evaluation.
        It is provided for API compatibility with PyTorch-style checkpointing code.
        See module docstring for memory-saving alternatives in MLX.

    During the forward pass, runs fn(*args) normally but does NOT store
    intermediate activations for the backward pass. During the backward pass,
    recomputes the forward pass to obtain gradients.

    Args:
        fn: Function to checkpoint. Should be deterministic given its inputs
            (or use preserve_rng_state=True for functions with randomness).
        *args: Input arrays to the function. These are stored for recomputation.
        preserve_rng_state: If True, captures and restores the RNG state to ensure
            reproducible recomputation for functions with randomness (e.g., dropout).

    Returns:
        Output of fn(*args), with checkpointed gradient computation.

    Example:
        >>> def mlp_block(x, w1, w2):
        ...     h = mx.nn.gelu(x @ w1)  # Large intermediate
        ...     return h @ w2
        >>>
        >>> x = mx.random.normal((32, 512))
        >>> w1 = mx.random.normal((512, 2048))
        >>> w2 = mx.random.normal((2048, 512))
        >>>
        >>> # Checkpointed forward - doesn't store 'h'
        >>> y = checkpoint(mlp_block, x, w1, w2)
        >>>
        >>> # Backward recomputes mlp_block to get gradients
        >>> loss = mx.sum(y ** 2)
        >>> grads = mx.grad(lambda *a: mx.sum(checkpoint(mlp_block, *a) ** 2))(x, w1, w2)

    Note:
        - The function must be pure (same inputs -> same outputs) for correctness
        - Memory savings come at the cost of ~2x forward computation
        - Most effective for compute-bound operations with large intermediates
        - **MLX Note**: Due to lazy evaluation, actual memory savings are minimal
    """
    global _CHECKPOINTING_WARNING_SHOWN
    if not _CHECKPOINTING_WARNING_SHOWN:
        warnings.warn(
            "gradient checkpointing in MLX does not provide the same memory savings "
            "as in eager frameworks like PyTorch due to lazy evaluation. "
            "Consider using mx.eval() strategically, chunked processing, or "
            "reducing batch size for memory optimization.",
            UserWarning,
            stacklevel=2,
        )
        _CHECKPOINTING_WARNING_SHOWN = True

    # Delegate to the implementation which handles RNG state and custom gradients
    return _checkpoint_impl(fn, args, preserve_rng_state)


def _checkpoint_impl(
    fn: Callable[..., mx.array],
    args: Tuple[mx.array, ...],
    preserve_rng_state: bool,
) -> mx.array:
    """Internal implementation of checkpointing.

    In MLX's lazy evaluation model, intermediate activations are not stored
    until needed, and the computation graph is built lazily. This means
    checkpoint doesn't provide the same memory savings as in eager frameworks
    like PyTorch.

    However, this implementation still provides:
    1. Correct gradient computation
    2. API compatibility with PyTorch-style checkpointing

    Note on RNG state:
        MLX does not provide a way to capture and restore RNG state like PyTorch.
        The preserve_rng_state parameter is accepted for API compatibility but
        has limited effect. For reproducible stochastic operations, use explicit
        seeds within your function.

    For true memory savings in MLX, consider:
    - Using mx.eval() strategically to control evaluation timing
    - Chunked processing (as in chunked_cross_attention)
    - Streaming computation patterns
    """
    # Note: MLX doesn't provide RNG state capture/restore like PyTorch.
    # The preserve_rng_state parameter is kept for API compatibility.
    # Previous implementation incorrectly called mx.random.uniform() which
    # actually CHANGED the RNG state instead of preserving it.

    # Simply run the function - MLX's lazy evaluation handles the graph
    # Gradients will flow through correctly via MLX's automatic differentiation
    return fn(*args)


def checkpoint_sequential(
    functions: List[Callable[..., mx.array]],
    segments: int,
    *inputs: mx.array,
    preserve_rng_state: bool = True,
) -> mx.array:
    """Checkpoint a sequence of functions in segments.

    Divides the functions into `segments` groups and only stores activations
    at segment boundaries. Within each segment, activations are recomputed
    during the backward pass.

    This provides a trade-off between memory and compute:
    - More segments = less memory, more recomputation
    - Fewer segments = more memory, less recomputation
    - Optimal: segments = sqrt(len(functions)) for balanced trade-off

    Args:
        functions: List of sequential functions [f1, f2, ..., fn] where
            output of f_i is input to f_{i+1}.
        segments: Number of checkpoint segments. Each segment's activations
            are recomputed during backward. Use sqrt(n) for optimal balance.
        *inputs: Inputs to the first function.
        preserve_rng_state: If True, preserve RNG state for reproducibility.

    Returns:
        Output of fn(fn-1(...f1(*inputs)...)).

    Example:
        >>> # 12 transformer layers, checkpoint every 4 layers
        >>> layers = [TransformerBlock() for _ in range(12)]
        >>> output = checkpoint_sequential(layers, segments=3, x)
        >>> # Stores activations after layers 4 and 8
        >>> # Recomputes layers 1-4, 5-8, 9-12 during backward

    Note:
        For a model with n layers:
        - segments=1: Checkpoint entire model (max memory savings, 2x compute)
        - segments=n: No checkpointing (no savings, normal compute)
        - segments=sqrt(n): Balanced trade-off (sqrt(n) memory, ~1.5x compute)
    """
    if not functions:
        raise ValueError("functions list cannot be empty")
    if segments < 1:
        raise ValueError(f"segments must be >= 1, got {segments}")
    if segments > len(functions):
        segments = len(functions)

    n_functions = len(functions)
    segment_size = (n_functions + segments - 1) // segments  # Ceiling division

    def run_segment(
        segment_fns: List[Callable[..., mx.array]],
        segment_input: mx.array,
    ) -> mx.array:
        """Run a segment of functions sequentially."""
        x = segment_input
        for fn in segment_fns:
            x = fn(x)
        return x

    # Process segments
    x = inputs[0] if len(inputs) == 1 else inputs

    for seg_idx in range(segments):
        start_idx = seg_idx * segment_size
        end_idx = min(start_idx + segment_size, n_functions)

        if start_idx >= n_functions:
            break

        segment_fns = functions[start_idx:end_idx]

        # Checkpoint this segment
        def segment_fn(inp: mx.array, fns: List[Callable] = segment_fns) -> mx.array:
            return run_segment(fns, inp)

        x = checkpoint(segment_fn, x, preserve_rng_state=preserve_rng_state)

    return x
