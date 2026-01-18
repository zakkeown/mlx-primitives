"""ANE-accelerated matrix multiplication.

This module provides matrix multiplication that can optionally
execute on the Apple Neural Engine for improved throughput during
inference workloads.
"""

from typing import Optional

import mlx.core as mx

from mlx_primitives.ane.detection import is_ane_available
from mlx_primitives.ane.dispatch import ComputeTarget, should_use_ane
from mlx_primitives.ane.model_cache import ModelSpec, get_model_cache


def ane_matmul(
    a: mx.array,
    b: mx.array,
    transpose_a: bool = False,
    transpose_b: bool = False,
    use_ane: str = "auto",
) -> mx.array:
    """Matrix multiplication with optional ANE offload.

    For inference workloads with fixed shapes, this can leverage the
    Neural Engine for improved throughput.

    Args:
        a: First matrix (M, K) or (batch, M, K).
        b: Second matrix (K, N) or (batch, K, N).
        transpose_a: Transpose first matrix before multiply.
        transpose_b: Transpose second matrix before multiply.
        use_ane: ANE usage mode:
            - "auto": Let dispatch logic decide (default).
            - "always": Force ANE (falls back to GPU if unavailable).
            - "never": Always use GPU.

    Returns:
        Result matrix (M, N) or (batch, M, N).

    Example:
        >>> a = mx.random.normal((1024, 512))
        >>> b = mx.random.normal((512, 1024))
        >>> c = ane_matmul(a, b)
    """
    # Determine dispatch target
    force_target = None
    if use_ane == "always":
        force_target = ComputeTarget.ANE
    elif use_ane == "never":
        force_target = ComputeTarget.GPU

    decision = should_use_ane(
        operation="matmul",
        input_shapes=[tuple(a.shape), tuple(b.shape)],
        is_training=False,
        force_target=force_target,
    )

    if decision.target == ComputeTarget.ANE:
        try:
            return _ane_matmul_impl(a, b, transpose_a, transpose_b)
        except Exception as e:
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("ane_matmul", e)

    # Standard MLX matmul
    if transpose_a:
        a = mx.swapaxes(a, -1, -2) if a.ndim >= 2 else a
    if transpose_b:
        b = mx.swapaxes(b, -1, -2) if b.ndim >= 2 else b

    return a @ b


def _ane_matmul_impl(
    a: mx.array,
    b: mx.array,
    transpose_a: bool,
    transpose_b: bool,
) -> mx.array:
    """Core ANE matmul implementation via Core ML.

    This compiles a Core ML model for the specific matrix shapes
    and executes it on the Neural Engine.
    """
    try:
        import coremltools as ct
    except ImportError:
        raise RuntimeError("coremltools required for ANE matmul")

    from mlx_primitives.ane.converters import mlx_to_coreml_input, coreml_to_mlx

    # Get shapes
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    # Create model spec
    spec = ModelSpec(
        operation="matmul",
        input_shapes=(a_shape, b_shape),
        dtype="float16",  # ANE prefers fp16
        params=(("transpose_a", transpose_a), ("transpose_b", transpose_b)),
    )

    # Get or compile model
    cache = get_model_cache()
    model = cache.get_or_compile(spec, _compile_matmul_model)

    # Prepare inputs
    # Convert to float16 for ANE efficiency
    a_fp16 = a.astype(mx.float16) if a.dtype != mx.float16 else a
    b_fp16 = b.astype(mx.float16) if b.dtype != mx.float16 else b

    inputs = mlx_to_coreml_input([a_fp16, b_fp16], ["a", "b"])

    # Run on ANE
    outputs = model.predict(inputs)

    # Convert back to MLX with original dtype
    result = coreml_to_mlx(outputs, "output", target_dtype=a.dtype)

    return result


def _compile_matmul_model(spec: ModelSpec) -> "ct.models.MLModel":
    """Compile a Core ML model for matmul targeting ANE.

    Args:
        spec: Model specification with shapes and parameters.

    Returns:
        Compiled Core ML model.
    """
    import coremltools as ct
    import numpy as np

    a_shape, b_shape = spec.input_shapes
    params = dict(spec.params)
    transpose_a = params.get("transpose_a", False)
    transpose_b = params.get("transpose_b", False)

    # Build using coremltools builder
    # For matmul, we use the neural network builder
    from coremltools.models import neural_network as nn_builder

    # Calculate output shape
    if transpose_a:
        m, k1 = a_shape[-1], a_shape[-2]
    else:
        m, k1 = a_shape[-2], a_shape[-1]

    if transpose_b:
        k2, n = b_shape[-1], b_shape[-2]
    else:
        k2, n = b_shape[-2], b_shape[-1]

    if k1 != k2:
        raise ValueError(f"Incompatible shapes for matmul: {a_shape} @ {b_shape}")

    # Handle batched matmul
    if len(a_shape) > 2:
        batch = a_shape[:-2]
        output_shape = batch + (m, n)
    else:
        output_shape = (m, n)

    # Build simple Core ML model using ML Program (newer API)
    try:
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.input_types import TensorType

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=a_shape, dtype=np.float16),
                mb.TensorSpec(shape=b_shape, dtype=np.float16),
            ]
        )
        def matmul_prog(a, b):
            if transpose_a:
                a = mb.transpose(x=a, perm=list(range(len(a_shape) - 2)) + [-1, -2])
            if transpose_b:
                b = mb.transpose(x=b, perm=list(range(len(b_shape) - 2)) + [-1, -2])
            return mb.matmul(x=a, y=b, name="output")

        model = ct.convert(
            matmul_prog,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
        )
        return model

    except Exception:
        # Fallback to older neural network API
        input_features = [
            ("a", ct.models.datatypes.Array(*a_shape)),
            ("b", ct.models.datatypes.Array(*b_shape)),
        ]
        output_features = [("output", ct.models.datatypes.Array(*output_shape))]

        builder = nn_builder.NeuralNetworkBuilder(
            input_features, output_features, disable_rank5_shape_mapping=True
        )

        # Add matmul layer
        builder.add_batched_mat_mul(
            name="matmul",
            input_names=["a", "b"],
            output_name="output",
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

        # Build and compile
        mlmodel = ct.models.MLModel(builder.spec)
        return mlmodel


def ane_linear(
    x: mx.array,
    weight: mx.array,
    bias: Optional[mx.array] = None,
    use_ane: str = "auto",
) -> mx.array:
    """Linear layer with optional ANE offload.

    Computes x @ weight.T + bias, optionally using ANE.

    Args:
        x: Input tensor (..., in_features).
        weight: Weight matrix (out_features, in_features).
        bias: Optional bias vector (out_features,).
        use_ane: ANE usage mode ("auto", "always", "never").

    Returns:
        Output tensor (..., out_features).
    """
    # Linear is matmul with transposed weight
    output = ane_matmul(x, weight, transpose_b=True, use_ane=use_ane)

    if bias is not None:
        output = output + bias

    return output
