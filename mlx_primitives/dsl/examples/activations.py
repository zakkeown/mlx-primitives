"""Activation function kernels using Metal-Triton DSL.

Includes:
- SiLU (Swish): x * sigmoid(x)
- GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
- GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
- ReLU: max(0, x)
- Fused variants

These are simple element-wise kernels demonstrating basic DSL usage
and transcendental function support.
"""

from mlx_primitives.dsl import metal_kernel, constexpr
import mlx_primitives.dsl as mt


@metal_kernel
def silu(
    x_ptr,
    y_ptr,
    N: constexpr,
):
    """SiLU (Swish) activation: y = x * sigmoid(x).

    SiLU is used in models like LLaMA, Mistral, and SwiGLU architectures.
    sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x_ptr: Input tensor pointer
        y_ptr: Output tensor pointer
        N: Total number of elements

    Grid: ((N + 255) // 256,)
    Threadgroup: 256
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid_x = 1.0 / (1.0 + mt.exp(-x))
        y = x * sigmoid_x
        mt.store(y_ptr + idx, y)


@metal_kernel
def gelu_tanh(
    x_ptr,
    y_ptr,
    N: constexpr,
):
    """GELU activation with tanh approximation.

    y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This is the fast approximation used in GPT-2, BERT, and many other models.
    Faster than exact GELU but slightly less accurate.

    Args:
        x_ptr: Input tensor pointer
        y_ptr: Output tensor pointer
        N: Total number of elements

    Grid: ((N + 255) // 256,)
    Threadgroup: 256
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)

        # Constants
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        coeff = 0.044715

        # Compute tanh approximation
        inner = sqrt_2_over_pi * (x + coeff * x * x * x)
        y = 0.5 * x * (1.0 + mt.tanh(inner))

        mt.store(y_ptr + idx, y)


@metal_kernel
def gelu_exact(
    x_ptr,
    y_ptr,
    N: constexpr,
):
    """GELU activation (exact formulation).

    y = 0.5 * x * (1 + erf(x / sqrt(2)))

    More accurate than tanh approximation but slower.
    Used when precision is more important than speed.

    Args:
        x_ptr: Input tensor pointer
        y_ptr: Output tensor pointer
        N: Total number of elements

    Grid: ((N + 255) // 256,)
    Threadgroup: 256
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)

        # sqrt(2) inverse for normalization
        inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)

        # Exact GELU using error function
        y = 0.5 * x * (1.0 + mt.erf(x * inv_sqrt2))

        mt.store(y_ptr + idx, y)


@metal_kernel
def quick_gelu(
    x_ptr,
    y_ptr,
    N: constexpr,
):
    """Quick GELU approximation.

    y = x * sigmoid(1.702 * x)

    Even faster approximation used in some models.
    sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x_ptr: Input tensor pointer
        y_ptr: Output tensor pointer
        N: Total number of elements

    Grid: ((N + 255) // 256,)
    Threadgroup: 256
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)
        # Quick GELU: x * sigmoid(1.702 * x)
        y = x / (1.0 + mt.exp(-1.702 * x))
        mt.store(y_ptr + idx, y)


@metal_kernel
def fused_silu_mul(
    x_ptr,
    gate_ptr,
    y_ptr,
    N: constexpr,
):
    """Fused SiLU + elementwise multiply (for SwiGLU).

    y = silu(x) * gate = (x * sigmoid(x)) * gate

    Common in LLaMA and Mistral for the SwiGLU activation in FFN:
        FFN(x) = SiLU(xW1) * (xW2)

    Fusing saves one memory round-trip.

    Args:
        x_ptr: Input tensor pointer (after W1 projection)
        gate_ptr: Gate tensor pointer (after W2 projection)
        y_ptr: Output tensor pointer
        N: Total number of elements

    Grid: ((N + 255) // 256,)
    Threadgroup: 256
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)
        gate = mt.load(gate_ptr + idx)

        # SiLU
        sigmoid_x = 1.0 / (1.0 + mt.exp(-x))
        silu_x = x * sigmoid_x

        # Multiply with gate
        y = silu_x * gate
        mt.store(y_ptr + idx, y)


@metal_kernel
def fused_gelu_mul(
    x_ptr,
    gate_ptr,
    y_ptr,
    N: constexpr,
):
    """Fused GELU + elementwise multiply (for GEGLU).

    y = gelu(x) * gate

    Similar to SwiGLU but using GELU activation.
    Fusing saves one memory round-trip.

    Args:
        x_ptr: Input tensor pointer
        gate_ptr: Gate tensor pointer
        y_ptr: Output tensor pointer
        N: Total number of elements

    Grid: ((N + 255) // 256,)
    Threadgroup: 256
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)
        gate = mt.load(gate_ptr + idx)

        # GELU (tanh approximation)
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        inner = sqrt_2_over_pi * (x + coeff * x * x * x)
        gelu_x = 0.5 * x * (1.0 + mt.tanh(inner))

        # Multiply with gate
        y = gelu_x * gate
        mt.store(y_ptr + idx, y)


@metal_kernel
def softplus(
    x_ptr,
    y_ptr,
    N: constexpr,
    beta: mt.float32 = 1.0,
    threshold: mt.float32 = 20.0,
):
    """Softplus activation: y = (1/beta) * log(1 + exp(beta * x)).

    For numerical stability, uses linear approximation when beta*x > threshold.

    Args:
        x_ptr: Input tensor pointer
        y_ptr: Output tensor pointer
        N: Total number of elements
        beta: Scaling factor (default 1.0)
        threshold: Above this value, use linear approximation (default 20.0)

    Grid: ((N + 255) // 256,)
    Threadgroup: 256
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)
        bx = beta * x

        # Numerically stable softplus
        if bx > threshold:
            # Linear approximation for large values
            y = x
        else:
            y = mt.log(1.0 + mt.exp(bx)) / beta

        mt.store(y_ptr + idx, y)


@metal_kernel
def mish(
    x_ptr,
    y_ptr,
    N: constexpr,
):
    """Mish activation: y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x))).

    Self-regularized non-monotonic activation function.

    Args:
        x_ptr: Input tensor pointer
        y_ptr: Output tensor pointer
        N: Total number of elements

    Grid: ((N + 255) // 256,)
    Threadgroup: 256
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)

        # Softplus: log(1 + exp(x)) - use numerically stable version
        softplus_x = mt.log(1.0 + mt.exp(x))
        if x > 20.0:
            softplus_x = x

        # Mish: x * tanh(softplus(x))
        y = x * mt.tanh(softplus_x)
        mt.store(y_ptr + idx, y)
