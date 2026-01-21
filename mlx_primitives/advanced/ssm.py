"""State Space Models for MLX.

This module provides SSM components:
- MambaBlock: Selective state space model (Mamba)
- S4Layer: Structured state space sequence model
- H3Layer/H3: Hungry Hungry Hippos hybrid attention-SSM
- SelectiveScan: Core selective scan operation

Performance Note:
    The MambaBlock uses parallel associative scan via Metal kernels,
    achieving O(log n) complexity for sequences up to 1024 tokens.

    For sequences > 1024 tokens, the implementation falls back to sequential
    O(n) computation. Consider processing in chunks for better performance.

    S4Layer and H3Layer still use sequential implementations as they have
    different state update structures not yet parallelized.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.constants import SSM_SEQUENTIAL_WARNING_THRESHOLD
from mlx_primitives.primitives.scan import selective_scan as parallel_selective_scan


def mamba_selective_scan(
    x: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: Optional[mx.array] = None,
    warn_on_long_seq: bool = True,
) -> mx.array:
    """Selective scan operation for Mamba-style state space models.

    This is the high-level API for Mamba selective scan, which includes:
    - Input validation and warnings
    - Automatic dispatch to parallel Metal kernels when beneficial

    For the low-level parallel scan primitive, see:
        mlx_primitives.primitives.scan.selective_scan

    Implements the core computation for Mamba:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t + D * x_t

    Where A_bar = exp(delta * A) and B_bar ≈ delta * B.

    Args:
        x: Input tensor (batch, seq_len, d_inner).
        delta: Time step delta (batch, seq_len, d_inner).
        A: State transition matrix (d_inner, d_state).
        B: Input matrix (batch, seq_len, d_state).
        C: Output matrix (batch, seq_len, d_state).
        D: Skip connection (d_inner,). If None, no skip connection is added.
        warn_on_long_seq: Whether to warn for long sequences. Default: True.

    Returns:
        Output tensor (batch, seq_len, d_inner).

    Note:
        This implementation uses a **parallel associative scan** via Metal kernels
        for sequences up to 1024 tokens, achieving O(log n) complexity.

        Performance characteristics:
        - seq_len <= 1024: O(log n) parallel Metal kernel (Blelloch algorithm)
        - seq_len > 1024: Falls back to O(n) sequential (multi-block not yet implemented)
        - Each step is O(batch * d_inner * d_state)

        For sequences longer than 1024, consider:
        1. Processing in chunks with state carry-over
        2. Using attention-based models with mx.fast.scaled_dot_product_attention

    References:
        - Mamba paper: https://arxiv.org/abs/2312.00752
        - Parallel scan algorithms: https://en.wikipedia.org/wiki/Prefix_sum

    Raises:
        ValueError: If input tensors have incorrect dimensions.
    """
    # Input validation
    if x.ndim != 3:
        raise ValueError(
            f"x must be 3D (batch, seq_len, d_inner), got {x.ndim}D with shape {x.shape}"
        )
    if delta.ndim != 3:
        raise ValueError(
            f"delta must be 3D (batch, seq_len, d_inner), got {delta.ndim}D with shape {delta.shape}"
        )
    if A.ndim != 2:
        raise ValueError(
            f"A must be 2D (d_inner, d_state), got {A.ndim}D with shape {A.shape}"
        )
    if B.ndim != 3:
        raise ValueError(
            f"B must be 3D (batch, seq_len, d_state), got {B.ndim}D with shape {B.shape}"
        )
    if C.ndim != 3:
        raise ValueError(
            f"C must be 3D (batch, seq_len, d_state), got {C.ndim}D with shape {C.shape}"
        )

    batch_size, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # For very long sequences, warn that Metal kernel may fall back to sequential
    if warn_on_long_seq and seq_len > 1024:
        warnings.warn(
            f"SSM scan over {seq_len} positions (>1024). "
            f"Multi-block Metal kernel not yet implemented; falling back to sequential. "
            "Consider processing in chunks for better performance.",
            stacklevel=2,
        )

    # Use the parallel selective scan implementation from primitives
    # This uses Metal kernels with O(log n) complexity for seq_len <= 1024
    return parallel_selective_scan(x, delta, A, B, C, D, use_metal=True)


# Backwards compatibility alias
selective_scan = mamba_selective_scan


class MambaBlock(nn.Module):
    """Mamba block - Selective State Space Model.

    Mamba uses input-dependent (selective) state space parameters,
    allowing it to selectively remember or forget information.

    Args:
        dims: Model dimension.
        d_state: SSM state dimension (default: 16).
        d_conv: Convolution kernel size (default: 4).
        expand: Expansion factor for inner dimension (default: 2).
        dt_rank: Rank for delta projection (default: "auto").

    Reference:
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
        https://arxiv.org/abs/2312.00752

    Example:
        >>> block = MambaBlock(dims=768)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y = block(x)
    """

    def __init__(
        self,
        dims: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
    ):
        super().__init__()

        self.dims = dims
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dims)

        if dt_rank == "auto":
            self.dt_rank = max(dims // 16, 1)
        else:
            self.dt_rank = dt_rank

        # Input projection (projects to 2 * d_inner for x and z)
        self.in_proj = nn.Linear(dims, self.d_inner * 2, bias=False)

        # Convolution - MLX Conv1d expects (batch, seq, channels) format
        # For depthwise conv, we use groups=d_inner
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM projections
        # x_proj projects x to (delta, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )

        # dt_proj projects dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameter (learnable, initialized specially)
        # A is typically initialized as -exp(linspace(log(dt_min), log(dt_max), d_state))
        A = mx.repeat(mx.arange(1, d_state + 1)[None, :], self.d_inner, axis=0)
        self.A_log = mx.log(A)  # Store log(A) for numerical stability

        # D parameter (skip connection)
        self.D = mx.ones((self.d_inner,))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through Mamba block.

        Args:
            x: Input tensor (batch, seq_len, dims).

        Returns:
            Output tensor (batch, seq_len, dims).

        Raises:
            ValueError: If input tensor has incorrect dimensions.
        """
        if x.ndim != 3:
            raise ValueError(
                f"MambaBlock expects 3D input (batch, seq_len, dims), "
                f"got {x.ndim}D with shape {x.shape}"
            )

        batch_size, seq_len, _ = x.shape

        # Input projection -> (x, z)
        xz = self.in_proj(x)
        x_branch, z = mx.split(xz, 2, axis=-1)

        # Convolution - MLX Conv1d expects (batch, seq, channels) which we already have
        x_branch = self.conv1d(x_branch)  # (batch, seq + padding, d_inner)
        x_branch = x_branch[:, :seq_len, :]  # Remove padding from seq dimension

        # Apply SiLU activation
        x_branch = nn.silu(x_branch)

        # SSM
        y = self._ssm(x_branch)

        # Gate with z
        z = nn.silu(z)
        output = y * z

        # Output projection
        return self.out_proj(output)

    def _ssm(self, x: mx.array) -> mx.array:
        """Apply selective state space model.

        Args:
            x: Input (batch, seq_len, d_inner).

        Returns:
            Output (batch, seq_len, d_inner).
        """
        batch_size, seq_len, _ = x.shape

        # Get A from log space
        A = -mx.exp(self.A_log)  # (d_inner, d_state)

        # Project x to get delta, B, C
        x_dbl = self.x_proj(x)  # (batch, seq, dt_rank + 2*d_state)

        # Split into components
        delta, B, C = mx.split(
            x_dbl,
            [self.dt_rank, self.dt_rank + self.d_state],
            axis=-1,
        )

        # Project delta to d_inner and apply softplus
        delta = self.dt_proj(delta)  # (batch, seq, d_inner)
        delta = nn.softplus(delta)

        # Run selective scan
        y = selective_scan(x, delta, A, B, C, self.D)

        return y


class S4Layer(nn.Module):
    """Structured State Space Sequence (S4) layer.

    S4 uses a structured parameterization of the state matrix
    based on the HiPPO framework for long-range dependencies.

    Args:
        dims: Model dimension.
        d_state: State dimension (default: 64).
        bidirectional: If True, process sequence in both directions.
        dropout: Dropout rate (default: 0.0).
        use_complex: If True, use complex diagonal (original S4).
            If False, use real diagonal only (S4D-Real variant, ~2x faster).

    Reference:
        "Efficiently Modeling Long Sequences with Structured State Spaces"
        https://arxiv.org/abs/2111.00396

    Performance Note:
        The use_complex=False variant (S4D-Real) achieves ~2x speedup
        by avoiding complex arithmetic while maintaining competitive
        performance for most tasks.
    """

    def __init__(
        self,
        dims: int,
        d_state: int = 64,
        bidirectional: bool = False,
        dropout: float = 0.0,
        use_complex: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.d_state = d_state
        self.bidirectional = bidirectional
        self.use_complex = use_complex

        # Learnable parameters for S4
        # Using diagonal approximation
        if use_complex:
            # Complex diagonal (original S4)
            self.A_real = mx.random.uniform(
                low=-0.5, high=0.5, shape=(dims, d_state)
            )
            self.A_imag = mx.random.uniform(
                low=-0.5, high=0.5, shape=(dims, d_state)
            )
        else:
            # Real-only diagonal (S4D-Real variant, faster)
            # Negative real parts for stability
            self.A_real = mx.random.uniform(
                low=-1.0, high=-0.01, shape=(dims, d_state)
            )
            self.A_imag = None

        # B and C parameters
        self.B = mx.random.normal((dims, d_state)) * 0.01
        self.C = mx.random.normal((dims, d_state)) * 0.01

        # D (skip connection)
        self.D = mx.ones((dims,))

        # Time step
        self.log_dt = mx.zeros((dims,))

        # Output mixing
        self.out_linear = nn.Linear(
            dims * (2 if bidirectional else 1), dims
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through S4 layer.

        Args:
            x: Input (batch, seq_len, dims).

        Returns:
            Output (batch, seq_len, dims).

        Raises:
            ValueError: If input tensor has incorrect dimensions.
        """
        if x.ndim != 3:
            raise ValueError(
                f"S4Layer expects 3D input (batch, seq_len, dims), "
                f"got {x.ndim}D with shape {x.shape}"
            )

        # Forward pass
        y_forward = self._s4_kernel(x)

        if self.bidirectional:
            # Backward pass (reverse sequence)
            x_rev = x[:, ::-1, :]
            y_backward = self._s4_kernel(x_rev)
            y_backward = y_backward[:, ::-1, :]

            y = mx.concatenate([y_forward, y_backward], axis=-1)
        else:
            y = y_forward

        # Output projection
        y = self.out_linear(y)

        if self.dropout is not None:
            y = self.dropout(y)

        return y

    def _s4_kernel(self, x: mx.array) -> mx.array:
        """Apply S4 kernel convolution.

        Args:
            x: Input (batch, seq_len, dims).

        Returns:
            Output (batch, seq_len, dims).

        Note:
            Uses real-only computation when use_complex=False for ~2x speedup.
        """
        if self.use_complex:
            return self._s4_kernel_complex(x)
        else:
            return self._s4_kernel_real(x)

    def _s4_kernel_real(self, x: mx.array) -> mx.array:
        """Real-only S4 kernel (S4D-Real variant, faster)."""
        batch_size, seq_len, _ = x.shape

        # Get discretized kernel
        dt = mx.exp(self.log_dt)

        # Real diagonal state matrix
        A_bar = mx.exp(dt[:, None] * self.A_real)  # Discretize

        # Precompute scaled B
        B_scaled = self.B * dt[:, None]

        # Initialize state (real, not complex)
        h = mx.zeros((batch_size, self.dims, self.d_state))
        outputs = []

        for t in range(seq_len):
            # Update state: h = A * h + B * x (all real operations)
            h = A_bar * h + B_scaled * x[:, t, :, None]

            # Compute output: y = sum(h * C, axis=-1)
            y_t = mx.sum(h * self.C, axis=-1)
            outputs.append(y_t)

        y = mx.stack(outputs, axis=1)

        # Add skip connection
        return y + x * self.D

    def _s4_kernel_complex(self, x: mx.array) -> mx.array:
        """Complex S4 kernel (original S4)."""
        batch_size, seq_len, _ = x.shape

        # Get discretized kernel
        dt = mx.exp(self.log_dt)

        # Complex diagonal state matrix
        A = self.A_real + 1j * self.A_imag
        A_bar = mx.exp(dt[:, None] * A)  # Discretize

        # Precompute scaled B
        B_scaled = self.B * dt[:, None]

        # Initialize state
        h = mx.zeros((batch_size, self.dims, self.d_state), dtype=mx.complex64)
        outputs = []

        for t in range(seq_len):
            # Update state: h = A * h + B * x
            h = A_bar * h + B_scaled * x[:, t, :, None].astype(mx.complex64)

            # Compute output: y = sum(real(h * C), axis=-1)
            y_t = mx.sum(mx.real(h * self.C), axis=-1)
            outputs.append(y_t)

        y = mx.stack(outputs, axis=1)

        # Add skip connection
        return y + x * self.D


class Mamba(nn.Module):
    """Full Mamba model with multiple blocks.

    Args:
        dims: Model dimension.
        n_layers: Number of Mamba blocks.
        d_state: SSM state dimension.
        d_conv: Convolution kernel size.
        expand: Expansion factor.
        vocab_size: Vocabulary size (optional, for language modeling).

    Example:
        >>> model = Mamba(dims=768, n_layers=24)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y = model(x)
    """

    def __init__(
        self,
        dims: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()

        self.dims = dims
        self.vocab_size = vocab_size

        # Optional embedding
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, dims)
        else:
            self.embedding = None

        # Mamba blocks with residual connections
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.RMSNorm(dims),
                    MambaBlock(dims, d_state, d_conv, expand),
                )
            )

        # Output norm
        self.norm = nn.RMSNorm(dims)

        # Optional output projection
        if vocab_size is not None:
            self.lm_head = nn.Linear(dims, vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through Mamba model.

        Args:
            x: Input tensor. If vocab_size is set, expects token IDs.
               Otherwise expects (batch, seq_len, dims).

        Returns:
            Output tensor. If vocab_size is set, returns logits.
        """
        # Embed if needed
        if self.embedding is not None:
            x = self.embedding(x)

        # Apply blocks with residual connections
        for layer in self.layers:
            x = x + layer(x)

        # Final norm
        x = self.norm(x)

        # Output projection if needed
        if self.lm_head is not None:
            x = self.lm_head(x)

        return x


class H3Layer(nn.Module):
    """H3 (Hungry Hungry Hippos) layer.

    H3 combines the efficiency of state space models with the expressiveness
    of attention by using two SSM layers with multiplicative interaction,
    similar to the structure of attention (Q * K^T * V).

    The key insight is that:
    - One SSM produces "keys" (shifted signal)
    - Another SSM produces "queries" (shifted signal)
    - The multiplicative interaction creates attention-like behavior
    - A final SSM produces the output

    Args:
        dims: Model dimension.
        d_state: SSM state dimension (default: 64).
        head_dim: Dimension per head for multi-head SSM (default: None, uses dims).
        num_heads: Number of heads (default: 1).
        dropout: Dropout rate (default: 0.0).
        use_fast_path: Use optimized implementation if available (default: True).

    Reference:
        "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
        https://arxiv.org/abs/2212.14052

    Example:
        >>> layer = H3Layer(dims=768, d_state=64)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y = layer(x)
    """

    def __init__(
        self,
        dims: int,
        d_state: int = 64,
        head_dim: Optional[int] = None,
        num_heads: int = 1,
        dropout: float = 0.0,
        use_fast_path: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.d_state = d_state
        self.head_dim = head_dim or dims
        self.num_heads = num_heads

        # Project to internal dimension
        self.head_dim_total = self.head_dim * num_heads

        # Input projections for Q, K, V analogues
        self.q_proj = nn.Linear(dims, self.head_dim_total, bias=False)
        self.k_proj = nn.Linear(dims, self.head_dim_total, bias=False)
        self.v_proj = nn.Linear(dims, self.head_dim_total, bias=False)

        # SSM for shift operations (like Q, K in attention)
        # These create the "shift" that allows looking at past context
        self._init_ssm_params()

        # Output projection
        self.out_proj = nn.Linear(self.head_dim_total, dims, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _init_ssm_params(self):
        """Initialize SSM parameters for diagonal state space."""
        # A is initialized as negative exponentials for stability
        # Shape: (head_dim, d_state) per head
        A_init = -mx.exp(
            mx.linspace(
                math.log(0.001),
                math.log(0.1),
                self.d_state
            )
        )
        self.A = mx.broadcast_to(
            A_init[None, :],
            (self.head_dim_total, self.d_state)
        )

        # B and C initialized randomly
        self.B_q = mx.random.normal((self.head_dim_total, self.d_state)) * 0.01
        self.B_k = mx.random.normal((self.head_dim_total, self.d_state)) * 0.01

        # Time step delta
        self.log_dt = mx.zeros((self.head_dim_total,))

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass through H3 layer.

        Args:
            x: Input tensor (batch, seq_len, dims).
            mask: Deprecated. H3 is inherently causal through its SSM structure.
                  If provided, a warning will be raised and the mask is ignored.

        Returns:
            Output tensor (batch, seq_len, dims).

        Note:
            Unlike attention, H3 achieves causality through its SSM state update,
            which naturally prevents positions from attending to future positions.
            Therefore, explicit causal masking is not needed.
        """
        if x.ndim != 3:
            raise ValueError(
                f"H3Layer expects 3D input (batch, seq_len, dims), "
                f"got {x.ndim}D with shape {x.shape}"
            )

        if mask is not None:
            warnings.warn(
                "H3Layer received a mask but H3 is inherently causal through its SSM "
                "structure. The mask will be ignored. Remove the mask parameter to "
                "suppress this warning.",
                stacklevel=2,
            )

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, head_dim_total)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply SSM shift to Q and K
        # This creates the "causal attention" effect
        q_shifted = self._ssm_shift(q, self.B_q)
        k_shifted = self._ssm_shift(k, self.B_k)

        # Multiplicative interaction (like attention scores)
        # In H3, this is: shifted_q * shifted_k * v
        # The shifts allow each position to "attend" to past positions
        output = q_shifted * k_shifted * v

        # Output projection
        output = self.out_proj(output)

        if self.dropout is not None:
            output = self.dropout(output)

        return output

    def _ssm_shift(self, x: mx.array, B: mx.array) -> mx.array:
        """Apply SSM-based shift operation.

        This is the core of H3 - using SSM to efficiently compute
        cumulative operations that mimic attention.

        Args:
            x: Input (batch, seq_len, head_dim_total).
            B: Input matrix (head_dim_total, d_state).

        Returns:
            Shifted output (batch, seq_len, head_dim_total).
        """
        batch_size, seq_len, d = x.shape

        # Get discretized SSM parameters
        dt = mx.exp(self.log_dt)  # (head_dim_total,)
        A_bar = mx.exp(dt[:, None] * self.A)  # (head_dim_total, d_state)
        B_bar = dt[:, None] * B  # (head_dim_total, d_state)

        # Initialize state
        h = mx.zeros((batch_size, d, self.d_state))

        # Sequential scan
        outputs = []
        for t in range(seq_len):
            # Update state: h = A_bar * h + B_bar * x_t
            h = A_bar * h + B_bar * x[:, t, :, None]

            # Output is sum over state dimension (simple aggregation)
            y = mx.sum(h, axis=-1)  # (batch, head_dim_total)
            outputs.append(y)

        return mx.stack(outputs, axis=1)


class H3Block(nn.Module):
    """H3 block with normalization and residual connection.

    A complete H3 block including layer normalization, the H3 layer,
    and MLP with residual connections.

    Args:
        dims: Model dimension.
        d_state: SSM state dimension (default: 64).
        mlp_ratio: MLP hidden dimension ratio (default: 4).
        dropout: Dropout rate (default: 0.0).

    Example:
        >>> block = H3Block(dims=768)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y = block(x)
    """

    def __init__(
        self,
        dims: int,
        d_state: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.RMSNorm(dims)
        self.h3 = H3Layer(dims, d_state, dropout=dropout)

        self.norm2 = nn.RMSNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(dims * mlp_ratio, dims),
        )
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def __call__(self, x: mx.array) -> mx.array:
        # H3 with residual
        h = self.norm1(x)
        h = self.h3(h)
        x = x + h

        # MLP with residual
        h = self.norm2(x)
        h = self.mlp(h)
        if self.dropout is not None:
            h = self.dropout(h)
        x = x + h

        return x


class H3(nn.Module):
    """Full H3 model with multiple blocks.

    Args:
        dims: Model dimension.
        n_layers: Number of H3 blocks.
        d_state: SSM state dimension (default: 64).
        mlp_ratio: MLP hidden dimension ratio (default: 4).
        dropout: Dropout rate (default: 0.0).
        vocab_size: Vocabulary size for language modeling (optional).

    Example:
        >>> model = H3(dims=768, n_layers=12)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y = model(x)
    """

    def __init__(
        self,
        dims: int,
        n_layers: int,
        d_state: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()

        self.dims = dims
        self.vocab_size = vocab_size

        # Optional embedding
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, dims)
        else:
            self.embedding = None

        # H3 blocks
        self.layers = [
            H3Block(dims, d_state, mlp_ratio, dropout)
            for _ in range(n_layers)
        ]

        # Output norm
        self.norm = nn.RMSNorm(dims)

        # Optional output projection
        if vocab_size is not None:
            self.lm_head = nn.Linear(dims, vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through H3 model.

        Args:
            x: Input tensor. If vocab_size is set, expects token IDs.
               Otherwise expects (batch, seq_len, dims).

        Returns:
            Output tensor. If vocab_size is set, returns logits.
        """
        # Embed if needed
        if self.embedding is not None:
            x = self.embedding(x)

        # Apply blocks
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.norm(x)

        # Output projection if needed
        if self.lm_head is not None:
            x = self.lm_head(x)

        return x