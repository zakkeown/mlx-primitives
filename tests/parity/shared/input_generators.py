"""Input generators for parity tests."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Size Configurations
# =============================================================================

SIZE_CONFIGS = {
    "tiny": {
        "attention": {"batch": 1, "seq": 64, "heads": 4, "head_dim": 32},
        "activation": {"batch": 1, "seq": 64, "dim": 256},
        "normalization": {"batch": 1, "seq": 64, "hidden": 256},
        "quantization": {"m": 64, "n": 64, "k": 64},
        "moe": {"batch": 1, "seq": 32, "dim": 128, "experts": 4, "top_k": 2},
        "pooling": {"batch": 1, "channels": 32, "height": 16, "width": 16},
        "embedding": {"batch": 1, "seq": 32, "vocab_size": 1000, "dim": 64},
        "scan": {"batch": 1, "seq": 64, "dim": 32},
        "cache": {"batch": 1, "seq": 64, "heads": 4, "head_dim": 32, "block_size": 16},
        "sampling": {"batch": 1, "vocab_size": 1000},
    },
    "small": {
        "attention": {"batch": 2, "seq": 256, "heads": 8, "head_dim": 64},
        "activation": {"batch": 4, "seq": 256, "dim": 1024},
        "normalization": {"batch": 4, "seq": 256, "hidden": 1024},
        "quantization": {"m": 256, "n": 256, "k": 256},
        "moe": {"batch": 2, "seq": 128, "dim": 512, "experts": 8, "top_k": 2},
        "pooling": {"batch": 4, "channels": 64, "height": 32, "width": 32},
        "embedding": {"batch": 4, "seq": 128, "vocab_size": 10000, "dim": 256},
        "scan": {"batch": 4, "seq": 256, "dim": 64},
        "cache": {"batch": 2, "seq": 256, "heads": 8, "head_dim": 64, "block_size": 32},
        "sampling": {"batch": 4, "vocab_size": 10000},
    },
    "medium": {
        "attention": {"batch": 4, "seq": 1024, "heads": 16, "head_dim": 64},
        "activation": {"batch": 8, "seq": 1024, "dim": 2048},
        "normalization": {"batch": 8, "seq": 1024, "hidden": 2048},
        "quantization": {"m": 1024, "n": 1024, "k": 1024},
        "moe": {"batch": 4, "seq": 512, "dim": 1024, "experts": 16, "top_k": 2},
        "pooling": {"batch": 8, "channels": 128, "height": 64, "width": 64},
        "embedding": {"batch": 8, "seq": 512, "vocab_size": 50000, "dim": 512},
        "scan": {"batch": 8, "seq": 1024, "dim": 128},
        "cache": {"batch": 4, "seq": 1024, "heads": 16, "head_dim": 64, "block_size": 64},
        "sampling": {"batch": 8, "vocab_size": 50000},
    },
    "large": {
        "attention": {"batch": 8, "seq": 4096, "heads": 32, "head_dim": 128},
        "activation": {"batch": 16, "seq": 4096, "dim": 4096},
        "normalization": {"batch": 16, "seq": 4096, "hidden": 4096},
        "quantization": {"m": 4096, "n": 4096, "k": 4096},
        "moe": {"batch": 8, "seq": 2048, "dim": 2048, "experts": 32, "top_k": 2},
        "pooling": {"batch": 16, "channels": 256, "height": 128, "width": 128},
        "embedding": {"batch": 16, "seq": 2048, "vocab_size": 100000, "dim": 1024},
        "scan": {"batch": 16, "seq": 4096, "dim": 256},
        "cache": {"batch": 8, "seq": 4096, "heads": 32, "head_dim": 128, "block_size": 128},
        "sampling": {"batch": 16, "vocab_size": 100000},
    },
}


EDGE_CASE_CONFIGS = {
    "empty": {
        "attention": {"batch": 0, "seq": 0, "heads": 4, "head_dim": 32},
        "activation": {"batch": 0, "seq": 0, "dim": 256},
    },
    "single_element": {
        "attention": {"batch": 1, "seq": 1, "heads": 1, "head_dim": 1},
        "activation": {"batch": 1, "seq": 1, "dim": 1},
    },
    "very_large": {
        "attention": {"batch": 1, "seq": 16384, "heads": 32, "head_dim": 128},
        "activation": {"batch": 1, "seq": 16384, "dim": 8192},
    },
    "non_contiguous": {
        "attention": {"batch": 2, "seq": 256, "heads": 8, "head_dim": 64, "stride": 2},
        "activation": {"batch": 4, "seq": 256, "dim": 1024, "stride": 2},
    },
    "powers_of_two": {
        "attention": {"batch": 2, "seq": 512, "heads": 8, "head_dim": 64},
        "activation": {"batch": 4, "seq": 512, "dim": 2048},
    },
}


# =============================================================================
# Input Generator Functions
# =============================================================================

def attention_inputs(
    batch: int,
    seq: int,
    heads: int,
    head_dim: int,
    dtype: str = "fp32",
    causal: bool = False,
    kv_seq: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    distribution: str = "normal",
) -> Dict[str, np.ndarray]:
    """Generate inputs for attention operations.

    Args:
        batch: Batch size.
        seq: Sequence length for queries.
        heads: Number of attention heads.
        head_dim: Dimension per head.
        dtype: Data type string.
        causal: Whether this is for causal attention.
        kv_seq: Sequence length for keys/values (defaults to seq).
        num_kv_heads: Number of KV heads for GQA/MQA (defaults to heads).
        distribution: Input distribution ("normal", "uniform").

    Returns:
        Dictionary with q, k, v arrays and metadata.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)
    kv_seq = kv_seq or seq
    num_kv_heads = num_kv_heads or heads

    if distribution == "uniform":
        q = np.random.uniform(-1, 1, (batch, seq, heads, head_dim)).astype(np_dtype)
        k = np.random.uniform(-1, 1, (batch, kv_seq, num_kv_heads, head_dim)).astype(np_dtype)
        v = np.random.uniform(-1, 1, (batch, kv_seq, num_kv_heads, head_dim)).astype(np_dtype)
    else:
        q = np.random.randn(batch, seq, heads, head_dim).astype(np_dtype)
        k = np.random.randn(batch, kv_seq, num_kv_heads, head_dim).astype(np_dtype)
        v = np.random.randn(batch, kv_seq, num_kv_heads, head_dim).astype(np_dtype)

    return {
        "q": q,
        "k": k,
        "v": v,
        "causal": causal,
        "scale": 1.0 / np.sqrt(head_dim),
    }


def activation_inputs(
    batch: int,
    seq: int,
    dim: int,
    dtype: str = "fp32",
    distribution: str = "normal",
    include_gate: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate inputs for activation functions.

    Args:
        batch: Batch size.
        seq: Sequence length.
        dim: Hidden dimension.
        dtype: Data type string.
        distribution: Input distribution.
        include_gate: Whether to include gate weights (for GLU variants).

    Returns:
        Dictionary with input arrays.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)

    if distribution == "uniform":
        x = np.random.uniform(-2, 2, (batch, seq, dim)).astype(np_dtype)
    else:
        x = np.random.randn(batch, seq, dim).astype(np_dtype)

    result = {"x": x}

    if include_gate:
        # For GLU variants: x is split or separate gate/up projections
        result["W_gate"] = np.random.randn(dim, dim).astype(np_dtype) * 0.02
        result["W_up"] = np.random.randn(dim, dim).astype(np_dtype) * 0.02

    return result


def normalization_inputs(
    batch: int,
    seq: int,
    hidden: int,
    dtype: str = "fp32",
    num_groups: int = 1,
    include_bias: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate inputs for normalization operations.

    Args:
        batch: Batch size.
        seq: Sequence length.
        hidden: Hidden dimension.
        dtype: Data type string.
        num_groups: Number of groups for GroupNorm.
        include_bias: Whether to include bias parameter.

    Returns:
        Dictionary with input arrays.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)

    x = np.random.randn(batch, seq, hidden).astype(np_dtype)
    weight = np.ones(hidden, dtype=np_dtype)

    result = {
        "x": x,
        "weight": weight,
        "eps": 1e-5,
    }

    if include_bias:
        result["bias"] = np.zeros(hidden, dtype=np_dtype)

    if num_groups > 1:
        result["num_groups"] = num_groups

    return result


def quantization_inputs(
    m: int,
    n: int,
    k: int,
    bits: int = 8,
    dtype: str = "fp32",
) -> Dict[str, np.ndarray]:
    """Generate inputs for quantization operations.

    Args:
        m: First dimension (batch * seq for linear).
        n: Output features.
        k: Input features.
        bits: Quantization bits (4 or 8).
        dtype: Data type string for input.

    Returns:
        Dictionary with input arrays.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)

    x = np.random.randn(m, k).astype(np_dtype)
    weight = np.random.randn(n, k).astype(np_dtype) * 0.02

    return {
        "x": x,
        "weight": weight,
        "bits": bits,
    }


def moe_inputs(
    batch: int,
    seq: int,
    dim: int,
    experts: int,
    top_k: int,
    dtype: str = "fp32",
) -> Dict[str, np.ndarray]:
    """Generate inputs for MoE operations.

    Args:
        batch: Batch size.
        seq: Sequence length.
        dim: Hidden dimension.
        experts: Number of experts.
        top_k: Number of experts per token.
        dtype: Data type string.

    Returns:
        Dictionary with input arrays.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)

    x = np.random.randn(batch, seq, dim).astype(np_dtype)
    router_logits = np.random.randn(batch * seq, experts).astype(np_dtype)

    return {
        "x": x,
        "router_logits": router_logits,
        "top_k": top_k,
        "num_experts": experts,
    }


def pooling_inputs(
    batch: int,
    channels: int,
    height: int,
    width: int,
    dtype: str = "fp32",
    output_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """Generate inputs for pooling operations.

    Args:
        batch: Batch size.
        channels: Number of channels.
        height: Input height.
        width: Input width.
        dtype: Data type string.
        output_size: Target output size for adaptive pooling.

    Returns:
        Dictionary with input arrays.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)

    x = np.random.randn(batch, channels, height, width).astype(np_dtype)

    result = {"x": x}
    if output_size is not None:
        result["output_size"] = output_size

    return result


def embedding_inputs(
    batch: int,
    seq: int,
    vocab_size: int,
    dim: int,
    dtype: str = "fp32",
    max_seq_len: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generate inputs for embedding operations.

    Args:
        batch: Batch size.
        seq: Sequence length.
        vocab_size: Vocabulary size.
        dim: Embedding dimension.
        dtype: Data type string.
        max_seq_len: Maximum sequence length for positional embeddings.

    Returns:
        Dictionary with input arrays.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)

    # Token indices
    indices = np.random.randint(0, vocab_size, (batch, seq))

    # Position indices
    positions = np.arange(seq)[None, :].repeat(batch, axis=0)

    # Embedding weights
    weight = np.random.randn(vocab_size, dim).astype(np_dtype) * 0.02

    return {
        "indices": indices,
        "positions": positions,
        "weight": weight,
        "dim": dim,
        "max_seq_len": max_seq_len or seq * 2,
    }


def scan_inputs(
    batch: int,
    seq: int,
    dim: int,
    dtype: str = "fp32",
    operator: str = "add",
) -> Dict[str, np.ndarray]:
    """Generate inputs for scan operations.

    Args:
        batch: Batch size.
        seq: Sequence length.
        dim: Feature dimension.
        dtype: Data type string.
        operator: Scan operator ("add", "mul", "ssm").

    Returns:
        Dictionary with input arrays.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)

    x = np.random.randn(batch, seq, dim).astype(np_dtype)

    result = {
        "x": x,
        "axis": 1,
        "operator": operator,
    }

    if operator == "ssm":
        # Additional SSM parameters
        result["A"] = np.random.randn(dim).astype(np_dtype) * 0.1
        result["B"] = np.random.randn(batch, seq, dim).astype(np_dtype)
        result["C"] = np.random.randn(batch, seq, dim).astype(np_dtype)
        result["D"] = np.random.randn(dim).astype(np_dtype)
        result["delta"] = np.abs(np.random.randn(batch, seq, dim).astype(np_dtype)) + 0.01

    return result


def cache_inputs(
    batch: int,
    seq: int,
    heads: int,
    head_dim: int,
    block_size: int,
    dtype: str = "fp32",
    num_blocks: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generate inputs for cache operations.

    Args:
        batch: Batch size.
        seq: Sequence length.
        heads: Number of attention heads.
        head_dim: Dimension per head.
        block_size: Block size for paged attention.
        dtype: Data type string.
        num_blocks: Number of blocks (computed from seq if not provided).

    Returns:
        Dictionary with input arrays.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)

    num_blocks = num_blocks or (seq + block_size - 1) // block_size

    # Query for current position
    q = np.random.randn(batch, 1, heads, head_dim).astype(np_dtype)

    # KV cache blocks
    k_cache = np.random.randn(num_blocks, block_size, heads, head_dim).astype(np_dtype)
    v_cache = np.random.randn(num_blocks, block_size, heads, head_dim).astype(np_dtype)

    # Block tables mapping sequences to blocks
    max_blocks_per_seq = num_blocks
    block_tables = np.zeros((batch, max_blocks_per_seq), dtype=np.int32)
    for b in range(batch):
        block_tables[b, :num_blocks] = np.arange(num_blocks)

    # Sequence lengths
    seq_lens = np.full(batch, seq, dtype=np.int32)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "block_size": block_size,
    }


def sampling_inputs(
    batch: int,
    vocab_size: int,
    dtype: str = "fp32",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Dict[str, np.ndarray]:
    """Generate inputs for sampling operations.

    Args:
        batch: Batch size.
        vocab_size: Vocabulary size.
        dtype: Data type string.
        temperature: Sampling temperature.
        top_k: Top-K for sampling.
        top_p: Top-P (nucleus) for sampling.

    Returns:
        Dictionary with input arrays.
    """
    np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}.get(dtype, np.float32)

    # Logits for next token prediction
    logits = np.random.randn(batch, vocab_size).astype(np_dtype)

    return {
        "logits": logits,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def get_size_config(size: str, category: str) -> Dict[str, Any]:
    """Get size configuration for a category.

    Args:
        size: Size name (tiny, small, medium, large).
        category: Category name (attention, activation, etc.).

    Returns:
        Configuration dictionary.
    """
    return SIZE_CONFIGS.get(size, SIZE_CONFIGS["small"]).get(category, {})


def get_edge_case_config(case: str, category: str) -> Dict[str, Any]:
    """Get edge case configuration for a category.

    Args:
        case: Edge case name.
        category: Category name.

    Returns:
        Configuration dictionary.
    """
    return EDGE_CASE_CONFIGS.get(case, {}).get(category, {})
