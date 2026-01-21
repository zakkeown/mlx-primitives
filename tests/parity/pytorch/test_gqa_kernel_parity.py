"""PyTorch parity tests for GQA kernel implementations."""

import math
import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance
from tests.parity.conftest import get_mlx_dtype, get_pytorch_dtype, HAS_PYTORCH

if HAS_PYTORCH:
    import torch
    import torch.nn.functional as F


# =============================================================================
# GQA-Specific Size Configurations
# =============================================================================

GQA_SIZE_CONFIGS = {
    "tiny": {
        "batch": 1, "seq_q": 64, "seq_kv": 64,
        "num_q_heads": 8, "num_kv_heads": 2, "head_dim": 32
    },
    "small": {
        "batch": 2, "seq_q": 256, "seq_kv": 256,
        "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64
    },
    "medium": {
        "batch": 4, "seq_q": 512, "seq_kv": 512,
        "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64
    },
    "large": {
        "batch": 2, "seq_q": 512, "seq_kv": 512,
        "num_q_heads": 64, "num_kv_heads": 8, "head_dim": 64  # Llama-2 70B style
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_to_mlx(x_np: np.ndarray, dtype: str) -> mx.array:
    """Convert numpy array to MLX with proper dtype."""
    x_mlx = mx.array(x_np)
    mlx_dtype = get_mlx_dtype(dtype)
    return x_mlx.astype(mlx_dtype)


def _convert_to_torch(x_np: np.ndarray, dtype: str) -> "torch.Tensor":
    """Convert numpy array to PyTorch with proper dtype."""
    x_torch = torch.from_numpy(x_np.astype(np.float32))
    torch_dtype = get_pytorch_dtype(dtype)
    return x_torch.to(torch_dtype)


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or PyTorch tensor to numpy."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_PYTORCH and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


# =============================================================================
# PyTorch Reference Implementations
# =============================================================================

def pytorch_gqa_attention(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    num_kv_groups: int,
    scale: float = None,
    causal: bool = False,
) -> "torch.Tensor":
    """PyTorch reference for GQA with K/V expansion.

    Args:
        q: Query tensor (batch, seq_q, num_q_heads, head_dim)
        k: Key tensor (batch, seq_kv, num_kv_heads, head_dim)
        v: Value tensor (batch, seq_kv, num_kv_heads, head_dim)
        num_kv_groups: Number of Q heads per KV head
        scale: Attention scale factor
        causal: Whether to apply causal masking

    Returns:
        Output tensor (batch, seq_q, num_q_heads, head_dim)
    """
    batch_size, seq_q, num_q_heads, head_dim = q.shape
    _, seq_kv, num_kv_heads, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Expand K and V: (batch, seq, kv_heads, dim) -> (batch, seq, q_heads, dim)
    # by repeating each KV head 'num_kv_groups' times
    k_expanded = k.unsqueeze(3).expand(
        batch_size, seq_kv, num_kv_heads, num_kv_groups, head_dim
    )
    k_expanded = k_expanded.reshape(batch_size, seq_kv, num_q_heads, head_dim)

    v_expanded = v.unsqueeze(3).expand(
        batch_size, seq_kv, num_kv_heads, num_kv_groups, head_dim
    )
    v_expanded = v_expanded.reshape(batch_size, seq_kv, num_q_heads, head_dim)

    # Transpose for attention: (batch, heads, seq, dim)
    q_t = q.transpose(1, 2)
    k_t = k_expanded.transpose(1, 2)
    v_t = v_expanded.transpose(1, 2)

    # Compute attention scores
    # Use float32 accumulation for numerical stability
    q_fp32 = q_t.float()
    k_fp32 = k_t.float()
    v_fp32 = v_t.float()

    scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * scale

    # Causal mask
    if causal:
        # Create upper triangular mask
        mask = torch.triu(
            torch.full((seq_q, seq_kv), float("-inf"), device=scores.device),
            diagonal=seq_kv - seq_q + 1,
        )
        scores = scores + mask

    # Softmax and output
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v_fp32)

    # Convert back to original dtype and transpose
    output = output.to(q.dtype)
    return output.transpose(1, 2)


# =============================================================================
# Fast GQA Attention (Metal Kernel) Parity Tests
# =============================================================================

class TestFastGQAAttentionParity:
    """Tests for fast_gqa_attention() Metal kernel parity."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("causal", [False, True])
    def test_forward_parity(self, size, dtype, causal, skip_without_pytorch):
        """Test fast_gqa_attention forward pass matches PyTorch."""
        from mlx_primitives.kernels.gqa_optimized import fast_gqa_attention

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq_q = config["seq_q"]
        seq_kv = config["seq_kv"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_q, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)

        # MLX
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        v_mlx = _convert_to_mlx(v_np, dtype)
        mlx_out = fast_gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups, causal=causal)
        mx.eval(mlx_out)

        # PyTorch reference
        q_torch = _convert_to_torch(q_np, dtype)
        k_torch = _convert_to_torch(k_np, dtype)
        v_torch = _convert_to_torch(v_np, dtype)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups, causal=causal)

        rtol, atol = get_tolerance("attention", "gqa", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_gqa_attention mismatch [{size}, {dtype}, causal={causal}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.parametrize("num_kv_groups", [1, 2, 4, 8])
    def test_various_group_ratios(self, num_kv_groups, skip_without_pytorch):
        """Test fast_gqa with different Q-to-KV head ratios."""
        from mlx_primitives.kernels.gqa_optimized import fast_gqa_attention

        batch, seq, head_dim = 2, 128, 64
        num_kv_heads = 4
        num_q_heads = num_kv_heads * num_kv_groups

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = fast_gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(mlx_out)

        q_torch = torch.from_numpy(q_np)
        k_torch = torch.from_numpy(k_np)
        v_torch = torch.from_numpy(v_np)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups)

        rtol, atol = get_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"fast_gqa with num_kv_groups={num_kv_groups} mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.parametrize("use_tiled", [True, False])
    def test_tiled_vs_simple_kernel(self, use_tiled, skip_without_pytorch):
        """Test both tiled and simple GQA kernels produce same results."""
        from mlx_primitives.kernels.gqa_optimized import fast_gqa_attention

        batch, seq, num_q_heads, num_kv_heads, head_dim = 2, 128, 16, 4, 64
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = fast_gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups, use_tiled=use_tiled)
        mx.eval(mlx_out)

        q_torch = torch.from_numpy(q_np)
        k_torch = torch.from_numpy(k_np)
        v_torch = torch.from_numpy(v_np)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups)

        rtol, atol = get_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"fast_gqa use_tiled={use_tiled} mismatch"
        )


# =============================================================================
# GQA Reference Implementation Parity Tests
# =============================================================================

class TestGQAAttentionReferenceParity:
    """Tests for gqa_attention_reference() parity."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test gqa_attention_reference matches PyTorch."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq_q = config["seq_q"]
        seq_kv = config["seq_kv"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_q, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)

        # MLX reference
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        v_mlx = _convert_to_mlx(v_np, dtype)
        mlx_out = gqa_attention_reference(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(mlx_out)

        # PyTorch reference
        q_torch = _convert_to_torch(q_np, dtype)
        k_torch = _convert_to_torch(k_np, dtype)
        v_torch = _convert_to_torch(v_np, dtype)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups)

        rtol, atol = get_tolerance("attention", "gqa", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"gqa_attention_reference mismatch [{size}, {dtype}]"
        )


# =============================================================================
# GQA Auto-Select Path Parity Tests
# =============================================================================

class TestGQAAttentionAutoSelectParity:
    """Tests for gqa_attention() with automatic path selection."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test gqa_attention() auto-selection matches PyTorch."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq_q = config["seq_q"]
        seq_kv = config["seq_kv"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_q, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)

        # MLX auto-select
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        v_mlx = _convert_to_mlx(v_np, dtype)
        mlx_out = gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(mlx_out)

        # PyTorch reference
        q_torch = _convert_to_torch(q_np, dtype)
        k_torch = _convert_to_torch(k_np, dtype)
        v_torch = _convert_to_torch(v_np, dtype)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups)

        rtol, atol = get_tolerance("attention", "gqa", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"gqa_attention auto-select mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    def test_reference_vs_optimized_consistency(self, skip_without_pytorch):
        """Verify optimized paths match reference implementation."""
        from mlx_primitives.kernels.gqa_optimized import (
            gqa_attention, gqa_attention_reference, fast_gqa_attention
        )

        batch, seq, num_q_heads, num_kv_heads, head_dim = 2, 128, 16, 4, 64
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)

        # Reference
        out_ref = gqa_attention_reference(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(out_ref)

        # Fast kernel
        out_fast = fast_gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(out_fast)

        # Auto-select
        out_auto = gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(out_auto)

        rtol, atol = get_tolerance("attention", "gqa", "fp32")

        np.testing.assert_allclose(
            _to_numpy(out_fast), _to_numpy(out_ref),
            rtol=rtol, atol=atol,
            err_msg="fast_gqa differs from reference"
        )

        # Note: auto may use MLX SDPA which has its own implementation
        # Just verify it's close to reference
        np.testing.assert_allclose(
            _to_numpy(out_auto), _to_numpy(out_ref),
            rtol=rtol, atol=atol,
            err_msg="gqa_attention auto differs from reference"
        )


# =============================================================================
# OptimizedGQA Module Parity Tests
# =============================================================================

class TestOptimizedGQAModuleParity:
    """Tests for OptimizedGQA nn.Module parity."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16"])
    def test_module_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test OptimizedGQA module forward pass."""
        from mlx_primitives.kernels.gqa_optimized import OptimizedGQA

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq = config["seq_q"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        dims = num_q_heads * head_dim

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        # Create MLX module
        gqa = OptimizedGQA(
            dims=dims,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=False,
        )
        mx.eval(gqa.parameters())

        # Extract weights for PyTorch comparison
        q_weight = np.array(gqa.q_proj.weight)
        k_weight = np.array(gqa.k_proj.weight)
        v_weight = np.array(gqa.v_proj.weight)
        out_weight = np.array(gqa.out_proj.weight)

        # MLX forward
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out, _ = gqa(x_mlx)
        mx.eval(mlx_out)

        # PyTorch manual forward (matching OptimizedGQA logic)
        x_torch = _convert_to_torch(x_np, dtype)

        # Project Q, K, V
        q_torch = x_torch @ _convert_to_torch(q_weight.T, dtype)
        k_torch = x_torch @ _convert_to_torch(k_weight.T, dtype)
        v_torch = x_torch @ _convert_to_torch(v_weight.T, dtype)

        # Reshape to (batch, seq, heads, dim)
        q_torch = q_torch.reshape(batch, seq, num_q_heads, head_dim)
        k_torch = k_torch.reshape(batch, seq, num_kv_heads, head_dim)
        v_torch = v_torch.reshape(batch, seq, num_kv_heads, head_dim)

        # GQA attention
        num_kv_groups = num_q_heads // num_kv_heads
        attn_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups)

        # Output projection
        attn_out = attn_out.reshape(batch, seq, -1)
        torch_out = attn_out @ _convert_to_torch(out_weight.T, dtype)

        rtol, atol = get_tolerance("attention", "gqa", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"OptimizedGQA module mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    def test_module_with_kv_cache(self, skip_without_pytorch):
        """Test OptimizedGQA with KV cache (incremental decoding)."""
        from mlx_primitives.kernels.gqa_optimized import OptimizedGQA

        batch, seq, num_q_heads, num_kv_heads, head_dim = 2, 32, 16, 4, 64
        dims = num_q_heads * head_dim

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        gqa = OptimizedGQA(
            dims=dims,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            causal=True,
        )
        mx.eval(gqa.parameters())

        x_mlx = mx.array(x_np)

        # First forward (full context)
        out1, cache = gqa(x_mlx)
        mx.eval(out1, cache[0], cache[1])

        # Second forward with single token (incremental)
        new_token = mx.array(np.random.randn(batch, 1, dims).astype(np.float32))
        out2, cache2 = gqa(new_token, cache=cache)
        mx.eval(out2, cache2[0], cache2[1])

        # Verify cache grew
        assert cache2[0].shape[1] == seq + 1, "KV cache should grow by 1"
        assert out2.shape == (batch, 1, dims), "Output should be single token"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestGQAEdgeCases:
    """Edge case tests for GQA implementations."""

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_single_token_decoding(self, skip_without_pytorch):
        """Test GQA with single query token (common in decoding)."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention

        batch, seq_kv, num_q_heads, num_kv_heads, head_dim = 2, 128, 32, 8, 64
        seq_q = 1  # Single token
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_q, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups, causal=False)
        mx.eval(mlx_out)

        q_torch = torch.from_numpy(q_np)
        k_torch = torch.from_numpy(k_np)
        v_torch = torch.from_numpy(v_np)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups, causal=False)

        rtol, atol = get_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg="Single token GQA mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_mqa_single_kv_head(self, skip_without_pytorch):
        """Test MQA (Multi-Query Attention) - single KV head."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention

        batch, seq, num_q_heads, head_dim = 2, 64, 8, 64
        num_kv_heads = 1  # MQA
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(mlx_out)

        q_torch = torch.from_numpy(q_np)
        k_torch = torch.from_numpy(k_np)
        v_torch = torch.from_numpy(v_np)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups)

        rtol, atol = get_tolerance("attention", "mqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg="MQA (single KV head) mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_standard_mha(self, skip_without_pytorch):
        """Test that GQA with groups=1 equals standard MHA."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention

        batch, seq, num_heads, head_dim = 2, 64, 8, 64
        num_kv_groups = 1  # Standard MHA

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(mlx_out)

        q_torch = torch.from_numpy(q_np)
        k_torch = torch.from_numpy(k_np)
        v_torch = torch.from_numpy(v_np)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups)

        rtol, atol = get_tolerance("attention", "flash_attention", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg="GQA with groups=1 should match MHA"
        )


# =============================================================================
# GQA Backward Pass Parity Tests
# =============================================================================

class TestGQABackwardParity:
    """Tests for GQA backward pass (gradient) parity with PyTorch."""

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_backward_parity(self, size, dtype, skip_without_pytorch):
        """Test GQA backward pass gradients match PyTorch."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq_q = config["seq_q"]
        seq_kv = config["seq_kv"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_q, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)

        # MLX backward
        def mlx_loss_fn(q, k, v):
            out = gqa_attention_reference(q, k, v, num_kv_groups)
            return mx.sum(out)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1, 2))
        q_grad, k_grad, v_grad = grad_fn(q_mlx, k_mlx, v_mlx)
        mx.eval(q_grad, k_grad, v_grad)

        # PyTorch backward
        q_torch = torch.from_numpy(q_np).requires_grad_(True)
        k_torch = torch.from_numpy(k_np).requires_grad_(True)
        v_torch = torch.from_numpy(v_np).requires_grad_(True)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_grad), q_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GQA Q gradient mismatch [{size}, {dtype}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_grad), k_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GQA K gradient mismatch [{size}, {dtype}]"
        )
        np.testing.assert_allclose(
            _to_numpy(v_grad), v_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GQA V gradient mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("num_kv_groups", [1, 2, 4, 8])
    def test_backward_various_group_ratios(self, num_kv_groups, skip_without_pytorch):
        """Test backward pass with different Q-to-KV head ratios."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        batch, seq, head_dim = 2, 64, 64
        num_kv_heads = 4
        num_q_heads = num_kv_heads * num_kv_groups

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

        # MLX backward
        def mlx_loss_fn(q, k, v):
            out = gqa_attention_reference(q, k, v, num_kv_groups)
            return mx.sum(out)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1, 2))
        q_grad, k_grad, v_grad = grad_fn(q_mlx, k_mlx, v_mlx)
        mx.eval(q_grad, k_grad, v_grad)

        # PyTorch backward
        q_torch = torch.from_numpy(q_np).requires_grad_(True)
        k_torch = torch.from_numpy(k_np).requires_grad_(True)
        v_torch = torch.from_numpy(v_np).requires_grad_(True)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_grad), q_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GQA Q grad mismatch [groups={num_kv_groups}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_grad), k_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GQA K grad mismatch [groups={num_kv_groups}]"
        )
        np.testing.assert_allclose(
            _to_numpy(v_grad), v_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GQA V grad mismatch [groups={num_kv_groups}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("causal", [False, True])
    def test_backward_causal_vs_non_causal(self, causal, skip_without_pytorch):
        """Test backward pass with and without causal masking."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        batch, seq, num_q_heads, num_kv_heads, head_dim = 2, 64, 16, 4, 64
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

        # MLX backward
        def mlx_loss_fn(q, k, v):
            out = gqa_attention_reference(q, k, v, num_kv_groups, causal=causal)
            return mx.sum(out)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1, 2))
        q_grad, k_grad, v_grad = grad_fn(q_mlx, k_mlx, v_mlx)
        mx.eval(q_grad, k_grad, v_grad)

        # PyTorch backward
        q_torch = torch.from_numpy(q_np).requires_grad_(True)
        k_torch = torch.from_numpy(k_np).requires_grad_(True)
        v_torch = torch.from_numpy(v_np).requires_grad_(True)
        torch_out = pytorch_gqa_attention(q_torch, k_torch, v_torch, num_kv_groups, causal=causal)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_grad), q_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GQA Q grad mismatch [causal={causal}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_grad), k_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GQA K grad mismatch [causal={causal}]"
        )
        np.testing.assert_allclose(
            _to_numpy(v_grad), v_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GQA V grad mismatch [causal={causal}]"
        )
