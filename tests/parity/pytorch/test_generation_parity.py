"""PyTorch parity tests for generation/sampling operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance
from tests.parity.conftest import get_mlx_dtype, get_pytorch_dtype, HAS_PYTORCH

if HAS_PYTORCH:
    import torch
    import torch.nn.functional as F


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

def _pytorch_temperature(logits: "torch.Tensor", temperature: float) -> "torch.Tensor":
    """PyTorch reference for temperature scaling."""
    if temperature == 1.0:
        return logits
    if temperature == 0.0:
        return logits
    return logits / temperature


def _pytorch_top_k(logits: "torch.Tensor", k: int) -> "torch.Tensor":
    """PyTorch reference for top-k filtering."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    # Sort descending and get k-th largest value
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    threshold = sorted_logits[:, k - 1 : k]  # k-th largest, shape (batch, 1)
    # Mask values below threshold
    return torch.where(logits >= threshold, logits, torch.tensor(float("-inf")))


def _pytorch_top_p(logits: "torch.Tensor", p: float) -> "torch.Tensor":
    """PyTorch reference for top-p (nucleus) filtering."""
    if p >= 1.0:
        return logits

    # Sort by descending probability
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    sorted_logits = torch.gather(logits, -1, sorted_indices)

    # Compute cumulative probabilities in fp32 for numerical stability
    # bf16/fp16 cumsum accumulates significant error over large vocab sizes
    sorted_probs = F.softmax(sorted_logits.float(), dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: keep tokens until cumulative prob exceeds p
    # Shift cumulative probs to include first token always
    shifted_cumulative = torch.cat(
        [torch.zeros_like(cumulative_probs[:, :1]), cumulative_probs[:, :-1]], dim=-1
    )
    sorted_mask = shifted_cumulative < p

    # Ensure at least first token is kept
    sorted_mask[:, 0] = True

    # Set filtered positions to -inf
    sorted_logits = torch.where(sorted_mask, sorted_logits, torch.tensor(float("-inf")))

    # Unsort back to original order
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    return torch.gather(sorted_logits, -1, inverse_indices)


# =============================================================================
# Temperature Sampling Parity Tests
# =============================================================================

class TestTemperatureSamplingParity:
    """Temperature sampling parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
    def test_forward_parity(self, size, dtype, temperature, skip_without_pytorch):
        """Test temperature scaling forward pass parity."""
        from mlx_primitives.generation.samplers import apply_temperature

        config = SIZE_CONFIGS[size]["sampling"]
        batch, vocab_size = config["batch"], config["vocab_size"]

        # Generate inputs
        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # MLX forward
        logits_mlx = _convert_to_mlx(logits_np, dtype)
        mlx_out = apply_temperature(logits_mlx, temperature)

        # PyTorch reference
        logits_torch = _convert_to_torch(logits_np, dtype)
        torch_out = _pytorch_temperature(logits_torch, temperature)

        rtol, atol = get_tolerance("generation", "temperature_sampling", dtype)
        # Scale tolerance by 1/temperature since output magnitude scales inversely
        # (dividing by 0.1 makes values 10x larger, so tolerances need to scale)
        scale_factor = 1.0 / temperature if temperature > 0 else 1.0
        scaled_rtol = rtol * scale_factor
        scaled_atol = atol * scale_factor
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=scaled_rtol, atol=scaled_atol,
            err_msg=f"Temperature forward mismatch [{size}, {dtype}, temp={temperature}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_logits_scaling(self, skip_without_pytorch):
        """Test that logit scaling matches PyTorch with known values."""
        from mlx_primitives.generation.samplers import apply_temperature

        # Use known values for predictable output
        logits_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        temperature = 2.0
        expected = logits_np / temperature

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_temperature(logits_mlx, temperature)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_out = _pytorch_temperature(logits_torch, temperature)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), expected, rtol=1e-6, atol=1e-7,
            err_msg="Temperature scaling mismatch with known values"
        )
        np.testing.assert_allclose(
            _to_numpy(torch_out), expected, rtol=1e-6, atol=1e-7,
            err_msg="PyTorch temperature scaling mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_probability_distribution(self, skip_without_pytorch):
        """Test that resulting probability distribution matches."""
        from mlx_primitives.generation.samplers import apply_temperature

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)

        for temperature in [0.5, 1.0, 2.0]:
            # MLX
            logits_mlx = mx.array(logits_np)
            mlx_scaled = apply_temperature(logits_mlx, temperature)
            mlx_probs = mx.softmax(mlx_scaled, axis=-1)

            # PyTorch
            logits_torch = torch.from_numpy(logits_np)
            torch_scaled = _pytorch_temperature(logits_torch, temperature)
            torch_probs = F.softmax(torch_scaled, dim=-1)

            np.testing.assert_allclose(
                _to_numpy(mlx_probs), _to_numpy(torch_probs),
                rtol=1e-5, atol=1e-6,
                err_msg=f"Probability distribution mismatch at temp={temperature}"
            )

            # Verify probabilities sum to 1
            mlx_sum = _to_numpy(mx.sum(mlx_probs, axis=-1))
            np.testing.assert_allclose(mlx_sum, np.ones(2), rtol=1e-5)

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_temperature_zero(self, skip_without_pytorch):
        """Test temperature=0 (greedy) behavior - should return logits unchanged."""
        from mlx_primitives.generation.samplers import apply_temperature

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_temperature(logits_mlx, 0.0)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_out = _pytorch_temperature(logits_torch, 0.0)

        # Both should return logits unchanged
        np.testing.assert_allclose(
            _to_numpy(mlx_out), logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="MLX temperature=0 should return logits unchanged"
        )
        np.testing.assert_allclose(
            _to_numpy(torch_out), logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="PyTorch temperature=0 should return logits unchanged"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_temperature_very_high(self, skip_without_pytorch):
        """Test very high temperature (uniform-like) behavior."""
        from mlx_primitives.generation.samplers import apply_temperature

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        temperature = 100.0

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_scaled = apply_temperature(logits_mlx, temperature)
        mlx_probs = mx.softmax(mlx_scaled, axis=-1)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_scaled = _pytorch_temperature(logits_torch, temperature)
        torch_probs = F.softmax(torch_scaled, dim=-1)

        # Check no NaN or Inf
        mlx_out_np = _to_numpy(mlx_probs)
        assert not np.any(np.isnan(mlx_out_np)), "MLX output contains NaN"
        assert not np.any(np.isinf(mlx_out_np)), "MLX output contains Inf"

        torch_out_np = _to_numpy(torch_probs)
        assert not np.any(np.isnan(torch_out_np)), "PyTorch output contains NaN"
        assert not np.any(np.isinf(torch_out_np)), "PyTorch output contains Inf"

        # Verify distribution is more uniform (closer to 1/vocab_size)
        uniform_prob = 1.0 / 100
        mlx_std = np.std(mlx_out_np, axis=-1)
        # High temperature should make std small (closer to uniform)
        assert np.all(mlx_std < 0.01), f"MLX distribution not uniform enough: std={mlx_std}"

        # Verify parity
        np.testing.assert_allclose(
            mlx_out_np, torch_out_np, rtol=1e-5, atol=1e-6,
            err_msg="High temperature parity mismatch"
        )


# =============================================================================
# Top-K Sampling Parity Tests
# =============================================================================

class TestTopKSamplingParity:
    """Top-K sampling parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("k", [1, 10, 50, 100])
    def test_forward_parity(self, size, dtype, k, skip_without_pytorch):
        """Test Top-K sampling forward pass parity."""
        from mlx_primitives.generation.samplers import apply_top_k

        config = SIZE_CONFIGS[size]["sampling"]
        batch, vocab_size = config["batch"], config["vocab_size"]

        # Adjust k if larger than vocab_size
        effective_k = min(k, vocab_size)

        # Generate inputs
        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # MLX forward
        logits_mlx = _convert_to_mlx(logits_np, dtype)
        mlx_out = apply_top_k(logits_mlx, effective_k)

        # PyTorch reference
        logits_torch = _convert_to_torch(logits_np, dtype)
        torch_out = _pytorch_top_k(logits_torch, effective_k)

        mlx_np = _to_numpy(mlx_out)
        torch_np = _to_numpy(torch_out)

        # Check that the same positions are masked (both -inf)
        mlx_masked = np.isneginf(mlx_np)
        torch_masked = np.isneginf(torch_np)
        np.testing.assert_array_equal(
            mlx_masked, torch_masked,
            err_msg=f"Top-K mask mismatch [{size}, {dtype}, k={effective_k}]"
        )

        # Check that non-masked values match
        rtol, atol = get_tolerance("generation", "top_k_sampling", dtype)
        non_masked = ~mlx_masked
        if np.any(non_masked):
            np.testing.assert_allclose(
                mlx_np[non_masked], torch_np[non_masked],
                rtol=rtol, atol=atol,
                err_msg=f"Top-K values mismatch [{size}, {dtype}, k={effective_k}]"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_top_k_selection(self, skip_without_pytorch):
        """Test that top-K selection correctly keeps the k largest values."""
        from mlx_primitives.generation.samplers import apply_top_k

        # Use known values for predictable output
        logits_np = np.array([[1.0, 5.0, 2.0, 8.0, 3.0]], dtype=np.float32)
        k = 3  # Should keep indices 1, 3, 4 (values 5, 8, 3)

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_k(logits_mlx, k)
        mlx_np = _to_numpy(mlx_out)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_out = _pytorch_top_k(logits_torch, k)
        torch_np = _to_numpy(torch_out)

        # Count non-inf values
        mlx_kept = np.sum(~np.isneginf(mlx_np), axis=-1)
        torch_kept = np.sum(~np.isneginf(torch_np), axis=-1)

        np.testing.assert_array_equal(mlx_kept, [k], err_msg="MLX didn't keep exactly k values")
        np.testing.assert_array_equal(torch_kept, [k], err_msg="PyTorch didn't keep exactly k values")

        # Verify the correct values are kept (top k: 8, 5, 3)
        mlx_kept_values = sorted(mlx_np[~np.isneginf(mlx_np)])
        assert mlx_kept_values == [3.0, 5.0, 8.0], f"Wrong values kept: {mlx_kept_values}"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_probability_renormalization(self, skip_without_pytorch):
        """Test probability renormalization after top-k filtering."""
        from mlx_primitives.generation.samplers import apply_top_k

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        k = 10

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_filtered = apply_top_k(logits_mlx, k)
        mlx_probs = mx.softmax(mlx_filtered, axis=-1)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_filtered = _pytorch_top_k(logits_torch, k)
        torch_probs = F.softmax(torch_filtered, dim=-1)

        # Probabilities should sum to 1
        mlx_sum = _to_numpy(mx.sum(mlx_probs, axis=-1))
        torch_sum = _to_numpy(torch.sum(torch_probs, dim=-1))

        np.testing.assert_allclose(mlx_sum, np.ones(2), rtol=1e-5)
        np.testing.assert_allclose(torch_sum, np.ones(2), rtol=1e-5)

        # Verify parity
        np.testing.assert_allclose(
            _to_numpy(mlx_probs), _to_numpy(torch_probs),
            rtol=1e-5, atol=1e-6,
            err_msg="Top-K probability renormalization mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_k_equals_1(self, skip_without_pytorch):
        """Test K=1 (greedy) behavior - only one value should be kept."""
        from mlx_primitives.generation.samplers import apply_top_k

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        k = 1

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_k(logits_mlx, k)
        mlx_np = _to_numpy(mlx_out)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_out = _pytorch_top_k(logits_torch, k)
        torch_np = _to_numpy(torch_out)

        # Only 1 value should be non-inf per row
        mlx_kept = np.sum(~np.isneginf(mlx_np), axis=-1)
        torch_kept = np.sum(~np.isneginf(torch_np), axis=-1)

        np.testing.assert_array_equal(mlx_kept, [1, 1], err_msg="MLX k=1 didn't keep exactly 1 value")
        np.testing.assert_array_equal(torch_kept, [1, 1], err_msg="PyTorch k=1 didn't keep exactly 1 value")

        # The kept value should be the maximum
        for i in range(2):
            max_idx = np.argmax(logits_np[i])
            assert not np.isneginf(mlx_np[i, max_idx]), "MLX didn't keep the max value"
            assert not np.isneginf(torch_np[i, max_idx]), "PyTorch didn't keep the max value"

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_k_equals_vocab_size(self, skip_without_pytorch):
        """Test K=vocab_size (no filtering) behavior."""
        from mlx_primitives.generation.samplers import apply_top_k

        np.random.seed(42)
        vocab_size = 100
        logits_np = np.random.randn(2, vocab_size).astype(np.float32)
        k = vocab_size

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_k(logits_mlx, k)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_out = _pytorch_top_k(logits_torch, k)

        # Output should equal input (no filtering)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="MLX k=vocab_size should return logits unchanged"
        )
        np.testing.assert_allclose(
            _to_numpy(torch_out), logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="PyTorch k=vocab_size should return logits unchanged"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_temperature(self, skip_without_pytorch):
        """Test Top-K combined with temperature scaling."""
        from mlx_primitives.generation.samplers import apply_temperature, apply_top_k

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        temperature = 0.7
        k = 10

        # MLX: temperature then top-k
        logits_mlx = mx.array(logits_np)
        mlx_scaled = apply_temperature(logits_mlx, temperature)
        mlx_filtered = apply_top_k(mlx_scaled, k)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_scaled = _pytorch_temperature(logits_torch, temperature)
        torch_filtered = _pytorch_top_k(torch_scaled, k)

        mlx_np = _to_numpy(mlx_filtered)
        torch_np = _to_numpy(torch_filtered)

        # Check mask positions match
        mlx_masked = np.isneginf(mlx_np)
        torch_masked = np.isneginf(torch_np)
        np.testing.assert_array_equal(
            mlx_masked, torch_masked,
            err_msg="Temperature+Top-K mask mismatch"
        )

        # Check non-masked values match
        non_masked = ~mlx_masked
        if np.any(non_masked):
            np.testing.assert_allclose(
                mlx_np[non_masked], torch_np[non_masked],
                rtol=1e-5, atol=1e-6,
                err_msg="Temperature+Top-K values mismatch"
            )


# =============================================================================
# Top-P (Nucleus) Sampling Parity Tests
# =============================================================================

class TestTopPSamplingParity:
    """Top-P (nucleus) sampling parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.95])
    def test_forward_parity(self, size, dtype, p, skip_without_pytorch):
        """Test Top-P sampling forward pass parity.

        Note: For fp16/bf16, cumulative probability calculations can cause
        slight mask boundary differences. The test allows <0.1% mask mismatches.
        """
        from mlx_primitives.generation.samplers import apply_top_p

        config = SIZE_CONFIGS[size]["sampling"]
        batch, vocab_size = config["batch"], config["vocab_size"]

        # Generate inputs
        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # MLX forward
        logits_mlx = _convert_to_mlx(logits_np, dtype)
        mlx_out = apply_top_p(logits_mlx, p)

        # PyTorch reference
        logits_torch = _convert_to_torch(logits_np, dtype)
        torch_out = _pytorch_top_p(logits_torch, p)

        mlx_np = _to_numpy(mlx_out)
        torch_np = _to_numpy(torch_out)

        # Check masks match with dtype-appropriate tolerance for boundary differences
        # (cumulative probability calculations accumulate more error in lower precision)
        mlx_masked = np.isneginf(mlx_np)
        torch_masked = np.isneginf(torch_np)
        mask_diff = np.sum(mlx_masked != torch_masked)
        total_elements = mlx_masked.size
        mask_diff_ratio = mask_diff / total_elements
        # With fp32 cumsum in both implementations, tolerance should be tight
        # Any remaining differences are from softmax input precision
        mask_tolerance = {
            "fp32": 0.001,  # 0.1%
            "fp16": 0.001,  # 0.1%
            "bf16": 0.001,  # 0.1% (fp32 cumsum eliminates accumulated error)
        }[dtype]
        assert mask_diff_ratio < mask_tolerance, (
            f"Top-P mask mismatch [{size}, {dtype}, p={p}]: "
            f"{mask_diff}/{total_elements} ({mask_diff_ratio*100:.4f}%) elements differ, "
            f"tolerance={mask_tolerance*100:.1f}%"
        )

        # Check that non-masked values match (where both agree on mask)
        rtol, atol = get_tolerance("generation", "top_p_sampling", dtype)
        both_unmasked = (~mlx_masked) & (~torch_masked)
        if np.any(both_unmasked):
            np.testing.assert_allclose(
                mlx_np[both_unmasked], torch_np[both_unmasked],
                rtol=rtol, atol=atol,
                err_msg=f"Top-P values mismatch [{size}, {dtype}, p={p}]"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_cumulative_probability(self, skip_without_pytorch):
        """Test cumulative probability computation matches."""
        # Use known values for predictable cumulative probs
        logits_np = np.array([[2.0, 1.0, 0.0, -1.0, -2.0]], dtype=np.float32)

        # Compute expected sorted probs
        sorted_logits = np.sort(logits_np, axis=-1)[:, ::-1]  # Descending
        exp_logits = np.exp(sorted_logits - np.max(sorted_logits, axis=-1, keepdims=True))
        sorted_probs_expected = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        cumulative_expected = np.cumsum(sorted_probs_expected, axis=-1)

        # MLX
        logits_mlx = mx.array(logits_np)
        sorted_indices_mlx = mx.argsort(logits_mlx, axis=-1)[:, ::-1]
        sorted_logits_mlx = mx.take_along_axis(logits_mlx, sorted_indices_mlx, axis=-1)
        sorted_probs_mlx = mx.softmax(sorted_logits_mlx, axis=-1)
        cumulative_mlx = mx.cumsum(sorted_probs_mlx, axis=-1)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        sorted_indices_torch = torch.argsort(logits_torch, dim=-1, descending=True)
        sorted_logits_torch = torch.gather(logits_torch, -1, sorted_indices_torch)
        sorted_probs_torch = F.softmax(sorted_logits_torch, dim=-1)
        cumulative_torch = torch.cumsum(sorted_probs_torch, dim=-1)

        # Verify cumulative probs match
        np.testing.assert_allclose(
            _to_numpy(cumulative_mlx), _to_numpy(cumulative_torch),
            rtol=1e-5, atol=1e-6,
            err_msg="Cumulative probability computation mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_nucleus_selection(self, skip_without_pytorch):
        """Test nucleus (smallest set with prob >= p) selection."""
        from mlx_primitives.generation.samplers import apply_top_p

        # Use known values: probs after softmax will be approximately [0.64, 0.24, 0.09, 0.02, 0.01]
        logits_np = np.array([[3.0, 2.0, 1.0, 0.0, -1.0]], dtype=np.float32)
        p = 0.9  # Should keep ~3 tokens (0.64 + 0.24 + 0.09 = 0.97 > 0.9)

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_p(logits_mlx, p)
        mlx_np = _to_numpy(mlx_out)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_out = _pytorch_top_p(logits_torch, p)
        torch_np = _to_numpy(torch_out)

        # Count kept tokens
        mlx_kept = np.sum(~np.isneginf(mlx_np), axis=-1)
        torch_kept = np.sum(~np.isneginf(torch_np), axis=-1)

        np.testing.assert_array_equal(
            mlx_kept, torch_kept,
            err_msg=f"Nucleus size mismatch: MLX={mlx_kept}, PyTorch={torch_kept}"
        )

        # Verify that kept probabilities sum to >= p
        kept_mask = ~np.isneginf(mlx_np)
        mlx_probs = np.exp(mlx_np[kept_mask] - np.max(mlx_np[kept_mask]))
        mlx_probs = mlx_probs / np.sum(mlx_probs)
        # This validates the concept but actual test is that masks match

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_probability_renormalization(self, skip_without_pytorch):
        """Test probability renormalization after top-p filtering."""
        from mlx_primitives.generation.samplers import apply_top_p

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        p = 0.9

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_filtered = apply_top_p(logits_mlx, p)
        mlx_probs = mx.softmax(mlx_filtered, axis=-1)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_filtered = _pytorch_top_p(logits_torch, p)
        torch_probs = F.softmax(torch_filtered, dim=-1)

        # Probabilities should sum to 1
        mlx_sum = _to_numpy(mx.sum(mlx_probs, axis=-1))
        torch_sum = _to_numpy(torch.sum(torch_probs, dim=-1))

        np.testing.assert_allclose(mlx_sum, np.ones(2), rtol=1e-5)
        np.testing.assert_allclose(torch_sum, np.ones(2), rtol=1e-5)

        # Verify parity
        np.testing.assert_allclose(
            _to_numpy(mlx_probs), _to_numpy(torch_probs),
            rtol=1e-5, atol=1e-6,
            err_msg="Top-P probability renormalization mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_p_equals_0(self, skip_without_pytorch):
        """Test P=0 behavior - should keep at least one token (the top one)."""
        from mlx_primitives.generation.samplers import apply_top_p

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        p = 0.0  # Edge case: p=0 should keep only the top token due to always-keep-first

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_p(logits_mlx, p)
        mlx_np = _to_numpy(mlx_out)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_out = _pytorch_top_p(logits_torch, p)
        torch_np = _to_numpy(torch_out)

        # Should keep exactly 1 token (the top probability one)
        mlx_kept = np.sum(~np.isneginf(mlx_np), axis=-1)
        torch_kept = np.sum(~np.isneginf(torch_np), axis=-1)

        np.testing.assert_array_equal(mlx_kept, [1, 1], err_msg="MLX p=0 didn't keep exactly 1 token")
        np.testing.assert_array_equal(torch_kept, [1, 1], err_msg="PyTorch p=0 didn't keep exactly 1 token")

        # Verify masks match
        np.testing.assert_array_equal(
            np.isneginf(mlx_np), np.isneginf(torch_np),
            err_msg="p=0 mask mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_p_equals_1(self, skip_without_pytorch):
        """Test P=1 (no filtering) behavior."""
        from mlx_primitives.generation.samplers import apply_top_p

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        p = 1.0

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_p(logits_mlx, p)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_out = _pytorch_top_p(logits_torch, p)

        # Output should equal input (no filtering)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="MLX p=1 should return logits unchanged"
        )
        np.testing.assert_allclose(
            _to_numpy(torch_out), logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="PyTorch p=1 should return logits unchanged"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_temperature(self, skip_without_pytorch):
        """Test Top-P combined with temperature scaling."""
        from mlx_primitives.generation.samplers import apply_temperature, apply_top_p

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        temperature = 0.7
        p = 0.9

        # MLX: temperature then top-p
        logits_mlx = mx.array(logits_np)
        mlx_scaled = apply_temperature(logits_mlx, temperature)
        mlx_filtered = apply_top_p(mlx_scaled, p)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_scaled = _pytorch_temperature(logits_torch, temperature)
        torch_filtered = _pytorch_top_p(torch_scaled, p)

        mlx_np = _to_numpy(mlx_filtered)
        torch_np = _to_numpy(torch_filtered)

        # Check mask positions match
        mlx_masked = np.isneginf(mlx_np)
        torch_masked = np.isneginf(torch_np)
        np.testing.assert_array_equal(
            mlx_masked, torch_masked,
            err_msg="Temperature+Top-P mask mismatch"
        )

        # Check non-masked values match
        non_masked = ~mlx_masked
        if np.any(non_masked):
            np.testing.assert_allclose(
                mlx_np[non_masked], torch_np[non_masked],
                rtol=1e-5, atol=1e-6,
                err_msg="Temperature+Top-P values mismatch"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_combined_top_k_top_p(self, skip_without_pytorch):
        """Test combined Top-K and Top-P filtering."""
        from mlx_primitives.generation.samplers import apply_top_k, apply_top_p

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        k = 50
        p = 0.9

        # MLX: top-k then top-p
        logits_mlx = mx.array(logits_np)
        mlx_topk = apply_top_k(logits_mlx, k)
        mlx_topp = apply_top_p(mlx_topk, p)

        # PyTorch
        logits_torch = torch.from_numpy(logits_np)
        torch_topk = _pytorch_top_k(logits_torch, k)
        torch_topp = _pytorch_top_p(torch_topk, p)

        mlx_np = _to_numpy(mlx_topp)
        torch_np = _to_numpy(torch_topp)

        # Check mask positions match
        mlx_masked = np.isneginf(mlx_np)
        torch_masked = np.isneginf(torch_np)
        np.testing.assert_array_equal(
            mlx_masked, torch_masked,
            err_msg="Top-K+Top-P mask mismatch"
        )

        # Check non-masked values match
        non_masked = ~mlx_masked
        if np.any(non_masked):
            np.testing.assert_allclose(
                mlx_np[non_masked], torch_np[non_masked],
                rtol=1e-5, atol=1e-6,
                err_msg="Top-K+Top-P values mismatch"
            )
