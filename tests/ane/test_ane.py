"""Tests for ANE (Apple Neural Engine) offload primitives."""

import pytest
import mlx.core as mx
import numpy as np

from mlx_primitives.ane import (
    ANECapabilities,
    ComputeTarget,
    DispatchDecision,
    ModelSpec,
    ane_linear,
    ane_matmul,
    get_ane_info,
    get_ane_tops,
    get_model_cache,
    get_recommended_target,
    is_ane_available,
    should_use_ane,
    supports_operation,
)


class TestANEDetection:
    """Tests for ANE capability detection."""

    def test_get_ane_info(self):
        """Test getting ANE capabilities."""
        info = get_ane_info()

        assert isinstance(info, ANECapabilities)
        assert isinstance(info.available, bool)
        assert info.tops >= 0
        assert isinstance(info.supported_dtypes, tuple)

    def test_is_ane_available(self):
        """Test ANE availability check."""
        available = is_ane_available()
        assert isinstance(available, bool)

    def test_get_ane_tops(self):
        """Test getting ANE TOPS."""
        tops = get_ane_tops()

        if is_ane_available():
            assert tops > 0
        else:
            assert tops == 0

    def test_supports_operation(self):
        """Test operation support checking."""
        # These should return bool regardless of ANE availability
        assert isinstance(supports_operation("matmul"), bool)
        assert isinstance(supports_operation("conv2d"), bool)
        assert isinstance(supports_operation("unknown_op"), bool)

        # Unknown ops should not be supported
        assert supports_operation("unknown_op") is False


class TestDispatchDecision:
    """Tests for dispatch decision logic."""

    def test_decision_dataclass(self):
        """Test DispatchDecision dataclass."""
        decision = DispatchDecision(
            target=ComputeTarget.GPU,
            reason="Test reason",
            estimated_speedup=1.5,
        )

        assert decision.target == ComputeTarget.GPU
        assert decision.reason == "Test reason"
        assert decision.estimated_speedup == 1.5

    def test_force_gpu(self):
        """Test forcing GPU target."""
        decision = should_use_ane(
            operation="matmul",
            input_shapes=[(1024, 512), (512, 1024)],
            force_target=ComputeTarget.GPU,
        )

        assert decision.target == ComputeTarget.GPU
        assert "Explicitly requested GPU" in decision.reason

    def test_training_uses_gpu(self):
        """Test that training mode always uses GPU."""
        decision = should_use_ane(
            operation="matmul",
            input_shapes=[(1024, 512), (512, 1024)],
            is_training=True,
        )

        assert decision.target == ComputeTarget.GPU
        assert "Training" in decision.reason or "inference" in decision.reason.lower()

    def test_unsupported_operation(self):
        """Test unsupported operation falls back to GPU."""
        decision = should_use_ane(
            operation="custom_unsupported_op",
            input_shapes=[(100, 100)],
        )

        assert decision.target == ComputeTarget.GPU
        assert "not supported" in decision.reason.lower()

    def test_small_tensor_uses_gpu(self):
        """Test small tensors use GPU due to overhead."""
        decision = should_use_ane(
            operation="matmul",
            input_shapes=[(10, 10), (10, 10)],  # Very small
        )

        assert decision.target == ComputeTarget.GPU
        assert "small" in decision.reason.lower() or "elements" in decision.reason.lower()

    def test_dynamic_shapes_use_gpu(self):
        """Test dynamic shapes are not supported."""
        decision = should_use_ane(
            operation="matmul",
            input_shapes=[(-1, 512), (512, 1024)],  # Dynamic dim
        )

        assert decision.target == ComputeTarget.GPU
        assert "Dynamic" in decision.reason or "shapes" in decision.reason.lower()

    def test_large_tensor_decision(self):
        """Test decision for large tensors."""
        decision = should_use_ane(
            operation="matmul",
            input_shapes=[(4096, 4096), (4096, 4096)],
        )

        # Should make a decision (either GPU or ANE)
        assert decision.target in (ComputeTarget.GPU, ComputeTarget.ANE)
        assert len(decision.reason) > 0


class TestRecommendedTarget:
    """Tests for get_recommended_target function."""

    def test_training_target(self):
        """Test recommended target for training."""
        target = get_recommended_target(
            operation="matmul",
            batch_size=32,
            seq_len=512,
            hidden_dim=768,
            is_training=True,
        )

        assert target == ComputeTarget.GPU

    def test_inference_target(self):
        """Test recommended target for inference."""
        target = get_recommended_target(
            operation="matmul",
            batch_size=1,
            seq_len=512,
            hidden_dim=768,
            is_training=False,
        )

        # Should be one of the valid targets
        assert target in (ComputeTarget.GPU, ComputeTarget.ANE)


class TestModelSpec:
    """Tests for ModelSpec dataclass."""

    def test_create_spec(self):
        """Test creating model spec."""
        spec = ModelSpec(
            operation="matmul",
            input_shapes=((1024, 512), (512, 1024)),
            dtype="float16",
            params=(("transpose_a", False), ("transpose_b", False)),
        )

        assert spec.operation == "matmul"
        assert spec.input_shapes == ((1024, 512), (512, 1024))
        assert spec.dtype == "float16"

    def test_cache_key_unique(self):
        """Test cache keys are unique for different specs."""
        spec1 = ModelSpec(
            operation="matmul",
            input_shapes=((1024, 512), (512, 1024)),
            dtype="float16",
            params=(),
        )

        spec2 = ModelSpec(
            operation="matmul",
            input_shapes=((512, 256), (256, 512)),  # Different shapes
            dtype="float16",
            params=(),
        )

        spec3 = ModelSpec(
            operation="matmul",
            input_shapes=((1024, 512), (512, 1024)),
            dtype="float32",  # Different dtype
            params=(),
        )

        assert spec1.cache_key() != spec2.cache_key()
        assert spec1.cache_key() != spec3.cache_key()

    def test_cache_key_consistent(self):
        """Test cache keys are consistent for same spec."""
        spec1 = ModelSpec(
            operation="matmul",
            input_shapes=((1024, 512), (512, 1024)),
            dtype="float16",
            params=(("transpose_a", False),),
        )

        spec2 = ModelSpec(
            operation="matmul",
            input_shapes=((1024, 512), (512, 1024)),
            dtype="float16",
            params=(("transpose_a", False),),
        )

        assert spec1.cache_key() == spec2.cache_key()

    def test_to_dict(self):
        """Test converting spec to dictionary."""
        spec = ModelSpec(
            operation="matmul",
            input_shapes=((1024, 512), (512, 1024)),
            dtype="float16",
            params=(("transpose_a", True),),
        )

        d = spec.to_dict()

        assert d["operation"] == "matmul"
        assert d["dtype"] == "float16"
        assert d["params"]["transpose_a"] is True


class TestModelCache:
    """Tests for CoreMLModelCache."""

    def test_get_cache(self):
        """Test getting model cache instance."""
        cache = get_model_cache()
        assert cache is not None

    def test_cache_stats(self):
        """Test getting cache statistics."""
        cache = get_model_cache()
        stats = cache.get_cache_stats()

        assert "memory_cache_count" in stats
        assert "disk_cache_count" in stats
        assert "disk_cache_size_mb" in stats


class TestANEMatmul:
    """Tests for ANE matmul primitive."""

    def test_matmul_never_ane(self):
        """Test matmul with ANE disabled."""
        a = mx.random.normal((64, 128))
        b = mx.random.normal((128, 256))
        mx.eval(a, b)

        result = ane_matmul(a, b, use_ane="never")

        assert result.shape == (64, 256)

        # Verify correctness against standard matmul
        expected = a @ b
        np.testing.assert_allclose(
            np.array(result), np.array(expected), rtol=1e-4, atol=1e-4
        )

    def test_matmul_auto(self):
        """Test matmul with auto dispatch."""
        a = mx.random.normal((256, 512))
        b = mx.random.normal((512, 256))
        mx.eval(a, b)

        result = ane_matmul(a, b, use_ane="auto")

        assert result.shape == (256, 256)

        # Verify correctness - use looser tolerance for auto mode
        # because ANE path uses float16 internally which has lower precision.
        # Max expected error is ~0.05 due to accumulated fp16 rounding.
        expected = a @ b
        np.testing.assert_allclose(
            np.array(result), np.array(expected), rtol=0.05, atol=0.05
        )

    def test_matmul_transpose_a(self):
        """Test matmul with transposed first matrix."""
        a = mx.random.normal((128, 64))  # Will be transposed to (64, 128)
        b = mx.random.normal((128, 256))
        mx.eval(a, b)

        result = ane_matmul(a, b, transpose_a=True, use_ane="never")

        assert result.shape == (64, 256)

        # Verify correctness
        expected = mx.swapaxes(a, -1, -2) @ b
        np.testing.assert_allclose(
            np.array(result), np.array(expected), rtol=1e-4, atol=1e-4
        )

    def test_matmul_transpose_b(self):
        """Test matmul with transposed second matrix."""
        a = mx.random.normal((64, 128))
        b = mx.random.normal((256, 128))  # Will be transposed to (128, 256)
        mx.eval(a, b)

        result = ane_matmul(a, b, transpose_b=True, use_ane="never")

        assert result.shape == (64, 256)

        # Verify correctness
        expected = a @ mx.swapaxes(b, -1, -2)
        np.testing.assert_allclose(
            np.array(result), np.array(expected), rtol=1e-4, atol=1e-4
        )

    def test_matmul_batched(self):
        """Test batched matmul."""
        a = mx.random.normal((4, 64, 128))
        b = mx.random.normal((4, 128, 64))
        mx.eval(a, b)

        result = ane_matmul(a, b, use_ane="never")

        assert result.shape == (4, 64, 64)

        # Verify correctness
        expected = a @ b
        np.testing.assert_allclose(
            np.array(result), np.array(expected), rtol=1e-4, atol=1e-4
        )


class TestANELinear:
    """Tests for ANE linear layer."""

    def test_linear_no_bias(self):
        """Test linear layer without bias."""
        x = mx.random.normal((32, 512))
        weight = mx.random.normal((256, 512))
        mx.eval(x, weight)

        result = ane_linear(x, weight, bias=None, use_ane="never")

        assert result.shape == (32, 256)

        # Verify correctness (linear is x @ W.T)
        expected = x @ mx.swapaxes(weight, -1, -2)
        np.testing.assert_allclose(
            np.array(result), np.array(expected), rtol=1e-4, atol=1e-4
        )

    def test_linear_with_bias(self):
        """Test linear layer with bias."""
        x = mx.random.normal((32, 512))
        weight = mx.random.normal((256, 512))
        bias = mx.random.normal((256,))
        mx.eval(x, weight, bias)

        result = ane_linear(x, weight, bias=bias, use_ane="never")

        assert result.shape == (32, 256)

        # Verify correctness
        expected = x @ mx.swapaxes(weight, -1, -2) + bias
        np.testing.assert_allclose(
            np.array(result), np.array(expected), rtol=1e-4, atol=1e-4
        )

    def test_linear_batched(self):
        """Test batched linear layer."""
        x = mx.random.normal((4, 32, 512))
        weight = mx.random.normal((256, 512))
        mx.eval(x, weight)

        result = ane_linear(x, weight, use_ane="never")

        assert result.shape == (4, 32, 256)


class TestANEConverters:
    """Tests for ANE tensor conversion utilities."""

    def test_mlx_to_coreml_input(self):
        """Test MLX to Core ML input conversion."""
        from mlx_primitives.ane.converters import mlx_to_coreml_input

        a = mx.random.normal((100, 100))
        b = mx.random.normal((100, 50))
        mx.eval(a, b)

        inputs = mlx_to_coreml_input([a, b], ["input_a", "input_b"])

        assert "input_a" in inputs
        assert "input_b" in inputs
        assert inputs["input_a"].shape == (100, 100)
        assert inputs["input_b"].shape == (100, 50)

    def test_coreml_to_mlx(self):
        """Test Core ML to MLX output conversion."""
        from mlx_primitives.ane.converters import coreml_to_mlx

        # Simulate Core ML output
        output = {"result": np.random.randn(50, 50).astype(np.float32)}

        result = coreml_to_mlx(output, "result", target_dtype=mx.float32)

        assert result.shape == (50, 50)
        assert result.dtype == mx.float32

    def test_prepare_for_ane(self):
        """Test tensor preparation for ANE."""
        from mlx_primitives.ane.converters import prepare_for_ane

        tensor = mx.random.normal((100, 100))

        prepared = prepare_for_ane(tensor, preferred_dtype="float16")

        assert prepared.dtype == mx.float16
        assert prepared.shape == (100, 100)
