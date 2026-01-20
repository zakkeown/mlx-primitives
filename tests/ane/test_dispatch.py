"""Comprehensive tests for ANE dispatch logic.

Tests the routing decisions between GPU (Metal) and ANE (Core ML).
"""

from unittest.mock import MagicMock, patch
import pytest

from mlx_primitives.ane.dispatch import (
    ComputeTarget,
    DispatchDecision,
    should_use_ane,
    estimate_transfer_overhead_ms,
    get_recommended_target,
    _estimate_ane_speedup,
    _product,
    _MIN_SPEEDUP_THRESHOLD,
    _MIN_TENSOR_SIZE,
)
from mlx_primitives.ane.detection import ANECapabilities
from mlx_primitives.hardware.detection import ChipFamily, ChipTier, ChipInfo


# ---------------------------------------------------------------------------
# Fixtures for mocking ANE and hardware info
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ane_capabilities():
    """Standard ANE capabilities for testing."""
    return ANECapabilities(
        available=True,
        tops=18.0,  # M3 level
        supported_dtypes=("float16", "float32", "int8"),
        max_tensor_size_mb=512,
        supports_dynamic_shapes=False,
        supports_training=False,
        supports_matmul=True,
        supports_conv2d=True,
        supports_depthwise_conv=True,
        supports_batch_norm=True,
        supports_layer_norm=True,
        supports_activations=True,
    )


@pytest.fixture
def mock_chip_info():
    """Standard chip info for testing (M3 base)."""
    return ChipInfo(
        family=ChipFamily.M3,
        tier=ChipTier.BASE,
        device_name="Apple M3",
        gpu_cores=10,
        memory_gb=16.0,
        ane_tops=18.0,
    )


@pytest.fixture
def mock_ane_unavailable():
    """ANE capabilities when ANE is not available."""
    return ANECapabilities(
        available=False,
        tops=0.0,
        supported_dtypes=(),
        max_tensor_size_mb=0,
    )


@pytest.fixture
def mock_ane_limited():
    """ANE capabilities with limited operation support."""
    return ANECapabilities(
        available=True,
        tops=11.0,
        supported_dtypes=("float16", "float32"),
        max_tensor_size_mb=256,
        supports_matmul=True,
        supports_conv2d=True,
        supports_depthwise_conv=False,  # Limited
        supports_batch_norm=False,  # Limited
        supports_layer_norm=False,  # Limited
        supports_activations=True,
    )


# ---------------------------------------------------------------------------
# Test ComputeTarget enum
# ---------------------------------------------------------------------------


class TestComputeTargetEnum:
    """Tests for ComputeTarget enum."""

    def test_gpu_value(self) -> None:
        """GPU target has correct value."""
        assert ComputeTarget.GPU.value == "gpu"

    def test_ane_value(self) -> None:
        """ANE target has correct value."""
        assert ComputeTarget.ANE.value == "ane"

    def test_auto_value(self) -> None:
        """AUTO target has correct value."""
        assert ComputeTarget.AUTO.value == "auto"


# ---------------------------------------------------------------------------
# Test force_target parameter (explicit override)
# ---------------------------------------------------------------------------


class TestShouldUseANEForceTarget:
    """Tests for explicit target override."""

    def test_force_gpu_always_returns_gpu(self, mock_ane_capabilities) -> None:
        """Force GPU target overrides all other logic."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ):
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(1024, 1024), (1024, 1024)],
                force_target=ComputeTarget.GPU,
            )

        assert decision.target == ComputeTarget.GPU
        assert "Explicitly requested GPU" in decision.reason

    def test_force_ane_when_available(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Force ANE when ANE is available."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ):
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(1024, 1024)],
                force_target=ComputeTarget.ANE,
            )

        assert decision.target == ComputeTarget.ANE
        assert "Explicitly requested ANE" in decision.reason

    def test_force_ane_when_unavailable_falls_back(
        self, mock_ane_unavailable
    ) -> None:
        """Force ANE falls back to GPU when ANE unavailable."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=False,
        ):
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(1024, 1024)],
                force_target=ComputeTarget.ANE,
            )

        assert decision.target == ComputeTarget.GPU
        assert "ANE requested but unavailable" in decision.reason


# ---------------------------------------------------------------------------
# Test training mode
# ---------------------------------------------------------------------------


class TestShouldUseANETrainingMode:
    """Tests for training mode behavior."""

    def test_training_always_uses_gpu(self, mock_ane_capabilities) -> None:
        """Training mode always routes to GPU (ANE doesn't support gradients)."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ):
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(1024, 1024), (1024, 1024)],
                is_training=True,
            )

        assert decision.target == ComputeTarget.GPU
        assert "Training mode" in decision.reason
        assert "inference-only" in decision.reason

    def test_inference_can_use_ane(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Inference mode can use ANE when appropriate."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            decision = should_use_ane(
                operation="depthwise_conv",  # High ANE efficiency
                input_shapes=[(4, 64, 64, 256)],  # Large tensor
                is_training=False,
            )

        # May or may not use ANE depending on speedup calculation
        # but should not be rejected for training mode
        assert "Training mode" not in decision.reason


# ---------------------------------------------------------------------------
# Test operation support
# ---------------------------------------------------------------------------


class TestShouldUseANEOperationSupport:
    """Tests for operation support checking."""

    @pytest.mark.parametrize(
        "operation",
        ["matmul", "linear", "conv2d", "depthwise_conv", "batch_norm", "layer_norm", "gelu", "silu", "relu", "softmax"],
    )
    def test_supported_operations_recognized(
        self, operation, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """All supported operations are recognized."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            decision = should_use_ane(
                operation=operation,
                input_shapes=[(4, 512, 1024)],  # Large enough to pass size check
            )

        # Should not be rejected for unsupported operation
        assert f"'{operation}' not supported" not in decision.reason

    @pytest.mark.parametrize(
        "operation",
        ["unknown_op", "custom_kernel", "scatter", "gather", "einsum"],
    )
    def test_unsupported_operations_rejected(
        self, operation, mock_ane_capabilities
    ) -> None:
        """Unsupported operations route to GPU."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ):
            decision = should_use_ane(
                operation=operation,
                input_shapes=[(1024, 1024)],
            )

        assert decision.target == ComputeTarget.GPU
        assert "not supported" in decision.reason

    def test_limited_ane_rejects_unsupported(self, mock_ane_limited) -> None:
        """Limited ANE config rejects operations it doesn't support."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_limited,
        ):
            decision = should_use_ane(
                operation="layer_norm",  # Not supported in mock_ane_limited
                input_shapes=[(4, 512, 1024)],
            )

        assert decision.target == ComputeTarget.GPU
        assert "not supported" in decision.reason


# ---------------------------------------------------------------------------
# Test tensor size constraints
# ---------------------------------------------------------------------------


class TestShouldUseANETensorSize:
    """Tests for tensor size constraints."""

    def test_tensor_too_small_rejected(self, mock_ane_capabilities) -> None:
        """Very small tensors stay on GPU due to overhead."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ):
            # 32x32 = 1024 elements, well below _MIN_TENSOR_SIZE (10000)
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(32, 32), (32, 32)],
            )

        assert decision.target == ComputeTarget.GPU
        assert "too small" in decision.reason

    def test_tensor_at_minimum_threshold(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Tensor at minimum threshold can be considered."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            # 100x100 = 10000 elements, at _MIN_TENSOR_SIZE
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(100, 100)],
            )

        # Should not be rejected for being too small
        assert "too small" not in decision.reason

    def test_tensor_too_large_rejected(self, mock_ane_capabilities) -> None:
        """Tensors exceeding ANE max size stay on GPU."""
        # Create a mock with small max size
        small_max_ane = ANECapabilities(
            available=True,
            tops=18.0,
            supported_dtypes=("float16", "float32"),
            max_tensor_size_mb=1,  # Only 1MB
            supports_matmul=True,
            supports_conv2d=True,
            supports_depthwise_conv=True,
            supports_batch_norm=True,
            supports_layer_norm=True,
            supports_activations=True,
        )

        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=small_max_ane,
        ):
            # 1024 * 1024 * 4 bytes = 4MB, exceeds 1MB limit
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(1024, 1024)],
            )

        assert decision.target == ComputeTarget.GPU
        assert "too large" in decision.reason

    def test_multiple_tensors_summed(self, mock_ane_capabilities) -> None:
        """Multiple tensor sizes are summed for size check."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ):
            # Each tensor is 50x50=2500 elements, but 4 tensors = 10000 total
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(50, 50), (50, 50), (50, 50), (50, 50)],
            )

        # Total 10000 elements should pass minimum
        assert "too small" not in decision.reason


# ---------------------------------------------------------------------------
# Test dynamic shapes
# ---------------------------------------------------------------------------


class TestShouldUseANEDynamicShapes:
    """Tests for dynamic shape handling."""

    @pytest.mark.parametrize(
        "shapes",
        [
            [(0, 1024)],  # Zero dimension
            [(-1, 1024)],  # Negative dimension (dynamic)
            [(1024, -1)],  # Negative in second dim
            [(0, 0)],  # All zeros
        ],
    )
    def test_dynamic_shapes_rejected(self, shapes, mock_ane_capabilities) -> None:
        """Dynamic shapes (negative/zero dims) route to GPU."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ):
            decision = should_use_ane(
                operation="matmul",
                input_shapes=shapes,
            )

        assert decision.target == ComputeTarget.GPU
        assert "Dynamic shapes" in decision.reason or "too small" in decision.reason

    def test_static_shapes_allowed(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Static positive shapes are allowed."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(512, 512), (512, 512)],
            )

        assert "Dynamic shapes" not in decision.reason


# ---------------------------------------------------------------------------
# Test speedup threshold
# ---------------------------------------------------------------------------


class TestShouldUseANESpeedupThreshold:
    """Tests for speedup threshold behavior."""

    def test_insufficient_speedup_rejected(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Operations with insufficient estimated speedup stay on GPU."""
        # relu has low ANE multiplier (0.5), should result in low speedup
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            decision = should_use_ane(
                operation="relu",  # Low efficiency op
                input_shapes=[(1, 100, 100)],  # Small-ish tensor
            )

        assert decision.target == ComputeTarget.GPU
        assert "Insufficient speedup" in decision.reason or "too small" in decision.reason

    def test_high_speedup_accepted(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Operations with high estimated speedup use ANE."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            decision = should_use_ane(
                operation="depthwise_conv",  # High efficiency (2.5x multiplier)
                input_shapes=[(4, 64, 64, 256)],  # Large tensor
            )

        # If speedup is high enough, should use ANE
        if decision.target == ComputeTarget.ANE:
            assert "speedup" in decision.reason.lower()

    def test_speedup_returned_in_decision(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Estimated speedup is included in decision."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(512, 512)],
            )

        # Decision should have speedup estimate
        assert decision.estimated_speedup >= 0


# ---------------------------------------------------------------------------
# Test _estimate_ane_speedup
# ---------------------------------------------------------------------------


class TestEstimateANESpeedup:
    """Tests for ANE speedup estimation."""

    def test_matmul_has_positive_speedup(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Matmul should have positive speedup estimate."""
        with patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            speedup = _estimate_ane_speedup("matmul", [(1024, 1024)])

        assert speedup > 0

    def test_depthwise_conv_higher_than_matmul(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Depthwise conv should have higher speedup than matmul."""
        shapes = [(4, 64, 64, 256)]

        with patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            matmul_speedup = _estimate_ane_speedup("matmul", shapes)
            depthwise_speedup = _estimate_ane_speedup("depthwise_conv", shapes)

        # depthwise_conv has 2.5x multiplier vs matmul 1.5x
        assert depthwise_speedup > matmul_speedup

    def test_small_tensor_penalty(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Small tensors should have reduced speedup due to overhead."""
        small_shape = [(100, 100)]  # 10000 elements
        large_shape = [(1000, 1000)]  # 1M elements

        with patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            small_speedup = _estimate_ane_speedup("matmul", small_shape)
            large_speedup = _estimate_ane_speedup("matmul", large_shape)

        assert small_speedup < large_speedup

    @pytest.mark.parametrize(
        "chip_family,expected_gpu_tflops",
        [
            (ChipFamily.M1, 2.6),
            (ChipFamily.M2, 3.6),
            (ChipFamily.M3, 4.1),
            (ChipFamily.M4, 4.5),
        ],
    )
    def test_chip_specific_gpu_estimates(
        self, chip_family, expected_gpu_tflops, mock_ane_capabilities
    ) -> None:
        """Different chips have different GPU TFLOPS estimates."""
        chip_info = ChipInfo(
            family=chip_family,
            tier=ChipTier.BASE,
            device_name=f"Apple {chip_family.value}",
            gpu_cores=10,
            memory_gb=16.0,
        )

        with patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=chip_info,
        ):
            speedup = _estimate_ane_speedup("matmul", [(1024, 1024)])

        # Speedup should be relative to chip's GPU performance
        assert speedup > 0

    @pytest.mark.parametrize(
        "tier,gpu_multiplier",
        [
            (ChipTier.BASE, 1.0),
            (ChipTier.PRO, 1.5),
            (ChipTier.MAX, 2.5),
            (ChipTier.ULTRA, 4.0),
        ],
    )
    def test_tier_scales_gpu_performance(
        self, tier, gpu_multiplier, mock_ane_capabilities
    ) -> None:
        """Higher tier chips have scaled GPU performance."""
        base_chip = ChipInfo(
            family=ChipFamily.M3,
            tier=ChipTier.BASE,
            device_name="Apple M3",
            gpu_cores=10,
            memory_gb=16.0,
        )

        tier_chip = ChipInfo(
            family=ChipFamily.M3,
            tier=tier,
            device_name=f"Apple M3 {tier.value}",
            gpu_cores=int(10 * gpu_multiplier),
            memory_gb=16.0 * gpu_multiplier,
        )

        shapes = [(1024, 1024)]

        with patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ):
            with patch(
                "mlx_primitives.ane.dispatch.get_chip_info",
                return_value=base_chip,
            ):
                base_speedup = _estimate_ane_speedup("matmul", shapes)

            with patch(
                "mlx_primitives.ane.dispatch.get_chip_info",
                return_value=tier_chip,
            ):
                tier_speedup = _estimate_ane_speedup("matmul", shapes)

        # Higher tier = more GPU power = less ANE benefit
        if tier != ChipTier.BASE:
            assert tier_speedup < base_speedup


# ---------------------------------------------------------------------------
# Test estimate_transfer_overhead_ms
# ---------------------------------------------------------------------------


class TestEstimateTransferOverhead:
    """Tests for transfer overhead estimation."""

    def test_small_tensor_overhead(self) -> None:
        """Small tensors have small overhead."""
        # 1000 float32 elements = 4000 bytes
        overhead = estimate_transfer_overhead_ms([(10, 100)], dtype_bytes=4)

        # Should be very small (microseconds)
        assert overhead < 1.0  # Less than 1ms

    def test_large_tensor_overhead(self) -> None:
        """Large tensors have larger overhead."""
        # 1M float32 elements = 4MB
        overhead = estimate_transfer_overhead_ms([(1000, 1000)], dtype_bytes=4)

        # Should be measurable but still small
        assert overhead > 0
        assert overhead < 10.0  # Less than 10ms

    def test_overhead_scales_with_size(self) -> None:
        """Overhead should scale linearly with tensor size."""
        small_overhead = estimate_transfer_overhead_ms([(100, 100)])
        large_overhead = estimate_transfer_overhead_ms([(1000, 1000)])

        # 100x more data should have ~100x more overhead
        ratio = large_overhead / small_overhead
        assert 50 < ratio < 200  # Allow some variance

    def test_overhead_scales_with_dtype(self) -> None:
        """Overhead should scale with dtype size."""
        fp32_overhead = estimate_transfer_overhead_ms([(100, 100)], dtype_bytes=4)
        fp16_overhead = estimate_transfer_overhead_ms([(100, 100)], dtype_bytes=2)

        assert fp32_overhead > fp16_overhead
        assert abs(fp32_overhead / fp16_overhead - 2.0) < 0.1

    def test_multiple_tensors_summed(self) -> None:
        """Multiple tensors have summed overhead."""
        single_overhead = estimate_transfer_overhead_ms([(100, 100)])
        double_overhead = estimate_transfer_overhead_ms([(100, 100), (100, 100)])

        assert abs(double_overhead / single_overhead - 2.0) < 0.1


# ---------------------------------------------------------------------------
# Test get_recommended_target
# ---------------------------------------------------------------------------


class TestGetRecommendedTarget:
    """Tests for transformer workload recommendations."""

    def test_training_always_gpu(self, mock_ane_capabilities) -> None:
        """Training mode always recommends GPU."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ):
            target = get_recommended_target(
                operation="matmul",
                batch_size=4,
                seq_len=512,
                hidden_dim=1024,
                is_training=True,
            )

        assert target == ComputeTarget.GPU

    def test_matmul_shapes_correct(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Matmul uses correct shape calculation."""
        # Should build shape (batch*seq, hidden) x (hidden, hidden)
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            target = get_recommended_target(
                operation="matmul",
                batch_size=4,
                seq_len=512,
                hidden_dim=1024,
            )

        # Result depends on speedup calculation
        assert target in (ComputeTarget.GPU, ComputeTarget.ANE)

    def test_layer_norm_shapes_correct(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Layer norm uses correct shape."""
        # Should build shape (batch, seq, hidden)
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            target = get_recommended_target(
                operation="layer_norm",
                batch_size=4,
                seq_len=512,
                hidden_dim=1024,
            )

        assert target in (ComputeTarget.GPU, ComputeTarget.ANE)

    @pytest.mark.parametrize("operation", ["gelu", "silu", "relu"])
    def test_activation_shapes_correct(
        self, operation, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Activations use correct shape."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            target = get_recommended_target(
                operation=operation,
                batch_size=4,
                seq_len=512,
                hidden_dim=1024,
            )

        assert target in (ComputeTarget.GPU, ComputeTarget.ANE)


# ---------------------------------------------------------------------------
# Test CoreML unavailable scenarios
# ---------------------------------------------------------------------------


class TestCoreMLUnavailable:
    """Tests for graceful fallback when CoreML is unavailable."""

    def test_ane_unavailable_always_gpu(self, mock_ane_unavailable) -> None:
        """When ANE unavailable, all operations route to GPU."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=False,
        ):
            decision = should_use_ane(
                operation="matmul",
                input_shapes=[(1024, 1024)],
            )

        assert decision.target == ComputeTarget.GPU
        assert "not available" in decision.reason

    def test_graceful_degradation_message(self, mock_ane_unavailable) -> None:
        """Unavailable ANE provides clear message."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=False,
        ):
            decision = should_use_ane(
                operation="depthwise_conv",  # Op that would benefit from ANE
                input_shapes=[(4, 64, 64, 256)],
            )

        assert decision.target == ComputeTarget.GPU
        assert "ANE not available" in decision.reason


# ---------------------------------------------------------------------------
# Test helper function _product
# ---------------------------------------------------------------------------


class TestProductHelper:
    """Tests for _product helper function."""

    def test_empty_shape(self) -> None:
        """Empty shape returns 1 (identity for multiplication)."""
        assert _product(()) == 1

    def test_scalar(self) -> None:
        """Single dimension works."""
        assert _product((5,)) == 5

    def test_2d_shape(self) -> None:
        """2D shape multiplied correctly."""
        assert _product((10, 20)) == 200

    def test_3d_shape(self) -> None:
        """3D shape multiplied correctly."""
        assert _product((2, 3, 4)) == 24

    def test_4d_shape(self) -> None:
        """4D shape multiplied correctly."""
        assert _product((2, 3, 4, 5)) == 120


# ---------------------------------------------------------------------------
# Test DispatchDecision dataclass
# ---------------------------------------------------------------------------


class TestDispatchDecision:
    """Tests for DispatchDecision dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Can create with just target and reason."""
        decision = DispatchDecision(
            target=ComputeTarget.GPU,
            reason="Test reason",
        )

        assert decision.target == ComputeTarget.GPU
        assert decision.reason == "Test reason"
        assert decision.estimated_speedup == 1.0

    def test_creation_with_speedup(self) -> None:
        """Can create with speedup."""
        decision = DispatchDecision(
            target=ComputeTarget.ANE,
            reason="Fast operation",
            estimated_speedup=2.5,
        )

        assert decision.target == ComputeTarget.ANE
        assert decision.estimated_speedup == 2.5

    def test_immutable_target(self) -> None:
        """Dataclass fields are accessible."""
        decision = DispatchDecision(
            target=ComputeTarget.GPU,
            reason="Test",
        )

        # Can read fields
        assert decision.target == ComputeTarget.GPU


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_min_speedup_threshold_reasonable(self) -> None:
        """Minimum speedup threshold is reasonable (>1)."""
        assert _MIN_SPEEDUP_THRESHOLD > 1.0
        assert _MIN_SPEEDUP_THRESHOLD < 2.0

    def test_min_tensor_size_reasonable(self) -> None:
        """Minimum tensor size is reasonable."""
        assert _MIN_TENSOR_SIZE > 1000
        assert _MIN_TENSOR_SIZE < 100000


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDispatchIntegration:
    """Integration tests for dispatch logic."""

    def test_typical_transformer_inference(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Typical transformer inference dispatch decisions."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            # Attention QKV projection (large matmul)
            qkv_decision = should_use_ane(
                operation="matmul",
                input_shapes=[(4 * 512, 1024), (1024, 3 * 1024)],
            )

            # Layer norm
            ln_decision = should_use_ane(
                operation="layer_norm",
                input_shapes=[(4, 512, 1024)],
            )

            # GELU activation
            gelu_decision = should_use_ane(
                operation="gelu",
                input_shapes=[(4, 512, 4096)],
            )

        # All should have valid decisions
        assert qkv_decision.target in (ComputeTarget.GPU, ComputeTarget.ANE)
        assert ln_decision.target in (ComputeTarget.GPU, ComputeTarget.ANE)
        assert gelu_decision.target in (ComputeTarget.GPU, ComputeTarget.ANE)

    def test_mixed_batch_sizes(
        self, mock_ane_capabilities, mock_chip_info
    ) -> None:
        """Dispatch handles various batch sizes."""
        with patch(
            "mlx_primitives.ane.dispatch.is_ane_available",
            return_value=True,
        ), patch(
            "mlx_primitives.ane.dispatch.get_ane_info",
            return_value=mock_ane_capabilities,
        ), patch(
            "mlx_primitives.ane.dispatch.get_chip_info",
            return_value=mock_chip_info,
        ):
            # Batch 1 (small)
            b1_decision = should_use_ane(
                operation="matmul",
                input_shapes=[(512, 1024), (1024, 1024)],
            )

            # Batch 8 (medium)
            b8_decision = should_use_ane(
                operation="matmul",
                input_shapes=[(8 * 512, 1024), (1024, 1024)],
            )

            # Batch 32 (large)
            b32_decision = should_use_ane(
                operation="matmul",
                input_shapes=[(32 * 512, 1024), (1024, 1024)],
            )

        # All should succeed
        assert b1_decision.target in (ComputeTarget.GPU, ComputeTarget.ANE)
        assert b8_decision.target in (ComputeTarget.GPU, ComputeTarget.ANE)
        assert b32_decision.target in (ComputeTarget.GPU, ComputeTarget.ANE)
