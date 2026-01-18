"""Tests for gradient checkpointing primitives."""

import pytest
import mlx.core as mx
import numpy as np

from mlx_primitives.training import checkpoint, checkpoint_sequential


class TestCheckpointCorrectness:
    """Verify checkpointed gradients match non-checkpointed."""

    def test_linear_gradient_matches(self) -> None:
        """Simple linear function gradient should match exactly."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))
        w = mx.random.normal((8, 16))

        def fn(x: mx.array, w: mx.array) -> mx.array:
            return x @ w

        # Standard gradient
        def loss_fn(x: mx.array, w: mx.array) -> mx.array:
            return mx.sum(fn(x, w) ** 2)

        grad_x, grad_w = mx.grad(loss_fn, argnums=(0, 1))(x, w)

        # Checkpointed gradient
        def loss_fn_ckpt(x: mx.array, w: mx.array) -> mx.array:
            return mx.sum(checkpoint(fn, x, w) ** 2)

        grad_x_ckpt, grad_w_ckpt = mx.grad(loss_fn_ckpt, argnums=(0, 1))(x, w)

        np.testing.assert_allclose(
            np.array(grad_x), np.array(grad_x_ckpt), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(grad_w), np.array(grad_w_ckpt), rtol=1e-5, atol=1e-6
        )

    def test_nonlinear_gradient_matches(self) -> None:
        """Nonlinear function with GELU should match."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))
        w = mx.random.normal((8, 16))

        def fn(x: mx.array, w: mx.array) -> mx.array:
            h = x @ w
            return mx.tanh(h) * h  # Nonlinear activation

        # Standard gradient
        def loss_fn(x: mx.array, w: mx.array) -> mx.array:
            return mx.sum(fn(x, w) ** 2)

        grad_x, grad_w = mx.grad(loss_fn, argnums=(0, 1))(x, w)

        # Checkpointed gradient
        def loss_fn_ckpt(x: mx.array, w: mx.array) -> mx.array:
            return mx.sum(checkpoint(fn, x, w) ** 2)

        grad_x_ckpt, grad_w_ckpt = mx.grad(loss_fn_ckpt, argnums=(0, 1))(x, w)

        np.testing.assert_allclose(
            np.array(grad_x), np.array(grad_x_ckpt), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(grad_w), np.array(grad_w_ckpt), rtol=1e-4, atol=1e-5
        )

    def test_multi_layer_gradient_matches(self) -> None:
        """Multi-layer MLP gradient should match."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))
        w1 = mx.random.normal((8, 16))
        w2 = mx.random.normal((16, 8))

        def fn(x: mx.array, w1: mx.array, w2: mx.array) -> mx.array:
            h = mx.maximum(x @ w1, 0)  # ReLU
            return h @ w2

        # Standard gradient
        def loss_fn(x: mx.array, w1: mx.array, w2: mx.array) -> mx.array:
            return mx.sum(fn(x, w1, w2) ** 2)

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(x, w1, w2)

        # Checkpointed gradient
        def loss_fn_ckpt(x: mx.array, w1: mx.array, w2: mx.array) -> mx.array:
            return mx.sum(checkpoint(fn, x, w1, w2) ** 2)

        grads_ckpt = mx.grad(loss_fn_ckpt, argnums=(0, 1, 2))(x, w1, w2)

        for g, g_ckpt in zip(grads, grads_ckpt):
            np.testing.assert_allclose(
                np.array(g), np.array(g_ckpt), rtol=1e-4, atol=1e-5
            )

    def test_forward_output_matches(self) -> None:
        """Forward pass output should be identical."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))
        w = mx.random.normal((8, 16))

        def fn(x: mx.array, w: mx.array) -> mx.array:
            return mx.tanh(x @ w)

        out_normal = fn(x, w)
        out_ckpt = checkpoint(fn, x, w)

        np.testing.assert_allclose(
            np.array(out_normal), np.array(out_ckpt), rtol=1e-6, atol=1e-7
        )


class TestCheckpointSequential:
    """Test checkpoint_sequential for segmented checkpointing."""

    def test_single_segment(self) -> None:
        """Single segment should work like regular checkpoint."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))

        def fn1(x: mx.array) -> mx.array:
            return mx.tanh(x)

        def fn2(x: mx.array) -> mx.array:
            return x * 2

        functions = [fn1, fn2]

        # Sequential execution
        out_seq = fn2(fn1(x))
        out_ckpt = checkpoint_sequential(functions, 1, x)

        np.testing.assert_allclose(
            np.array(out_seq), np.array(out_ckpt), rtol=1e-6, atol=1e-7
        )

    def test_multiple_segments(self) -> None:
        """Multiple segments should produce correct output."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))

        functions = [
            lambda y: mx.tanh(y),
            lambda y: y * 2,
            lambda y: mx.maximum(y, 0),
            lambda y: y + 1,
        ]

        # Sequential execution
        out_seq = x
        for fn in functions:
            out_seq = fn(out_seq)

        # With 2 segments (2 functions each)
        out_ckpt = checkpoint_sequential(functions, 2, x)

        np.testing.assert_allclose(
            np.array(out_seq), np.array(out_ckpt), rtol=1e-5, atol=1e-6
        )

    def test_gradient_with_segments(self) -> None:
        """Gradients should match with segmented checkpointing."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))

        def fn1(y: mx.array) -> mx.array:
            return mx.tanh(y)

        def fn2(y: mx.array) -> mx.array:
            return y * 2

        def fn3(y: mx.array) -> mx.array:
            return mx.sin(y)

        functions = [fn1, fn2, fn3]

        # Normal gradient
        def loss_normal(x: mx.array) -> mx.array:
            y = x
            for fn in functions:
                y = fn(y)
            return mx.sum(y ** 2)

        grad_normal = mx.grad(loss_normal)(x)

        # Checkpointed gradient
        def loss_ckpt(x: mx.array) -> mx.array:
            y = checkpoint_sequential(functions, 2, x)
            return mx.sum(y ** 2)

        grad_ckpt = mx.grad(loss_ckpt)(x)

        np.testing.assert_allclose(
            np.array(grad_normal), np.array(grad_ckpt), rtol=1e-4, atol=1e-5
        )

    def test_more_segments_than_functions(self) -> None:
        """Should handle segments > num_functions gracefully."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))

        functions = [lambda y: y * 2, lambda y: y + 1]

        # Request 10 segments for 2 functions
        out = checkpoint_sequential(functions, 10, x)
        expected = functions[1](functions[0](x))

        np.testing.assert_allclose(
            np.array(out), np.array(expected), rtol=1e-6, atol=1e-7
        )

    def test_empty_functions_raises(self) -> None:
        """Should raise error for empty function list."""
        x = mx.random.normal((4, 8))
        with pytest.raises(ValueError, match="cannot be empty"):
            checkpoint_sequential([], 1, x)

    def test_invalid_segments_raises(self) -> None:
        """Should raise error for invalid segment count."""
        x = mx.random.normal((4, 8))
        functions = [lambda y: y * 2]
        with pytest.raises(ValueError, match="must be >= 1"):
            checkpoint_sequential(functions, 0, x)


class TestCheckpointEdgeCases:
    """Test edge cases and special scenarios."""

    def test_identity_function(self) -> None:
        """Identity function should work correctly."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))

        def identity(x: mx.array) -> mx.array:
            return x

        out = checkpoint(identity, x)
        np.testing.assert_array_equal(np.array(x), np.array(out))

    def test_scalar_output(self) -> None:
        """Functions returning scalars should work."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))

        def scalar_fn(x: mx.array) -> mx.array:
            return mx.sum(x ** 2)

        out_normal = scalar_fn(x)
        out_ckpt = checkpoint(scalar_fn, x)

        np.testing.assert_allclose(
            np.array(out_normal), np.array(out_ckpt), rtol=1e-6
        )

    def test_tuple_output(self) -> None:
        """Functions with single array output should work."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))

        # Note: checkpoint expects single array output
        def fn(x: mx.array) -> mx.array:
            return x * 2 + 1

        out = checkpoint(fn, x)
        expected = x * 2 + 1

        np.testing.assert_allclose(
            np.array(out), np.array(expected), rtol=1e-6
        )

    def test_nested_checkpoint(self) -> None:
        """Nested checkpoints should work correctly."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8))
        w = mx.random.normal((8, 8))

        def inner_fn(x: mx.array, w: mx.array) -> mx.array:
            return mx.tanh(x @ w)

        def outer_fn(x: mx.array, w: mx.array) -> mx.array:
            return checkpoint(inner_fn, x, w) * 2

        out = checkpoint(outer_fn, x, w)
        expected = mx.tanh(x @ w) * 2

        np.testing.assert_allclose(
            np.array(out), np.array(expected), rtol=1e-5, atol=1e-6
        )


class TestCheckpointNumericalStability:
    """Test numerical stability of checkpointing."""

    def test_large_values(self) -> None:
        """Should handle large input values."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8)) * 100

        def fn(x: mx.array) -> mx.array:
            return mx.tanh(x / 100) * 100

        out = checkpoint(fn, x)
        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()

    def test_small_values(self) -> None:
        """Should handle small input values."""
        mx.random.seed(42)
        x = mx.random.normal((4, 8)) * 1e-6

        def fn(x: mx.array) -> mx.array:
            return x * 1e6

        out = checkpoint(fn, x)
        assert not mx.any(mx.isnan(out)).item()


@pytest.mark.benchmark
class TestCheckpointBenchmarks:
    """Benchmark tests for checkpointing."""

    def test_deep_network(self) -> None:
        """Test checkpointing on a deeper network."""
        mx.random.seed(42)
        x = mx.random.normal((32, 256))

        # Create a sequence of linear transformations
        weights = [mx.random.normal((256, 256)) for _ in range(8)]

        def layer(x: mx.array, w: mx.array) -> mx.array:
            return mx.maximum(x @ w, 0)  # ReLU

        def model(x: mx.array) -> mx.array:
            for w in weights:
                x = layer(x, w)
            return x

        def checkpointed_model(x: mx.array) -> mx.array:
            for w in weights:
                x = checkpoint(layer, x, w)
            return x

        # Verify outputs match
        out_normal = model(x)
        out_ckpt = checkpointed_model(x)

        np.testing.assert_allclose(
            np.array(out_normal), np.array(out_ckpt), rtol=1e-4, atol=1e-5
        )

        # Verify gradients
        def loss_normal(x: mx.array) -> mx.array:
            return mx.sum(model(x) ** 2)

        def loss_ckpt(x: mx.array) -> mx.array:
            return mx.sum(checkpointed_model(x) ** 2)

        grad_normal = mx.grad(loss_normal)(x)
        grad_ckpt = mx.grad(loss_ckpt)(x)

        np.testing.assert_allclose(
            np.array(grad_normal), np.array(grad_ckpt), rtol=1e-3, atol=1e-4
        )
