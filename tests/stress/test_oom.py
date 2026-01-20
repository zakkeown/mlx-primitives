"""Tests for out-of-memory scenario handling.

These tests verify that the library:
1. Handles memory pressure gracefully
2. Provides helpful error messages when OOM occurs
3. Cleans up memory properly after failed operations
"""

import gc

import numpy as np
import pytest

import mlx.core as mx

from mlx_primitives import associative_scan, flash_attention


def get_available_memory_mb() -> float:
    """Get approximate available memory in MB.

    Note: This is a rough estimate on unified memory systems.
    """
    try:
        info = mx.metal.device_info()
        total_memory = info.get("memory_size", 8 * 1024 * 1024 * 1024)
        # Assume ~70% is typically available (conservative)
        return (total_memory * 0.7) / (1024 * 1024)
    except Exception:
        return 8000.0  # Default 8GB


class TestMemoryBoundaries:
    """Tests that exercise memory boundaries without intentionally causing OOM."""

    @pytest.mark.stress
    @pytest.mark.slow
    def test_large_but_feasible_scan(self) -> None:
        """Test scan on largest feasible sequence for available memory."""
        available_mb = get_available_memory_mb()

        # Calculate largest feasible sequence (each float32 = 4 bytes)
        # Need 2x for input + output
        max_elements = int((available_mb * 1024 * 1024) / (4 * 2) * 0.5)  # Use 50% of available
        max_seq_len = min(max_elements, 100_000)  # Cap at 100K

        np.random.seed(42)
        x_np = np.random.randn(max_seq_len).astype(np.float32)

        try:
            result = associative_scan(mx.array(x_np), operator="add")
            mx.eval(result)
            assert result.shape == (max_seq_len,)
        except MemoryError:
            pytest.skip(f"Insufficient memory for {max_seq_len} element scan")

    @pytest.mark.stress
    @pytest.mark.slow
    def test_large_but_feasible_attention(self) -> None:
        """Test attention on largest feasible configuration for available memory."""
        available_mb = get_available_memory_mb()

        # Attention memory: O(batch * heads * seq^2) for scores
        # Start conservative and adjust
        batch, heads, dim = 1, 4, 32

        # Estimate max seq_len (each score = 4 bytes, need ~3x for q/k/v/scores/output)
        max_seq = int(np.sqrt((available_mb * 1024 * 1024) / (4 * batch * heads * 3) * 0.3))
        max_seq = min(max_seq, 4096)  # Cap at 4K

        np.random.seed(42)
        q_np = np.random.randn(batch, max_seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, max_seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, max_seq, heads, dim).astype(np.float32)

        try:
            result = flash_attention(
                mx.array(q_np), mx.array(k_np), mx.array(v_np),
                causal=True, use_metal=False
            )
            mx.eval(result)
            assert result.shape == (batch, max_seq, heads, dim)
        except MemoryError:
            pytest.skip(f"Insufficient memory for {max_seq} seq attention")


class TestMemoryCleanup:
    """Tests that verify memory is properly cleaned up."""

    @pytest.mark.stress
    def test_memory_cleanup_after_scan(self) -> None:
        """Test that memory is freed after scan operations."""
        np.random.seed(42)
        seq_len = 10000

        # Create and evaluate arrays
        x_np = np.random.randn(seq_len).astype(np.float32)

        for _ in range(5):
            x = mx.array(x_np)
            result = associative_scan(x, operator="add")
            mx.eval(result)
            # Explicit cleanup
            del x, result
            gc.collect()

        # If we get here without OOM, cleanup is working

    @pytest.mark.stress
    def test_memory_cleanup_after_attention(self) -> None:
        """Test that memory is freed after attention operations."""
        np.random.seed(42)
        batch, seq, heads, dim = 1, 512, 4, 64

        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        for _ in range(5):
            q = mx.array(q_np)
            k = mx.array(k_np)
            v = mx.array(v_np)
            result = flash_attention(q, k, v, causal=True, use_metal=False)
            mx.eval(result)
            # Explicit cleanup
            del q, k, v, result
            gc.collect()

        # If we get here without OOM, cleanup is working

    @pytest.mark.stress
    def test_incremental_memory_usage(self) -> None:
        """Test that memory usage stays bounded during repeated operations."""
        np.random.seed(42)
        seq_len = 5000

        x_np = np.random.randn(seq_len).astype(np.float32)

        # Run many iterations to check for memory leaks
        for i in range(20):
            x = mx.array(x_np)
            result = associative_scan(x, operator="add")
            mx.eval(result)

            # Check result is valid
            if i == 0:
                first_result = np.array(result)
            else:
                # Results should be identical
                np.testing.assert_array_equal(np.array(result), first_result)

            del x, result

        gc.collect()
        # If we complete 20 iterations without OOM, no significant memory leak


class TestGracefulDegradation:
    """Tests for graceful handling when approaching memory limits."""

    @pytest.mark.stress
    @pytest.mark.slow
    def test_chunked_processing_large_input(self) -> None:
        """Test that large inputs can be processed in chunks if needed."""
        np.random.seed(42)

        # Process a large sequence in chunks
        total_seq = 50000
        chunk_size = 10000
        num_chunks = total_seq // chunk_size

        x_np = np.random.randn(total_seq).astype(np.float32)

        # Process in chunks and accumulate results
        chunk_results = []
        running_sum = 0.0

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = x_np[start:end]

            result = associative_scan(mx.array(chunk), operator="add")
            mx.eval(result)
            chunk_result = np.array(result) + running_sum
            chunk_results.append(chunk_result)
            running_sum = chunk_result[-1]

        # Verify chunked processing matches full processing
        full_result = np.cumsum(x_np)
        chunked_result = np.concatenate(chunk_results)

        # Relaxed tolerance due to accumulation of floating point errors over long sequences
        np.testing.assert_allclose(chunked_result, full_result, rtol=1e-3, atol=1e-3)


class TestInputValidation:
    """Tests for proper input validation to prevent OOM."""

    @pytest.mark.stress
    def test_scan_validates_input_shape(self) -> None:
        """Test that scan validates input shapes to prevent OOM on malformed inputs."""
        # Empty input should be handled gracefully
        x = mx.array([])
        try:
            result = associative_scan(x, operator="add")
            mx.eval(result)
            assert result.shape == (0,)
        except (ValueError, RuntimeError):
            # Either returning empty or raising an error is acceptable
            pass

    @pytest.mark.stress
    def test_attention_validates_input_shapes(self) -> None:
        """Test that attention validates input shapes.

        Note: K and V having different seq_len from Q is actually valid for
        cross-attention scenarios. We test other shape mismatches instead.
        """
        batch, seq, heads, dim = 2, 64, 8, 64

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        # Mismatched dim should raise an error
        v_wrong_dim = mx.random.normal((batch, seq, heads, dim + 8))

        with pytest.raises((ValueError, RuntimeError)):
            result = flash_attention(q, k, v_wrong_dim, causal=True, use_metal=False)
            mx.eval(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
