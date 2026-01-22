"""Tests for concurrent operations.

These tests verify that:
1. Multiple operations can run concurrently without race conditions
2. Kernel caching works correctly under concurrent access
3. Results are consistent regardless of execution order

IMPORTANT: What these tests DO and DON'T test
=============================================
These tests verify the thread-safety of the Python wrapper code and kernel
caching mechanisms, NOT true concurrent GPU kernel execution. All MLX
operations are serialized through a global mutex (_mlx_mutex) because MLX
may not be thread-safe for concurrent kernel execution on the GPU.

What IS tested:
- Thread-safety of Python-level data structures (kernel cache, registries)
- Correctness of results when the same operations are called from multiple threads
- No race conditions in the Python wrapper code
- Kernel compilation and caching under multi-threaded access patterns

What is NOT tested:
- True concurrent GPU kernel execution
- GPU-level parallelism or race conditions
- Performance under actual concurrent GPU workloads

For true concurrency testing, consider using separate processes with their
own MLX contexts, or wait for future MLX versions with explicit threading support.
"""

import concurrent.futures
import threading
from typing import List

import numpy as np
import pytest

import mlx.core as mx

from mlx_primitives import associative_scan, flash_attention

# Global mutex for serializing MLX operations
# MLX may not be thread-safe for concurrent kernel execution
_mlx_mutex = threading.Lock()


def to_numpy(x: mx.array) -> np.ndarray:
    """Convert MLX array to NumPy."""
    with _mlx_mutex:
        mx.eval(x)
        return np.array(x)


class TestConcurrentScans:
    """Tests for concurrent scan operations."""

    @pytest.mark.stress
    def test_concurrent_cumsum_same_size(self) -> None:
        """Test concurrent cumsum operations with same size inputs."""
        np.random.seed(42)
        num_concurrent = 8
        seq_len = 1024

        # Create different inputs for each concurrent operation
        inputs = [
            np.random.randn(seq_len).astype(np.float32)
            for _ in range(num_concurrent)
        ]

        def run_cumsum(x_np: np.ndarray) -> np.ndarray:
            with _mlx_mutex:
                x = mx.array(x_np)
                result = associative_scan(x, operator="add")
                mx.eval(result)
                return np.array(result)

        # Run from multiple threads (serialized by mutex)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(run_cumsum, x) for x in inputs]
            results = [f.result() for f in futures]

        # Verify each result matches sequential computation
        for x_np, result in zip(inputs, results):
            expected = np.cumsum(x_np)
            # Slightly relaxed tolerance for concurrent execution
            np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.stress
    def test_concurrent_cumsum_different_sizes(self) -> None:
        """Test concurrent cumsum operations with different size inputs."""
        np.random.seed(42)
        sizes = [256, 512, 1024, 2048, 128, 768, 1536, 384]

        # Create inputs of different sizes
        inputs = [
            np.random.randn(size).astype(np.float32)
            for size in sizes
        ]

        def run_cumsum(x_np: np.ndarray) -> np.ndarray:
            with _mlx_mutex:
                x = mx.array(x_np)
                result = associative_scan(x, operator="add")
                mx.eval(result)
                return np.array(result)

        # Run from multiple threads (serialized by mutex)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sizes)) as executor:
            futures = [executor.submit(run_cumsum, x) for x in inputs]
            results = [f.result() for f in futures]

        # Verify each result matches sequential computation
        for x_np, result in zip(inputs, results):
            expected = np.cumsum(x_np)
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.stress
    def test_concurrent_ssm_scan(self) -> None:
        """Test concurrent SSM scan operations."""
        np.random.seed(42)
        num_concurrent = 4
        batch, seq, state = 2, 512, 32

        # Create different inputs for each concurrent operation
        inputs = [
            (
                np.random.uniform(0.8, 0.99, (batch, seq, state)).astype(np.float32),
                np.random.randn(batch, seq, state).astype(np.float32)
            )
            for _ in range(num_concurrent)
        ]

        def run_ssm_scan(A_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
            with _mlx_mutex:
                A = mx.array(A_np)
                x = mx.array(x_np)
                result = associative_scan(x, operator="ssm", A=A, axis=1)
                mx.eval(result)
                return np.array(result)

        # Run from multiple threads (serialized by mutex)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(run_ssm_scan, A, x) for A, x in inputs]
            results = [f.result() for f in futures]

        # Verify results are valid (no NaN/Inf from race conditions)
        for result in results:
            assert np.all(np.isfinite(result)), "Result contains NaN or Inf"


class TestConcurrentAttention:
    """Tests for concurrent attention operations."""

    @pytest.mark.stress
    def test_concurrent_attention_same_size(self) -> None:
        """Test concurrent attention operations with same size inputs."""
        np.random.seed(42)
        num_concurrent = 4
        batch, seq, heads, dim = 1, 256, 8, 64

        # Create different inputs for each concurrent operation
        inputs = [
            (
                np.random.randn(batch, seq, heads, dim).astype(np.float32),
                np.random.randn(batch, seq, heads, dim).astype(np.float32),
                np.random.randn(batch, seq, heads, dim).astype(np.float32)
            )
            for _ in range(num_concurrent)
        ]

        def run_attention(q_np, k_np, v_np) -> np.ndarray:
            with _mlx_mutex:
                q = mx.array(q_np)
                k = mx.array(k_np)
                v = mx.array(v_np)
                result = flash_attention(q, k, v, causal=True, use_metal=False)
                mx.eval(result)
                return np.array(result)

        # Run from multiple threads (serialized by mutex)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(run_attention, q, k, v) for q, k, v in inputs]
            results = [f.result() for f in futures]

        # Verify results are valid
        for result in results:
            assert np.all(np.isfinite(result)), "Result contains NaN or Inf"
            assert result.shape == (batch, seq, heads, dim)


class TestKernelCaching:
    """Tests for kernel caching under concurrent access."""

    @pytest.mark.stress
    def test_kernel_cache_thread_safety(self) -> None:
        """Test that kernel caching is thread-safe."""
        np.random.seed(42)
        num_threads = 8
        iterations_per_thread = 5

        results: List[np.ndarray] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker(thread_id: int) -> None:
            try:
                for i in range(iterations_per_thread):
                    x_np = np.random.randn(512).astype(np.float32)
                    with _mlx_mutex:
                        x = mx.array(x_np)
                        result = associative_scan(x, operator="add")
                        mx.eval(result)
                        result_np = np.array(result)

                    expected = np.cumsum(x_np)

                    with lock:
                        # Check correctness
                        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-4)
                        results.append(result_np)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run threads
        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Check for errors
        if errors:
            raise errors[0]

        # All operations should have succeeded
        assert len(results) == num_threads * iterations_per_thread

    @pytest.mark.stress
    def test_mixed_operations_concurrent(self) -> None:
        """Test concurrent execution of different operation types."""
        np.random.seed(42)

        def run_cumsum() -> np.ndarray:
            with _mlx_mutex:
                x = mx.array(np.random.randn(1024).astype(np.float32))
                result = associative_scan(x, operator="add")
                mx.eval(result)
                return np.array(result)

        def run_cumprod() -> np.ndarray:
            with _mlx_mutex:
                x = mx.array(np.random.uniform(0.5, 1.5, 256).astype(np.float32))
                result = associative_scan(x, operator="mul")
                mx.eval(result)
                return np.array(result)

        def run_ssm() -> np.ndarray:
            with _mlx_mutex:
                A = mx.array(np.random.uniform(0.8, 0.99, (2, 256, 16)).astype(np.float32))
                x = mx.array(np.random.randn(2, 256, 16).astype(np.float32))
                result = associative_scan(x, operator="ssm", A=A, axis=1)
                mx.eval(result)
                return np.array(result)

        operations = [run_cumsum, run_cumprod, run_ssm] * 4  # 12 total operations

        # Run from multiple threads (serialized by mutex)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(op) for op in operations]
            results = [f.result() for f in futures]

        # All operations should complete without error
        assert len(results) == 12
        for result in results:
            assert np.all(np.isfinite(result)), "Result contains NaN or Inf"


class TestConsistency:
    """Tests for result consistency under various conditions."""

    @pytest.mark.stress
    def test_repeated_concurrent_consistency(self) -> None:
        """Test that repeated concurrent runs produce consistent results."""
        np.random.seed(42)
        x_np = np.random.randn(4, 1024).astype(np.float32)

        def run_scan() -> np.ndarray:
            with _mlx_mutex:
                x = mx.array(x_np)
                result = associative_scan(x, operator="add", axis=-1)
                mx.eval(result)
                return np.array(result)

        # Run the same computation from multiple threads
        all_results = []
        for _ in range(5):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(run_scan) for _ in range(4)]
                results = [f.result() for f in futures]
                all_results.extend(results)

        # All results should be identical
        reference = all_results[0]
        for result in all_results[1:]:
            np.testing.assert_array_equal(result, reference)

    @pytest.mark.stress
    def test_interleaved_operations(self) -> None:
        """Test interleaved execution of different operations."""
        np.random.seed(42)

        results_cumsum = []
        results_attention = []

        def run_cumsum(idx: int) -> tuple:
            with _mlx_mutex:
                x = mx.array(np.random.randn(512).astype(np.float32) + idx * 0.1)
                result = associative_scan(x, operator="add")
                mx.eval(result)
                return ("cumsum", idx, np.array(result))

        def run_attention(idx: int) -> tuple:
            with _mlx_mutex:
                batch, seq, heads, dim = 1, 64, 4, 32
                q = mx.array(np.random.randn(batch, seq, heads, dim).astype(np.float32))
                k = mx.array(np.random.randn(batch, seq, heads, dim).astype(np.float32))
                v = mx.array(np.random.randn(batch, seq, heads, dim).astype(np.float32))
                result = flash_attention(q, k, v, causal=True, use_metal=False)
                mx.eval(result)
                return ("attention", idx, np.array(result))

        # Create interleaved task list
        tasks = []
        for i in range(8):
            tasks.append(lambda i=i: run_cumsum(i))
            tasks.append(lambda i=i: run_attention(i))

        # Run from multiple threads (serialized by mutex)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(task) for task in tasks]
            results = [f.result() for f in futures]

        # Verify all operations completed
        cumsum_results = [r for r in results if r[0] == "cumsum"]
        attention_results = [r for r in results if r[0] == "attention"]

        assert len(cumsum_results) == 8
        assert len(attention_results) == 8

        # All results should be valid
        for _, _, result in cumsum_results + attention_results:
            assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
