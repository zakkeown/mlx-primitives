"""Memory profiling utilities for mlx-primitives benchmarks."""

import time
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any
from contextlib import contextmanager

import mlx.core as mx


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage."""

    timestamp: float
    allocated_mb: float
    peak_mb: float
    label: Optional[str] = None


@dataclass
class MemoryProfile:
    """Complete memory profile for a benchmark."""

    name: str
    snapshots: List[MemorySnapshot]
    peak_allocated_mb: float
    final_allocated_mb: float
    memory_delta_mb: float  # Change during execution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "peak_allocated_mb": self.peak_allocated_mb,
            "final_allocated_mb": self.final_allocated_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "num_snapshots": len(self.snapshots),
        }


class MemoryProfiler:
    """Profile memory usage during benchmark execution."""

    def __init__(self):
        """Initialize memory profiler."""
        self.snapshots: List[MemorySnapshot] = []
        self._tracking = False
        self._start_allocated = 0.0

    _warned_no_metal: bool = False

    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage.

        Note: MLX uses unified memory, so we track via Metal memory stats
        when available, falling back to process memory.
        """
        # Force synchronization to ensure accurate measurement
        mx.synchronize()

        # Try various MLX metal memory APIs
        try:
            # MLX >= 0.20 provides metal.get_active_memory() and metal.get_peak_memory()
            if hasattr(mx, "metal"):
                metal = mx.metal
                allocated = 0.0
                peak = 0.0

                if hasattr(metal, "get_active_memory"):
                    allocated = metal.get_active_memory() / (1024 ** 2)
                if hasattr(metal, "get_peak_memory"):
                    peak = metal.get_peak_memory() / (1024 ** 2)

                if allocated > 0 or peak > 0:
                    return {
                        "allocated_mb": allocated,
                        "peak_mb": peak,
                    }

                # Try older API format
                if hasattr(metal, "get_memory_info"):
                    info = metal.get_memory_info()
                    return {
                        "allocated_mb": info.get("allocated", 0) / (1024 ** 2),
                        "peak_mb": info.get("peak", 0) / (1024 ** 2),
                    }
        except Exception:
            pass

        # Warn once if metal memory tracking is unavailable
        if not MemoryProfiler._warned_no_metal:
            import warnings
            warnings.warn(
                "MLX metal memory tracking unavailable. Memory values will be 0. "
                "Use estimate_tensor_memory_mb() for theoretical estimates.",
                RuntimeWarning,
            )
            MemoryProfiler._warned_no_metal = True

        # Fallback: return zeros (caller should use estimate functions)
        return {
            "allocated_mb": 0.0,
            "peak_mb": 0.0,
        }

    def snapshot(self, label: Optional[str] = None) -> MemorySnapshot:
        """Take a memory snapshot.

        Args:
            label: Optional label for this snapshot

        Returns:
            MemorySnapshot with current memory usage
        """
        mx.synchronize()
        info = self._get_memory_info()

        snapshot = MemorySnapshot(
            timestamp=time.perf_counter(),
            allocated_mb=info["allocated_mb"],
            peak_mb=info["peak_mb"],
            label=label,
        )

        if self._tracking:
            self.snapshots.append(snapshot)

        return snapshot

    def _reset_peak_memory(self) -> None:
        """Reset peak memory counter if available."""
        try:
            if hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
                mx.metal.reset_peak_memory()
        except Exception:
            pass

    def start_tracking(self):
        """Start tracking memory snapshots."""
        self._tracking = True
        self.snapshots = []
        self._reset_peak_memory()  # Reset peak for accurate measurement
        self._start_allocated = self._get_memory_info()["allocated_mb"]
        self.snapshot("start")

    def stop_tracking(self) -> MemoryProfile:
        """Stop tracking and return profile.

        Returns:
            MemoryProfile with collected snapshots
        """
        self.snapshot("end")
        self._tracking = False

        if not self.snapshots:
            return MemoryProfile(
                name="unknown",
                snapshots=[],
                peak_allocated_mb=0.0,
                final_allocated_mb=0.0,
                memory_delta_mb=0.0,
            )

        peak_mb = max(s.peak_mb for s in self.snapshots)
        final_mb = self.snapshots[-1].allocated_mb

        return MemoryProfile(
            name="profile",
            snapshots=self.snapshots,
            peak_allocated_mb=peak_mb,
            final_allocated_mb=final_mb,
            memory_delta_mb=final_mb - self._start_allocated,
        )

    @contextmanager
    def track(self, name: str = "profile"):
        """Context manager for memory tracking.

        Example:
            with profiler.track("attention") as profile:
                output = attention(x)
            print(f"Peak memory: {profile.peak_allocated_mb} MB")
        """
        self.start_tracking()
        profile_container = {"profile": None}

        try:
            yield profile_container
        finally:
            profile = self.stop_tracking()
            profile.name = name
            profile_container["profile"] = profile

    def profile_function(
        self,
        func: Callable,
        name: str = "function",
        warmup: int = 1,
        snapshot_interval_ms: float = 10.0,
    ) -> MemoryProfile:
        """Profile memory usage of a function.

        Args:
            func: Function to profile
            name: Name for the profile
            warmup: Number of warmup iterations
            snapshot_interval_ms: Interval between snapshots

        Returns:
            MemoryProfile with memory usage data
        """
        # Warmup
        for _ in range(warmup):
            func()
            mx.synchronize()

        # Profile
        self.start_tracking()
        self.snapshot("before_call")

        func()
        mx.synchronize()

        self.snapshot("after_call")
        profile = self.stop_tracking()
        profile.name = name

        return profile

    def compare_memory(
        self,
        implementations: Dict[str, Callable],
        warmup: int = 1,
    ) -> Dict[str, MemoryProfile]:
        """Compare memory usage of multiple implementations.

        Args:
            implementations: Dict of name -> function
            warmup: Number of warmup iterations

        Returns:
            Dict of name -> MemoryProfile
        """
        results = {}

        for name, func in implementations.items():
            # Force cleanup between implementations
            mx.synchronize()

            profile = self.profile_function(func, name=name, warmup=warmup)
            results[name] = profile

        return results


def estimate_tensor_memory_mb(shape: tuple, dtype: mx.Dtype = mx.float32) -> float:
    """Estimate memory usage of a tensor.

    Args:
        shape: Tensor shape
        dtype: Data type

    Returns:
        Estimated memory in MB
    """
    # Get dtype size in bytes
    dtype_sizes = {
        mx.float32: 4,
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.int32: 4,
        mx.int16: 2,
        mx.int8: 1,
        mx.uint8: 1,
        mx.bool_: 1,
    }

    size = dtype_sizes.get(dtype, 4)
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    return (num_elements * size) / (1024 ** 2)


def estimate_attention_memory_mb(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: mx.Dtype = mx.float32,
    include_qkv: bool = True,
    include_attention_matrix: bool = True,
) -> Dict[str, float]:
    """Estimate memory usage for attention computation.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type
        include_qkv: Include Q, K, V memory
        include_attention_matrix: Include attention matrix memory

    Returns:
        Dict with memory estimates for each component
    """
    estimates = {}

    if include_qkv:
        # Q, K, V each have shape [batch, num_heads, seq_len, head_dim]
        qkv_shape = (batch_size, num_heads, seq_len, head_dim)
        qkv_mem = estimate_tensor_memory_mb(qkv_shape, dtype)
        estimates["query_mb"] = qkv_mem
        estimates["key_mb"] = qkv_mem
        estimates["value_mb"] = qkv_mem

    if include_attention_matrix:
        # Attention matrix: [batch, num_heads, seq_len, seq_len]
        attn_shape = (batch_size, num_heads, seq_len, seq_len)
        estimates["attention_matrix_mb"] = estimate_tensor_memory_mb(attn_shape, dtype)

    # Output: same as Q
    output_shape = (batch_size, num_heads, seq_len, head_dim)
    estimates["output_mb"] = estimate_tensor_memory_mb(output_shape, dtype)

    estimates["total_mb"] = sum(estimates.values())

    return estimates


def memory_scaling_test(
    func_factory: Callable[[int], Callable],
    scales: List[int],
    scale_name: str = "scale",
) -> List[Dict[str, Any]]:
    """Test memory scaling across different sizes.

    Args:
        func_factory: Function that creates benchmark function for given scale
        scales: List of scale values to test
        scale_name: Name of the scale parameter

    Returns:
        List of dicts with scale and memory info
    """
    profiler = MemoryProfiler()
    results = []

    for scale in scales:
        func = func_factory(scale)

        # Clear any cached memory
        mx.synchronize()

        profile = profiler.profile_function(func, name=f"{scale_name}_{scale}")

        results.append({
            scale_name: scale,
            "peak_memory_mb": profile.peak_allocated_mb,
            "memory_delta_mb": profile.memory_delta_mb,
        })

    return results
