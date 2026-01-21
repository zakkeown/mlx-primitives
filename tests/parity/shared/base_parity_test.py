"""Base classes for parity testing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import mlx.core as mx


@dataclass
class ParityTestConfig:
    """Configuration for a parity test.

    Attributes:
        sizes: List of size configurations to test (tiny, small, medium, large).
        dtypes: List of dtypes to test (fp32, fp16, bf16).
        edge_cases: List of edge case configurations to test.
        tolerances: Dict mapping dtype to (rtol, atol) tuples.
        test_backward: Whether to test backward pass / gradients.
        test_edge_cases: Whether to test edge cases.
    """

    sizes: List[str] = field(default_factory=lambda: ["tiny", "small", "medium", "large"])
    dtypes: List[str] = field(default_factory=lambda: ["fp32", "fp16", "bf16"])
    edge_cases: List[str] = field(default_factory=lambda: ["empty", "single_element", "very_large"])
    tolerances: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    test_backward: bool = True
    test_edge_cases: bool = True

    def get_tolerance(self, dtype: str) -> Tuple[float, float]:
        """Get tolerance for a given dtype.

        Args:
            dtype: The dtype string (fp32, fp16, bf16).

        Returns:
            Tuple of (rtol, atol).
        """
        defaults = {
            "fp32": (1e-5, 1e-6),
            "fp16": (1e-3, 1e-3),
            "bf16": (1e-2, 1e-2),
        }
        return self.tolerances.get(dtype, defaults.get(dtype, (1e-4, 1e-4)))


class ParityTestCase(ABC):
    """Base class for all parity tests.

    Subclasses should implement the abstract methods to define:
    - How to generate inputs for the operation
    - How to run the MLX implementation
    - How to run the reference implementation
    """

    config: ParityTestConfig

    def __init__(self, config: Optional[ParityTestConfig] = None):
        """Initialize the parity test case.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ParityTestConfig()

    @abstractmethod
    def generate_inputs(self, size_config: str, dtype: str) -> Dict[str, np.ndarray]:
        """Generate inputs for the operation.

        Args:
            size_config: Size configuration (tiny, small, medium, large).
            dtype: Data type (fp32, fp16, bf16).

        Returns:
            Dictionary of input arrays as numpy arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def run_mlx_forward(self, inputs: Dict[str, mx.array]) -> mx.array:
        """Run the MLX implementation forward pass.

        Args:
            inputs: Dictionary of MLX input arrays.

        Returns:
            MLX array output.
        """
        raise NotImplementedError

    @abstractmethod
    def run_reference_forward(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run the reference implementation forward pass.

        Args:
            inputs: Dictionary of numpy input arrays.

        Returns:
            Numpy array output.
        """
        raise NotImplementedError

    def convert_inputs_to_mlx(
        self, inputs: Dict[str, np.ndarray], dtype: str
    ) -> Dict[str, mx.array]:
        """Convert numpy inputs to MLX arrays.

        Args:
            inputs: Dictionary of numpy arrays.
            dtype: Target dtype string.

        Returns:
            Dictionary of MLX arrays.
        """
        dtype_map = {
            "fp32": mx.float32,
            "fp16": mx.float16,
            "bf16": mx.bfloat16,
        }
        mlx_dtype = dtype_map.get(dtype, mx.float32)

        return {
            key: mx.array(val).astype(mlx_dtype)
            for key, val in inputs.items()
        }

    def compare_outputs(
        self,
        mlx_out: mx.array,
        ref_out: np.ndarray,
        rtol: float,
        atol: float,
    ) -> bool:
        """Compare MLX output to reference output.

        Args:
            mlx_out: MLX array output.
            ref_out: Reference numpy array output.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            True if outputs match within tolerance.
        """
        mlx_np = np.array(mlx_out)
        return np.allclose(mlx_np, ref_out, rtol=rtol, atol=atol)

    def assert_outputs_close(
        self,
        mlx_out: mx.array,
        ref_out: np.ndarray,
        rtol: float,
        atol: float,
        msg: str = "",
    ) -> None:
        """Assert that MLX output matches reference output.

        Args:
            mlx_out: MLX array output.
            ref_out: Reference numpy array output.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            msg: Optional message for assertion error.
        """
        mlx_np = np.array(mlx_out)
        np.testing.assert_allclose(mlx_np, ref_out, rtol=rtol, atol=atol, err_msg=msg)

    def test_forward_parity(self, size_config: str, dtype: str) -> None:
        """Test forward pass parity.

        Args:
            size_config: Size configuration.
            dtype: Data type.
        """
        # Generate inputs
        inputs_np = self.generate_inputs(size_config, dtype)

        # Convert to MLX
        inputs_mlx = self.convert_inputs_to_mlx(inputs_np, dtype)

        # Run MLX forward
        mlx_out = self.run_mlx_forward(inputs_mlx)

        # Run reference forward
        ref_out = self.run_reference_forward(inputs_np)

        # Compare
        rtol, atol = self.config.get_tolerance(dtype)
        self.assert_outputs_close(
            mlx_out, ref_out, rtol, atol,
            msg=f"Forward parity failed for size={size_config}, dtype={dtype}"
        )

    def test_backward_parity(self, size_config: str, dtype: str = "fp32") -> None:
        """Test backward pass (gradient) parity.

        Args:
            size_config: Size configuration.
            dtype: Data type (typically fp32 for gradient tests).
        """
        # Stub - subclasses should implement
        raise NotImplementedError("Backward parity test not implemented")


class ForwardParityMixin:
    """Mixin providing forward pass parity test utilities."""

    def run_forward_parity_test(
        self,
        mlx_fn: Callable,
        ref_fn: Callable,
        inputs_np: Dict[str, np.ndarray],
        dtype: str,
        rtol: float,
        atol: float,
    ) -> None:
        """Run a forward parity test.

        Args:
            mlx_fn: MLX implementation function.
            ref_fn: Reference implementation function.
            inputs_np: Numpy inputs dictionary.
            dtype: Data type string.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
        """
        # Convert to MLX dtype
        dtype_map = {
            "fp32": mx.float32,
            "fp16": mx.float16,
            "bf16": mx.bfloat16,
        }
        mlx_dtype = dtype_map.get(dtype, mx.float32)

        # Convert inputs
        inputs_mlx = {
            key: mx.array(val).astype(mlx_dtype)
            for key, val in inputs_np.items()
        }

        # Run both implementations
        mlx_out = mlx_fn(**inputs_mlx)
        ref_out = ref_fn(**inputs_np)

        # Compare
        mlx_np = np.array(mlx_out)
        np.testing.assert_allclose(mlx_np, ref_out, rtol=rtol, atol=atol)


class BackwardParityMixin:
    """Mixin providing backward pass (gradient) parity test utilities."""

    def run_backward_parity_test(
        self,
        mlx_fn: Callable,
        ref_fn: Callable,
        inputs_np: Dict[str, np.ndarray],
        grad_output_np: np.ndarray,
        wrt: List[str],
        rtol: float,
        atol: float,
    ) -> None:
        """Run a backward parity test.

        Args:
            mlx_fn: MLX implementation function.
            ref_fn: Reference implementation function.
            inputs_np: Numpy inputs dictionary.
            grad_output_np: Gradient of output (upstream gradient).
            wrt: List of input names to compute gradients with respect to.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
        """
        # Stub - implementation depends on framework-specific gradient computation
        raise NotImplementedError("Backward parity test not implemented")

    def compute_numerical_gradient(
        self,
        fn: Callable,
        inputs: Dict[str, np.ndarray],
        wrt: str,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute numerical gradient using finite differences.

        Args:
            fn: Function to differentiate.
            inputs: Input dictionary.
            wrt: Input name to compute gradient with respect to.
            eps: Epsilon for finite differences.

        Returns:
            Numerical gradient as numpy array.
        """
        x = inputs[wrt].copy()
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            old_value = x[idx]

            # f(x + eps)
            x[idx] = old_value + eps
            inputs_plus = {**inputs, wrt: x}
            f_plus = fn(**inputs_plus)
            if isinstance(f_plus, mx.array):
                f_plus = np.array(f_plus)

            # f(x - eps)
            x[idx] = old_value - eps
            inputs_minus = {**inputs, wrt: x}
            f_minus = fn(**inputs_minus)
            if isinstance(f_minus, mx.array):
                f_minus = np.array(f_minus)

            # Gradient
            grad[idx] = np.sum(f_plus - f_minus) / (2 * eps)

            # Restore
            x[idx] = old_value
            it.iternext()

        return grad
