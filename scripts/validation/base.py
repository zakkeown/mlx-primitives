"""Base classes for golden file generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import json
import numpy as np

# Backend type for reference implementations
Backend = Literal["numpy", "pytorch", "both"]


@dataclass
class TestConfig:
    """Configuration for a single test case."""

    name: str
    inputs: Dict[str, np.ndarray]
    params: Dict[str, Any]
    expected_outputs: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    tolerance: Optional["ToleranceConfig"] = None  # Per-test tolerance override


@dataclass
class ToleranceConfig:
    """Tolerance configuration for numerical comparison."""

    # Standard relative/absolute tolerances
    rtol_fp32: float = 1e-5
    atol_fp32: float = 1e-6
    rtol_fp16: float = 1e-3
    atol_fp16: float = 1e-4
    rtol_bf16: float = 1e-2
    atol_bf16: float = 1e-3

    # Additional max/mean diff tolerances for unstable operations
    max_diff_fp32: Optional[float] = None
    max_diff_fp16: Optional[float] = None
    mean_diff_fp32: Optional[float] = None
    mean_diff_fp16: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rtol_fp32": self.rtol_fp32,
            "atol_fp32": self.atol_fp32,
            "rtol_fp16": self.rtol_fp16,
            "atol_fp16": self.atol_fp16,
            "rtol_bf16": self.rtol_bf16,
            "atol_bf16": self.atol_bf16,
            "max_diff_fp32": self.max_diff_fp32,
            "max_diff_fp16": self.max_diff_fp16,
            "mean_diff_fp32": self.mean_diff_fp32,
            "mean_diff_fp16": self.mean_diff_fp16,
        }


class GoldenGenerator(ABC):
    """Base class for golden file generators.

    Subclasses implement operation-specific golden file generation by:
    1. Defining tolerance configuration via get_tolerance_config()
    2. Defining test configurations via get_test_configs()
    3. Implementing NumPy reference via generate_numpy_reference() (required)
    4. Optionally implementing PyTorch reference via generate_pytorch_reference()

    The NumPy backend is the default and enables validation without PyTorch.
    The PyTorch backend can be used for cross-validation.
    """

    def __init__(
        self,
        output_dir: Path,
        seed: int = 42,
        dtype: str = "float32",
        backend: Backend = "numpy",
    ):
        """Initialize generator.

        Args:
            output_dir: Directory to save golden files
            seed: Random seed for reproducibility
            dtype: Default dtype for generated tensors
            backend: Reference implementation backend ("numpy", "pytorch", "both")
        """
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.dtype = dtype
        self.backend = backend
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this generator (e.g., 'gelu', 'rmsnorm')."""
        pass

    @abstractmethod
    def get_tolerance_config(self) -> ToleranceConfig:
        """Return tolerance configuration for this operation type."""
        pass

    @abstractmethod
    def get_test_configs(self) -> List[Dict[str, Any]]:
        """Return list of test configurations (shapes, params, etc.).

        Each config dict should contain at minimum:
        - 'name': unique identifier for this test case
        - Shape/size parameters needed for the operation
        - Any operation-specific parameters
        """
        pass

    @abstractmethod
    def generate_numpy_reference(self, config: Dict[str, Any]) -> TestConfig:
        """Generate NumPy reference output for a single config.

        This is the primary reference implementation and must be implemented.

        Args:
            config: Test configuration from get_test_configs()

        Returns:
            TestConfig with inputs, expected outputs, and metadata
        """
        pass

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        """Generate PyTorch reference output for a single config.

        Optional method for cross-validation. Default raises NotImplementedError.

        Args:
            config: Test configuration from get_test_configs()

        Returns:
            TestConfig with inputs, expected outputs, and metadata
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement generate_pytorch_reference(). "
            "Use backend='numpy' or implement this method for PyTorch support."
        )

    def generate_reference(self, config: Dict[str, Any]) -> TestConfig:
        """Generate reference output using the configured backend.

        Args:
            config: Test configuration from get_test_configs()

        Returns:
            TestConfig with inputs, expected outputs, and metadata
        """
        if self.backend == "numpy":
            return self.generate_numpy_reference(config)
        elif self.backend == "pytorch":
            return self.generate_pytorch_reference(config)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def validate_implementations(
        self,
        config: Dict[str, Any],
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> Tuple[bool, Optional[str]]:
        """Cross-validate NumPy and PyTorch produce same results.

        Args:
            config: Test configuration to validate
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison

        Returns:
            Tuple of (success, error_message)
        """
        try:
            numpy_result = self.generate_numpy_reference(config)
            pytorch_result = self.generate_pytorch_reference(config)
        except NotImplementedError as e:
            return False, str(e)

        # Compare outputs
        for key in numpy_result.expected_outputs:
            if key not in pytorch_result.expected_outputs:
                return False, f"Output key '{key}' missing in PyTorch result"

            np_out = numpy_result.expected_outputs[key]
            pt_out = pytorch_result.expected_outputs[key]

            if np_out.shape != pt_out.shape:
                return False, f"Shape mismatch for '{key}': {np_out.shape} vs {pt_out.shape}"

            if not np.allclose(np_out, pt_out, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(np_out - pt_out))
                return False, f"Values differ for '{key}': max_diff={max_diff:.2e}"

        return True, None

    def generate_all(self, validate: bool = False) -> List[Path]:
        """Generate all golden files for this operation.

        Args:
            validate: If True and backend is 'both', cross-validate implementations

        Returns:
            List of paths to generated .npz files
        """
        np.random.seed(self.seed)

        # Only import torch if actually using it
        if self.backend in ("pytorch", "both"):
            try:
                import torch
                torch.manual_seed(self.seed)
            except ImportError:
                if self.backend == "pytorch":
                    raise ImportError(
                        "PyTorch is required for backend='pytorch'. "
                        "Install with: pip install torch"
                    )
                # For 'both', fall back to numpy-only with a warning
                print(f"  WARNING: PyTorch not available, using NumPy only for {self.name}")
                self.backend = "numpy"

        saved_files = []
        configs = self.get_test_configs()

        print(f"  Generating {len(configs)} test cases for {self.name} (backend={self.backend})...")

        for config in configs:
            try:
                # Generate with configured backend
                if self.backend == "both":
                    # Use NumPy for the golden file
                    test_config = self.generate_numpy_reference(config)

                    # Optionally validate against PyTorch
                    if validate:
                        success, error = self.validate_implementations(config)
                        if not success:
                            print(f"    VALIDATION WARNING {config.get('name', 'unknown')}: {error}")
                else:
                    test_config = self.generate_reference(config)

                path = self._save_golden(test_config)
                saved_files.append(path)
                print(f"    Saved: {path.name}")
            except Exception as e:
                print(f"    ERROR generating {config.get('name', 'unknown')}: {e}")

        return saved_files

    def _save_golden(self, test_config: TestConfig) -> Path:
        """Save test config to .npz file.

        File format:
        - Input arrays: stored directly
        - Expected outputs: prefixed with 'expected_'
        - Metadata: stored as JSON string in '__metadata__'
        """
        save_dict = {}

        # Save inputs
        for key, value in test_config.inputs.items():
            save_dict[key] = value

        # Save expected outputs with prefix
        for key, value in test_config.expected_outputs.items():
            save_dict[f"expected_{key}"] = value

        # Save metadata including tolerance config
        # Use per-test tolerance if provided, otherwise fall back to generator default
        tolerance = test_config.tolerance if test_config.tolerance else self.get_tolerance_config()
        metadata = {
            **test_config.metadata,
            "params": test_config.params,
            "tolerance": tolerance.to_dict(),
            "generator": self.name,
            "seed": self.seed,
        }
        save_dict["__metadata__"] = np.array([json.dumps(metadata)])

        path = self.output_dir / f"{test_config.name}.npz"
        np.savez(path, **save_dict)
        return path


class EdgeCaseMixin:
    """Mixin providing standard edge cases for numerical testing."""

    @staticmethod
    def get_edge_case_inputs(
        shape: Tuple[int, ...], dtype: np.dtype = np.float32
    ) -> Dict[str, np.ndarray]:
        """Generate edge case input tensors.

        Returns dict with keys:
        - 'zeros': all zeros
        - 'ones': all ones
        - 'small': very small values (1e-7)
        - 'large': large values (1e4)
        - 'negative': all negative ones
        - 'mixed': random mixed values
        """
        np.random.seed(42)  # Reproducible
        return {
            "zeros": np.zeros(shape, dtype=dtype),
            "ones": np.ones(shape, dtype=dtype),
            "small": np.full(shape, 1e-7, dtype=dtype),
            "large": np.full(shape, 1e4, dtype=dtype),
            "negative": np.full(shape, -1.0, dtype=dtype),
            "mixed": np.random.randn(*shape).astype(dtype),
        }

    @staticmethod
    def get_edge_case_configs(
        base_name: str,
        base_shape: Tuple[int, ...],
        include_long_seq: bool = True,
        max_seq_len: int = 4096,
    ) -> List[Dict[str, Any]]:
        """Generate standard edge case configurations.

        Args:
            base_name: Base name for test cases
            base_shape: Base shape tuple (batch, seq, dims) or similar
            include_long_seq: Whether to include long sequence test
            max_seq_len: Maximum sequence length for long seq test

        Returns:
            List of edge case config dicts
        """
        configs = [
            {"name": f"{base_name}_zeros", "shape": base_shape, "input_type": "zeros"},
            {"name": f"{base_name}_small", "shape": base_shape, "input_type": "small"},
            {"name": f"{base_name}_large", "shape": base_shape, "input_type": "large"},
            {
                "name": f"{base_name}_negative",
                "shape": base_shape,
                "input_type": "negative",
            },
            {
                "name": f"{base_name}_single",
                "shape": (1,) * len(base_shape),
                "input_type": "random",
            },
        ]

        if include_long_seq and len(base_shape) >= 2:
            long_shape = list(base_shape)
            long_shape[1] = max_seq_len  # Assume dim 1 is sequence
            configs.append(
                {
                    "name": f"{base_name}_long_seq",
                    "shape": tuple(long_shape),
                    "input_type": "random",
                }
            )

        return configs


def generate_input_tensor(
    shape: Tuple[int, ...],
    input_type: str = "random",
    dtype: np.dtype = np.float32,
    seed: int = 42,
) -> np.ndarray:
    """Generate input tensor based on type.

    Args:
        shape: Tensor shape
        input_type: One of 'random', 'zeros', 'ones', 'small', 'large', 'negative'
        dtype: NumPy dtype
        seed: Random seed for reproducibility

    Returns:
        NumPy array with specified characteristics
    """
    np.random.seed(seed)

    if input_type == "zeros":
        return np.zeros(shape, dtype=dtype)
    elif input_type == "ones":
        return np.ones(shape, dtype=dtype)
    elif input_type == "small":
        return np.full(shape, 1e-7, dtype=dtype)
    elif input_type == "large":
        return np.full(shape, 1e4, dtype=dtype)
    elif input_type == "negative":
        return np.full(shape, -1.0, dtype=dtype)
    elif input_type == "random":
        return np.random.randn(*shape).astype(dtype)
    elif input_type == "uniform":
        return np.random.uniform(-1, 1, shape).astype(dtype)
    elif input_type == "positive":
        return np.abs(np.random.randn(*shape).astype(dtype)) + 0.1
    else:
        raise ValueError(f"Unknown input_type: {input_type}")
