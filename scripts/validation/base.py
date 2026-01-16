"""Base classes for golden file generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np


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
    3. Implementing PyTorch reference via generate_pytorch_reference()
    """

    def __init__(
        self,
        output_dir: Path,
        seed: int = 42,
        dtype: str = "float32",
    ):
        """Initialize generator.

        Args:
            output_dir: Directory to save golden files
            seed: Random seed for reproducibility
            dtype: Default dtype for generated tensors
        """
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.dtype = dtype
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
    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        """Generate PyTorch reference output for a single config.

        Args:
            config: Test configuration from get_test_configs()

        Returns:
            TestConfig with inputs, expected outputs, and metadata
        """
        pass

    def generate_all(self) -> List[Path]:
        """Generate all golden files for this operation.

        Returns:
            List of paths to generated .npz files
        """
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        saved_files = []
        configs = self.get_test_configs()

        print(f"  Generating {len(configs)} test cases for {self.name}...")

        for config in configs:
            try:
                test_config = self.generate_pytorch_reference(config)
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
