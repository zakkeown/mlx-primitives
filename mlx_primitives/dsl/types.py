"""Type system for Metal-Triton DSL.

Provides static type annotations that map to Metal types:
- Scalar types: float32, float16, int32, uint32, bool_
- Pointer types: Pointer[T]
- Compile-time constants: constexpr[T]
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar, Union, Optional, Any

T = TypeVar("T")


class MetalDType(Enum):
    """Metal data types."""
    FLOAT32 = "float"
    FLOAT16 = "half"
    BFLOAT16 = "bfloat"
    INT32 = "int"
    INT16 = "short"
    INT8 = "char"
    UINT32 = "uint"
    UINT16 = "ushort"
    UINT8 = "uchar"
    BOOL = "bool"

    @property
    def size_bytes(self) -> int:
        """Size in bytes."""
        sizes = {
            MetalDType.FLOAT32: 4,
            MetalDType.FLOAT16: 2,
            MetalDType.BFLOAT16: 2,
            MetalDType.INT32: 4,
            MetalDType.INT16: 2,
            MetalDType.INT8: 1,
            MetalDType.UINT32: 4,
            MetalDType.UINT16: 2,
            MetalDType.UINT8: 1,
            MetalDType.BOOL: 1,
        }
        return sizes[self]

    @property
    def metal_name(self) -> str:
        """Metal type name."""
        return self.value


@dataclass
class DType:
    """DSL data type wrapper."""
    metal_dtype: MetalDType

    def __repr__(self) -> str:
        return f"mt.{self.metal_dtype.name.lower()}"

    @property
    def metal_name(self) -> str:
        return self.metal_dtype.metal_name


# Scalar type singletons
float32 = DType(MetalDType.FLOAT32)
float16 = DType(MetalDType.FLOAT16)
bfloat16 = DType(MetalDType.BFLOAT16)
int32 = DType(MetalDType.INT32)
int16 = DType(MetalDType.INT16)
int8 = DType(MetalDType.INT8)
uint32 = DType(MetalDType.UINT32)
uint16 = DType(MetalDType.UINT16)
uint8 = DType(MetalDType.UINT8)
bool_ = DType(MetalDType.BOOL)

# Alias for dtype parameter in functions
dtype = DType


@dataclass
class Pointer(Generic[T]):
    """Pointer type for device memory.

    Usage in kernel signatures:
        def kernel(a_ptr: mt.Pointer[mt.float32], ...):
            val = mt.load(a_ptr + offset)
    """
    element_type: DType

    def __class_getitem__(cls, item: DType) -> "Pointer":
        """Enable Pointer[float32] syntax."""
        return Pointer(element_type=item)

    @property
    def metal_name(self) -> str:
        return f"device {self.element_type.metal_name}*"

    def __repr__(self) -> str:
        return f"mt.Pointer[{self.element_type}]"


@dataclass
class SharedMemoryAlloc:
    """Shared (threadgroup) memory allocation.

    Represents a block of memory shared across all threads in a threadgroup.
    Apple Silicon has 32KB threadgroup memory.

    Usage:
        # Allocate shared memory for a tile
        shared_k = mt.shared_memory(BLOCK_N, head_dim, dtype=mt.float32)

        # Load into shared memory
        mt.load_shared(shared_k, K_ptr + offset, size)

        # Synchronize before reading
        mt.threadgroup_barrier()

        # Read from shared
        k_val = mt.load(shared_k + local_idx)
    """
    name: str
    shape: tuple[Union[int, str], ...]
    dtype: DType
    padding: int = 4  # Default padding for bank conflict avoidance

    @property
    def size(self) -> str:
        """Total size expression including padding."""
        if len(self.shape) == 1:
            return f"({self.shape[0]} + {self.padding})"
        elif len(self.shape) == 2:
            # 2D: pad the inner dimension
            return f"({self.shape[0]}) * ({self.shape[1]} + {self.padding})"
        else:
            # Multi-dimensional
            sizes = [str(s) for s in self.shape]
            # Add padding to innermost dimension
            sizes[-1] = f"({sizes[-1]} + {self.padding})"
            return " * ".join(f"({s})" for s in sizes)

    @property
    def metal_declaration(self) -> str:
        """Metal threadgroup declaration."""
        return f"threadgroup {self.dtype.metal_name} {self.name}[{self.size}]"

    def __repr__(self) -> str:
        return f"SharedMemory({self.name}, {self.shape}, {self.dtype})"


class ConstExprMeta(type):
    """Metaclass to enable constexpr[T] syntax."""

    def __getitem__(cls, item: Union[DType, type]) -> "ConstExpr":
        if isinstance(item, DType):
            return ConstExpr(dtype=item)
        elif item == int:
            return ConstExpr(dtype=uint32)
        elif item == float:
            return ConstExpr(dtype=float32)
        elif item == bool:
            return ConstExpr(dtype=bool_)
        else:
            raise TypeError(f"Invalid constexpr type: {item}")


@dataclass
class ConstExpr(metaclass=ConstExprMeta):
    """Compile-time constant type.

    Used for template parameters that are known at compile time:
        def kernel(..., BLOCK_SIZE: mt.constexpr = 256):
            # BLOCK_SIZE is substituted at Metal compile time

    Can also specify explicit type:
        N: mt.constexpr[mt.uint32]
    """
    dtype: Optional[DType] = None

    def __repr__(self) -> str:
        if self.dtype:
            return f"mt.constexpr[{self.dtype}]"
        return "mt.constexpr"


# Singleton for untyped constexpr
constexpr = ConstExpr()


@dataclass
class TensorType:
    """Tensor type for intermediate values in kernels.

    Represents a block of values with known shape and dtype.
    """
    shape: tuple[Union[int, str], ...]  # Can include constexpr names
    dtype: DType

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        shape_str = ", ".join(str(s) for s in self.shape)
        return f"Tensor[({shape_str}), {self.dtype}]"


@dataclass
class SharedMemoryType:
    """Threadgroup (shared) memory allocation.

    Declares memory shared across threads in a threadgroup:
        shared_a: mt.SharedMemory[mt.float32, BLOCK_SIZE]
    """
    dtype: DType
    size: Union[int, str]  # Size or constexpr name
    padding: int = 4  # Default padding for bank conflict avoidance

    @property
    def metal_declaration(self) -> str:
        padded_size = f"{self.size} + {self.padding}" if self.padding else str(self.size)
        return f"threadgroup {self.dtype.metal_name} shared[{padded_size}]"


def infer_dtype(value: Any) -> DType:
    """Infer DSL dtype from Python value."""
    if isinstance(value, bool):
        return bool_
    elif isinstance(value, int):
        return int32
    elif isinstance(value, float):
        return float32
    elif isinstance(value, DType):
        return value
    else:
        raise TypeError(f"Cannot infer dtype from {type(value)}")


def dtype_from_annotation(annotation: Any) -> Optional[DType]:
    """Extract dtype from type annotation."""
    if isinstance(annotation, DType):
        return annotation
    elif isinstance(annotation, Pointer):
        return annotation.element_type
    elif isinstance(annotation, ConstExpr):
        return annotation.dtype
    return None
