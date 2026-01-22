"""Custom exceptions for MLX Primitives.

This module provides a hierarchy of exceptions for specific error conditions,
enabling more precise exception handling throughout the codebase.
"""


class MLXPrimitivesError(Exception):
    """Base exception for all MLX Primitives errors."""

    pass


class MetalKernelError(MLXPrimitivesError):
    """Error in Metal kernel compilation or execution.

    Raised when a Metal kernel fails to compile, execute, or produces
    invalid results.
    """

    pass


class HardwareDetectionError(MLXPrimitivesError):
    """Error detecting hardware capabilities.

    Raised when hardware detection fails due to missing drivers,
    sandbox restrictions, or unavailable APIs.
    """

    pass


class QuantizationError(MLXPrimitivesError):
    """Error during quantization operations.

    Raised when quantization fails due to numerical issues
    (e.g., singular Hessian) or invalid parameters.
    """

    pass


class CoreMLError(MLXPrimitivesError):
    """Error in Core ML operations.

    Raised when Core ML model compilation, conversion, or execution fails.
    """

    pass


class MMapError(MLXPrimitivesError):
    """Error in memory-mapped file operations.

    Raised when memory mapping fails due to file access issues,
    insufficient memory, or invalid parameters.
    """

    pass
