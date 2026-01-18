"""Main compilation entry point for Metal-Triton.

MLX-first design: generates Metal code compatible with mx.fast.metal_kernel.
Orchestrates: Python AST → IR → Metal body → MLX kernel
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_primitives.dsl.decorators import KernelDefinition, Config

from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
from mlx_primitives.dsl.compiler.codegen import generate_mlx_body, detect_output_params, detect_input_params, detect_uses_atomics
from mlx_primitives.dsl.compiler.metal_ir import IRFunction, IRType


@dataclass
class CompiledKernelInfo:
    """Information about a compiled kernel for execution."""
    mlx_kernel: Any
    metal_source: str
    input_names: list[str]
    output_names: list[str]
    output_indices: list[int]  # Indices in original parameter list
    inout_names: list[str]  # Parameters that are both input and output
    inout_indices: list[int]  # Indices of inout parameters
    uses_atomics: bool  # Whether kernel uses atomic operations


def compile_kernel(
    kernel_def: "KernelDefinition",
    config: "Config",
    debug: bool = False,
) -> tuple[str, CompiledKernelInfo]:
    """Compile a kernel definition to MLX Metal kernel.

    MLX-first: generates function body that works with mx.fast.metal_kernel's
    auto-generated function signature.

    Args:
        kernel_def: Parsed kernel definition
        config: Configuration for constexpr values
        debug: Include debug info in generated code

    Returns:
        Tuple of (metal_source, CompiledKernelInfo)
    """
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError("MLX not available. Install with: pip install mlx")

    # Parse Python AST to IR
    ir_func = parse_kernel(
        kernel_def.source_code,
        kernel_def.parameters,
        kernel_def.constexpr_params,
    )

    # Detect which parameters are inputs (have loads) and outputs (have stores)
    output_param_names = set(detect_output_params(ir_func))
    input_param_names = set(detect_input_params(ir_func))

    # Inout params are both loaded from AND stored to
    inout_param_names = output_param_names & input_param_names

    # Build input/output names for MLX
    # - Pure inputs: params that are only loaded (not stored)
    # - Pure outputs: params that are only stored (not loaded)
    # - Inout: params that are both - use original name for input, renamed for output
    #   (MLX can't have same name in both input and output)
    input_names = []
    output_names = []
    output_indices = []
    inout_names = []
    inout_indices = []
    inout_renames = {}  # Maps original name to output name

    for idx, param in enumerate(ir_func.parameters):
        is_output = param.name in output_param_names
        is_input = param.name in input_param_names
        is_inout = param.name in inout_param_names

        if is_inout:
            # Inout param: input uses original name, output uses renamed version
            input_names.append(param.name)
            out_name = f"{param.name}_out"
            output_names.append(out_name)
            output_indices.append(idx)
            inout_names.append(param.name)
            inout_indices.append(idx)
            inout_renames[param.name] = out_name
        elif is_output:
            # Pure output: only in output_names
            output_names.append(param.name)
            output_indices.append(idx)
        else:
            # Pure input (or non-pointer param): only in input_names
            input_names.append(param.name)

    # Detect if kernel uses atomic operations
    uses_atomics = detect_uses_atomics(ir_func)

    # Generate Metal function body (not full kernel - MLX generates signature)
    # Pass inout_renames so stores to inout params use the renamed output
    metal_body = generate_mlx_body(ir_func, input_names, output_names, inout_renames)

    if debug:
        metal_body = f"// Metal-Triton kernel: {ir_func.name}\n" + metal_body

    # Create MLX kernel
    try:
        mlx_kernel = mx.fast.metal_kernel(
            name=ir_func.name,
            input_names=input_names,
            output_names=output_names,
            source=metal_body,
            atomic_outputs=uses_atomics,
        )
    except Exception as e:
        raise CompilationError(
            f"Failed to compile Metal kernel:\n{e}\n\nGenerated source:\n{metal_body}"
        ) from e

    kernel_info = CompiledKernelInfo(
        mlx_kernel=mlx_kernel,
        metal_source=metal_body,
        input_names=input_names,
        output_names=output_names,
        output_indices=output_indices,
        inout_names=inout_names,
        inout_indices=inout_indices,
        uses_atomics=uses_atomics,
    )

    return metal_body, kernel_info


def execute_kernel(
    kernel_info: CompiledKernelInfo,
    args: tuple,
    grid: tuple,
    config: "Config",
) -> Any:
    """Execute a compiled MLX kernel.

    Args:
        kernel_info: Compiled kernel information
        args: Input arguments (tensors and scalars)
        grid: (grid_x, grid_y, grid_z) - ignored in MLX mode, computed from output shape
        config: Configuration (currently unused in MLX mode)

    Returns:
        Output tensor(s)
    """
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError("MLX not available")

    # Determine which indices are pure outputs vs inout
    # - Pure outputs: in output_indices but NOT in inout_indices
    # - Inout: in inout_indices (need to be passed as input too)
    output_indices_set = set(kernel_info.output_indices)
    inout_indices_set = set(kernel_info.inout_indices)

    inputs = []
    output_shapes = []
    output_dtypes = []

    for idx, arg in enumerate(args):
        is_output = idx in output_indices_set
        is_inout = idx in inout_indices_set

        if is_inout:
            # Inout param: add to inputs AND use for output shape/dtype
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                inputs.append(arg)  # Add to inputs (for reading)
                output_shapes.append(arg.shape)
                output_dtypes.append(arg.dtype)
            else:
                raise ValueError(f"Inout argument at index {idx} must be an MLX array")
        elif is_output:
            # Pure output: only use for shape/dtype, don't add to inputs
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                output_shapes.append(arg.shape)
                output_dtypes.append(arg.dtype)
            else:
                raise ValueError(f"Output argument at index {idx} must be an MLX array")
        else:
            # Pure input
            inputs.append(arg)

    # Get threadgroup size from config (default: 8 warps * 32 threads = 256)
    # Apple Silicon SIMD width is 32 threads
    num_warps = getattr(config, 'num_warps', 8) if config else 8
    threads_per_block = num_warps * 32

    # User's grid is in Triton terms (number of threadgroups/blocks)
    # MLX expects grid as total threads. Convert by multiplying by threads_per_block.
    if len(grid) == 1:
        user_grid_threads = (grid[0] * threads_per_block, 1, 1)
    elif len(grid) == 2:
        user_grid_threads = (grid[0] * threads_per_block, grid[1], 1)
    else:
        user_grid_threads = (grid[0] * threads_per_block, grid[1], grid[2])

    if output_shapes:
        # Calculate total elements from first output
        total_elements = 1
        for dim in output_shapes[0]:
            total_elements *= dim
        output_grid = (total_elements, 1, 1)

        # For reductions and similar, user-provided grid may be larger
        # (based on input size). Use whichever is larger.
        user_total = user_grid_threads[0] * user_grid_threads[1] * user_grid_threads[2]
        if user_total > total_elements:
            actual_grid = user_grid_threads
        else:
            actual_grid = output_grid
    else:
        # Use user-provided grid if no outputs detected
        actual_grid = user_grid_threads

    # Threadgroup size = threads per block
    threadgroup_size = (threads_per_block, 1, 1)

    # Execute kernel
    # init_value=0 ensures outputs are zero-initialized (required for atomic ops)
    result = kernel_info.mlx_kernel(
        inputs=inputs,
        grid=actual_grid,
        threadgroup=threadgroup_size,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        init_value=0,
        stream=mx.default_stream(mx.default_device()),
    )

    return result


class CompilationError(Exception):
    """Error during kernel compilation."""
    pass
