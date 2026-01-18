"""Metal code generation from IR.

Takes IRProgram/IRFunction and emits Metal Shading Language code.
"""

from __future__ import annotations
from typing import Optional

from mlx_primitives.dsl.compiler.metal_ir import (
    IRProgram, IRFunction, IRNode, IRParameter, IRType,
)


class MetalCodeGenerator:
    """Generate Metal source code from IR."""

    def __init__(self, debug: bool = False):
        self.debug = debug

    def generate(self, ir_func: IRFunction) -> str:
        """Generate complete Metal source from IR function."""
        program = IRProgram(
            includes=[
                "#include <metal_stdlib>",
                "using namespace metal;",
            ],
            functions=[ir_func],
        )

        source = program.emit()

        if self.debug:
            # Add debug comments
            source = self._add_debug_info(source, ir_func)

        return source

    def generate_mlx_body(self, ir_func: IRFunction) -> str:
        """Generate MLX-compatible Metal body (no function signature).

        MLX's metal_kernel auto-generates the function signature,
        so we only need to emit the body.
        """
        lines = []

        # Emit shared memory declarations
        if ir_func.shared_memory_decls:
            lines.append("// Shared memory allocations")
            for decl in ir_func.shared_memory_decls:
                lines.append(decl.emit(indent=0))
            lines.append("")

        # Emit block pointer declarations (if any not emitted inline)
        if ir_func.block_ptr_decls:
            lines.append("// Block pointer declarations")
            for decl in ir_func.block_ptr_decls:
                lines.append(decl.emit(indent=0))
            lines.append("")

        # Emit body
        for node in ir_func.body:
            lines.append(node.emit(indent=0))

        return "\n".join(lines)

    def _add_debug_info(self, source: str, ir_func: IRFunction) -> str:
        """Add debug comments to source."""
        header = f"""// Metal-Triton generated kernel
// Kernel: {ir_func.name}
// Parameters: {len(ir_func.parameters)}
// Debug mode enabled

"""
        return header + source


def generate_metal(ir_func: IRFunction, debug: bool = False) -> str:
    """Generate Metal source code from IR function.

    Args:
        ir_func: IR function to compile
        debug: Include debug comments

    Returns:
        Metal shader source code
    """
    generator = MetalCodeGenerator(debug=debug)
    return generator.generate(ir_func)


def generate_mlx_body(
    ir_func: IRFunction,
    input_names: list[str],
    output_names: list[str],
    inout_renames: Optional[dict[str, str]] = None,
) -> str:
    """Generate MLX-compatible Metal function body.

    MLX's metal_kernel auto-generates the function signature with:
    - device const T* for inputs
    - device T* for outputs
    - Standard thread position variables

    We only need to emit the body code.

    Args:
        ir_func: IR function to compile
        input_names: Names of input parameters
        output_names: Names of output parameters
        inout_renames: Mapping from original param name to output name for inout params

    Returns:
        Metal function body source code
    """
    generator = MLXCodeGenerator(input_names, output_names, inout_renames or {})
    return generator.generate_body(ir_func)


class MLXCodeGenerator:
    """Generate MLX-compatible Metal function body."""

    def __init__(self, input_names: list[str], output_names: list[str], inout_renames: dict[str, str]):
        self.input_names = set(input_names)
        self.output_names = set(output_names)
        self.inout_renames = inout_renames

    def generate_body(self, ir_func: IRFunction) -> str:
        """Generate the function body for MLX metal_kernel."""
        lines = []

        # MLX provides these automatically:
        # - thread_position_in_grid (uint3)
        # - threads_per_grid (uint3)
        # - thread_position_in_threadgroup (uint3)
        # - threads_per_threadgroup (uint3)
        # - threadgroup_position_in_grid (uint3)
        # - threadgroups_per_grid (uint3)
        # - thread_index_in_simdgroup (uint)
        # - simdgroup_index_in_threadgroup (uint)

        # Emit shared memory declarations
        if ir_func.shared_memory_decls:
            for decl in ir_func.shared_memory_decls:
                lines.append(decl.emit(indent=0))
            lines.append("")

        # Emit body statements
        for node in ir_func.body:
            emitted = node.emit(indent=0)
            if emitted:
                lines.append(emitted)

        # Post-process: rename inout store targets
        # e.g., "y_ptr[idx] = value" -> "y_ptr_out[idx] = value"
        # But NOT loads like "float y = y_ptr[idx]" - those should stay as y_ptr
        result = "\n".join(lines)
        for orig_name, out_name in self.inout_renames.items():
            import re
            # Match stores: "name[expr] = value" at the start of a line (after optional whitespace)
            # This pattern: line start, optional whitespace, name, [...], spaces, =
            result = re.sub(
                rf'^(\s*){re.escape(orig_name)}\[([^\]]+)\](\s*=)',
                rf'\1{out_name}[\2]\3',
                result,
                flags=re.MULTILINE
            )
            # Also match stores inside if blocks: "{ name[expr] = value; }"
            result = re.sub(
                rf'\{{\s*{re.escape(orig_name)}\[([^\]]+)\](\s*=)',
                rf'{{ {out_name}[\1]\2',
                result
            )

        return result


def detect_uses_atomics(ir_func: IRFunction) -> bool:
    """Detect if the kernel uses any atomic operations.

    Returns True if the kernel uses atomic_add, atomic_max, etc.
    This determines whether atomic_outputs should be set on the MLX kernel.
    """
    from mlx_primitives.dsl.compiler.metal_ir import IRCall, IRExprStatement

    def check_call(node: IRCall) -> bool:
        return node.func_name in ("atomic_add", "atomic_max", "atomic_min", "atomic_cas")

    def find_atomics(nodes: list[IRNode]) -> bool:
        for node in nodes:
            if isinstance(node, IRCall):
                if check_call(node):
                    return True
            elif isinstance(node, IRExprStatement):
                if isinstance(node.expr, IRCall) and check_call(node.expr):
                    return True
            # Recurse into compound nodes
            if hasattr(node, 'body') and find_atomics(node.body):
                return True
            if hasattr(node, 'orelse') and find_atomics(node.orelse):
                return True
        return False

    return find_atomics(ir_func.body)


def detect_output_params(ir_func: IRFunction) -> list[str]:
    """Detect which parameters are outputs (have stores/atomics written to them).

    Analyzes the IR to find parameters that are store destinations or
    targets of atomic operations.
    """
    from mlx_primitives.dsl.compiler.metal_ir import (
        IRStore, IRSubscriptAssign, IRBlockStore, IRCall, IRExprStatement,
    )

    outputs = set()

    def check_call(node: IRCall):
        """Check if an IRCall is an atomic operation that writes to an output."""
        if node.func_name in ("atomic_add", "atomic_max", "atomic_min", "atomic_cas"):
            if node.args:
                ptr_name = _extract_base_name(node.args[0])
                if ptr_name:
                    outputs.add(ptr_name)

    def find_stores(nodes: list[IRNode]):
        for node in nodes:
            if isinstance(node, IRStore):
                # Extract base pointer name from store ptr
                ptr_name = _extract_base_name(node.ptr)
                if ptr_name:
                    outputs.add(ptr_name)
            elif isinstance(node, IRSubscriptAssign):
                arr_name = _extract_base_name(node.array)
                if arr_name:
                    outputs.add(arr_name)
            elif isinstance(node, IRBlockStore):
                # Block stores go to the block pointer's base
                pass
            elif isinstance(node, IRCall):
                # Detect atomic operations which write to outputs
                check_call(node)
            elif isinstance(node, IRExprStatement):
                # Unwrap expression statements and check inner expression
                if isinstance(node.expr, IRCall):
                    check_call(node.expr)

            # Recurse into compound nodes
            if hasattr(node, 'body'):
                find_stores(node.body)
            if hasattr(node, 'orelse'):
                find_stores(node.orelse)

    find_stores(ir_func.body)

    # Filter to only include actual parameters
    param_names = {p.name for p in ir_func.parameters if p.type_info.ir_type == IRType.POINTER}
    return [p for p in param_names if p in outputs]


def _extract_base_name(node: IRNode) -> Optional[str]:
    """Extract the base variable name from an expression."""
    from mlx_primitives.dsl.compiler.metal_ir import IRVariable, IRBinaryOp

    if isinstance(node, IRVariable):
        return node.name
    elif isinstance(node, IRBinaryOp):
        # For pointer + offset, get the left side
        return _extract_base_name(node.left)
    return None


def detect_input_params(ir_func: IRFunction) -> list[str]:
    """Detect which parameters are read (have loads from them).

    Analyzes the IR to find parameters that are load sources.
    """
    from mlx_primitives.dsl.compiler.metal_ir import IRLoad, IRBlockLoad, IRBlockLoadWithRef

    inputs = set()

    def find_loads(nodes: list[IRNode]):
        for node in nodes:
            if isinstance(node, IRLoad):
                # Extract base pointer name from load ptr
                ptr_name = _extract_base_name(node.ptr)
                if ptr_name:
                    inputs.add(ptr_name)
            elif isinstance(node, (IRBlockLoad, IRBlockLoadWithRef)):
                # Block loads - would need to track block pointer bases
                pass

            # Recurse into compound nodes
            if hasattr(node, 'body'):
                find_loads(node.body)
            if hasattr(node, 'orelse'):
                find_loads(node.orelse)
            if hasattr(node, 'value'):
                find_loads([node.value] if hasattr(node.value, 'emit') else [])

    find_loads(ir_func.body)

    # Filter to only include actual pointer parameters
    param_names = {p.name for p in ir_func.parameters if p.type_info.ir_type == IRType.POINTER}
    return [p for p in param_names if p in inputs]
