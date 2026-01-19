"""Intermediate representation for Metal-Triton compiler.

Provides a typed IR between Python AST and Metal code generation.
Each node knows how to emit Metal code.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Any

from mlx_primitives.dsl.types import DType, MetalDType, float32, int32, uint32


class IRType(Enum):
    """IR type categories."""
    SCALAR = "scalar"
    POINTER = "pointer"
    ARRAY = "array"
    VOID = "void"


@dataclass
class TypeInfo:
    """Type information for IR nodes."""
    ir_type: IRType
    dtype: DType = field(default_factory=lambda: float32)
    shape: Optional[tuple[Union[int, str], ...]] = None
    is_const: bool = False

    @property
    def metal_type(self) -> str:
        """Get Metal type string."""
        if self.ir_type == IRType.VOID:
            return "void"
        elif self.ir_type == IRType.POINTER:
            const = "const " if self.is_const else ""
            return f"device {const}{self.dtype.metal_name}*"
        elif self.ir_type == IRType.ARRAY:
            # Arrays in registers
            if self.shape and len(self.shape) == 1:
                return f"{self.dtype.metal_name}"
            return self.dtype.metal_name
        else:
            return self.dtype.metal_name


@dataclass
class IRNode(ABC):
    """Base class for all IR nodes."""

    @abstractmethod
    def emit(self, indent: int = 0) -> str:
        """Emit Metal code for this node."""
        pass

    def _indent(self, level: int) -> str:
        return "    " * level


@dataclass
class IRLiteral(IRNode):
    """Literal value (int, float, bool)."""
    value: Any
    type_info: TypeInfo

    def emit(self, indent: int = 0) -> str:
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        elif isinstance(self.value, float):
            if self.value == float('inf'):
                return "INFINITY"
            elif self.value == float('-inf'):
                return "-INFINITY"
            return f"{self.value}f"
        return str(self.value)


@dataclass
class IRVariable(IRNode):
    """Variable reference."""
    name: str
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        return self.name


@dataclass
class IRBinaryOp(IRNode):
    """Binary operation (a + b, a * b, etc.)."""
    op: str  # +, -, *, /, %, <, >, <=, >=, ==, !=, &&, ||
    left: IRNode
    right: IRNode
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        left = self.left.emit()
        right = self.right.emit()

        # Handle pointer arithmetic
        if self.op == "+" and hasattr(self.left, "type_info"):
            if self.left.type_info and self.left.type_info.ir_type == IRType.POINTER:
                return f"({left} + {right})"

        return f"({left} {self.op} {right})"


@dataclass
class IRUnaryOp(IRNode):
    """Unary operation (-x, !x)."""
    op: str  # -, !, ~
    operand: IRNode
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        return f"({self.op}{self.operand.emit()})"


@dataclass
class IRSubscript(IRNode):
    """Array subscript operation (arr[idx])."""
    array: IRNode
    index: IRNode
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        return f"{self.array.emit()}[{self.index.emit()}]"


@dataclass
class IRSubscriptAssign(IRNode):
    """Subscript assignment (arr[idx] = value)."""
    array: IRNode
    index: IRNode
    value: IRNode

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        return f"{ind}{self.array.emit()}[{self.index.emit()}] = {self.value.emit()};"


@dataclass
class IRExprStatement(IRNode):
    """Expression used as a statement (adds semicolon)."""
    expr: IRNode

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        return f"{ind}{self.expr.emit()};"


@dataclass
class IRCall(IRNode):
    """Function call."""
    func_name: str
    args: list[IRNode]
    type_info: Optional[TypeInfo] = None

    # Mapping from DSL functions to Metal (takes emitted string args)
    METAL_FUNCTIONS = {
        # Math functions
        "exp": lambda args: f"exp({args[0]})",
        "log": lambda args: f"log({args[0]})",
        "sqrt": lambda args: f"sqrt({args[0]})",
        "rsqrt": lambda args: f"rsqrt({args[0]})",
        "abs": lambda args: f"abs({args[0]})",
        "maximum": lambda args: f"max({args[0]}, {args[1]})",
        "minimum": lambda args: f"min({args[0]}, {args[1]})",
        "where": lambda args: f"({args[0]} ? {args[1]} : {args[2]})",
        "fma": lambda args: f"fma({args[0]}, {args[1]}, {args[2]})",
        "tanh": lambda args: f"tanh({args[0]})",
        "erf": lambda args: f"erf({args[0]})",
        "cos": lambda args: f"cos({args[0]})",
        "sin": lambda args: f"sin({args[0]})",

        # SIMD operations
        "simd_shuffle_down": lambda args: f"simd_shuffle_down({args[0]}, {args[1]})",
        "simd_shuffle_up": lambda args: f"simd_shuffle_up({args[0]}, {args[1]})",
        "simd_shuffle_xor": lambda args: f"simd_shuffle_xor({args[0]}, {args[1]})",
        "simd_broadcast": lambda args: f"simd_broadcast({args[0]}, {args[1]})",
        "simd_sum": lambda args: f"simd_sum({args[0]})",
        "simd_max": lambda args: f"simd_max({args[0]})",
        "simd_min": lambda args: f"simd_min({args[0]})",

        # Vector constructors
        "vec2": lambda args: f"float2({args[0]}, {args[1]})",
        "vec4": lambda args: f"float4({args[0]}, {args[1]}, {args[2]}, {args[3]})",

        # Atomics
        "atomic_add": lambda args: f"atomic_fetch_add_explicit((device atomic_float*){args[0]}, {args[1]}, memory_order_relaxed)",
    }

    def emit(self, indent: int = 0) -> str:
        args_str = [arg.emit() for arg in self.args]

        # Handle thread indexing functions specially
        # Maps Triton-style program/thread concepts to Metal's threading model:
        # - program_id = threadgroup index (like CUDA block idx)
        # - thread_id_in_threadgroup = thread within block (like CUDA thread idx)
        if self.func_name == "program_id":
            axis = self._get_axis_from_arg(0)
            # Threadgroup index = block/program index
            return f"threadgroup_position_in_grid.{'xyz'[axis]}"
        elif self.func_name == "num_programs":
            axis = self._get_axis_from_arg(0)
            # Number of threadgroups in grid
            return f"threadgroups_per_grid.{'xyz'[axis]}"
        elif self.func_name == "thread_id_in_threadgroup":
            # Thread index within this threadgroup
            return "thread_position_in_threadgroup.x"
        elif self.func_name == "threads_per_threadgroup":
            # Number of threads per threadgroup (block size)
            return "threads_per_threadgroup.x"
        elif self.func_name == "simd_lane_id":
            return "thread_index_in_simdgroup"
        elif self.func_name == "simd_group_id":
            return "simdgroup_index_in_threadgroup"
        elif self.func_name in self.METAL_FUNCTIONS:
            # Other special DSL functions
            return self.METAL_FUNCTIONS[self.func_name](args_str)
        else:
            # Regular function call
            return f"{self.func_name}({', '.join(args_str)})"

    def _get_axis_from_arg(self, idx: int) -> int:
        """Extract integer axis value from argument."""
        if idx < len(self.args):
            arg = self.args[idx]
            if isinstance(arg, IRLiteral):
                return int(arg.value)
            # Try to parse from emitted string
            try:
                return int(arg.emit())
            except ValueError:
                pass
        return 0  # Default to x axis


@dataclass
class IRLoad(IRNode):
    """Memory load operation."""
    ptr: IRNode
    mask: Optional[IRNode] = None
    other: Optional[IRNode] = None
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        # For MLX compatibility, use array indexing: base[offset]
        # Parse the ptr expression to extract base and offset
        base, offset = self._extract_base_offset(self.ptr)

        if self.mask:
            mask_expr = self.mask.emit()
            other_expr = self.other.emit() if self.other else "0"
            return f"({mask_expr} ? {base}[{offset}] : {other_expr})"
        return f"{base}[{offset}]"

    def _extract_base_offset(self, node: IRNode) -> tuple[str, str]:
        """Extract base pointer and offset from a pointer expression."""
        if isinstance(node, IRBinaryOp) and node.op == "+":
            # ptr + offset
            base = node.left.emit()
            offset = node.right.emit()
            return base, offset
        else:
            # Just a pointer, offset is 0
            return node.emit(), "0"


@dataclass
class IRStore(IRNode):
    """Memory store operation."""
    ptr: IRNode
    value: IRNode
    mask: Optional[IRNode] = None

    def emit(self, indent: int = 0) -> str:
        # For MLX compatibility, use array indexing: base[offset] = value
        base, offset = self._extract_base_offset(self.ptr)
        val_expr = self.value.emit()
        ind = self._indent(indent)
        if self.mask:
            mask_expr = self.mask.emit()
            return f"{ind}if ({mask_expr}) {{ {base}[{offset}] = {val_expr}; }}"
        return f"{ind}{base}[{offset}] = {val_expr};"

    def _extract_base_offset(self, node: IRNode) -> tuple[str, str]:
        """Extract base pointer and offset from a pointer expression."""
        if isinstance(node, IRBinaryOp) and node.op == "+":
            base = node.left.emit()
            offset = node.right.emit()
            return base, offset
        else:
            return node.emit(), "0"


@dataclass
class IRAssign(IRNode):
    """Variable assignment."""
    target: IRVariable
    value: IRNode
    is_declaration: bool = False
    type_info: Optional[TypeInfo] = None
    is_shared_memory: bool = False  # True if assigning threadgroup memory

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        val_expr = self.value.emit()
        if self.is_declaration and self.type_info:
            # Check if this is a shared memory pointer
            if self.is_shared_memory or isinstance(self.value, IRSharedMemoryRef):
                type_str = f"threadgroup {self.type_info.dtype.metal_name}*"
            else:
                type_str = self.type_info.metal_type
            return f"{ind}{type_str} {self.target.name} = {val_expr};"
        return f"{ind}{self.target.name} = {val_expr};"


@dataclass
class IRIf(IRNode):
    """If statement."""
    condition: IRNode
    body: list[IRNode]
    orelse: list[IRNode] = field(default_factory=list)

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        lines = [f"{ind}if ({self.condition.emit()}) {{"]

        for node in self.body:
            lines.append(node.emit(indent + 1))

        if self.orelse:
            lines.append(f"{ind}}} else {{")
            for node in self.orelse:
                lines.append(node.emit(indent + 1))

        lines.append(f"{ind}}}")
        return "\n".join(lines)


@dataclass
class IRFor(IRNode):
    """For loop."""
    var: IRVariable
    start: IRNode
    end: IRNode
    step: IRNode
    body: list[IRNode]

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        var_name = self.var.name
        start_expr = self.start.emit()
        end_expr = self.end.emit()
        step_expr = self.step.emit()

        lines = [f"{ind}for (uint {var_name} = {start_expr}; {var_name} < {end_expr}; {var_name} += {step_expr}) {{"]

        for node in self.body:
            lines.append(node.emit(indent + 1))

        lines.append(f"{ind}}}")
        return "\n".join(lines)


@dataclass
class IRWhile(IRNode):
    """While loop."""
    condition: IRNode
    body: list[IRNode]

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        lines = [f"{ind}while ({self.condition.emit()}) {{"]

        for node in self.body:
            lines.append(node.emit(indent + 1))

        lines.append(f"{ind}}}")
        return "\n".join(lines)


@dataclass
class IRReturn(IRNode):
    """Return statement (early exit)."""
    value: Optional[IRNode] = None

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        if self.value:
            return f"{ind}return {self.value.emit()};"
        return f"{ind}return;"


@dataclass
class IRContinue(IRNode):
    """Continue statement (skip to next iteration)."""

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        return f"{ind}continue;"


@dataclass
class IRBreak(IRNode):
    """Break statement (exit loop)."""

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        return f"{ind}break;"


@dataclass
class IRBarrier(IRNode):
    """Threadgroup barrier."""
    mem_flags: str = "mem_threadgroup"

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        return f"{ind}threadgroup_barrier(mem_flags::{self.mem_flags});"


@dataclass
class IRSharedMemoryDecl(IRNode):
    """Shared (threadgroup) memory declaration."""
    name: str
    dtype: DType
    size_expr: str  # Size expression (may include constexpr names)
    padding: int = 4

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        # Add padding to avoid bank conflicts
        if self.padding > 0:
            size = f"({self.size_expr}) + {self.padding}"
        else:
            size = self.size_expr
        return f"{ind}threadgroup {self.dtype.metal_name} {self.name}[{size}];"


@dataclass
class IRSharedMemoryRef(IRNode):
    """Reference to shared memory (for indexing)."""
    name: str
    index: Optional[IRNode] = None
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        if self.index:
            return f"{self.name}[{self.index.emit()}]"
        return self.name


@dataclass
class IRArrayDecl(IRNode):
    """Local (register) array declaration.

    Generates Metal array declarations for stack-allocated arrays.
    Used for accumulators and working arrays that live in thread-local registers.

    Example output:
        float acc[128] = {};           // Zero-initialized
        float acc[head_dim] = {};      // Constexpr size
        float m_i[N];                  // With init loop for special values
        for (uint _i = 0; _i < N; _i++) { m_i[_i] = -INFINITY; }
    """
    name: str
    dtype: DType
    size_expr: str  # Size expression: "128" or "head_dim" (constexpr)
    init_value: Optional[float] = None  # None = zero-init with {}, else fill value
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        type_name = self.dtype.metal_name

        if self.init_value is None:
            # Zero initialization using aggregate initialization
            return f"{ind}{type_name} {self.name}[{self.size_expr}] = {{}};"
        elif self.init_value == float('-inf'):
            # Need explicit loop for -INFINITY
            return (f"{ind}{type_name} {self.name}[{self.size_expr}];\n"
                    f"{ind}for (uint _i = 0; _i < {self.size_expr}; _i++) {{ {self.name}[_i] = -INFINITY; }}")
        elif self.init_value == float('inf'):
            return (f"{ind}{type_name} {self.name}[{self.size_expr}];\n"
                    f"{ind}for (uint _i = 0; _i < {self.size_expr}; _i++) {{ {self.name}[_i] = INFINITY; }}")
        else:
            # Explicit fill value - need loop
            return (f"{ind}{type_name} {self.name}[{self.size_expr}];\n"
                    f"{ind}for (uint _i = 0; _i < {self.size_expr}; _i++) {{ {self.name}[_i] = {self.init_value}f; }}")


@dataclass
class IRArrayRef(IRNode):
    """Reference to a local array variable.

    When used directly, emits just the array name.
    Array indexing is handled by IRSubscript.
    """
    name: str
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        return self.name


@dataclass
class IRComment(IRNode):
    """Code comment for debugging."""
    text: str

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        return f"{ind}// {self.text}"


# =============================================================================
# Block Pointer Support (Triton-style tiled access)
# =============================================================================

@dataclass
class IRBlockPtrDecl(IRNode):
    """Block pointer declaration with metadata."""
    name: str
    base_ptr: IRNode
    shape: tuple[IRNode, ...]      # (M, N) shape of underlying tensor
    strides: tuple[IRNode, ...]    # (stride_m, stride_n)
    offsets: tuple[IRNode, ...]    # (offset_m, offset_n) starting offsets
    block_shape: tuple[IRNode, ...]  # (BLOCK_M, BLOCK_N)
    dtype: DType = field(default_factory=lambda: float32)

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        lines = []

        # Emit struct-like block pointer as individual variables
        lines.append(f"{ind}// Block pointer: {self.name}")
        lines.append(f"{ind}device {self.dtype.metal_name}* {self.name}_base = {self.base_ptr.emit()};")

        # Shape
        for i, s in enumerate(self.shape):
            lines.append(f"{ind}uint {self.name}_shape_{i} = {s.emit()};")

        # Strides
        for i, s in enumerate(self.strides):
            lines.append(f"{ind}uint {self.name}_stride_{i} = {s.emit()};")

        # Offsets (mutable - can be advanced)
        for i, o in enumerate(self.offsets):
            lines.append(f"{ind}uint {self.name}_offset_{i} = {o.emit()};")

        # Block shape
        for i, b in enumerate(self.block_shape):
            lines.append(f"{ind}uint {self.name}_block_{i} = {b.emit()};")

        return "\n".join(lines)


@dataclass
class IRBlockPtrRef(IRNode):
    """Reference to an existing block pointer."""
    name: str
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        return self.name


@dataclass
class IRBlockPtrAdvance(IRNode):
    """Advance a block pointer by offsets."""
    block_ptr_name: str
    offsets: tuple[IRNode, ...]

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        lines = []
        for i, offset in enumerate(self.offsets):
            offset_val = offset.emit()
            if offset_val != "0":
                lines.append(f"{ind}{self.block_ptr_name}_offset_{i} += {offset_val};")
        return "\n".join(lines) if lines else f"{ind}// advance: no change"


@dataclass
class IRBlockLoad(IRNode):
    """Load a 2D block using block pointer into shared memory."""
    block_ptr_name: str
    target_shared: str  # Name of shared memory to load into
    boundary_check: Optional[tuple[int, ...]] = None
    padding_value: float = 0.0
    dtype: DType = field(default_factory=lambda: float32)

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        ind2 = self._indent(indent + 1)
        ind3 = self._indent(indent + 2)

        bp = self.block_ptr_name
        lines = []

        lines.append(f"{ind}// Block load: {bp} -> {self.target_shared}")
        lines.append(f"{ind}{{")

        # Cooperative load: each thread loads multiple elements
        lines.append(f"{ind2}uint _tid = thread_position_in_threadgroup.x;")
        lines.append(f"{ind2}uint _block_size = {bp}_block_0 * {bp}_block_1;")
        lines.append(f"{ind2}uint _threads = threads_per_threadgroup.x;")

        lines.append(f"{ind2}for (uint _i = _tid; _i < _block_size; _i += _threads) {{")

        # Compute 2D indices within block
        lines.append(f"{ind3}uint _row = _i / {bp}_block_1;")
        lines.append(f"{ind3}uint _col = _i % {bp}_block_1;")

        # Compute global indices
        lines.append(f"{ind3}uint _global_row = {bp}_offset_0 + _row;")
        lines.append(f"{ind3}uint _global_col = {bp}_offset_1 + _col;")

        # Bounds check
        check_dims = self.boundary_check if self.boundary_check else (0, 1)
        conditions = []
        if 0 in check_dims:
            conditions.append(f"_global_row < {bp}_shape_0")
        if 1 in check_dims:
            conditions.append(f"_global_col < {bp}_shape_1")

        if conditions:
            cond_str = " && ".join(conditions)
            lines.append(f"{ind3}if ({cond_str}) {{")
            ind4 = self._indent(indent + 3)
            lines.append(f"{ind4}uint _src_idx = _global_row * {bp}_stride_0 + _global_col * {bp}_stride_1;")
            lines.append(f"{ind4}{self.target_shared}[_i] = {bp}_base[_src_idx];")
            lines.append(f"{ind3}}} else {{")
            lines.append(f"{ind4}{self.target_shared}[_i] = {self.padding_value}f;")
            lines.append(f"{ind3}}}")
        else:
            lines.append(f"{ind3}uint _src_idx = _global_row * {bp}_stride_0 + _global_col * {bp}_stride_1;")
            lines.append(f"{ind3}{self.target_shared}[_i] = {bp}_base[_src_idx];")

        lines.append(f"{ind2}}}")
        lines.append(f"{ind}}}")

        return "\n".join(lines)


@dataclass
class IRBlockStore(IRNode):
    """Store a 2D block from shared memory using block pointer."""
    block_ptr_name: str
    source_shared: str  # Name of shared memory to store from
    boundary_check: Optional[tuple[int, ...]] = None
    dtype: DType = field(default_factory=lambda: float32)

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        ind2 = self._indent(indent + 1)
        ind3 = self._indent(indent + 2)

        bp = self.block_ptr_name
        lines = []

        lines.append(f"{ind}// Block store: {self.source_shared} -> {bp}")
        lines.append(f"{ind}{{")

        # Cooperative store
        lines.append(f"{ind2}uint _tid = thread_position_in_threadgroup.x;")
        lines.append(f"{ind2}uint _block_size = {bp}_block_0 * {bp}_block_1;")
        lines.append(f"{ind2}uint _threads = threads_per_threadgroup.x;")

        lines.append(f"{ind2}for (uint _i = _tid; _i < _block_size; _i += _threads) {{")

        # Compute 2D indices
        lines.append(f"{ind3}uint _row = _i / {bp}_block_1;")
        lines.append(f"{ind3}uint _col = _i % {bp}_block_1;")
        lines.append(f"{ind3}uint _global_row = {bp}_offset_0 + _row;")
        lines.append(f"{ind3}uint _global_col = {bp}_offset_1 + _col;")

        # Bounds check
        check_dims = self.boundary_check if self.boundary_check else (0, 1)
        conditions = []
        if 0 in check_dims:
            conditions.append(f"_global_row < {bp}_shape_0")
        if 1 in check_dims:
            conditions.append(f"_global_col < {bp}_shape_1")

        if conditions:
            cond_str = " && ".join(conditions)
            lines.append(f"{ind3}if ({cond_str}) {{")
            ind4 = self._indent(indent + 3)
            lines.append(f"{ind4}uint _dst_idx = _global_row * {bp}_stride_0 + _global_col * {bp}_stride_1;")
            lines.append(f"{ind4}{bp}_base[_dst_idx] = {self.source_shared}[_i];")
            lines.append(f"{ind3}}}")
        else:
            lines.append(f"{ind3}uint _dst_idx = _global_row * {bp}_stride_0 + _global_col * {bp}_stride_1;")
            lines.append(f"{ind3}{bp}_base[_dst_idx] = {self.source_shared}[_i];")

        lines.append(f"{ind2}}}")
        lines.append(f"{ind}}}")

        return "\n".join(lines)


@dataclass
class IRBlockLoadWithRef(IRNode):
    """Block load that also provides a reference to the loaded data.

    When used as a statement, emits the load code.
    When used as a value, returns a reference to the target shared memory.
    """
    block_ptr_name: str
    target_shared: str
    boundary_check: Optional[tuple[int, ...]] = None
    padding_value: float = 0.0
    dtype: DType = field(default_factory=lambda: float32)

    def emit(self, indent: int = 0) -> str:
        """Emit the block load code as a statement."""
        load = IRBlockLoad(
            block_ptr_name=self.block_ptr_name,
            target_shared=self.target_shared,
            boundary_check=self.boundary_check,
            padding_value=self.padding_value,
            dtype=self.dtype,
        )
        return load.emit(indent)

    def emit_as_value(self) -> str:
        """Emit as a value reference (pointer to shared memory)."""
        return self.target_shared


@dataclass
class IRBlockLoadAssign(IRNode):
    """Block load followed by variable alias assignment.

    Emits:
    1. The block load code
    2. A pointer alias: threadgroup float* var_name = target_shared;
    """
    load: IRBlockLoadWithRef
    var_name: str

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        lines = []
        # Emit the load
        lines.append(self.load.emit(indent))
        # Emit the alias
        lines.append(f"{ind}threadgroup {self.load.dtype.metal_name}* {self.var_name} = {self.load.target_shared};")
        return "\n".join(lines)


@dataclass
class IRParameter:
    """Kernel parameter."""
    name: str
    type_info: TypeInfo
    buffer_index: Optional[int] = None
    is_constexpr: bool = False


@dataclass
class IRFunction(IRNode):
    """Complete kernel function."""
    name: str
    parameters: list[IRParameter]
    body: list[IRNode]
    shared_memory: list[tuple[str, TypeInfo, Union[int, str]]] = field(default_factory=list)
    shared_memory_decls: list["IRSharedMemoryDecl"] = field(default_factory=list)
    block_ptr_decls: list["IRBlockPtrDecl"] = field(default_factory=list)

    def emit(self, indent: int = 0) -> str:
        lines = []

        # Function signature
        params = []
        buffer_idx = 0

        for param in self.parameters:
            if param.is_constexpr:
                # Constexpr becomes template parameter or constant
                params.append(f"constant {param.type_info.metal_type}& {param.name} [[buffer({buffer_idx})]]")
            elif param.type_info.ir_type == IRType.POINTER:
                params.append(f"{param.type_info.metal_type} {param.name} [[buffer({buffer_idx})]]")
            else:
                params.append(f"constant {param.type_info.metal_type}& {param.name} [[buffer({buffer_idx})]]")
            buffer_idx += 1

        # Add thread position arguments
        params.append("uint3 _tg_pos [[threadgroup_position_in_grid]]")
        params.append("uint3 _t_pos [[thread_position_in_threadgroup]]")
        params.append("uint3 _tgs [[threadgroups_per_grid]]")
        params.append("uint3 _tpt [[threads_per_threadgroup]]")
        params.append("uint _simd_lane [[thread_index_in_simdgroup]]")
        params.append("uint _simd_group [[simdgroup_index_in_threadgroup]]")

        # Shared memory parameters (old style - passed as function params)
        for i, (name, type_info, size) in enumerate(self.shared_memory):
            params.append(f"threadgroup {type_info.dtype.metal_name}* {name} [[threadgroup({i})]]")

        param_str = ",\n    ".join(params)
        lines.append(f"kernel void {self.name}(")
        lines.append(f"    {param_str}")
        lines.append(") {")

        # Convenience aliases for thread positions
        lines.append("    // Thread position aliases")
        lines.append("    uint3 threadgroup_position_in_grid = _tg_pos;")
        lines.append("    uint3 thread_position_in_threadgroup = _t_pos;")
        lines.append("    uint3 threadgroups_per_grid = _tgs;")
        lines.append("    uint3 threads_per_threadgroup = _tpt;")
        lines.append("    uint thread_index_in_simdgroup = _simd_lane;")
        lines.append("    uint simdgroup_index_in_threadgroup = _simd_group;")
        lines.append("")

        # Emit shared memory declarations (new style - declared in function body)
        if self.shared_memory_decls:
            lines.append("    // Shared memory allocations")
            for decl in self.shared_memory_decls:
                lines.append(decl.emit(indent=1))
            lines.append("")

        # Note: Block pointer declarations are emitted inline where they're defined
        # (not at the top of the function) to ensure variables they reference exist

        # Body
        for node in self.body:
            lines.append(node.emit(indent=1))

        lines.append("}")

        return "\n".join(lines)


@dataclass
class IRCast(IRNode):
    """Type cast operation."""
    value: IRNode
    target_dtype: DType
    reinterpret: bool = False  # as_type vs regular cast
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        val = self.value.emit()
        target = self.target_dtype.metal_name

        if self.reinterpret:
            return f"as_type<{target}>({val})"
        else:
            # Metal uses function-style casts
            return f"{target}({val})"


@dataclass
class IRLoadVec(IRNode):
    """Vector load operation (load_vec2, load_vec4)."""
    ptr: IRNode
    width: int  # 2 or 4
    mask: Optional[IRNode] = None
    other: Optional[IRNode] = None
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        vec_type = f"float{self.width}"
        ptr_expr = self.ptr.emit()

        if self.mask:
            mask_expr = self.mask.emit()
            other_expr = self.other.emit() if self.other else f"{vec_type}(0)"
            return f"({mask_expr} ? *((device {vec_type}*)({ptr_expr})) : {other_expr})"

        return f"*((device {vec_type}*)({ptr_expr}))"


@dataclass
class IRStoreVec(IRNode):
    """Vector store operation."""
    ptr: IRNode
    value: IRNode
    width: int  # 2 or 4
    mask: Optional[IRNode] = None

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        vec_type = f"float{self.width}"
        ptr_expr = self.ptr.emit()
        val_expr = self.value.emit()

        if self.mask:
            mask_expr = self.mask.emit()
            return f"{ind}if ({mask_expr}) {{ *((device {vec_type}*)({ptr_expr})) = {val_expr}; }}"

        return f"{ind}*((device {vec_type}*)({ptr_expr})) = {val_expr};"


@dataclass
class IRSwizzle(IRNode):
    """Vector swizzle operation."""
    vec: IRNode
    pattern: str  # e.g., "xy", "wzyx"
    type_info: Optional[TypeInfo] = None

    def emit(self, indent: int = 0) -> str:
        return f"{self.vec.emit()}.{self.pattern}"


@dataclass
class IRStaticFor(IRNode):
    """Statically unrolled for loop."""
    var_name: str
    start: int
    end: int
    step: int
    body: list[IRNode]

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        lines = []

        # Generate unrolled iterations
        for i in range(self.start, self.end, self.step):
            lines.append(f"{ind}// Unrolled iteration {self.var_name} = {i}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}    const uint {self.var_name} = {i};")
            for stmt in self.body:
                lines.append(stmt.emit(indent + 1))
            lines.append(f"{ind}}}")

        return "\n".join(lines)


@dataclass
class IRPragmaUnrollFor(IRNode):
    """For loop with pragma unroll hint."""
    var: IRVariable
    start: IRNode
    end: IRNode
    step: IRNode
    body: list[IRNode]
    unroll_factor: int = 0  # 0 = full unroll

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        var_name = self.var.name
        start_expr = self.start.emit()
        end_expr = self.end.emit()
        step_expr = self.step.emit()

        lines = []

        # Emit pragma
        if self.unroll_factor == 0:
            lines.append(f"{ind}#pragma unroll")
        else:
            lines.append(f"{ind}#pragma unroll({self.unroll_factor})")

        lines.append(f"{ind}for (uint {var_name} = {start_expr}; {var_name} < {end_expr}; {var_name} += {step_expr}) {{")

        for node in self.body:
            lines.append(node.emit(indent + 1))

        lines.append(f"{ind}}}")
        return "\n".join(lines)


@dataclass
class IRStaticAssert(IRNode):
    """Compile-time assertion."""
    condition: IRNode
    message: str = ""

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        cond = self.condition.emit()
        return f'{ind}static_assert({cond}, "{self.message}");'


@dataclass
class IRDebugPrint(IRNode):
    """Debug print statement (Metal printf)."""
    fmt: str
    args: list[IRNode]

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)

        if self.args:
            args_str = ", ".join(arg.emit() for arg in self.args)
            return f'{ind}printf("{self.fmt}\\n", {args_str});'
        else:
            return f'{ind}printf("{self.fmt}\\n");'


@dataclass
class IRCooperativeLoad(IRNode):
    """Cooperative load from device memory into shared memory.

    All threads in the threadgroup participate in loading data.
    Generates a loop where each thread loads multiple elements.
    """
    shared_name: str  # Target shared memory name
    src_ptr: IRNode  # Source pointer expression
    count: IRNode  # Number of elements to load
    stride: Optional[IRNode] = None  # Element stride (default: 1)
    dtype: DType = field(default_factory=lambda: float32)

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        ind2 = self._indent(indent + 1)
        ind3 = self._indent(indent + 2)

        lines = []
        lines.append(f"{ind}// Cooperative load into {self.shared_name}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind2}uint _tid = thread_position_in_threadgroup.x;")
        lines.append(f"{ind2}uint _count = {self.count.emit()};")
        lines.append(f"{ind2}uint _threads = threads_per_threadgroup.x;")

        stride_expr = self.stride.emit() if self.stride else "1"

        lines.append(f"{ind2}for (uint _i = _tid; _i < _count; _i += _threads) {{")
        lines.append(f"{ind3}{self.shared_name}[_i] = ({self.src_ptr.emit()})[_i * {stride_expr}];")
        lines.append(f"{ind2}}}")
        lines.append(f"{ind}}}")

        return "\n".join(lines)


@dataclass
class IRCooperativeStore(IRNode):
    """Cooperative store from shared memory to device memory.

    All threads in the threadgroup participate in storing data.
    """
    dst_ptr: IRNode  # Destination pointer expression
    shared_name: str  # Source shared memory name
    count: IRNode  # Number of elements to store
    stride: Optional[IRNode] = None  # Element stride (default: 1)
    dtype: DType = field(default_factory=lambda: float32)

    def emit(self, indent: int = 0) -> str:
        ind = self._indent(indent)
        ind2 = self._indent(indent + 1)
        ind3 = self._indent(indent + 2)

        lines = []
        lines.append(f"{ind}// Cooperative store from {self.shared_name}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind2}uint _tid = thread_position_in_threadgroup.x;")
        lines.append(f"{ind2}uint _count = {self.count.emit()};")
        lines.append(f"{ind2}uint _threads = threads_per_threadgroup.x;")

        stride_expr = self.stride.emit() if self.stride else "1"

        lines.append(f"{ind2}for (uint _i = _tid; _i < _count; _i += _threads) {{")
        lines.append(f"{ind3}({self.dst_ptr.emit()})[_i * {stride_expr}] = {self.shared_name}[_i];")
        lines.append(f"{ind2}}}")
        lines.append(f"{ind}}}")

        return "\n".join(lines)


@dataclass
class IRProgram(IRNode):
    """Complete Metal program (header + kernel)."""
    includes: list[str] = field(default_factory=lambda: [
        "#include <metal_stdlib>",
        "using namespace metal;",
    ])
    functions: list[IRFunction] = field(default_factory=list)

    def emit(self, indent: int = 0) -> str:
        lines = []

        # Headers
        for inc in self.includes:
            lines.append(inc)
        lines.append("")

        # Functions
        for func in self.functions:
            lines.append(func.emit())
            lines.append("")

        return "\n".join(lines)
