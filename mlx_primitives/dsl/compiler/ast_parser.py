"""Parse Python AST into Metal IR.

Transforms @metal_kernel decorated functions into IR suitable
for Metal code generation.
"""

from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Any, Optional

from mlx_primitives.dsl.types import (
    DType, MetalDType, float32, int32, uint32, bool_,
    Pointer, ConstExpr, constexpr,
)
from mlx_primitives.dsl.compiler.metal_ir import (
    IRNode, IRProgram, IRFunction, IRParameter, IRVariable, IRLiteral,
    IRBinaryOp, IRUnaryOp, IRCall, IRLoad, IRStore, IRAssign, IRSubscript, IRSubscriptAssign,
    IRIf, IRFor, IRWhile, IRReturn, IRContinue, IRBreak, IRBarrier, IRComment,
    IRSharedMemoryDecl, IRSharedMemoryRef,
    IRBlockPtrDecl, IRBlockPtrRef, IRBlockPtrAdvance, IRBlockLoad, IRBlockStore,
    IRBlockLoadWithRef, IRBlockLoadAssign, IRExprStatement,
    IRCast, IRLoadVec, IRStoreVec, IRStaticFor,
    TypeInfo, IRType,
)


class CompilationError(Exception):
    """Error during kernel compilation."""
    pass


class ASTParser(ast.NodeVisitor):
    """Parse Python AST into Metal IR."""

    # DSL module attributes that are function calls
    DSL_FUNCTIONS = {
        "program_id", "num_programs",
        "thread_id_in_threadgroup", "threads_per_threadgroup",
        "simd_lane_id", "simd_group_id",
        "load", "store",
        "shared_memory", "load_shared", "store_shared",
        "make_block_ptr", "advance", "load_block", "store_block",
        "zeros", "full", "arange",
        "dot", "trans",
        "maximum", "minimum", "exp", "log", "sqrt", "abs", "where",
        "fma", "tanh", "erf", "rsqrt", "cos", "sin",  # Math ops
        "cast", "reinterpret_cast",  # Type casting
        "sum", "max", "min",
        "threadgroup_barrier", "simd_barrier",
        "simd_shuffle_down", "simd_shuffle_up", "simd_shuffle_xor",
        "simd_broadcast", "simd_sum", "simd_max", "simd_min",
        "atomic_add", "atomic_max", "atomic_min", "atomic_cas",
        "static_for",  # Loop control
        "load_vec2", "load_vec4", "store_vec2", "store_vec4",  # Vector ops
        "vec2", "vec4", "swizzle",
        "debug_print", "static_assert",  # Debug
    }

    # Binary operators
    BINOP_MAP = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Mod: "%",
        ast.FloorDiv: "/",  # Integer division
        ast.BitAnd: "&",
        ast.BitOr: "|",
        ast.BitXor: "^",
        ast.LShift: "<<",
        ast.RShift: ">>",
    }

    COMPARE_MAP = {
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Eq: "==",
        ast.NotEq: "!=",
    }

    BOOLOP_MAP = {
        ast.And: "&&",
        ast.Or: "||",
    }

    UNARYOP_MAP = {
        ast.USub: "-",
        ast.Not: "!",
        ast.Invert: "~",
        ast.UAdd: "+",
    }

    def __init__(self, parameters: list[tuple[str, Any]], constexpr_params: list[str]):
        self.parameters = parameters
        self.constexpr_params = constexpr_params
        self.local_vars: dict[str, TypeInfo] = {}
        self.declared_vars: set[str] = set()
        self.ir_parameters: list[IRParameter] = []
        self.shared_memory_decls: list[IRSharedMemoryDecl] = []
        self._shared_memory_counter = 0

        # Block pointer tracking
        self.block_ptr_decls: list[IRBlockPtrDecl] = []
        self._block_ptr_counter = 0
        self._block_ptr_names: dict[str, str] = {}  # Maps variable name to internal block ptr name

        # Build parameter type info
        self._build_parameter_info()

    def _build_parameter_info(self) -> None:
        """Build type information for parameters."""
        for name, annotation in self.parameters:
            is_constexpr = name in self.constexpr_params
            type_info = self._annotation_to_type_info(annotation, name, is_constexpr)
            self.ir_parameters.append(IRParameter(
                name=name,
                type_info=type_info,
                is_constexpr=is_constexpr,
            ))
            self.local_vars[name] = type_info

    def _annotation_to_type_info(self, annotation: Any, name: str, is_constexpr: bool = False) -> TypeInfo:
        """Convert Python type annotation to TypeInfo."""
        # If marked as constexpr, default to uint32 scalar
        if is_constexpr:
            if isinstance(annotation, ConstExpr) and annotation.dtype:
                return TypeInfo(ir_type=IRType.SCALAR, dtype=annotation.dtype, is_const=True)
            return TypeInfo(ir_type=IRType.SCALAR, dtype=uint32, is_const=True)

        if annotation is inspect.Parameter.empty or annotation is None:
            # Default to pointer for unannotated params
            return TypeInfo(ir_type=IRType.POINTER, dtype=float32)

        if isinstance(annotation, DType):
            return TypeInfo(ir_type=IRType.SCALAR, dtype=annotation)

        if isinstance(annotation, Pointer):
            return TypeInfo(ir_type=IRType.POINTER, dtype=annotation.element_type)

        if isinstance(annotation, ConstExpr) or annotation is constexpr:
            dtype = annotation.dtype if isinstance(annotation, ConstExpr) and annotation.dtype else uint32
            return TypeInfo(ir_type=IRType.SCALAR, dtype=dtype, is_const=True)

        # Handle string annotations like "mt.constexpr"
        if isinstance(annotation, str):
            if "constexpr" in annotation:
                return TypeInfo(ir_type=IRType.SCALAR, dtype=uint32, is_const=True)
            if "Pointer" in annotation:
                return TypeInfo(ir_type=IRType.POINTER, dtype=float32)

        # Default: assume pointer for function params without clear annotation
        return TypeInfo(ir_type=IRType.POINTER, dtype=float32)

    def parse(self, source: str) -> IRFunction:
        """Parse kernel source into IR."""
        # Dedent to handle decorator indentation
        source = textwrap.dedent(source)

        # Parse AST
        tree = ast.parse(source)

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        if func_def is None:
            raise CompilationError("No function definition found")

        # Parse body
        body = []
        for stmt in func_def.body:
            # Skip docstrings
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    continue
            ir_node = self.visit(stmt)
            if ir_node is not None:
                if isinstance(ir_node, list):
                    body.extend(ir_node)
                else:
                    body.append(ir_node)

        # Prepend shared memory declarations to body
        shared_decls = list(self.shared_memory_decls)

        # Prepend block pointer declarations
        block_ptr_decls = list(self.block_ptr_decls)

        return IRFunction(
            name=func_def.name,
            parameters=self.ir_parameters,
            body=body,
            shared_memory_decls=shared_decls,
            block_ptr_decls=block_ptr_decls,
        )

    def visit_Assign(self, node: ast.Assign) -> IRNode:
        """Handle variable assignment."""
        if len(node.targets) != 1:
            raise CompilationError("Multiple assignment targets not supported")

        target = node.targets[0]

        if isinstance(target, ast.Tuple):
            # Tuple unpacking: a, b = expr
            # For now, simple handling
            raise CompilationError("Tuple unpacking not yet supported")

        if isinstance(target, ast.Subscript):
            # Subscript assignment: arr[idx] = value
            return self._handle_subscript_assign(target, node.value)

        if not isinstance(target, ast.Name):
            raise CompilationError(f"Assignment target must be a name, got {type(target)}")

        var_name = target.id
        value = self.visit(node.value)

        # Special case: assigning block pointer reference
        # Track the mapping from variable name to internal block ptr name
        # and emit the block pointer declaration inline
        if isinstance(value, IRBlockPtrRef):
            self._block_ptr_names[var_name] = value.name
            # Find and return the declaration (will be emitted inline)
            bp_decl = next((d for d in self.block_ptr_decls if d.name == value.name), None)
            if bp_decl:
                # Move from pending to emitted list
                self.block_ptr_decls.remove(bp_decl)
                if not hasattr(self, '_emitted_block_ptrs'):
                    self._emitted_block_ptrs = []
                self._emitted_block_ptrs.append(bp_decl)
                return bp_decl
            return IRComment(f"Block pointer alias: {var_name} = {value.name}")

        # Special case: advance returns a block pointer reference
        # This is an in-place update so we just emit the advance
        if isinstance(value, IRBlockPtrAdvance):
            return value

        # Special case: load_block returns data in shared memory
        # Emit the load and create a variable alias to the shared memory
        if isinstance(value, IRBlockLoadWithRef):
            # Track that this variable maps to the shared memory
            self.local_vars[var_name] = TypeInfo(ir_type=IRType.POINTER, dtype=value.dtype)
            self.declared_vars.add(var_name)
            # Create a compound statement: load + alias
            return IRBlockLoadAssign(
                load=value,
                var_name=var_name,
            )

        # Special case: assigning shared memory reference
        # The variable becomes an alias for the shared memory
        if isinstance(value, IRSharedMemoryRef):
            # Create a pointer type for the shared memory
            shared_decl = next(
                (d for d in self.shared_memory_decls if d.name == value.name),
                None
            )
            if shared_decl:
                type_info = TypeInfo(ir_type=IRType.POINTER, dtype=shared_decl.dtype)
                self.local_vars[var_name] = type_info
                self.declared_vars.add(var_name)
                # Return assignment as threadgroup pointer
                return IRAssign(
                    target=IRVariable(var_name, type_info),
                    value=value,
                    is_declaration=True,
                    type_info=type_info,
                )

        # Determine if this is a declaration
        is_declaration = var_name not in self.declared_vars
        if is_declaration:
            self.declared_vars.add(var_name)

        # Infer type from value
        type_info = self._infer_type(value)
        self.local_vars[var_name] = type_info

        return IRAssign(
            target=IRVariable(var_name, type_info),
            value=value,
            is_declaration=is_declaration,
            type_info=type_info,
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> IRNode:
        """Handle augmented assignment (+=, -=, etc.)."""
        target = node.target
        if not isinstance(target, ast.Name):
            raise CompilationError("Augmented assignment target must be a name")

        var_name = target.id
        op = self.BINOP_MAP.get(type(node.op))
        if op is None:
            raise CompilationError(f"Unsupported augmented assignment operator: {type(node.op)}")

        value = self.visit(node.value)
        var = IRVariable(var_name, self.local_vars.get(var_name))

        return IRAssign(
            target=var,
            value=IRBinaryOp(op, var, value),
            is_declaration=False,
            type_info=self.local_vars.get(var_name),
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> IRNode:
        """Handle annotated assignment: x: mt.float32 = ..."""
        if not isinstance(node.target, ast.Name):
            raise CompilationError("Annotated assignment target must be a name")

        var_name = node.target.id
        type_info = self._parse_type_annotation(node.annotation)

        self.declared_vars.add(var_name)
        self.local_vars[var_name] = type_info

        if node.value is None:
            # Declaration only
            return IRAssign(
                target=IRVariable(var_name, type_info),
                value=IRLiteral(0, type_info),
                is_declaration=True,
                type_info=type_info,
            )

        value = self.visit(node.value)
        return IRAssign(
            target=IRVariable(var_name, type_info),
            value=value,
            is_declaration=True,
            type_info=type_info,
        )

    def visit_Expr(self, node: ast.Expr) -> Optional[IRNode]:
        """Handle expression statements."""
        result = self.visit(node.value)

        # If it's a store or barrier, return as-is (they handle their own semicolons)
        if isinstance(result, (IRStore, IRBarrier)):
            return result

        # If it's a call that's a statement (like mt.threadgroup_barrier())
        if isinstance(result, IRCall):
            if result.func_name == "threadgroup_barrier":
                return IRBarrier()
            # Wrap function calls used as statements (e.g., atomic_add)
            return IRExprStatement(result)

        # Wrap other expressions used as statements
        if result is not None:
            return IRExprStatement(result)

        return result

    def visit_If(self, node: ast.If) -> IRNode:
        """Handle if statements."""
        condition = self.visit(node.test)

        body = []
        for stmt in node.body:
            ir_node = self.visit(stmt)
            if ir_node is not None:
                body.append(ir_node)

        orelse = []
        for stmt in node.orelse:
            ir_node = self.visit(stmt)
            if ir_node is not None:
                orelse.append(ir_node)

        return IRIf(condition, body, orelse)

    def visit_For(self, node: ast.For) -> IRNode:
        """Handle for loops (range-based or static_for)."""
        if not isinstance(node.target, ast.Name):
            raise CompilationError("For loop target must be a simple name")

        var_name = node.target.id

        # Parse iterator
        if not isinstance(node.iter, ast.Call):
            raise CompilationError("For loop must use range() or mt.static_for()")

        func = node.iter.func

        # Check for mt.static_for() - fully unrolled at compile time
        is_static_for = False
        if isinstance(func, ast.Attribute):
            if func.attr == "static_for":
                is_static_for = True

        if is_static_for:
            return self._handle_static_for(node, var_name)

        # Handle regular range()
        if isinstance(func, ast.Name) and func.id == "range":
            args = node.iter.args
            if len(args) == 1:
                start = IRLiteral(0, TypeInfo(IRType.SCALAR, uint32))
                end = self.visit(args[0])
                step = IRLiteral(1, TypeInfo(IRType.SCALAR, uint32))
            elif len(args) == 2:
                start = self.visit(args[0])
                end = self.visit(args[1])
                step = IRLiteral(1, TypeInfo(IRType.SCALAR, uint32))
            elif len(args) == 3:
                start = self.visit(args[0])
                end = self.visit(args[1])
                step = self.visit(args[2])
            else:
                raise CompilationError("range() takes 1-3 arguments")
        else:
            raise CompilationError("For loop must use range() or mt.static_for()")

        # Mark variable as declared
        self.declared_vars.add(var_name)
        self.local_vars[var_name] = TypeInfo(IRType.SCALAR, uint32)

        body = []
        for stmt in node.body:
            ir_node = self.visit(stmt)
            if ir_node is not None:
                body.append(ir_node)

        return IRFor(
            var=IRVariable(var_name, TypeInfo(IRType.SCALAR, uint32)),
            start=start,
            end=end,
            step=step,
            body=body,
        )

    def _handle_static_for(self, node: ast.For, var_name: str) -> IRStaticFor:
        """Handle mt.static_for() - fully unrolled loop.

        static_for requires compile-time constant bounds.
        The loop is fully unrolled in the generated Metal code.
        """
        args = node.iter.args

        # Parse bounds - must be compile-time constants
        if len(args) == 2:
            start_val = self._extract_constant(args[0])
            end_val = self._extract_constant(args[1])
            step_val = 1
        elif len(args) == 3:
            start_val = self._extract_constant(args[0])
            end_val = self._extract_constant(args[1])
            step_val = self._extract_constant(args[2])
        else:
            raise CompilationError("static_for() takes 2-3 arguments")

        if start_val is None or end_val is None or step_val is None:
            raise CompilationError("static_for() bounds must be compile-time constants")

        # Mark variable as declared
        self.declared_vars.add(var_name)
        self.local_vars[var_name] = TypeInfo(IRType.SCALAR, uint32)

        # Parse body
        body = []
        for stmt in node.body:
            ir_node = self.visit(stmt)
            if ir_node is not None:
                body.append(ir_node)

        return IRStaticFor(
            var_name=var_name,
            start=start_val,
            end=end_val,
            step=step_val,
            body=body,
        )

    def _extract_constant(self, node: ast.expr) -> Optional[int]:
        """Extract a compile-time constant integer from an AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._extract_constant(node.operand)
            if inner is not None:
                return -inner
        return None

    def visit_While(self, node: ast.While) -> IRNode:
        """Handle while loops."""
        condition = self.visit(node.test)

        body = []
        for stmt in node.body:
            ir_node = self.visit(stmt)
            if ir_node is not None:
                body.append(ir_node)

        return IRWhile(condition, body)

    def visit_Return(self, node: ast.Return) -> IRNode:
        """Handle return statements (early exit)."""
        if node.value is None:
            return IRReturn()
        return IRReturn(self.visit(node.value))

    def visit_Continue(self, node: ast.Continue) -> IRNode:
        """Handle continue statements."""
        return IRContinue()

    def visit_Break(self, node: ast.Break) -> IRNode:
        """Handle break statements."""
        return IRBreak()

    def visit_BinOp(self, node: ast.BinOp) -> IRNode:
        """Handle binary operations."""
        op = self.BINOP_MAP.get(type(node.op))
        if op is None:
            raise CompilationError(f"Unsupported binary operator: {type(node.op)}")

        left = self.visit(node.left)
        right = self.visit(node.right)

        return IRBinaryOp(op, left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> IRNode:
        """Handle unary operations."""
        op = self.UNARYOP_MAP.get(type(node.op))
        if op is None:
            raise CompilationError(f"Unsupported unary operator: {type(node.op)}")

        operand = self.visit(node.operand)
        return IRUnaryOp(op, operand)

    def visit_Compare(self, node: ast.Compare) -> IRNode:
        """Handle comparisons."""
        if len(node.ops) != 1:
            raise CompilationError("Chained comparisons not supported")

        op = self.COMPARE_MAP.get(type(node.ops[0]))
        if op is None:
            raise CompilationError(f"Unsupported comparison: {type(node.ops[0])}")

        left = self.visit(node.left)
        right = self.visit(node.comparators[0])

        return IRBinaryOp(op, left, right)

    def visit_BoolOp(self, node: ast.BoolOp) -> IRNode:
        """Handle boolean operations (and, or)."""
        op = self.BOOLOP_MAP.get(type(node.op))
        if op is None:
            raise CompilationError(f"Unsupported boolean operator: {type(node.op)}")

        # Chain multiple operands
        result = self.visit(node.values[0])
        for value in node.values[1:]:
            result = IRBinaryOp(op, result, self.visit(value))

        return result

    def visit_Name(self, node: ast.Name) -> IRNode:
        """Handle variable references."""
        return IRVariable(node.id, self.local_vars.get(node.id))

    def visit_Constant(self, node: ast.Constant) -> IRNode:
        """Handle literals."""
        value = node.value

        if isinstance(value, bool):
            type_info = TypeInfo(IRType.SCALAR, bool_)
        elif isinstance(value, int):
            type_info = TypeInfo(IRType.SCALAR, int32)
        elif isinstance(value, str):
            # String literals - handle special cases like '-inf'
            if value in ('-inf', 'inf', '-infinity', 'infinity'):
                float_val = float(value)
                type_info = TypeInfo(IRType.SCALAR, float32)
                return IRLiteral(float_val, type_info)
            # Otherwise, store as string for error messages etc.
            type_info = TypeInfo(IRType.SCALAR, int32)
            return IRLiteral(0, type_info)  # Default to 0 for unsupported strings
        elif isinstance(value, float):
            type_info = TypeInfo(IRType.SCALAR, float32)
        else:
            raise CompilationError(f"Unsupported constant type: {type(value)}")

        return IRLiteral(value, type_info)

    def visit_Call(self, node: ast.Call) -> IRNode:
        """Handle function calls."""
        func_name = self._get_call_name(node.func)

        # Handle DSL primitives
        if func_name in self.DSL_FUNCTIONS:
            return self._handle_dsl_call(func_name, node)

        # Handle float(), int() conversions
        if func_name == "float":
            if node.args:
                # Check for compile-time constant like float('-inf')
                arg_node = node.args[0]
                if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                    try:
                        float_val = float(arg_node.value)
                        return IRLiteral(float_val, TypeInfo(IRType.SCALAR, float32))
                    except ValueError:
                        pass
                arg = self.visit(arg_node)
                # If arg is already a literal, convert it
                if isinstance(arg, IRLiteral):
                    return IRLiteral(float(arg.value), TypeInfo(IRType.SCALAR, float32))
                return IRCall("float", [arg])
            return IRLiteral(0.0, TypeInfo(IRType.SCALAR, float32))

        if func_name == "int":
            if node.args:
                arg = self.visit(node.args[0])
                if isinstance(arg, IRLiteral):
                    return IRLiteral(int(arg.value), TypeInfo(IRType.SCALAR, int32))
                return IRCall("int", [arg])
            return IRLiteral(0, TypeInfo(IRType.SCALAR, int32))

        # Generic function call
        args = [self.visit(arg) for arg in node.args]
        return IRCall(func_name, args)

    def _get_call_name(self, node: ast.expr) -> str:
        """Extract function name from call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle mt.load, mt.store, etc.
            if isinstance(node.value, ast.Name):
                if node.value.id in ("mt", "metal_triton"):
                    return node.attr
            # Return full qualified name
            return f"{self._get_call_name(node.value)}.{node.attr}"
        raise CompilationError(f"Cannot get function name from {type(node)}")

    def _handle_dsl_call(self, name: str, node: ast.Call) -> IRNode:
        """Handle DSL-specific function calls."""
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}

        if name == "load":
            mask = kwargs.get("mask")
            other = kwargs.get("other")
            return IRLoad(args[0], mask, other)

        if name == "store":
            mask = kwargs.get("mask")
            return IRStore(args[0], args[1], mask)

        if name == "shared_memory":
            return self._handle_shared_memory(node)

        if name == "load_shared":
            # Cooperative load into shared memory
            # shared, src_ptr, count
            return self._handle_load_shared(args, kwargs)

        if name == "store_shared":
            # Cooperative store from shared memory
            return self._handle_store_shared(args, kwargs)

        if name == "make_block_ptr":
            return self._handle_make_block_ptr(node)

        if name == "advance":
            return self._handle_advance(node)

        if name == "load_block":
            return self._handle_load_block(node)

        if name == "store_block":
            return self._handle_store_block(node)

        if name == "threadgroup_barrier":
            return IRBarrier()

        if name in ("zeros", "full", "arange"):
            # These create local arrays - handled specially
            return IRCall(name, args)

        # Type casting
        if name == "cast":
            if len(args) < 2:
                raise CompilationError("cast() requires value and target dtype")
            target_dtype = self._parse_dtype_arg(node.args[1])
            return IRCast(args[0], target_dtype, reinterpret=False)

        if name == "reinterpret_cast":
            if len(args) < 2:
                raise CompilationError("reinterpret_cast() requires value and target dtype")
            target_dtype = self._parse_dtype_arg(node.args[1])
            return IRCast(args[0], target_dtype, reinterpret=True)

        # Vector operations
        if name == "load_vec2":
            mask = kwargs.get("mask")
            other = kwargs.get("other")
            return IRLoadVec(args[0], width=2, mask=mask, other=other)

        if name == "load_vec4":
            mask = kwargs.get("mask")
            other = kwargs.get("other")
            return IRLoadVec(args[0], width=4, mask=mask, other=other)

        if name == "store_vec2":
            mask = kwargs.get("mask")
            return IRStoreVec(args[0], args[1], width=2, mask=mask)

        if name == "store_vec4":
            mask = kwargs.get("mask")
            return IRStoreVec(args[0], args[1], width=4, mask=mask)

        # Most DSL functions are direct mappings
        return IRCall(name, args)

    def _parse_dtype_arg(self, node: ast.expr) -> DType:
        """Parse a dtype argument (e.g., mt.float16)."""
        if isinstance(node, ast.Attribute):
            dtype_name = node.attr
            if dtype_name == "float32":
                return float32
            elif dtype_name == "float16":
                from mlx_primitives.dsl.types import float16
                return float16
            elif dtype_name == "int32":
                return int32
            elif dtype_name == "uint32":
                return uint32
            elif dtype_name == "bool_":
                return bool_
        raise CompilationError(f"Unknown dtype: {ast.dump(node)}")

    def _handle_shared_memory(self, node: ast.Call) -> IRSharedMemoryRef:
        """Handle shared_memory allocation."""
        # Parse shape from positional args
        shape_parts = []
        for arg in node.args:
            if isinstance(arg, ast.Constant):
                shape_parts.append(str(arg.value))
            elif isinstance(arg, ast.Name):
                shape_parts.append(arg.id)
            else:
                shape_parts.append(self.visit(arg).emit())

        # Parse dtype from kwargs
        dtype = float32
        padding = 4
        for kw in node.keywords:
            if kw.arg == "dtype":
                # Handle mt.float32 etc.
                if isinstance(kw.value, ast.Attribute):
                    dtype_name = kw.value.attr
                    if dtype_name == "float32":
                        dtype = float32
                    elif dtype_name == "float16":
                        from mlx_primitives.dsl.types import float16
                        dtype = float16
                    elif dtype_name == "int32":
                        dtype = int32
                    elif dtype_name == "uint32":
                        dtype = uint32
            elif kw.arg == "padding":
                if isinstance(kw.value, ast.Constant):
                    padding = kw.value.value

        # Generate unique name
        shared_name = f"_shared_{self._shared_memory_counter}"
        self._shared_memory_counter += 1

        # Compute size expression
        if len(shape_parts) == 1:
            size_expr = shape_parts[0]
        else:
            size_expr = " * ".join(f"({s})" for s in shape_parts)

        # Create declaration
        decl = IRSharedMemoryDecl(
            name=shared_name,
            dtype=dtype,
            size_expr=size_expr,
            padding=padding,
        )
        self.shared_memory_decls.append(decl)

        # Track in local vars
        self.local_vars[shared_name] = TypeInfo(
            ir_type=IRType.POINTER,
            dtype=dtype,
        )

        # Return reference
        return IRSharedMemoryRef(shared_name)

    def _handle_load_shared(self, args: list[IRNode], kwargs: dict) -> IRNode:
        """Generate cooperative load into shared memory."""
        # This generates a loop where all threads participate
        # For now, return a placeholder that will be expanded
        return IRCall("__load_shared", args)

    def _handle_store_shared(self, args: list[IRNode], kwargs: dict) -> IRNode:
        """Generate cooperative store from shared memory."""
        return IRCall("__store_shared", args)

    def _handle_subscript_assign(self, target: ast.Subscript, value_node: ast.expr) -> IRNode:
        """Handle subscript assignment: arr[idx] = value."""
        array = self.visit(target.value)
        index = self.visit(target.slice)
        value = self.visit(value_node)

        # Generate: arr[idx] = value;
        return IRSubscriptAssign(array, index, value)

    def _handle_make_block_ptr(self, node: ast.Call) -> IRBlockPtrRef:
        """Handle make_block_ptr allocation."""
        # Parse arguments: base, shape, strides, offsets, block_shape, order=None
        # All except base are tuples

        # Parse keyword arguments first
        base_ptr = None
        shape = None
        strides = None
        offsets = None
        block_shape = None
        dtype = float32

        for kw in node.keywords:
            if kw.arg == "base":
                base_ptr = self.visit(kw.value)
            elif kw.arg == "shape":
                shape = self._parse_tuple_arg(kw.value)
            elif kw.arg == "strides":
                strides = self._parse_tuple_arg(kw.value)
            elif kw.arg == "offsets":
                offsets = self._parse_tuple_arg(kw.value)
            elif kw.arg == "block_shape":
                block_shape = self._parse_tuple_arg(kw.value)

        # Also check positional args
        if len(node.args) > 0 and base_ptr is None:
            base_ptr = self.visit(node.args[0])
        if len(node.args) > 1 and shape is None:
            shape = self._parse_tuple_arg(node.args[1])
        if len(node.args) > 2 and strides is None:
            strides = self._parse_tuple_arg(node.args[2])
        if len(node.args) > 3 and offsets is None:
            offsets = self._parse_tuple_arg(node.args[3])
        if len(node.args) > 4 and block_shape is None:
            block_shape = self._parse_tuple_arg(node.args[4])

        if base_ptr is None:
            raise CompilationError("make_block_ptr requires base pointer argument")
        if shape is None or strides is None or offsets is None or block_shape is None:
            raise CompilationError("make_block_ptr requires shape, strides, offsets, and block_shape")

        # Generate unique name
        bp_name = f"_bp_{self._block_ptr_counter}"
        self._block_ptr_counter += 1

        # Create declaration
        decl = IRBlockPtrDecl(
            name=bp_name,
            base_ptr=base_ptr,
            shape=tuple(shape),
            strides=tuple(strides),
            offsets=tuple(offsets),
            block_shape=tuple(block_shape),
            dtype=dtype,
        )
        self.block_ptr_decls.append(decl)

        # Return reference
        return IRBlockPtrRef(bp_name)

    def _parse_tuple_arg(self, node: ast.expr) -> list[IRNode]:
        """Parse a tuple argument into list of IR nodes."""
        if isinstance(node, ast.Tuple):
            return [self.visit(elt) for elt in node.elts]
        elif isinstance(node, ast.List):
            return [self.visit(elt) for elt in node.elts]
        else:
            # Single element
            return [self.visit(node)]

    def _handle_advance(self, node: ast.Call) -> IRBlockPtrAdvance:
        """Handle advance(block_ptr, offsets)."""
        if len(node.args) < 2:
            raise CompilationError("advance requires block_ptr and offsets")

        # First arg should be a block pointer reference
        bp_arg = self.visit(node.args[0])
        if isinstance(bp_arg, IRBlockPtrRef):
            bp_name = bp_arg.name
        elif isinstance(bp_arg, IRVariable):
            # Look up the internal block ptr name
            bp_name = self._block_ptr_names.get(bp_arg.name, bp_arg.name)
        else:
            raise CompilationError(f"advance first arg must be block pointer, got {type(bp_arg)}")

        # Second arg is tuple of offsets - parse from AST directly
        offset_nodes = self._parse_tuple_arg(node.args[1])

        return IRBlockPtrAdvance(bp_name, tuple(offset_nodes))

    def _handle_load_block(self, node: ast.Call) -> IRNode:
        """Handle load_block(block_ptr, boundary_check=..., padding_option=...).

        Returns a compound node that both loads data and provides a reference to the result.
        """
        if not node.args:
            raise CompilationError("load_block requires block_ptr argument")

        bp_arg = self.visit(node.args[0])
        if isinstance(bp_arg, IRBlockPtrRef):
            bp_name = bp_arg.name
        elif isinstance(bp_arg, IRVariable):
            bp_name = self._block_ptr_names.get(bp_arg.name, bp_arg.name)
        else:
            raise CompilationError(f"load_block first arg must be block pointer, got {type(bp_arg)}")

        # Parse kwargs
        boundary_check = None
        padding_value = 0.0

        for kw in node.keywords:
            if kw.arg == "boundary_check":
                if isinstance(kw.value, ast.Tuple):
                    boundary_check = tuple(
                        elt.value if isinstance(elt, ast.Constant) else 0
                        for elt in kw.value.elts
                    )
            elif kw.arg == "padding_option":
                if isinstance(kw.value, ast.Constant):
                    if kw.value.value == "nan":
                        padding_value = float('nan')

        # Find the block pointer declaration to get block_shape
        # We look in the original list AND any already-emitted decls
        all_decls = list(self.block_ptr_decls) + list(getattr(self, '_emitted_block_ptrs', []))
        bp_decl = next((d for d in all_decls if d.name == bp_name), None)

        # Allocate shared memory for the block
        target_shared = f"_block_load_{self._shared_memory_counter}"
        self._shared_memory_counter += 1

        if bp_decl:
            block_size_parts = [b.emit() for b in bp_decl.block_shape]
            size_expr = " * ".join(f"({s})" for s in block_size_parts)
            dtype = bp_decl.dtype
        else:
            # Fallback - this shouldn't happen if usage is correct
            size_expr = "256"  # Default size
            dtype = float32

        shared_decl = IRSharedMemoryDecl(
            name=target_shared,
            dtype=dtype,
            size_expr=size_expr,
            padding=4,
        )
        self.shared_memory_decls.append(shared_decl)

        # Track that this shared memory is the result of this load
        self.local_vars[target_shared] = TypeInfo(ir_type=IRType.POINTER, dtype=dtype)

        # Return a wrapper that combines the load and reference
        return IRBlockLoadWithRef(
            block_ptr_name=bp_name,
            target_shared=target_shared,
            boundary_check=boundary_check,
            padding_value=padding_value,
            dtype=dtype,
        )

    def _handle_store_block(self, node: ast.Call) -> IRBlockStore:
        """Handle store_block(block_ptr, value, boundary_check=...)."""
        if len(node.args) < 2:
            raise CompilationError("store_block requires block_ptr and value arguments")

        bp_arg = self.visit(node.args[0])
        if isinstance(bp_arg, IRBlockPtrRef):
            bp_name = bp_arg.name
        elif isinstance(bp_arg, IRVariable):
            bp_name = self._block_ptr_names.get(bp_arg.name, bp_arg.name)
        else:
            raise CompilationError(f"store_block first arg must be block pointer")

        value_arg = self.visit(node.args[1])

        # Get source shared memory name
        if isinstance(value_arg, IRVariable):
            source_shared = value_arg.name
        elif isinstance(value_arg, IRSharedMemoryRef):
            source_shared = value_arg.name
        else:
            raise CompilationError("store_block value must be a variable or shared memory reference")

        # Parse kwargs
        boundary_check = None
        for kw in node.keywords:
            if kw.arg == "boundary_check":
                if isinstance(kw.value, ast.Tuple):
                    boundary_check = tuple(
                        elt.value if isinstance(elt, ast.Constant) else 0
                        for elt in kw.value.elts
                    )

        return IRBlockStore(
            block_ptr_name=bp_name,
            source_shared=source_shared,
            boundary_check=boundary_check,
        )

    def visit_Subscript(self, node: ast.Subscript) -> IRNode:
        """Handle subscript operations (array[idx])."""
        value = self.visit(node.value)
        index = self.visit(node.slice)

        # Check if this is shared memory or a local array (use array indexing)
        # vs a device pointer (use pointer arithmetic for load/store)
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            # Check if variable is shared memory or a local array
            if var_name in self.local_vars:
                type_info = self.local_vars[var_name]
                # Use array indexing for shared memory pointers
                if type_info.ir_type == IRType.POINTER:
                    # Check if it's a shared memory alias
                    is_shared = any(
                        var_name == assign_target
                        for assign_target in self.declared_vars
                        if var_name in self.local_vars
                    )
                    # For shared memory and local arrays, use array indexing
                    return IRSubscript(value, index, type_info)

        # Default: treat as pointer arithmetic for device memory loads
        return IRBinaryOp("+", value, index)

    def visit_Attribute(self, node: ast.Attribute) -> IRNode:
        """Handle attribute access."""
        # Handle mt.float32, mt.constexpr, etc.
        if isinstance(node.value, ast.Name) and node.value.id in ("mt", "metal_triton"):
            return IRVariable(node.attr)

        value = self.visit(node.value)
        return IRCall(f"__getattr__", [value, IRLiteral(node.attr, TypeInfo(IRType.SCALAR))])

    def visit_IfExp(self, node: ast.IfExp) -> IRNode:
        """Handle ternary expressions: x if cond else y."""
        condition = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)

        return IRCall("where", [condition, body, orelse])

    # Functions that return uint (thread indexing)
    UINT_RETURNING_FUNCTIONS = {
        "program_id", "num_programs",
        "thread_id_in_threadgroup", "threads_per_threadgroup",
        "simd_lane_id", "simd_group_id",
    }

    def _infer_type(self, node: IRNode) -> TypeInfo:
        """Infer type from IR node."""
        if isinstance(node, IRLiteral):
            return node.type_info
        elif isinstance(node, IRVariable):
            return node.type_info or TypeInfo(IRType.SCALAR, float32)
        elif isinstance(node, IRCall):
            # Thread indexing functions return uint
            if node.func_name in self.UINT_RETURNING_FUNCTIONS:
                return TypeInfo(IRType.SCALAR, uint32)
            # Most math functions return float
            return TypeInfo(IRType.SCALAR, float32)
        elif isinstance(node, IRBinaryOp):
            # Return type of left operand
            return self._infer_type(node.left)
        elif isinstance(node, IRLoad):
            return TypeInfo(IRType.SCALAR, float32)
        else:
            return TypeInfo(IRType.SCALAR, float32)

    def _parse_type_annotation(self, node: ast.expr) -> TypeInfo:
        """Parse type annotation AST node."""
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("mt", "metal_triton"):
                name = node.attr
                if name == "float32":
                    return TypeInfo(IRType.SCALAR, float32)
                elif name == "float16":
                    from mlx_primitives.dsl.types import float16
                    return TypeInfo(IRType.SCALAR, float16)
                elif name == "int32":
                    return TypeInfo(IRType.SCALAR, int32)
                elif name == "uint32":
                    return TypeInfo(IRType.SCALAR, uint32)
                elif name == "bool_":
                    return TypeInfo(IRType.SCALAR, bool_)

        return TypeInfo(IRType.SCALAR, float32)


def parse_kernel(source: str, parameters: list[tuple[str, Any]], constexpr_params: list[str]) -> IRFunction:
    """Parse kernel source into IR.

    Args:
        source: Python source code of kernel function
        parameters: List of (name, annotation) tuples
        constexpr_params: Names of constexpr parameters

    Returns:
        IRFunction representing the kernel
    """
    parser = ASTParser(parameters, constexpr_params)
    return parser.parse(source)
