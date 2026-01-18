"""Metal-Triton compiler: Python AST → Metal code.

Pipeline:
1. ast_parser: Parse Python function → IR nodes
2. metal_ir: Intermediate representation
3. codegen: IR → Metal source code
"""

from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
from mlx_primitives.dsl.compiler.metal_ir import (
    IRNode,
    IRProgram,
    IRFunction,
    IRVariable,
    IRBinaryOp,
    IRUnaryOp,
    IRCall,
    IRLoad,
    IRStore,
    IRIf,
    IRFor,
    IRReturn,
)
from mlx_primitives.dsl.compiler.codegen import MetalCodeGenerator
from mlx_primitives.dsl.compiler.compile import compile_kernel, execute_kernel

__all__ = [
    "parse_kernel",
    "compile_kernel",
    "execute_kernel",
    "MetalCodeGenerator",
    # IR types
    "IRNode",
    "IRProgram",
    "IRFunction",
    "IRVariable",
    "IRBinaryOp",
    "IRUnaryOp",
    "IRCall",
    "IRLoad",
    "IRStore",
    "IRIf",
    "IRFor",
    "IRReturn",
]
