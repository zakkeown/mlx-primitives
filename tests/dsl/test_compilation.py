"""Tests for Metal-Triton DSL compilation pipeline.

Tests that the DSL correctly parses Python and generates valid Metal code.
"""

import pytest


def _mlx_available() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core
        return True
    except ImportError:
        return False


class TestASTParser:
    """Test Python AST to IR conversion."""

    def test_parse_simple_kernel(self):
        """Test parsing a simple kernel."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel

        source = '''
@metal_kernel
def vector_add(a_ptr, b_ptr, c_ptr, N):
    pid = mt.program_id(0)
    idx = pid * 256 + mt.thread_id_in_threadgroup()
    if idx < N:
        a = mt.load(a_ptr + idx)
        b = mt.load(b_ptr + idx)
        mt.store(c_ptr + idx, a + b)
'''
        parameters = [
            ("a_ptr", None),
            ("b_ptr", None),
            ("c_ptr", None),
            ("N", None),
        ]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)

        assert ir_func.name == "vector_add"
        assert len(ir_func.parameters) == 4
        assert len(ir_func.body) > 0

    def test_parse_for_loop(self):
        """Test parsing for loops."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel

        source = '''
def sum_kernel(x_ptr, out_ptr, N):
    total = 0.0
    for i in range(N):
        total = total + mt.load(x_ptr + i)
    mt.store(out_ptr, total)
'''
        parameters = [("x_ptr", None), ("out_ptr", None), ("N", None)]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)

        # Should have assignment and for loop
        assert any("IRFor" in str(type(node)) for node in ir_func.body)


class TestMetalCodeGen:
    """Test Metal code generation."""

    def test_generate_simple_kernel(self):
        """Test generating Metal from simple IR."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def vector_add(a_ptr, b_ptr, c_ptr, N):
    pid = mt.program_id(0)
    idx = pid * 256 + mt.thread_id_in_threadgroup()
    if idx < N:
        a = mt.load(a_ptr + idx)
        b = mt.load(b_ptr + idx)
        mt.store(c_ptr + idx, a + b)
'''
        parameters = [
            ("a_ptr", None),
            ("b_ptr", None),
            ("c_ptr", None),
            ("N", None),
        ]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        # Check that generated code contains expected elements
        assert "#include <metal_stdlib>" in metal_source
        assert "kernel void vector_add" in metal_source
        assert "threadgroup_position_in_grid" in metal_source

    def test_generate_with_simd(self):
        """Test generating Metal with SIMD operations."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def simd_sum(x_ptr, out_ptr, N):
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()

    val = mt.load(x_ptr + tid)
    val = val + mt.simd_shuffle_down(val, 16)
    val = val + mt.simd_shuffle_down(val, 8)

    if simd_lane == 0:
        mt.atomic_add(out_ptr, val)
'''
        parameters = [("x_ptr", None), ("out_ptr", None), ("N", None)]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        # Check SIMD operations are present
        assert "simd_shuffle_down" in metal_source
        assert "thread_index_in_simdgroup" in metal_source


class TestEndToEnd:
    """End-to-end tests with actual kernel execution."""

    @pytest.mark.skipif(
        not _mlx_available(),
        reason="MLX not available"
    )
    def test_vector_add_execution(self):
        """Test actual execution of vector_add kernel."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.vector_ops import vector_add

        N = 1024
        a = mx.random.normal((N,))
        b = mx.random.normal((N,))
        c = mx.zeros((N,))  # Template for output shape/dtype

        # Inspect generated Metal
        metal_source = vector_add.inspect_metal()
        print("\nGenerated Metal:")
        print(metal_source)

        # Execute the kernel
        result = vector_add(a, b, c, N=N, grid=(8,))

        # Unpack result
        if isinstance(result, list):
            result = result[0]

        # Verify
        mx.eval(result)
        expected = a + b
        mx.eval(expected)

        assert mx.allclose(result, expected)


class TestBlockPointers:
    """Test block pointer compilation."""

    def test_make_block_ptr(self):
        """Test make_block_ptr compiles correctly."""
        from mlx_primitives.dsl.examples.matmul import matmul_tiled

        metal_source = matmul_tiled.inspect_metal()

        # Check block pointer declarations
        assert "_bp_0_base" in metal_source
        assert "_bp_0_shape_0" in metal_source
        assert "_bp_0_stride_0" in metal_source
        assert "_bp_0_offset_0" in metal_source
        assert "_bp_0_block_0" in metal_source

    def test_advance(self):
        """Test advance updates block pointer offsets."""
        from mlx_primitives.dsl.examples.matmul import matmul_tiled

        metal_source = matmul_tiled.inspect_metal()

        # Check advance generates offset updates
        assert "_bp_0_offset_1 += BLOCK_K" in metal_source
        assert "_bp_1_offset_0 += BLOCK_K" in metal_source

    def test_load_block(self):
        """Test load_block generates cooperative load."""
        from mlx_primitives.dsl.examples.matmul import matmul_tiled

        metal_source = matmul_tiled.inspect_metal()

        # Check block load code is generated
        assert "// Block load:" in metal_source
        assert "_block_load_1" in metal_source

    def test_store_block(self):
        """Test store_block generates cooperative store."""
        from mlx_primitives.dsl.examples.matmul import matmul_tiled

        metal_source = matmul_tiled.inspect_metal()

        # Check block store code is generated
        assert "// Block store:" in metal_source


class TestInspectMetal:
    """Test Metal code inspection."""

    def test_inspect_vector_add(self):
        """Inspect generated Metal for vector_add."""
        from mlx_primitives.dsl.examples.vector_ops import vector_add

        metal_source = vector_add.inspect_metal()

        print("\n" + "=" * 60)
        print("Generated Metal for vector_add:")
        print("=" * 60)
        print(metal_source)
        print("=" * 60)

        # Basic validation
        assert "kernel void vector_add" in metal_source
        assert "#include <metal_stdlib>" in metal_source


if __name__ == "__main__":
    # Run a quick test
    test = TestInspectMetal()
    test.test_inspect_vector_add()

    test2 = TestMetalCodeGen()
    test2.test_generate_simple_kernel()
    print("\nAll tests passed!")
