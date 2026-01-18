"""Tests for new DSL primitives.

Tests compilation of new primitives (fma, tanh, erf, cos, sin, rsqrt, cast,
static_for, vector ops, etc.) to Metal code.
"""

import pytest


def _mlx_available() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core
        return True
    except ImportError:
        return False


class TestMathPrimitives:
    """Test new math primitives compile correctly."""

    def test_fma(self):
        """Test fused multiply-add compiles to Metal fma()."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def fma_kernel(a_ptr, b_ptr, c_ptr, out_ptr, N):
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    idx = pid * 256 + tid
    if idx < N:
        a = mt.load(a_ptr + idx)
        b = mt.load(b_ptr + idx)
        c = mt.load(c_ptr + idx)
        result = mt.fma(a, b, c)
        mt.store(out_ptr + idx, result)
'''
        parameters = [
            ("a_ptr", None), ("b_ptr", None), ("c_ptr", None),
            ("out_ptr", None), ("N", None)
        ]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        assert "fma(" in metal_source

    def test_tanh(self):
        """Test tanh compiles to Metal tanh()."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def tanh_kernel(x_ptr, out_ptr, N):
    idx = mt.program_id(0) * 256 + mt.thread_id_in_threadgroup()
    if idx < N:
        x = mt.load(x_ptr + idx)
        mt.store(out_ptr + idx, mt.tanh(x))
'''
        parameters = [("x_ptr", None), ("out_ptr", None), ("N", None)]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        assert "tanh(" in metal_source

    def test_erf(self):
        """Test erf compiles to Metal erf()."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def erf_kernel(x_ptr, out_ptr, N):
    idx = mt.program_id(0) * 256 + mt.thread_id_in_threadgroup()
    if idx < N:
        x = mt.load(x_ptr + idx)
        mt.store(out_ptr + idx, mt.erf(x))
'''
        parameters = [("x_ptr", None), ("out_ptr", None), ("N", None)]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        assert "erf(" in metal_source

    def test_cos_sin(self):
        """Test cos/sin compile to Metal cos()/sin()."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def trig_kernel(x_ptr, cos_out_ptr, sin_out_ptr, N):
    idx = mt.program_id(0) * 256 + mt.thread_id_in_threadgroup()
    if idx < N:
        x = mt.load(x_ptr + idx)
        mt.store(cos_out_ptr + idx, mt.cos(x))
        mt.store(sin_out_ptr + idx, mt.sin(x))
'''
        parameters = [
            ("x_ptr", None), ("cos_out_ptr", None),
            ("sin_out_ptr", None), ("N", None)
        ]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        assert "cos(" in metal_source
        assert "sin(" in metal_source

    def test_rsqrt(self):
        """Test rsqrt compiles to Metal rsqrt()."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def rsqrt_kernel(x_ptr, out_ptr, N):
    idx = mt.program_id(0) * 256 + mt.thread_id_in_threadgroup()
    if idx < N:
        x = mt.load(x_ptr + idx)
        mt.store(out_ptr + idx, mt.rsqrt(x))
'''
        parameters = [("x_ptr", None), ("out_ptr", None), ("N", None)]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        assert "rsqrt(" in metal_source


class TestTypeCasting:
    """Test type casting primitives."""

    def test_cast_function_exists(self):
        """Test cast function is exported and callable."""
        from mlx_primitives.dsl import cast

        # Should be a callable marker function
        assert callable(cast)

    def test_cast_compiles(self):
        """Test cast compiles to Metal type cast."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def cast_kernel(x_ptr, out_ptr, N):
    idx = mt.program_id(0) * 256 + mt.thread_id_in_threadgroup()
    if idx < N:
        x = mt.load(x_ptr + idx)
        casted = mt.cast(x, mt.float16)
        mt.store(out_ptr + idx, casted)
'''
        parameters = [("x_ptr", None), ("out_ptr", None), ("N", None)]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        # Should contain Metal type cast: half(x)
        assert "half(" in metal_source


class TestLoopControl:
    """Test loop control primitives."""

    def test_static_for_function_exists(self):
        """Test static_for function is exported."""
        from mlx_primitives.dsl import static_for, unroll

        assert callable(static_for)
        # unroll is a class
        assert unroll is not None

    def test_regular_range_loop(self):
        """Test regular range() loops work."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel

        source = '''
def loop_kernel(x_ptr, out_ptr, N):
    acc = 0.0
    for i in range(4):
        acc = acc + mt.load(x_ptr + i)
    mt.store(out_ptr, acc)
'''
        parameters = [("x_ptr", None), ("out_ptr", None), ("N", None)]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        assert ir_func.name == "loop_kernel"

    def test_static_for_parsing(self):
        """Test static_for is recognized during parsing."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def unrolled_kernel(x_ptr, out_ptr, N):
    acc = 0.0
    for i in mt.static_for(0, 4):
        acc = acc + mt.load(x_ptr + i)
    mt.store(out_ptr, acc)
'''
        parameters = [("x_ptr", None), ("out_ptr", None), ("N", None)]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        # Should contain unrolled iterations
        assert "Unrolled iteration" in metal_source
        # Should have 4 iterations (0, 1, 2, 3)
        assert "i = 0" in metal_source
        assert "i = 1" in metal_source
        assert "i = 2" in metal_source
        assert "i = 3" in metal_source


class TestVectorOperations:
    """Test vector operation primitives."""

    def test_vec4_construction(self):
        """Test vec4 construction compiles."""
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal

        source = '''
def vec4_kernel(out_ptr, N):
    idx = mt.program_id(0) * 256 + mt.thread_id_in_threadgroup()
    if idx < N:
        v = mt.vec4(1.0, 2.0, 3.0, 4.0)
        mt.store_vec4(out_ptr + idx * 4, v)
'''
        parameters = [("out_ptr", None), ("N", None)]
        constexpr_params = ["N"]

        ir_func = parse_kernel(source, parameters, constexpr_params)
        metal_source = generate_metal(ir_func)

        assert "float4(" in metal_source or "vec4" in metal_source


class TestExampleKernels:
    """Test example kernel compilation."""

    def test_silu_compiles(self):
        """Test SiLU kernel compiles."""
        from mlx_primitives.dsl.examples.activations import silu

        metal_source = silu.inspect_metal()

        # Check essential elements
        assert "kernel void silu" in metal_source
        assert "exp(" in metal_source

    def test_gelu_tanh_compiles(self):
        """Test GELU (tanh) kernel compiles."""
        from mlx_primitives.dsl.examples.activations import gelu_tanh

        metal_source = gelu_tanh.inspect_metal()

        assert "kernel void gelu_tanh" in metal_source
        assert "tanh(" in metal_source

    def test_gelu_exact_compiles(self):
        """Test GELU (exact) kernel compiles."""
        from mlx_primitives.dsl.examples.activations import gelu_exact

        metal_source = gelu_exact.inspect_metal()

        assert "kernel void gelu_exact" in metal_source
        assert "erf(" in metal_source

    def test_layer_norm_compiles(self):
        """Test LayerNorm kernel compiles."""
        from mlx_primitives.dsl.examples.normalization import layer_norm

        metal_source = layer_norm.inspect_metal()

        assert "kernel void layer_norm" in metal_source
        # Should have SIMD shuffle for reduction
        assert "simd_shuffle_down" in metal_source

    def test_rms_norm_compiles(self):
        """Test RMSNorm kernel compiles."""
        from mlx_primitives.dsl.examples.normalization import rms_norm

        metal_source = rms_norm.inspect_metal()

        assert "kernel void rms_norm" in metal_source
        assert "sqrt(" in metal_source

    def test_rope_forward_compiles(self):
        """Test RoPE forward kernel compiles."""
        from mlx_primitives.dsl.examples.rope import rope_forward

        metal_source = rope_forward.inspect_metal()

        assert "kernel void rope_forward" in metal_source

    def test_rope_inline_compiles(self):
        """Test RoPE inline kernel compiles."""
        from mlx_primitives.dsl.examples.rope import rope_inline

        metal_source = rope_inline.inspect_metal()

        assert "kernel void rope_inline" in metal_source
        assert "cos(" in metal_source
        assert "sin(" in metal_source

    def test_fused_add_layer_norm_compiles(self):
        """Test fused add + LayerNorm kernel compiles."""
        from mlx_primitives.dsl.examples.normalization import fused_add_layer_norm

        metal_source = fused_add_layer_norm.inspect_metal()

        assert "kernel void fused_add_layer_norm" in metal_source


class TestExecutionCorrectness:
    """End-to-end correctness tests."""

    @pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
    def test_silu_correctness(self):
        """Test SiLU produces correct results."""
        import mlx.core as mx
        import numpy as np
        from mlx_primitives.dsl.examples.activations import silu

        N = 1024
        x = mx.random.normal((N,))
        y = mx.zeros((N,))

        result = silu(x, y, N=N, grid=((N + 255) // 256,))
        if isinstance(result, list):
            result = result[0]

        mx.eval(result)

        # Reference: x * sigmoid(x)
        x_np = np.array(x)
        expected = x_np * (1.0 / (1.0 + np.exp(-x_np)))

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
    def test_rope_precompute_cache(self):
        """Test RoPE cache precomputation."""
        import numpy as np
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        max_seq_len = 128
        head_dim = 64
        base = 10000.0

        cos_cache, sin_cache = precompute_rope_cache(max_seq_len, head_dim, base)

        assert cos_cache.shape == (max_seq_len, head_dim // 2)
        assert sin_cache.shape == (max_seq_len, head_dim // 2)

        # Verify values are in valid range
        assert np.all(cos_cache >= -1.0) and np.all(cos_cache <= 1.0)
        assert np.all(sin_cache >= -1.0) and np.all(sin_cache <= 1.0)


class TestExports:
    """Test that all primitives are properly exported."""

    def test_main_exports(self):
        """Test main __init__.py exports new primitives."""
        from mlx_primitives.dsl import (
            # Math
            fma, tanh, erf, cos, sin, rsqrt,
            # Type casting
            cast, reinterpret_cast,
            # Vectors
            vec2, vec4, load_vec2, load_vec4, store_vec2, store_vec4, swizzle,
            # Loop control
            static_for, unroll,
            # Debug
            static_assert, debug_print,
        )

        # All should be callable (marker functions)
        assert callable(fma)
        assert callable(tanh)
        assert callable(erf)
        assert callable(cos)
        assert callable(sin)
        assert callable(rsqrt)
        assert callable(cast)
        assert callable(reinterpret_cast)
        assert callable(vec2)
        assert callable(vec4)
        assert callable(load_vec2)
        assert callable(load_vec4)
        assert callable(store_vec2)
        assert callable(store_vec4)
        assert callable(swizzle)
        assert callable(static_for)
        assert callable(static_assert)
        assert callable(debug_print)

    def test_example_exports(self):
        """Test example kernels are exported."""
        from mlx_primitives.dsl.examples import (
            silu, gelu_tanh, gelu_exact, quick_gelu,
            fused_silu_mul, fused_gelu_mul,
            layer_norm, rms_norm,
            fused_add_layer_norm, fused_add_rms_norm,
            rope_forward, rope_inline, rope_qk_fused, rope_neox,
        )

        # All should be callable compiled kernels
        assert callable(silu)
        assert callable(gelu_tanh)
        assert callable(layer_norm)
        assert callable(rope_forward)


if __name__ == "__main__":
    # Quick smoke test
    test = TestMathPrimitives()
    test.test_fma()
    test.test_tanh()
    test.test_cos_sin()
    print("Math primitives tests passed!")

    test2 = TestExampleKernels()
    test2.test_silu_compiles()
    test2.test_layer_norm_compiles()
    test2.test_rope_forward_compiles()
    print("Example kernel tests passed!")

    test3 = TestExports()
    test3.test_main_exports()
    test3.test_example_exports()
    print("Export tests passed!")

    print("\nAll primitive tests passed!")
