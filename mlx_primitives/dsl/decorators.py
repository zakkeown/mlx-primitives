"""Decorators for Metal-Triton DSL.

@metal_kernel: Mark a function as a Metal kernel to be compiled
@autotune: Enable auto-tuning over configuration space
Config: Configuration for auto-tuning
"""

from __future__ import annotations
import functools
import inspect
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, TypeVar, Sequence

F = TypeVar("F", bound=Callable)


@dataclass
class Config:
    """Configuration for auto-tuning.

    Defines a specific set of compile-time constants to try.

    Example:
        Config(BLOCK_M=32, BLOCK_N=32, num_warps=4)
    """
    num_warps: int = 4
    num_stages: int = 1
    # Arbitrary constexpr values
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, num_warps: int = 4, num_stages: int = 1, **kwargs: Any):
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.kwargs = kwargs

    def __repr__(self) -> str:
        parts = [f"num_warps={self.num_warps}"]
        if self.num_stages != 1:
            parts.append(f"num_stages={self.num_stages}")
        for k, v in self.kwargs.items():
            parts.append(f"{k}={v}")
        return f"Config({', '.join(parts)})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if key == "num_warps":
            return self.num_warps
        if key == "num_stages":
            return self.num_stages
        return self.kwargs.get(key, default)


@dataclass
class KernelDefinition:
    """Parsed kernel definition with metadata."""
    name: str
    func: Callable
    source_code: str
    parameters: list[tuple[str, Any]]  # (name, annotation)
    constexpr_params: list[str]
    configs: Optional[list[Config]] = None
    autotune_keys: Optional[list[str]] = None
    autotune_warmup: int = 3
    autotune_rep: int = 10


class CompiledKernel:
    """Wrapper for a compiled Metal kernel.

    Handles:
    - Lazy compilation on first call
    - Configuration selection (if auto-tuned)
    - Grid/threadgroup calculation
    - MLX kernel invocation
    """

    def __init__(
        self,
        kernel_def: KernelDefinition,
        debug: bool = False,
    ):
        self.kernel_def = kernel_def
        self.debug = debug
        self._compiled_cache: dict[tuple, Any] = {}
        self._metal_source_cache: dict[tuple, str] = {}
        self._best_config: Optional[Config] = None
        # Auto-tuning state - use lock for thread-safe cache access
        self._autotune_results: dict[tuple, Config] = {}  # (name, tuning_key) -> Config
        self._persistent_cache: bool = True
        self._cache_lock = threading.Lock()  # Protects _autotune_results and _compiled_cache

    def __call__(self, *args, grid: tuple, **kwargs) -> Any:
        """Execute the kernel.

        Args:
            *args: Tensor/scalar arguments
            grid: (grid_x, grid_y, grid_z) - number of threadgroups
            **kwargs: Constexpr values and other options

        Returns:
            Output tensor(s)
        """
        from mlx_primitives.dsl.compiler import compile_kernel, execute_kernel

        # Determine configuration (thread-safe via _cache_lock)
        config = self._select_config(args, kwargs, grid)

        # Get or compile kernel for this configuration (thread-safe)
        cache_key = self._make_cache_key(config)
        with self._cache_lock:
            if cache_key not in self._compiled_cache:
                metal_source, kernel_info = compile_kernel(
                    self.kernel_def,
                    config,
                    debug=self.debug,
                )
                self._compiled_cache[cache_key] = kernel_info
                self._metal_source_cache[cache_key] = metal_source

            kernel_info = self._compiled_cache[cache_key]

        # Build full args list, adding constexpr values in order
        # Args are positional (tensors), kwargs contain constexpr values and grid
        full_args = list(args)
        for name in kernel_info.input_names:
            if name in kwargs:
                # Constexpr parameter passed as kwarg
                full_args.append(kwargs[name])

        # Execute
        return execute_kernel(kernel_info, tuple(full_args), grid, config)

    def _select_config(self, args: tuple, kwargs: dict, grid: tuple) -> Config:
        """Select configuration for this invocation.

        Order of precedence:
        1. Already-tuned best config for this tuning key
        2. Run auto-tuning if configs are provided and key matches
        3. Default config from kwargs

        Thread-safe: uses _cache_lock to protect cache access.
        """
        # If we have a globally set best config, use it
        if self._best_config is not None:
            return self._best_config

        # If no configs to tune, create default from kwargs
        if not self.kernel_def.configs:
            return Config(**{k: v for k, v in kwargs.items()
                            if k in self.kernel_def.constexpr_params})

        # Compute tuning key from kwargs
        tuning_key = self._compute_tuning_key(kwargs)
        cache_key = (self.kernel_def.name, tuning_key)

        # Thread-safe cache access
        with self._cache_lock:
            # Check local cache first
            if cache_key in self._autotune_results:
                return self._autotune_results[cache_key]

            # Check global cache (including disk persistence)
            from mlx_primitives.dsl.autotuner import get_autotune_cache
            cached = get_autotune_cache().get(self.kernel_def.name, tuning_key)
            if cached is not None:
                self._autotune_results[cache_key] = cached
                return cached

            # Run auto-tuning (still inside lock to prevent duplicate tuning)
            best_config = self._run_autotune(args, kwargs, grid, tuning_key)
            self._autotune_results[cache_key] = best_config
            return best_config

    def _compute_tuning_key(self, kwargs: dict) -> tuple:
        """Compute the tuning key from current invocation kwargs.

        Uses the key parameter names specified in @autotune decorator.
        """
        if not self.kernel_def.autotune_keys:
            return ()

        key_values = []
        for key_name in self.kernel_def.autotune_keys:
            if key_name in kwargs:
                key_values.append(kwargs[key_name])
            else:
                key_values.append(None)
        return tuple(key_values)

    def _run_autotune(
        self,
        args: tuple,
        kwargs: dict,
        grid: tuple,
        tuning_key: tuple,
    ) -> Config:
        """Execute auto-tuning over all configs.

        Note: Caller must hold _cache_lock to prevent concurrent auto-tuning.
        """
        from mlx_primitives.dsl.autotuner import DSLAutoTuner, get_autotune_cache

        tuner = DSLAutoTuner(
            warmup=self.kernel_def.autotune_warmup,
            rep=self.kernel_def.autotune_rep,
        )

        result = tuner.tune(
            kernel=self,
            configs=self.kernel_def.configs,
            args=args,
            kwargs=kwargs,
            grid=grid,
            tuning_key=tuning_key,
        )

        # Cache result
        if self._persistent_cache:
            get_autotune_cache().put(
                self.kernel_def.name,
                tuning_key,
                result.best_config,
                result,
            )

        return result.best_config

    def _make_cache_key(self, config: Config) -> tuple:
        """Create hashable cache key from config."""
        items = [("num_warps", config.num_warps), ("num_stages", config.num_stages)]
        items.extend(sorted(config.kwargs.items()))
        return tuple(items)

    def inspect_metal(self, config: Optional[Config] = None, full: bool = True) -> str:
        """Get generated Metal source code.

        Useful for debugging. Does not compile to MLX kernel.

        Args:
            config: Configuration to use
            full: If True, show full standalone kernel. If False, show MLX-compatible body.
        """
        from mlx_primitives.dsl.compiler.ast_parser import parse_kernel
        from mlx_primitives.dsl.compiler.codegen import generate_metal, generate_mlx_body, detect_output_params

        if config is None:
            # For inspection, use default config or first available
            if self.kernel_def.configs:
                config = self.kernel_def.configs[0]
            else:
                config = Config()

        cache_key = (self._make_cache_key(config), full)
        if cache_key not in self._metal_source_cache:
            ir_func = parse_kernel(
                self.kernel_def.source_code,
                self.kernel_def.parameters,
                self.kernel_def.constexpr_params,
            )

            if full:
                # Full standalone kernel (for inspection/debugging)
                metal_source = generate_metal(ir_func, debug=True)
            else:
                # MLX-compatible body only
                output_params = detect_output_params(ir_func)
                input_names = [p.name for p in ir_func.parameters if p.name not in output_params]
                output_names = [p.name for p in ir_func.parameters if p.name in output_params]
                metal_source = generate_mlx_body(ir_func, input_names, output_names)
                metal_source = f"// MLX kernel body: {ir_func.name}\n// Inputs: {input_names}\n// Outputs: {output_names}\n\n{metal_source}"

            self._metal_source_cache[cache_key] = metal_source

        return self._metal_source_cache[cache_key]

    def warmup(self, *args, grid: tuple, **kwargs) -> None:
        """Trigger compilation without execution."""
        config = self._select_config(args, kwargs)
        cache_key = self._make_cache_key(config)
        if cache_key not in self._compiled_cache:
            from mlx_primitives.dsl.compiler import compile_kernel
            metal_source, kernel_info = compile_kernel(
                self.kernel_def,
                config,
                debug=self.debug,
            )
            self._compiled_cache[cache_key] = kernel_info
            self._metal_source_cache[cache_key] = metal_source


def metal_kernel(
    func: Optional[F] = None,
    *,
    debug: bool = False,
) -> CompiledKernel | Callable[[F], CompiledKernel]:
    """Decorator to mark a function as a Metal kernel.

    The decorated function's body is parsed and compiled to Metal
    shader code. DSL primitives (mt.load, mt.store, etc.) are
    recognized and translated to Metal equivalents.

    Args:
        func: The kernel function
        debug: If True, include debug info in generated Metal

    Example:
        @metal_kernel
        def vector_add(a_ptr, b_ptr, c_ptr, N: mt.constexpr):
            pid = mt.program_id(0)
            idx = pid * 256 + mt.thread_id_in_threadgroup()
            if idx < N:
                a = mt.load(a_ptr + idx)
                b = mt.load(b_ptr + idx)
                mt.store(c_ptr + idx, a + b)

        # Use:
        vector_add(a, b, c, N=len(a), grid=(num_blocks,))
    """
    def decorator(fn: F) -> CompiledKernel:
        # Get source code
        source = inspect.getsource(fn)

        # Parse signature
        sig = inspect.signature(fn)
        parameters = []
        constexpr_params = []

        for name, param in sig.parameters.items():
            annotation = param.annotation
            parameters.append((name, annotation))

            # Check if constexpr
            from mlx_primitives.dsl.types import ConstExpr, constexpr as constexpr_type
            if isinstance(annotation, ConstExpr) or annotation is constexpr_type:
                constexpr_params.append(name)
            elif hasattr(annotation, "__origin__"):
                # Handle constexpr[T] generic form
                pass

        kernel_def = KernelDefinition(
            name=fn.__name__,
            func=fn,
            source_code=source,
            parameters=parameters,
            constexpr_params=constexpr_params,
        )

        return CompiledKernel(kernel_def, debug=debug)

    if func is not None:
        return decorator(func)
    return decorator


def autotune(
    configs: Sequence[Config],
    key: Optional[Sequence[str]] = None,
    warmup: int = 3,
    rep: int = 10,
    persistent_cache: bool = True,
) -> Callable[[F], CompiledKernel]:
    """Decorator to enable auto-tuning over configurations.

    At runtime, benchmarks all configurations and selects the fastest.
    Results are cached based on the key parameters.

    Args:
        configs: List of Config objects to try
        key: Parameter names that determine tuning key
        warmup: Warmup iterations before timing
        rep: Repetitions for timing
        persistent_cache: Whether to save results to disk for future sessions

    Example:
        @autotune(
            configs=[
                Config(BLOCK_M=32, BLOCK_N=32),
                Config(BLOCK_M=64, BLOCK_N=32),
                Config(BLOCK_M=64, BLOCK_N=64),
            ],
            key=['seq_len', 'head_dim'],
        )
        @metal_kernel
        def flash_attention(...):
            ...
    """
    def decorator(kernel: CompiledKernel) -> CompiledKernel:
        # Update kernel definition with autotune info
        kernel.kernel_def.configs = list(configs)
        kernel.kernel_def.autotune_keys = list(key) if key else None
        kernel.kernel_def.autotune_warmup = warmup
        kernel.kernel_def.autotune_rep = rep
        kernel._persistent_cache = persistent_cache
        return kernel

    return decorator
