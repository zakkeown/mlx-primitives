"""Generation benchmark suite for sampling and batching operations."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, GenerationSizes
from benchmarks.utils import BenchmarkResult, benchmark_fn


class GenerationBenchmarks:
    """Benchmark suite for generation/sampling operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[GenerationSizes] = None,
    ):
        """Initialize generation benchmarks.

        Args:
            config: Benchmark configuration.
            sizes: Size configurations for generation benchmarks.
        """
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or GenerationSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all generation benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []
        results.extend(self.run_sampling_benchmarks())
        results.extend(self.run_batching_benchmarks())
        return results

    def run_sampling_benchmarks(self) -> list[BenchmarkResult]:
        """Run sampling operation benchmarks."""
        results = []

        for batch_size in self.sizes.batch_sizes[:4]:
            for vocab_size in self.sizes.vocab_sizes[:2]:
                # Temperature sampling
                result = self._benchmark_temperature(batch_size, vocab_size, temperature=0.7)
                if result:
                    results.append(result)

                # Top-K sampling
                for top_k in self.sizes.top_k_values[:2]:
                    result = self._benchmark_top_k(batch_size, vocab_size, top_k)
                    if result:
                        results.append(result)

                # Top-P (nucleus) sampling
                for top_p in self.sizes.top_p_values[:2]:
                    result = self._benchmark_top_p(batch_size, vocab_size, top_p)
                    if result:
                        results.append(result)

                # Repetition penalty
                result = self._benchmark_repetition_penalty(batch_size, vocab_size)
                if result:
                    results.append(result)

        return results

    def run_batching_benchmarks(self) -> list[BenchmarkResult]:
        """Run batching operation benchmarks."""
        results = []

        for batch_size in self.sizes.batch_sizes[:4]:
            seq_len = 512

            # Create attention mask
            result = self._benchmark_create_attention_mask(batch_size, seq_len)
            if result:
                results.append(result)

            # Unbatch outputs
            result = self._benchmark_unbatch_outputs(batch_size, seq_len)
            if result:
                results.append(result)

        return results

    def _benchmark_temperature(
        self,
        batch_size: int,
        vocab_size: int,
        temperature: float,
    ) -> Optional[BenchmarkResult]:
        """Benchmark temperature scaling."""
        try:
            from mlx_primitives.generation import apply_temperature
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        logits = mx.random.normal((batch_size, vocab_size))

        def fn():
            return apply_temperature(logits, temperature)

        name = f"temperature_b{batch_size}_v{vocab_size}_t{temperature}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "temperature": temperature,
            "type": "sampling",
            "operation": "temperature",
        }
        return result

    def _benchmark_top_k(
        self,
        batch_size: int,
        vocab_size: int,
        top_k: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark top-k filtering."""
        try:
            from mlx_primitives.generation import apply_top_k
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        logits = mx.random.normal((batch_size, vocab_size))

        def fn():
            return apply_top_k(logits, top_k)

        name = f"top_k_b{batch_size}_v{vocab_size}_k{top_k}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "top_k": top_k,
            "type": "sampling",
            "operation": "top_k",
        }
        return result

    def _benchmark_top_p(
        self,
        batch_size: int,
        vocab_size: int,
        top_p: float,
    ) -> Optional[BenchmarkResult]:
        """Benchmark top-p (nucleus) filtering."""
        try:
            from mlx_primitives.generation import apply_top_p
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        logits = mx.random.normal((batch_size, vocab_size))

        def fn():
            return apply_top_p(logits, top_p)

        name = f"top_p_b{batch_size}_v{vocab_size}_p{top_p}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "top_p": top_p,
            "type": "sampling",
            "operation": "top_p",
        }
        return result

    def _benchmark_repetition_penalty(
        self,
        batch_size: int,
        vocab_size: int,
        penalty: float = 1.2,
    ) -> Optional[BenchmarkResult]:
        """Benchmark repetition penalty."""
        try:
            from mlx_primitives.generation import apply_repetition_penalty_batch
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        logits = mx.random.normal((batch_size, vocab_size))

        # Create some input IDs representing previous tokens (as list of lists)
        context_length = 100
        input_ids_arr = mx.random.randint(0, vocab_size, shape=(batch_size, context_length))
        mx.eval(input_ids_arr)
        generated_tokens = [input_ids_arr[i].tolist() for i in range(batch_size)]

        def fn():
            return apply_repetition_penalty_batch(logits, generated_tokens, penalty)

        name = f"repetition_penalty_b{batch_size}_v{vocab_size}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "penalty": penalty,
            "context_length": context_length,
            "type": "sampling",
            "operation": "repetition_penalty",
        }
        return result

    def _benchmark_create_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark attention mask creation."""
        try:
            from mlx_primitives.generation import create_attention_mask
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        # Variable length sequences
        lengths = mx.random.randint(1, seq_len, shape=(batch_size,))

        def fn():
            return create_attention_mask(lengths, seq_len)

        name = f"create_attn_mask_b{batch_size}_s{seq_len}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "type": "batching",
            "operation": "create_attention_mask",
        }
        return result

    def _benchmark_unbatch_outputs(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark output unbatching."""
        try:
            from mlx_primitives.generation import unbatch_outputs
            from mlx_primitives.generation.batch_manager import BatchedSequences, PaddingSide
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        # Batched outputs
        hidden_dim = 512
        outputs = mx.random.normal((batch_size, seq_len, hidden_dim))

        # Create proper BatchedSequences object
        sequence_lengths = mx.random.randint(1, seq_len, shape=(batch_size,))
        mx.eval(sequence_lengths)
        input_ids = mx.random.randint(0, 32000, shape=(batch_size, seq_len))
        attention_mask = mx.ones((batch_size, seq_len))
        position_ids = mx.broadcast_to(mx.arange(seq_len), (batch_size, seq_len))

        batch = BatchedSequences(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sequence_lengths=sequence_lengths,
            batch_indices=list(range(batch_size)),
            max_seqlen=seq_len,
            padding_side=PaddingSide.RIGHT,
        )

        def fn():
            return unbatch_outputs(outputs, batch)

        name = f"unbatch_outputs_b{batch_size}_s{seq_len}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "type": "batching",
            "operation": "unbatch_outputs",
        }
        return result
