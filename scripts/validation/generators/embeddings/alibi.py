"""ALiBi embedding generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class ALiBiEmbeddingGenerator(GoldenGenerator):
    """Generate golden files for ALiBi (Attention with Linear Biases).

    ALiBi adds position-dependent linear biases to attention scores:
    bias[i,j] = -slope * |i - j|

    Different heads use different slopes (geometric sequence).
    """

    @property
    def name(self) -> str:
        return "alibi_embedding"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_standard"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "alibi_small", "seq": 128, "heads": 8},
            {"name": "alibi_medium", "seq": 512, "heads": 8},
            {"name": "alibi_large", "seq": 2048, "heads": 12},
            {"name": "alibi_many_heads", "seq": 256, "heads": 32},
            {"name": "alibi_single_head", "seq": 256, "heads": 1},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        seq = config["seq"]
        heads = config["heads"]

        def get_alibi_slopes(num_heads):
            """Get slopes for ALiBi attention bias.

            Slopes form a geometric sequence: 2^(-8/n), 2^(-16/n), ...
            """
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(np.log2(n) - 3)))
                ratio = start
                return [start * (ratio ** i) for i in range(n)]

            # Handle non-power-of-2 head counts
            if num_heads == 1:
                return [0.5]

            if (num_heads & (num_heads - 1)) == 0:  # Power of 2
                return get_slopes_power_of_2(num_heads)
            else:
                # Interpolate between closest powers of 2
                closest_power_of_2 = 2 ** int(np.log2(num_heads))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
                extra_slopes = extra_slopes[0::2][:num_heads - closest_power_of_2]
                return slopes + extra_slopes

        slopes = torch.tensor(get_alibi_slopes(heads)).float()

        # Create position difference matrix: bias[i,j] = -slope * |i - j|
        positions = torch.arange(seq).float()
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)

        # Compute biases for each head
        # Shape: (heads, seq, seq)
        alibi_bias = -slopes.view(-1, 1, 1) * pos_diff.abs().unsqueeze(0)

        return TestConfig(
            name=config["name"],
            inputs={"slopes": slopes.numpy()},
            params={"seq": seq, "heads": heads},
            expected_outputs={"alibi_bias": alibi_bias.numpy()},
            metadata={"seq": seq, "heads": heads},
        )
