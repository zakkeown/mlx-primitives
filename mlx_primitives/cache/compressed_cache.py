"""Compressed KV Cache for memory-efficient inference.

This module provides a KV cache implementation that uses compression
techniques (quantization, pruning, or clustering) to reduce memory usage
while maintaining acceptable quality for attention computation.

Reference:
    "Scissorhands: Exploiting the Persistence of Importance Hypothesis
    for LLM KV Cache Compression at Test Time"
    https://arxiv.org/abs/2305.17118
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx


class CompressedKVCache:
    """Compressed KV cache using quantization or other compression.

    Reduces memory usage by compressing cached keys and values
    while maintaining acceptable quality for attention computation.

    Args:
        num_layers: Number of transformer layers.
        max_batch_size: Maximum batch size.
        max_seq_len: Maximum sequence length.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        compression: Compression method ('quantize', 'prune', 'cluster').
        bits: Bits for quantization (default: 4).
        keep_ratio: Ratio of KV pairs to keep for pruning (default: 0.5).
        dtype: Data type for computation.

    Example:
        >>> cache = CompressedKVCache(
        ...     num_layers=32,
        ...     max_batch_size=1,
        ...     max_seq_len=8192,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     compression='quantize',
        ...     bits=4,
        ... )
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        compression: str = "quantize",
        bits: int = 4,
        keep_ratio: float = 0.5,
        dtype: mx.Dtype = mx.float16,
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compression = compression
        self.bits = bits
        self.keep_ratio = keep_ratio
        self.dtype = dtype

        # Initialize based on compression method
        if compression == "quantize":
            self._init_quantized_cache()
        elif compression == "prune":
            self._init_pruned_cache()
        elif compression == "cluster":
            self._init_clustered_cache()
        else:
            raise ValueError(f"Unknown compression method: {compression}")

        self.seq_len = 0

    def _init_quantized_cache(self) -> None:
        """Initialize quantized cache storage."""
        cache_shape = (
            self.max_batch_size,
            self.num_heads,
            self.max_seq_len,
            self.head_dim,
        )

        # Quantized storage (int8 or packed int4)
        if self.bits == 8:
            self.k_cache_q = [
                mx.zeros(cache_shape, dtype=mx.int8) for _ in range(self.num_layers)
            ]
            self.v_cache_q = [
                mx.zeros(cache_shape, dtype=mx.int8) for _ in range(self.num_layers)
            ]
        else:
            # Pack multiple values per byte for sub-8-bit
            packed_dim = (self.head_dim * self.bits + 7) // 8
            packed_shape = (
                self.max_batch_size,
                self.num_heads,
                self.max_seq_len,
                packed_dim,
            )
            self.k_cache_q = [
                mx.zeros(packed_shape, dtype=mx.uint8) for _ in range(self.num_layers)
            ]
            self.v_cache_q = [
                mx.zeros(packed_shape, dtype=mx.uint8) for _ in range(self.num_layers)
            ]

        # Scales per position (for dequantization)
        scale_shape = (self.max_batch_size, self.num_heads, self.max_seq_len, 1)
        self.k_scales = [
            mx.ones(scale_shape, dtype=mx.float32) for _ in range(self.num_layers)
        ]
        self.v_scales = [
            mx.ones(scale_shape, dtype=mx.float32) for _ in range(self.num_layers)
        ]

    def _init_pruned_cache(self) -> None:
        """Initialize pruned cache storage."""
        # Keep only top-k positions based on attention importance
        self.keep_len = int(self.max_seq_len * self.keep_ratio)

        # Initialize as empty lists - will be populated on first update
        self.k_cache = [None for _ in range(self.num_layers)]
        self.v_cache = [None for _ in range(self.num_layers)]
        self.importance_scores = [None for _ in range(self.num_layers)]

        # Track actual sequence lengths per layer
        self.pruned_seq_lens = [0 for _ in range(self.num_layers)]

        # Mapping from cache index to original position
        self.position_map = [
            mx.zeros(
                (self.max_batch_size, self.num_heads, self.keep_len), dtype=mx.int32
            )
            for _ in range(self.num_layers)
        ]

    def _init_clustered_cache(self) -> None:
        """Initialize clustered cache storage."""
        # Cluster similar KV pairs together
        num_clusters = int(self.max_seq_len * self.keep_ratio)
        cache_shape = (
            self.max_batch_size,
            self.num_heads,
            num_clusters,
            self.head_dim,
        )

        self.k_centroids = [
            mx.zeros(cache_shape, dtype=self.dtype) for _ in range(self.num_layers)
        ]
        self.v_centroids = [
            mx.zeros(cache_shape, dtype=self.dtype) for _ in range(self.num_layers)
        ]

        # Cluster assignments
        self.cluster_assignments = [
            mx.zeros(
                (self.max_batch_size, self.num_heads, self.max_seq_len), dtype=mx.int32
            )
            for _ in range(self.num_layers)
        ]
        self.cluster_counts = [
            mx.zeros((self.max_batch_size, self.num_heads, num_clusters))
            for _ in range(self.num_layers)
        ]

    def reset(self) -> None:
        """Reset the cache."""
        self.seq_len = 0
        if self.compression == "quantize":
            for i in range(self.num_layers):
                self.k_cache_q[i] = mx.zeros_like(self.k_cache_q[i])
                self.v_cache_q[i] = mx.zeros_like(self.v_cache_q[i])
        elif self.compression == "prune":
            for i in range(self.num_layers):
                self.k_cache[i] = None
                self.v_cache[i] = None
                self.importance_scores[i] = None
                self.pruned_seq_lens[i] = 0
        elif self.compression == "cluster":
            for i in range(self.num_layers):
                self.k_centroids[i] = mx.zeros_like(self.k_centroids[i])
                self.v_centroids[i] = mx.zeros_like(self.v_centroids[i])

    def _quantize_8bit(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize to 8-bit with per-token scaling."""
        # Per-token absmax quantization
        absmax = mx.max(mx.abs(x), axis=-1, keepdims=True)
        scale = absmax / 127.0
        scale = mx.maximum(scale, mx.array(1e-8))

        x_q = mx.round(x / scale)
        x_q = mx.clip(x_q, -128, 127).astype(mx.int8)

        return x_q, scale

    def _dequantize_8bit(self, x_q: mx.array, scale: mx.array) -> mx.array:
        """Dequantize from 8-bit."""
        return x_q.astype(mx.float32) * scale

    def _quantize_4bit(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize to 4-bit with per-token scaling."""
        absmax = mx.max(mx.abs(x), axis=-1, keepdims=True)
        scale = absmax / 7.0
        scale = mx.maximum(scale, mx.array(1e-8))

        x_q = mx.round(x / scale)
        x_q = mx.clip(x_q, -8, 7).astype(mx.int8)

        # Pack pairs of 4-bit values into bytes
        batch, heads, seq, dim = x_q.shape
        x_q_list = x_q.reshape(-1, dim).tolist()

        packed = []
        for row in x_q_list:
            packed_row = []
            for j in range(0, len(row), 2):
                low = row[j] & 0xF
                high = (row[j + 1] & 0xF) if j + 1 < len(row) else 0
                packed_row.append((high << 4) | low)
            packed.append(packed_row)

        packed_arr = mx.array(packed, dtype=mx.uint8)
        packed_arr = packed_arr.reshape(batch, heads, seq, -1)

        return packed_arr, scale

    def _dequantize_4bit(self, x_packed: mx.array, scale: mx.array) -> mx.array:
        """Dequantize from 4-bit."""
        batch, heads, seq, packed_dim = x_packed.shape

        # Unpack
        packed_list = x_packed.reshape(-1, packed_dim).tolist()

        unpacked = []
        for packed_row in packed_list:
            row = []
            for byte in packed_row:
                low = byte & 0xF
                high = (byte >> 4) & 0xF
                low = low - 16 if low > 7 else low
                high = high - 16 if high > 7 else high
                row.extend([low, high])
            unpacked.append(row[: self.head_dim])

        x = mx.array(unpacked, dtype=mx.float32)
        x = x.reshape(batch, heads, seq, self.head_dim)

        return x * scale

    def get(self, layer_idx: int) -> Tuple[mx.array, mx.array]:
        """Get cached K, V tensors.

        Args:
            layer_idx: Layer index.

        Returns:
            Tuple of (K, V) tensors.
        """
        if self.seq_len == 0:
            return (
                mx.zeros(
                    (self.max_batch_size, self.num_heads, 0, self.head_dim),
                    dtype=self.dtype,
                ),
                mx.zeros(
                    (self.max_batch_size, self.num_heads, 0, self.head_dim),
                    dtype=self.dtype,
                ),
            )

        if self.compression == "quantize":
            if self.bits == 8:
                k = self._dequantize_8bit(
                    self.k_cache_q[layer_idx][:, :, : self.seq_len, :],
                    self.k_scales[layer_idx][:, :, : self.seq_len, :],
                )
                v = self._dequantize_8bit(
                    self.v_cache_q[layer_idx][:, :, : self.seq_len, :],
                    self.v_scales[layer_idx][:, :, : self.seq_len, :],
                )
            else:
                k = self._dequantize_4bit(
                    self.k_cache_q[layer_idx][:, :, : self.seq_len, :],
                    self.k_scales[layer_idx][:, :, : self.seq_len, :],
                )
                v = self._dequantize_4bit(
                    self.v_cache_q[layer_idx][:, :, : self.seq_len, :],
                    self.v_scales[layer_idx][:, :, : self.seq_len, :],
                )
            return k.astype(self.dtype), v.astype(self.dtype)

        elif self.compression == "prune":
            effective_len = min(self.seq_len, self.k_cache[layer_idx].shape[2])
            return (
                self.k_cache[layer_idx][:, :, :effective_len, :],
                self.v_cache[layer_idx][:, :, :effective_len, :],
            )

        elif self.compression == "cluster":
            # Return cluster centroids
            return (
                self.k_centroids[layer_idx],
                self.v_centroids[layer_idx],
            )

        return mx.zeros((0,)), mx.zeros((0,))

    def update(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
        attention_scores: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with new K, V.

        Args:
            layer_idx: Layer index.
            new_k: New key tensor (batch, heads, new_len, head_dim).
            new_v: New value tensor.
            attention_scores: Optional attention scores for importance-based pruning.

        Returns:
            Tuple of full (K, V) after update.
        """
        new_len = new_k.shape[2]
        start_pos = self.seq_len

        if self.compression == "quantize":
            return self._update_quantized(layer_idx, new_k, new_v, start_pos, new_len)
        elif self.compression == "prune":
            return self._update_pruned(layer_idx, new_k, new_v, attention_scores)
        elif self.compression == "cluster":
            return self._update_clustered(layer_idx, new_k, new_v)

        return new_k, new_v

    def _update_quantized(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
        start_pos: int,
        new_len: int,
    ) -> Tuple[mx.array, mx.array]:
        """Update quantized cache."""
        end_pos = start_pos + new_len

        if self.bits == 8:
            k_q, k_scale = self._quantize_8bit(new_k)
            v_q, v_scale = self._quantize_8bit(new_v)

            # Update using vectorized operations
            positions = mx.arange(self.max_seq_len)
            update_mask = (positions >= start_pos) & (positions < end_pos)
            update_mask = update_mask[None, None, :, None]  # (1, 1, seq, 1)

            # Broadcast new values to full cache shape
            k_q_full = mx.zeros_like(self.k_cache_q[layer_idx])
            v_q_full = mx.zeros_like(self.v_cache_q[layer_idx])
            k_scale_full = mx.ones_like(self.k_scales[layer_idx])
            v_scale_full = mx.ones_like(self.v_scales[layer_idx])

            # Place new values at correct positions
            k_q_full = k_q_full.at[:, :, start_pos:end_pos, :].add(k_q)
            v_q_full = v_q_full.at[:, :, start_pos:end_pos, :].add(v_q)
            k_scale_full = k_scale_full.at[:, :, start_pos:end_pos, :].add(k_scale - 1)
            v_scale_full = v_scale_full.at[:, :, start_pos:end_pos, :].add(v_scale - 1)

            self.k_cache_q[layer_idx] = mx.where(
                update_mask, k_q_full, self.k_cache_q[layer_idx]
            )
            self.v_cache_q[layer_idx] = mx.where(
                update_mask, v_q_full, self.v_cache_q[layer_idx]
            )
            self.k_scales[layer_idx] = mx.where(
                update_mask, k_scale_full, self.k_scales[layer_idx]
            )
            self.v_scales[layer_idx] = mx.where(
                update_mask, v_scale_full, self.v_scales[layer_idx]
            )
        else:
            k_q, k_scale = self._quantize_4bit(new_k)
            v_q, v_scale = self._quantize_4bit(new_v)

            # Update packed cache
            positions = mx.arange(self.max_seq_len)
            update_mask = (positions >= start_pos) & (positions < end_pos)
            update_mask_q = update_mask[None, None, :, None]
            update_mask_s = update_mask[None, None, :, None]

            # Use where for update
            self.k_cache_q[layer_idx] = mx.where(
                update_mask_q[..., : k_q.shape[-1]],
                mx.concatenate(
                    [
                        self.k_cache_q[layer_idx][:, :, :start_pos, :],
                        k_q,
                        self.k_cache_q[layer_idx][:, :, end_pos:, :],
                    ],
                    axis=2,
                )[:, :, : self.max_seq_len, :],
                self.k_cache_q[layer_idx],
            )
            self.v_cache_q[layer_idx] = mx.where(
                update_mask_q[..., : v_q.shape[-1]],
                mx.concatenate(
                    [
                        self.v_cache_q[layer_idx][:, :, :start_pos, :],
                        v_q,
                        self.v_cache_q[layer_idx][:, :, end_pos:, :],
                    ],
                    axis=2,
                )[:, :, : self.max_seq_len, :],
                self.v_cache_q[layer_idx],
            )

            self.k_scales[layer_idx] = mx.where(
                update_mask_s,
                mx.concatenate(
                    [
                        self.k_scales[layer_idx][:, :, :start_pos, :],
                        k_scale,
                        self.k_scales[layer_idx][:, :, end_pos:, :],
                    ],
                    axis=2,
                )[:, :, : self.max_seq_len, :],
                self.k_scales[layer_idx],
            )
            self.v_scales[layer_idx] = mx.where(
                update_mask_s,
                mx.concatenate(
                    [
                        self.v_scales[layer_idx][:, :, :start_pos, :],
                        v_scale,
                        self.v_scales[layer_idx][:, :, end_pos:, :],
                    ],
                    axis=2,
                )[:, :, : self.max_seq_len, :],
                self.v_scales[layer_idx],
            )

        if layer_idx == 0:
            self.seq_len = end_pos

        return self.get(layer_idx)

    def _update_pruned(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
        attention_scores: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Update pruned cache based on importance."""
        new_len = new_k.shape[2]

        # Compute importance scores if not provided
        if attention_scores is None:
            # Use L2 norm as proxy for importance
            scores = mx.sum(new_k**2, axis=-1)  # (batch, heads, new_len)
        else:
            scores = mx.mean(attention_scores, axis=-2)  # Average over query positions

        # First update - initialize cache
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
            self.importance_scores[layer_idx] = scores
            self.pruned_seq_lens[layer_idx] = new_len
        else:
            # Combine existing and new
            current_k = self.k_cache[layer_idx]
            current_v = self.v_cache[layer_idx]
            current_scores = self.importance_scores[layer_idx]

            # Merge new KV pairs
            all_k = mx.concatenate([current_k, new_k], axis=2)
            all_v = mx.concatenate([current_v, new_v], axis=2)
            all_scores = mx.concatenate([current_scores, scores], axis=2)

            # Keep top-k based on importance if exceeds keep_len
            total_len = all_k.shape[2]
            if total_len > self.keep_len:
                # Simplified: take most recent keep_len positions
                # Full implementation would sort and gather by importance
                self.k_cache[layer_idx] = all_k[:, :, -self.keep_len :, :]
                self.v_cache[layer_idx] = all_v[:, :, -self.keep_len :, :]
                self.importance_scores[layer_idx] = all_scores[:, :, -self.keep_len :]
                self.pruned_seq_lens[layer_idx] = self.keep_len
            else:
                self.k_cache[layer_idx] = all_k
                self.v_cache[layer_idx] = all_v
                self.importance_scores[layer_idx] = all_scores
                self.pruned_seq_lens[layer_idx] = total_len

        if layer_idx == 0:
            self.seq_len += new_len

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def _update_clustered(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update clustered cache using online k-means."""
        new_len = new_k.shape[2]

        for t in range(new_len):
            k_t = new_k[:, :, t : t + 1, :]  # (batch, heads, 1, head_dim)
            v_t = new_v[:, :, t : t + 1, :]

            # Find nearest cluster
            distances = mx.sum(
                (self.k_centroids[layer_idx] - k_t) ** 2, axis=-1
            )  # (batch, heads, num_clusters)

            nearest = mx.argmin(distances, axis=-1)  # (batch, heads)

            # Update cluster centroids (online update)
            # c_new = (c_old * n + x) / (n + 1)
            for b in range(new_k.shape[0]):
                for h in range(new_k.shape[1]):
                    c = int(nearest[b, h])
                    count = float(self.cluster_counts[layer_idx][b, h, c])
                    new_count = count + 1

                    # Update K centroid
                    old_k = self.k_centroids[layer_idx][b, h, c, :]
                    self.k_centroids[layer_idx] = self.k_centroids[layer_idx].at[
                        b, h, c, :
                    ].add((k_t[b, h, 0, :] - old_k) / new_count)

                    # Update V centroid
                    old_v = self.v_centroids[layer_idx][b, h, c, :]
                    self.v_centroids[layer_idx] = self.v_centroids[layer_idx].at[
                        b, h, c, :
                    ].add((v_t[b, h, 0, :] - old_v) / new_count)

                    # Update count
                    self.cluster_counts[layer_idx] = self.cluster_counts[layer_idx].at[
                        b, h, c
                    ].add(1)

        if layer_idx == 0:
            self.seq_len += new_len

        return self.k_centroids[layer_idx], self.v_centroids[layer_idx]

    @property
    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        if self.compression == "quantize":
            if self.bits == 8:
                bytes_per_element = 1  # int8
            else:
                bytes_per_element = 0.5  # 4-bit packed
            cache_size = (
                self.max_batch_size
                * self.num_heads
                * self.max_seq_len
                * self.head_dim
                * bytes_per_element
                * 2
                * self.num_layers  # K + V
            )
            scale_size = (
                self.max_batch_size
                * self.num_heads
                * self.max_seq_len
                * 4
                * 2
                * self.num_layers  # float32 scales
            )
            return (cache_size + scale_size) / (1024 * 1024)

        elif self.compression == "prune":
            keep_len = int(self.max_seq_len * self.keep_ratio)
            cache_size = (
                self.max_batch_size
                * self.num_heads
                * keep_len
                * self.head_dim
                * 2
                * 2
                * self.num_layers  # float16
            )
            return cache_size / (1024 * 1024)

        elif self.compression == "cluster":
            num_clusters = int(self.max_seq_len * self.keep_ratio)
            cache_size = (
                self.max_batch_size
                * self.num_heads
                * num_clusters
                * self.head_dim
                * 2
                * 2
                * self.num_layers
            )
            return cache_size / (1024 * 1024)

        return 0.0

    @property
    def compression_ratio(self) -> float:
        """Get compression ratio vs full precision cache."""
        full_size = (
            self.max_batch_size
            * self.num_heads
            * self.max_seq_len
            * self.head_dim
            * 2
            * 2
            * self.num_layers  # float16 K + V
        )
        compressed_size = self.memory_usage_mb * 1024 * 1024
        return full_size / max(compressed_size, 1)
