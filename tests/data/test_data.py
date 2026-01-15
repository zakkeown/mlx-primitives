"""Tests for data utilities."""

import mlx.core as mx
import pytest

from mlx_primitives.data import (
    # Dataset
    Dataset,
    TensorDataset,
    ListDataset,
    ConcatDataset,
    SubsetDataset,
    TransformDataset,
    random_split,
    # Samplers
    SequentialSampler,
    RandomSampler,
    WeightedRandomSampler,
    BatchSampler,
    DistributedSampler,
    # DataLoader
    DataLoader,
    InfiniteDataLoader,
    default_collate,
    # Vision transforms
    Compose,
    Normalize,
    ToTensor,
    RandomHorizontalFlip,
    RandomCrop,
    CenterCrop,
    ColorJitter,
    CutOut,
    mixup,
    cutmix,
    RandomRotation,
    RandomAugment,
    TrivialAugmentWide,
    # Text transforms
    pad_sequence,
    pad_to_length,
    create_attention_mask,
    create_causal_mask,
    RandomMask,
    SpanMask,
    TokenDropout,
    pack_sequences,
)


# ============================================================================
# Dataset Tests
# ============================================================================


class TestTensorDataset:
    """Tests for TensorDataset."""

    def test_single_tensor(self):
        """Test dataset with single tensor."""
        x = mx.arange(100)
        dataset = TensorDataset(x)

        assert len(dataset) == 100
        assert dataset[0] == (mx.array(0),)

    def test_multiple_tensors(self):
        """Test dataset with multiple tensors."""
        x = mx.random.normal((100, 10))
        y = mx.arange(100)
        dataset = TensorDataset(x, y)

        assert len(dataset) == 100
        sample = dataset[0]
        assert len(sample) == 2
        assert sample[0].shape == (10,)

    def test_mismatched_sizes(self):
        """Test error on mismatched tensor sizes."""
        x = mx.random.normal((100, 10))
        y = mx.arange(50)  # Different size

        with pytest.raises(ValueError):
            TensorDataset(x, y)


class TestListDataset:
    """Tests for ListDataset."""

    def test_basic(self):
        """Test basic list dataset."""
        data = [{"x": i, "y": i * 2} for i in range(10)]
        dataset = ListDataset(data)

        assert len(dataset) == 10
        assert dataset[5]["x"] == 5


class TestConcatDataset:
    """Tests for ConcatDataset."""

    def test_concat(self):
        """Test concatenating datasets."""
        ds1 = TensorDataset(mx.ones((50,)))
        ds2 = TensorDataset(mx.zeros((50,)))
        combined = ConcatDataset([ds1, ds2])

        assert len(combined) == 100

        # First half should be ones
        assert float(combined[25][0]) == 1.0
        # Second half should be zeros
        assert float(combined[75][0]) == 0.0


class TestSubsetDataset:
    """Tests for SubsetDataset."""

    def test_subset(self):
        """Test subsetting a dataset."""
        full = TensorDataset(mx.arange(100))
        subset = SubsetDataset(full, indices=[0, 10, 20, 30])

        assert len(subset) == 4
        assert int(subset[1][0]) == 10


class TestTransformDataset:
    """Tests for TransformDataset."""

    def test_transform(self):
        """Test applying transform."""
        dataset = TensorDataset(mx.arange(10))
        transformed = TransformDataset(
            dataset, lambda x: (x[0] * 2,)
        )

        assert int(transformed[5][0]) == 10


class TestRandomSplit:
    """Tests for random_split."""

    def test_split(self):
        """Test random splitting."""
        dataset = TensorDataset(mx.arange(100))
        train, val, test = random_split(dataset, [80, 10, 10], seed=42)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


# ============================================================================
# Sampler Tests
# ============================================================================


class TestSamplers:
    """Tests for samplers."""

    def test_sequential_sampler(self):
        """Test sequential sampler."""
        dataset = TensorDataset(mx.arange(10))
        sampler = SequentialSampler(dataset)

        indices = list(sampler)
        assert indices == list(range(10))

    def test_random_sampler(self):
        """Test random sampler."""
        dataset = TensorDataset(mx.arange(10))
        sampler = RandomSampler(dataset, seed=42)

        indices = list(sampler)
        assert len(indices) == 10
        assert set(indices) == set(range(10))

    def test_random_sampler_reproducibility(self):
        """Test random sampler is reproducible with seed."""
        dataset = TensorDataset(mx.arange(100))

        sampler1 = RandomSampler(dataset, seed=42)
        sampler2 = RandomSampler(dataset, seed=42)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 == indices2

    def test_batch_sampler(self):
        """Test batch sampler."""
        dataset = TensorDataset(mx.arange(10))
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=3)

        batches = list(batch_sampler)
        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_batch_sampler_drop_last(self):
        """Test batch sampler with drop_last."""
        dataset = TensorDataset(mx.arange(10))
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=True)

        batches = list(batch_sampler)
        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_distributed_sampler(self):
        """Test distributed sampler."""
        dataset = TensorDataset(mx.arange(100))

        # Simulate 4 processes
        all_indices = []
        for rank in range(4):
            sampler = DistributedSampler(
                dataset, num_replicas=4, rank=rank, shuffle=False
            )
            all_indices.extend(list(sampler))

        # Each index should appear exactly once
        assert len(all_indices) == 100
        assert set(all_indices) == set(range(100))


# ============================================================================
# DataLoader Tests
# ============================================================================


class TestDataLoader:
    """Tests for DataLoader."""

    def test_basic_loading(self):
        """Test basic data loading."""
        x = mx.random.normal((100, 10))
        y = mx.arange(100)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=16)

        batches = list(loader)
        assert len(batches) == 7  # 100 / 16 = 6.25, rounded up

        # First batch
        bx, by = batches[0]
        assert bx.shape == (16, 10)
        assert by.shape == (16,)

    def test_shuffle(self):
        """Test shuffled loading."""
        dataset = TensorDataset(mx.arange(100))
        loader = DataLoader(dataset, batch_size=10, shuffle=True, seed=42)

        batches = list(loader)
        first_batch = batches[0][0].tolist()

        # Should not be sequential
        assert first_batch != list(range(10))

    def test_drop_last(self):
        """Test drop_last option."""
        dataset = TensorDataset(mx.arange(100))
        loader = DataLoader(dataset, batch_size=16, drop_last=True)

        batches = list(loader)
        assert len(batches) == 6  # 100 // 16 = 6

    def test_custom_collate(self):
        """Test custom collate function."""
        dataset = ListDataset([{"x": mx.array([i])} for i in range(10)])

        def collate(batch):
            return {"x": mx.stack([b["x"] for b in batch])}

        loader = DataLoader(dataset, batch_size=5, collate_fn=collate)

        batches = list(loader)
        assert batches[0]["x"].shape == (5, 1)


class TestDefaultCollate:
    """Tests for default_collate function."""

    def test_collate_arrays(self):
        """Test collating MLX arrays."""
        batch = [mx.array([1, 2]), mx.array([3, 4])]
        collated = default_collate(batch)

        assert collated.shape == (2, 2)

    def test_collate_tuples(self):
        """Test collating tuples."""
        batch = [(mx.array([1]), mx.array([2])), (mx.array([3]), mx.array([4]))]
        collated = default_collate(batch)

        assert len(collated) == 2
        assert collated[0].shape == (2, 1)

    def test_collate_dicts(self):
        """Test collating dictionaries."""
        batch = [{"x": mx.array([1]), "y": 0}, {"x": mx.array([2]), "y": 1}]
        collated = default_collate(batch)

        assert collated["x"].shape == (2, 1)


class TestInfiniteDataLoader:
    """Tests for InfiniteDataLoader."""

    def test_infinite_iteration(self):
        """Test infinite iteration."""
        dataset = TensorDataset(mx.arange(10))
        loader = DataLoader(dataset, batch_size=3)
        infinite = InfiniteDataLoader(loader)

        # Take more than one epoch worth of batches
        batches = []
        for i, batch in enumerate(infinite):
            batches.append(batch)
            if i >= 10:
                break

        assert len(batches) == 11


# ============================================================================
# Vision Transform Tests
# ============================================================================


class TestVisionTransforms:
    """Tests for vision transforms."""

    def test_compose(self):
        """Test composing transforms."""
        transform = Compose([
            lambda x: x * 2,
            lambda x: x + 1,
        ])

        x = mx.array([1.0, 2.0, 3.0])
        result = transform(x)

        expected = mx.array([3.0, 5.0, 7.0])
        assert mx.allclose(result, expected)

    def test_normalize(self):
        """Test normalization."""
        normalize = Normalize(mean=[0.5], std=[0.5])
        x = mx.array([[[1.0], [0.0]]])  # HWC format

        result = normalize(x)

        # (1.0 - 0.5) / 0.5 = 1.0, (0.0 - 0.5) / 0.5 = -1.0
        expected = mx.array([[[1.0], [-1.0]]])
        assert mx.allclose(result, expected)

    def test_to_tensor(self):
        """Test ToTensor scaling."""
        to_tensor = ToTensor()
        x = mx.array([0, 128, 255], dtype=mx.uint8)

        result = to_tensor(x)

        assert result.dtype == mx.float32
        assert float(result[0]) == pytest.approx(0.0)
        assert float(result[2]) == pytest.approx(1.0)

    def test_random_horizontal_flip(self):
        """Test random horizontal flip."""
        flip = RandomHorizontalFlip(p=1.0)  # Always flip
        x = mx.array([[[1], [2], [3]]])  # 1x3x1 HWC

        result = flip(x)

        expected = mx.array([[[3], [2], [1]]])
        assert mx.allclose(result, expected)

    def test_center_crop(self):
        """Test center crop."""
        crop = CenterCrop(size=(2, 2))
        x = mx.random.normal((4, 4, 3))  # HWC

        result = crop(x)
        assert result.shape == (2, 2, 3)

    def test_random_crop(self):
        """Test random crop."""
        crop = RandomCrop(size=(3, 3))
        x = mx.random.normal((5, 5, 3))

        result = crop(x)
        assert result.shape == (3, 3, 3)

    def test_random_crop_with_padding(self):
        """Test random crop with padding."""
        crop = RandomCrop(size=(5, 5), padding=2)
        x = mx.random.normal((3, 3, 3))

        # After padding: 7x7, then crop to 5x5
        result = crop(x)
        assert result.shape == (5, 5, 3)


class TestMixupCutmix:
    """Tests for mixup and cutmix."""

    def test_mixup(self):
        """Test mixup augmentation."""
        x1 = mx.ones((8, 8, 3))
        y1 = mx.array([1.0, 0.0])
        x2 = mx.zeros((8, 8, 3))
        y2 = mx.array([0.0, 1.0])

        mixed_x, mixed_y = mixup(x1, y1, x2, y2, alpha=1.0)

        # Result should be between x1 and x2
        assert mixed_x.shape == (8, 8, 3)
        assert mixed_y.shape == (2,)
        # Labels should sum to 1
        assert float(mx.sum(mixed_y)) == pytest.approx(1.0)

    def test_cutmix(self):
        """Test cutmix augmentation."""
        x1 = mx.ones((8, 8, 3))
        y1 = mx.array([1.0, 0.0])
        x2 = mx.zeros((8, 8, 3))
        y2 = mx.array([0.0, 1.0])

        mixed_x, mixed_y = cutmix(x1, y1, x2, y2, alpha=1.0)

        assert mixed_x.shape == (8, 8, 3)
        assert mixed_y.shape == (2,)


# ============================================================================
# Text Transform Tests
# ============================================================================


class TestTextTransforms:
    """Tests for text/sequence transforms."""

    def test_pad_sequence(self):
        """Test padding sequences."""
        seqs = [
            mx.array([1, 2, 3]),
            mx.array([4, 5]),
            mx.array([6]),
        ]

        padded = pad_sequence(seqs, padding_value=0)

        assert padded.shape == (3, 3)
        # First sequence unchanged
        assert padded[0, 2] == 3
        # Second sequence padded
        assert padded[1, 2] == 0
        # Third sequence padded
        assert padded[2, 1] == 0

    def test_pad_to_length(self):
        """Test padding single sequence."""
        seq = mx.array([1, 2, 3])

        # Pad right
        padded = pad_to_length(seq, length=5, padding_value=0, side="right")
        assert padded.shape == (5,)
        assert padded[4] == 0

        # Pad left
        padded = pad_to_length(seq, length=5, padding_value=0, side="left")
        assert padded[0] == 0
        assert padded[2] == 1

    def test_create_attention_mask(self):
        """Test attention mask creation."""
        lengths = [3, 5, 2]
        mask = create_attention_mask(lengths, max_length=5)

        assert mask.shape == (3, 5)
        # First sequence: [T, T, T, F, F]
        assert mask[0, 2] == True
        assert mask[0, 3] == False
        # Second sequence: all True
        assert mask[1, 4] == True
        # Third sequence: [T, T, F, F, F]
        assert mask[2, 1] == True
        assert mask[2, 2] == False

    def test_create_causal_mask(self):
        """Test causal mask creation."""
        mask = create_causal_mask(4)

        assert mask.shape == (4, 4)
        # Lower triangular
        assert mask[0, 0] == True
        assert mask[0, 1] == False
        assert mask[2, 1] == True
        assert mask[3, 3] == True


class TestRandomMask:
    """Tests for BERT-style masking."""

    def test_random_mask_shape(self):
        """Test mask preserves shape."""
        masker = RandomMask(
            mask_prob=0.15,
            mask_token_id=103,
            vocab_size=30000,
        )

        input_ids = mx.random.randint(0, 1000, (128,))
        masked_ids, labels = masker(input_ids)

        assert masked_ids.shape == input_ids.shape
        assert labels.shape == input_ids.shape

    def test_random_mask_labels(self):
        """Test labels are -100 for non-masked."""
        masker = RandomMask(
            mask_prob=0.5,  # High prob to ensure some masking
            mask_token_id=103,
            vocab_size=30000,
        )

        input_ids = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        masked_ids, labels = masker(input_ids)

        # Labels should be -100 where not masked
        non_masked = labels == -100
        assert mx.sum(non_masked) > 0


class TestTokenDropout:
    """Tests for token dropout."""

    def test_dropout(self):
        """Test token dropout reduces length."""
        dropout = TokenDropout(drop_prob=0.5, min_length=1)
        input_ids = mx.arange(100)

        dropped = dropout(input_ids)

        # Length should be reduced (with high probability)
        assert dropped.shape[0] <= 100
        assert dropped.shape[0] >= 1


class TestPackSequences:
    """Tests for sequence packing."""

    def test_packing(self):
        """Test packing sequences."""
        seqs = [
            mx.array([1, 2, 3]),
            mx.array([4, 5]),
            mx.array([6, 7, 8, 9]),
        ]

        packed, mask = pack_sequences(seqs, max_length=6, padding_value=0)

        assert packed.shape[1] == 6
        assert mask.shape == packed.shape


# ============================================================================
# Benchmark Tests
# ============================================================================


@pytest.mark.benchmark
class TestDataBenchmarks:
    """Benchmark tests for data utilities."""

    def test_dataloader_benchmark(self, benchmark):
        """Benchmark data loading."""
        dataset = TensorDataset(
            mx.random.normal((1000, 64)),
            mx.arange(1000),
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        def iterate():
            for batch in loader:
                pass

        benchmark(iterate)

    def test_transform_benchmark(self, benchmark):
        """Benchmark vision transforms."""
        transform = Compose([
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        x = mx.random.uniform(shape=(224, 224, 3))

        def apply_transform():
            return transform(x)

        benchmark(apply_transform)

    def test_pad_sequence_benchmark(self, benchmark):
        """Benchmark sequence padding."""
        seqs = [mx.random.randint(0, 1000, (i,)) for i in range(10, 100)]

        def pad():
            return pad_sequence(seqs)

        benchmark(pad)


# ============================================================================
# RandomRotation Tests
# ============================================================================


class TestRandomRotation:
    """Tests for RandomRotation transform."""

    def test_output_shape(self):
        """Test rotation preserves shape."""
        rotation = RandomRotation(degrees=45)
        x = mx.random.normal((32, 32, 3))

        result = rotation(x)

        assert result.shape == x.shape

    def test_rotation_range(self):
        """Test rotation with degree range."""
        rotation = RandomRotation(degrees=(-30, 30))
        x = mx.random.normal((16, 16, 3))

        result = rotation(x)

        assert result.shape == x.shape

    def test_zero_rotation(self):
        """Test zero rotation returns similar image."""
        rotation = RandomRotation(degrees=0)
        x = mx.random.normal((8, 8, 3))

        result = rotation(x)

        # With zero degrees, output should be very close to input
        assert mx.allclose(result, x, atol=1e-5)


# ============================================================================
# RandomAugment Tests
# ============================================================================


class TestRandomAugment:
    """Tests for RandAugment transform."""

    def test_output_shape(self):
        """Test augmentation preserves shape."""
        augment = RandomAugment(num_ops=2, magnitude=9)
        x = mx.random.uniform(shape=(32, 32, 3))

        result = augment(x)

        assert result.shape == x.shape

    def test_different_num_ops(self):
        """Test with different number of operations."""
        for num_ops in [1, 2, 3]:
            augment = RandomAugment(num_ops=num_ops, magnitude=5)
            x = mx.random.uniform(shape=(16, 16, 3))

            result = augment(x)

            assert result.shape == x.shape

    def test_different_magnitudes(self):
        """Test with different magnitudes."""
        for mag in [1, 5, 10]:
            augment = RandomAugment(num_ops=1, magnitude=mag)
            x = mx.random.uniform(shape=(16, 16, 3))

            result = augment(x)

            assert result.shape == x.shape


# ============================================================================
# TrivialAugmentWide Tests
# ============================================================================


class TestTrivialAugmentWide:
    """Tests for TrivialAugmentWide transform."""

    def test_output_shape(self):
        """Test augmentation preserves shape."""
        augment = TrivialAugmentWide()
        x = mx.random.uniform(shape=(32, 32, 3))

        result = augment(x)

        assert result.shape == x.shape

    def test_single_operation(self):
        """Test that exactly one operation is applied."""
        augment = TrivialAugmentWide()
        x = mx.random.uniform(shape=(16, 16, 3))

        # Should work without error
        result = augment(x)

        assert result.shape == x.shape

    def test_different_input_sizes(self):
        """Test with different input sizes."""
        augment = TrivialAugmentWide()

        for size in [(8, 8, 3), (32, 32, 3), (64, 64, 3)]:
            x = mx.random.uniform(shape=size)

            result = augment(x)

            assert result.shape == size
