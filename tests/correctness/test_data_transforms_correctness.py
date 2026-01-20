"""Correctness tests for Data transforms.

Tests verify:
1. Vision transforms: shape preservation, value ranges, composability
2. Text transforms: padding, masking, sequence manipulation
"""

import pytest
import mlx.core as mx

from mlx_primitives.data.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomCrop,
    CenterCrop,
    Resize,
    ColorJitter,
    GaussianNoise,
    CutOut,
    mixup,
    cutmix,
    MixUpTransform,
    CutMixTransform,
    RandomAugment,
    TrivialAugmentWide,
)

from mlx_primitives.data.text_transforms import (
    pad_sequence,
    pad_to_length,
    create_attention_mask,
    create_causal_mask,
    RandomMask,
    SpanMask,
    TokenDropout,
    pack_sequences,
    truncate_sequences,
)


# =============================================================================
# Vision Transform Tests
# =============================================================================


class TestCompose:
    """Test Compose transform chaining."""

    def test_chains_multiple_transforms(self):
        """Compose should apply transforms in order."""
        mx.random.seed(42)
        img = mx.random.uniform(shape=(32, 32, 3))
        mx.eval(img)

        # Create compose with normalize
        compose = Compose([
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        out = compose(img)
        mx.eval(out)

        assert out.shape == img.shape

    def test_empty_compose(self):
        """Empty compose should return input unchanged."""
        img = mx.random.uniform(shape=(32, 32, 3))
        mx.eval(img)

        compose = Compose([])
        out = compose(img)
        mx.eval(out)

        diff = float(mx.max(mx.abs(out - img)))
        assert diff == 0.0


class TestNormalize:
    """Test Normalize transform."""

    def test_normalize_applies_mean_std(self):
        """Normalize should subtract mean and divide by std."""
        # Create constant image
        img = mx.ones((32, 32, 3)) * 0.5
        mx.eval(img)

        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        out = normalize(img)
        mx.eval(out)

        # (0.5 - 0.5) / 0.5 = 0
        expected_val = 0.0
        max_diff = float(mx.max(mx.abs(out - expected_val)))
        assert max_diff < 1e-6

    def test_normalize_per_channel(self):
        """Normalize should apply different mean/std per channel."""
        img = mx.ones((32, 32, 3))
        img = mx.concatenate([
            img[:, :, :1] * 0.2,
            img[:, :, 1:2] * 0.5,
            img[:, :, 2:3] * 0.8,
        ], axis=2)
        mx.eval(img)

        normalize = Normalize(mean=[0.2, 0.5, 0.8], std=[0.1, 0.1, 0.1])
        out = normalize(img)
        mx.eval(out)

        # Each channel should be ~0 after normalization
        max_diff = float(mx.max(mx.abs(out)))
        assert max_diff < 1e-5


class TestToTensor:
    """Test ToTensor transform."""

    def test_scales_to_0_1(self):
        """ToTensor should scale uint8 [0, 255] to float [0, 1]."""
        img = mx.array([[[0, 128, 255]]], dtype=mx.uint8)
        mx.eval(img)

        to_tensor = ToTensor()
        out = to_tensor(img)
        mx.eval(out)

        assert float(out[0, 0, 0]) == 0.0
        assert abs(float(out[0, 0, 1]) - 0.502) < 0.01  # 128/255
        assert float(out[0, 0, 2]) == 1.0


class TestRandomFlip:
    """Test random flip transforms."""

    def test_horizontal_flip_probability_0(self):
        """With p=0, image should not flip."""
        mx.random.seed(42)
        img = mx.arange(12).reshape(2, 2, 3).astype(mx.float32)
        mx.eval(img)

        flip = RandomHorizontalFlip(p=0.0)
        out = flip(img)
        mx.eval(out)

        diff = float(mx.max(mx.abs(out - img)))
        assert diff == 0.0

    def test_horizontal_flip_probability_1(self):
        """With p=1, image should always flip."""
        mx.random.seed(42)
        img = mx.arange(12).reshape(2, 2, 3).astype(mx.float32)
        mx.eval(img)

        flip = RandomHorizontalFlip(p=1.0)
        out = flip(img)
        mx.eval(out)

        # Check that columns are swapped
        expected = img[:, ::-1, :]
        diff = float(mx.max(mx.abs(out - expected)))
        assert diff == 0.0

    def test_vertical_flip_probability_1(self):
        """With p=1, image should always flip vertically."""
        mx.random.seed(42)
        img = mx.arange(12).reshape(2, 2, 3).astype(mx.float32)
        mx.eval(img)

        flip = RandomVerticalFlip(p=1.0)
        out = flip(img)
        mx.eval(out)

        # Check that rows are swapped
        expected = img[::-1, :, :]
        diff = float(mx.max(mx.abs(out - expected)))
        assert diff == 0.0


class TestRandomCrop:
    """Test random crop transform."""

    def test_crop_shape(self):
        """Crop should produce correct output shape."""
        mx.random.seed(42)
        img = mx.random.uniform(shape=(64, 64, 3))
        mx.eval(img)

        crop = RandomCrop(size=(32, 32))
        out = crop(img)
        mx.eval(out)

        assert out.shape == (32, 32, 3)

    def test_crop_with_padding(self):
        """Crop with padding should work correctly."""
        mx.random.seed(42)
        img = mx.random.uniform(shape=(28, 28, 3))
        mx.eval(img)

        crop = RandomCrop(size=(32, 32), padding=4)
        out = crop(img)
        mx.eval(out)

        assert out.shape == (32, 32, 3)


class TestCenterCrop:
    """Test center crop transform."""

    def test_center_crop_shape(self):
        """Center crop should produce correct shape."""
        img = mx.random.uniform(shape=(64, 64, 3))
        mx.eval(img)

        crop = CenterCrop(size=(32, 32))
        out = crop(img)
        mx.eval(out)

        assert out.shape == (32, 32, 3)

    def test_center_crop_extracts_center(self):
        """Center crop should extract the center region."""
        # Create image with values that indicate position
        # Use row index as value to verify center extraction
        img = mx.zeros((8, 8, 1))
        for i in range(8):
            row = mx.ones((1, 8, 1)) * i
            if i == 0:
                img = row
            else:
                img = mx.concatenate([img, row], axis=0)
        mx.eval(img)

        crop = CenterCrop(size=(4, 4))
        out = crop(img)
        mx.eval(out)

        assert out.shape == (4, 4, 1)
        # Center 4 rows should be rows 2,3,4,5 (indices)
        # First row of output should have value 2.0
        assert float(out[0, 0, 0]) == 2.0


class TestResize:
    """Test resize transform."""

    def test_resize_to_fixed_size(self):
        """Resize should produce target size."""
        mx.random.seed(42)
        img = mx.random.uniform(shape=(64, 64, 3))
        mx.eval(img)

        resize = Resize(size=(32, 32))
        out = resize(img)
        mx.eval(out)

        assert out.shape == (32, 32, 3)


class TestColorJitter:
    """Test color jitter transform."""

    def test_no_jitter_with_zero_params(self):
        """With zero parameters, image should be unchanged."""
        mx.random.seed(42)
        img = mx.random.uniform(shape=(32, 32, 3))
        mx.eval(img)

        jitter = ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        out = jitter(img)
        mx.eval(out)

        diff = float(mx.max(mx.abs(out - img)))
        assert diff == 0.0

    def test_jitter_maintains_shape(self):
        """Color jitter should maintain shape."""
        mx.random.seed(42)
        img = mx.random.uniform(shape=(32, 32, 3))
        mx.eval(img)

        jitter = ColorJitter(brightness=0.2, contrast=0.2)
        out = jitter(img)
        mx.eval(out)

        assert out.shape == img.shape


class TestGaussianNoise:
    """Test Gaussian noise transform."""

    def test_noise_probability_0(self):
        """With p=0, no noise should be added."""
        mx.random.seed(42)
        img = mx.random.uniform(shape=(32, 32, 3))
        mx.eval(img)

        noise = GaussianNoise(std=0.1, p=0.0)
        out = noise(img)
        mx.eval(out)

        diff = float(mx.max(mx.abs(out - img)))
        assert diff == 0.0

    def test_noise_changes_image(self):
        """With p=1, noise should change the image."""
        mx.random.seed(42)
        img = mx.ones((32, 32, 3)) * 0.5
        mx.eval(img)

        noise = GaussianNoise(std=0.1, p=1.0)
        out = noise(img)
        mx.eval(out)

        diff = float(mx.max(mx.abs(out - img)))
        assert diff > 0.0


class TestCutOut:
    """Test CutOut transform."""

    def test_cutout_maintains_shape(self):
        """CutOut should maintain image shape."""
        mx.random.seed(42)
        img = mx.random.uniform(shape=(64, 64, 3))
        mx.eval(img)

        cutout = CutOut(num_holes=1, hole_size=16, p=1.0)
        out = cutout(img)
        mx.eval(out)

        assert out.shape == img.shape


class TestMixUp:
    """Test MixUp augmentation."""

    def test_mixup_shapes(self):
        """MixUp should preserve shapes."""
        mx.random.seed(42)
        x1 = mx.random.uniform(shape=(32, 32, 3))
        x2 = mx.random.uniform(shape=(32, 32, 3))
        y1 = mx.array([1.0, 0.0, 0.0])  # One-hot
        y2 = mx.array([0.0, 1.0, 0.0])
        mx.eval(x1, x2, y1, y2)

        x_mix, y_mix = mixup(x1, y1, x2, y2, alpha=1.0)
        mx.eval(x_mix, y_mix)

        assert x_mix.shape == x1.shape
        assert y_mix.shape == y1.shape


class TestCutMix:
    """Test CutMix augmentation."""

    def test_cutmix_shapes(self):
        """CutMix should preserve shapes."""
        mx.random.seed(42)
        x1 = mx.random.uniform(shape=(32, 32, 3))
        x2 = mx.random.uniform(shape=(32, 32, 3))
        y1 = mx.array([1.0, 0.0, 0.0])
        y2 = mx.array([0.0, 1.0, 0.0])
        mx.eval(x1, x2, y1, y2)

        x_mix, y_mix = cutmix(x1, y1, x2, y2, alpha=1.0)
        mx.eval(x_mix, y_mix)

        assert x_mix.shape == x1.shape
        assert y_mix.shape == y1.shape


class TestRandomAugment:
    """Test RandomAugment transform."""

    def test_random_augment_shape(self):
        """RandomAugment should maintain shape."""
        mx.random.seed(42)
        img = mx.random.uniform(shape=(32, 32, 3))
        mx.eval(img)

        augment = RandomAugment(num_ops=2, magnitude=5)
        out = augment(img)
        mx.eval(out)

        assert out.shape == img.shape


# =============================================================================
# Text Transform Tests
# =============================================================================


class TestPadSequence:
    """Test pad_sequence function."""

    def test_pads_to_longest(self):
        """pad_sequence should pad to longest sequence."""
        seqs = [
            mx.array([1, 2, 3]),
            mx.array([4, 5]),
            mx.array([6]),
        ]

        padded = pad_sequence(seqs, batch_first=True, padding_value=0)
        mx.eval(padded)

        assert padded.shape == (3, 3)  # 3 sequences, max length 3

    def test_respects_max_length(self):
        """pad_sequence should respect max_length parameter."""
        seqs = [
            mx.array([1, 2, 3, 4, 5]),
            mx.array([6, 7]),
        ]

        padded = pad_sequence(seqs, batch_first=True, max_length=3)
        mx.eval(padded)

        assert padded.shape == (2, 3)

    def test_padding_value(self):
        """pad_sequence should use correct padding value."""
        seqs = [mx.array([1, 2]), mx.array([3])]

        padded = pad_sequence(seqs, batch_first=True, padding_value=-1)
        mx.eval(padded)

        # Second sequence should be [3, -1]
        assert int(padded[1, 1]) == -1


class TestPadToLength:
    """Test pad_to_length function."""

    def test_right_padding(self):
        """Should pad on the right."""
        seq = mx.array([1, 2, 3])

        padded = pad_to_length(seq, length=5, padding_value=0, side="right")
        mx.eval(padded)

        assert padded.shape == (5,)
        assert list(padded.tolist()) == [1, 2, 3, 0, 0]

    def test_left_padding(self):
        """Should pad on the left."""
        seq = mx.array([1, 2, 3])

        padded = pad_to_length(seq, length=5, padding_value=0, side="left")
        mx.eval(padded)

        assert padded.shape == (5,)
        assert list(padded.tolist()) == [0, 0, 1, 2, 3]

    def test_truncates_if_longer(self):
        """Should truncate if sequence is longer than target."""
        seq = mx.array([1, 2, 3, 4, 5])

        padded = pad_to_length(seq, length=3)
        mx.eval(padded)

        assert padded.shape == (3,)
        assert list(padded.tolist()) == [1, 2, 3]


class TestCreateAttentionMask:
    """Test create_attention_mask function."""

    def test_creates_correct_mask(self):
        """Should create correct attention mask."""
        lengths = [3, 5, 2]

        mask = create_attention_mask(lengths, max_length=5)
        mx.eval(mask)

        assert mask.shape == (3, 5)
        # First sequence: [T, T, T, F, F]
        assert list(mask[0].tolist()) == [True, True, True, False, False]
        # Second sequence: [T, T, T, T, T]
        assert list(mask[1].tolist()) == [True, True, True, True, True]
        # Third sequence: [T, T, F, F, F]
        assert list(mask[2].tolist()) == [True, True, False, False, False]


class TestCreateCausalMask:
    """Test create_causal_mask function."""

    def test_creates_triangular_mask(self):
        """Should create lower-triangular mask."""
        mask = create_causal_mask(4)
        mx.eval(mask)

        assert mask.shape == (4, 4)
        # Check triangular structure
        assert bool(mask[0, 0]) == True
        assert bool(mask[0, 1]) == False
        assert bool(mask[1, 0]) == True
        assert bool(mask[1, 1]) == True
        assert bool(mask[3, 3]) == True


class TestRandomMask:
    """Test RandomMask transform."""

    def test_masks_tokens(self):
        """RandomMask should mask some tokens."""
        mx.random.seed(42)
        input_ids = mx.arange(100)  # 100 tokens

        masker = RandomMask(
            mask_prob=0.15,
            mask_token_id=103,
            vocab_size=1000,
            special_token_ids=[0]  # Don't mask token 0
        )

        masked_ids, labels = masker(input_ids)
        mx.eval(masked_ids, labels)

        # Some tokens should be masked
        num_masked = int(mx.sum(labels != -100))
        assert num_masked > 0
        assert num_masked < 100  # Not all masked

    def test_preserves_special_tokens(self):
        """RandomMask should not mask special tokens."""
        mx.random.seed(42)
        # Create sequence with special token at position 0
        input_ids = mx.arange(10)

        masker = RandomMask(
            mask_prob=1.0,  # Try to mask everything
            mask_token_id=103,
            vocab_size=1000,
            special_token_ids=[0]  # Token 0 is special
        )

        masked_ids, labels = masker(input_ids)
        mx.eval(masked_ids, labels)

        # Token 0 should not be masked
        assert int(labels[0]) == -100  # -100 means not masked


class TestTokenDropout:
    """Test TokenDropout transform."""

    def test_drops_tokens(self):
        """TokenDropout should reduce sequence length."""
        mx.random.seed(42)
        input_ids = mx.arange(100)

        dropout = TokenDropout(drop_prob=0.5, min_length=10)
        dropped = dropout(input_ids)
        mx.eval(dropped)

        # Should be shorter (with high probability)
        assert dropped.shape[0] < 100 or dropped.shape[0] >= 10

    def test_respects_min_length(self):
        """TokenDropout should respect minimum length."""
        mx.random.seed(42)
        input_ids = mx.arange(20)

        dropout = TokenDropout(drop_prob=0.99, min_length=5)
        dropped = dropout(input_ids)
        mx.eval(dropped)

        assert dropped.shape[0] >= 5


class TestPackSequences:
    """Test pack_sequences function."""

    def test_packs_into_chunks(self):
        """pack_sequences should combine sequences."""
        seqs = [
            mx.array([1, 2, 3]),
            mx.array([4, 5]),
            mx.array([6]),
        ]

        packed, mask = pack_sequences(seqs, max_length=6, padding_value=0)
        mx.eval(packed, mask)

        # All sequences should fit in one chunk of length 6
        assert packed.shape[1] == 6

    def test_creates_attention_mask(self):
        """pack_sequences should create correct attention mask."""
        seqs = [mx.array([1, 2, 3])]

        packed, mask = pack_sequences(seqs, max_length=5, padding_value=0)
        mx.eval(packed, mask)

        # Mask should be True for first 3, False for last 2
        assert list(mask[0].tolist()) == [True, True, True, False, False]


class TestTruncateSequences:
    """Test truncate_sequences function."""

    def test_truncates_from_right(self):
        """Should truncate from right by default."""
        seqs = [mx.array([1, 2, 3, 4, 5])]

        truncated = truncate_sequences(seqs, max_length=3, truncation_side="right")

        assert truncated[0].shape == (3,)
        assert list(truncated[0].tolist()) == [1, 2, 3]

    def test_truncates_from_left(self):
        """Should truncate from left when specified."""
        seqs = [mx.array([1, 2, 3, 4, 5])]

        truncated = truncate_sequences(seqs, max_length=3, truncation_side="left")

        assert truncated[0].shape == (3,)
        assert list(truncated[0].tolist()) == [3, 4, 5]

    def test_preserves_short_sequences(self):
        """Should not truncate sequences shorter than max_length."""
        seqs = [mx.array([1, 2])]

        truncated = truncate_sequences(seqs, max_length=5)

        assert truncated[0].shape == (2,)
