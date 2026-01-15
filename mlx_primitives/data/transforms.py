"""Vision transforms for MLX.

This module provides image transforms and augmentations:
- Compose: Chain multiple transforms
- Normalize: Channel-wise normalization
- RandomCrop: Random cropping
- RandomHorizontalFlip: Random horizontal flip
- RandomVerticalFlip: Random vertical flip
- RandomRotation: Random rotation
- ColorJitter: Random color adjustments
- MixUp: MixUp augmentation
- CutMix: CutMix augmentation
- CutOut: Random erasing
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import mlx.core as mx


class Compose:
    """Compose multiple transforms together.

    Args:
        transforms: List of transforms to apply in order.

    Example:
        >>> transform = Compose([
        ...     RandomHorizontalFlip(),
        ...     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ... ])
        >>> augmented = transform(image)
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: mx.array) -> mx.array:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class Normalize:
    """Normalize a tensor with mean and standard deviation.

    Works for both single images (H, W, C) or (C, H, W) and batches (N, H, W, C).

    Args:
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
        channel_first: If True, expects (C, H, W) format (default: False).

    Example:
        >>> # ImageNet normalization
        >>> normalize = Normalize(
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225]
        ... )
        >>> normalized = normalize(image)
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        channel_first: bool = False,
    ):
        self.mean = mx.array(mean, dtype=mx.float32)
        self.std = mx.array(std, dtype=mx.float32)
        self.channel_first = channel_first

    def __call__(self, x: mx.array) -> mx.array:
        if self.channel_first:
            # (C, H, W) or (N, C, H, W)
            mean = self.mean.reshape(-1, 1, 1)
            std = self.std.reshape(-1, 1, 1)
        else:
            # (H, W, C) or (N, H, W, C)
            mean = self.mean
            std = self.std

        return (x - mean) / std

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean.tolist()}, std={self.std.tolist()})"


class ToTensor:
    """Convert image to MLX tensor and scale to [0, 1].

    Assumes input is in uint8 format [0, 255].

    Args:
        dtype: Output dtype (default: float32).

    Example:
        >>> to_tensor = ToTensor()
        >>> tensor = to_tensor(uint8_image)  # [0, 1] float32
    """

    def __init__(self, dtype: mx.Dtype = mx.float32):
        self.dtype = dtype

    def __call__(self, x: mx.array) -> mx.array:
        return x.astype(self.dtype) / 255.0

    def __repr__(self) -> str:
        return f"ToTensor(dtype={self.dtype})"


class RandomHorizontalFlip:
    """Randomly flip image horizontally.

    Args:
        p: Probability of flip (default: 0.5).

    Example:
        >>> flip = RandomHorizontalFlip(p=0.5)
        >>> flipped = flip(image)
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        if mx.random.uniform() < self.p:
            # Flip along width axis (axis=-2 for HWC)
            # MLX doesn't have flip, so we use indexing
            return x[..., ::-1, :]
        return x

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """Randomly flip image vertically.

    Args:
        p: Probability of flip (default: 0.5).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        if mx.random.uniform() < self.p:
            # Flip along height axis
            return x[..., ::-1, :, :]
        return x

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class RandomRotation:
    """Randomly rotate image by an angle.

    Args:
        degrees: Range of degrees to select from. If single number, range is (-degrees, +degrees).
        expand: If True, expands output to fit rotated image (default: False).
        center: Center of rotation. If None, uses image center.
        fill: Fill value for areas outside the rotated image (default: 0).
        p: Probability of applying rotation (default: 1.0).

    Example:
        >>> rotation = RandomRotation(degrees=15)
        >>> rotated = rotation(image)
    """

    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        expand: bool = False,
        center: Optional[Tuple[float, float]] = None,
        fill: float = 0.0,
        p: float = 1.0,
    ):
        if isinstance(degrees, (int, float)):
            self.degrees = (-float(degrees), float(degrees))
        else:
            self.degrees = (float(degrees[0]), float(degrees[1]))
        self.expand = expand
        self.center = center
        self.fill = fill
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        if mx.random.uniform() >= self.p:
            return x

        # Sample random angle
        angle = float(mx.random.uniform(self.degrees[0], self.degrees[1]))
        angle_rad = math.radians(angle)

        return self._rotate(x, angle_rad)

    def _rotate(self, x: mx.array, angle: float) -> mx.array:
        """Rotate image by angle (in radians) using bilinear interpolation."""
        h, w = x.shape[-3], x.shape[-2]

        # Compute rotation center
        if self.center is not None:
            cx, cy = self.center
        else:
            cx, cy = w / 2, h / 2

        # Compute rotation matrix components
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Create coordinate grids for output
        y_out = mx.arange(h, dtype=mx.float32) - cy
        x_out = mx.arange(w, dtype=mx.float32) - cx

        # Create meshgrid
        yy, xx = mx.meshgrid(y_out, x_out, indexing='ij')

        # Apply inverse rotation to get source coordinates
        x_src = cos_a * xx + sin_a * yy + cx
        y_src = -sin_a * xx + cos_a * yy + cy

        # Bilinear interpolation
        x0 = mx.floor(x_src).astype(mx.int32)
        x1 = x0 + 1
        y0 = mx.floor(y_src).astype(mx.int32)
        y1 = y0 + 1

        # Compute interpolation weights
        wa = (x1.astype(mx.float32) - x_src) * (y1.astype(mx.float32) - y_src)
        wb = (x1.astype(mx.float32) - x_src) * (y_src - y0.astype(mx.float32))
        wc = (x_src - x0.astype(mx.float32)) * (y1.astype(mx.float32) - y_src)
        wd = (x_src - x0.astype(mx.float32)) * (y_src - y0.astype(mx.float32))

        # Clip coordinates to valid range
        x0_clip = mx.clip(x0, 0, w - 1)
        x1_clip = mx.clip(x1, 0, w - 1)
        y0_clip = mx.clip(y0, 0, h - 1)
        y1_clip = mx.clip(y1, 0, h - 1)

        # Gather pixel values (HWC format)
        # Handle both single image and batch
        if x.ndim == 3:
            Ia = x[y0_clip, x0_clip, :]
            Ib = x[y1_clip, x0_clip, :]
            Ic = x[y0_clip, x1_clip, :]
            Id = x[y1_clip, x1_clip, :]
        else:
            # Batch case
            Ia = x[..., y0_clip, x0_clip, :]
            Ib = x[..., y1_clip, x0_clip, :]
            Ic = x[..., y0_clip, x1_clip, :]
            Id = x[..., y1_clip, x1_clip, :]

        # Interpolate
        wa = wa[..., None]
        wb = wb[..., None]
        wc = wc[..., None]
        wd = wd[..., None]

        result = wa * Ia + wb * Ib + wc * Ic + wd * Id

        # Create mask for out-of-bounds pixels
        valid_mask = (x_src >= 0) & (x_src < w) & (y_src >= 0) & (y_src < h)
        valid_mask = valid_mask[..., None]

        # Fill out-of-bounds with fill value
        result = mx.where(valid_mask, result, self.fill)

        return result

    def __repr__(self) -> str:
        return f"RandomRotation(degrees={self.degrees}, expand={self.expand})"


class RandomCrop:
    """Randomly crop image to given size.

    Args:
        size: Target size as (H, W) or single int for square.
        padding: Optional padding to apply before cropping.
        pad_value: Value for padding (default: 0).

    Example:
        >>> crop = RandomCrop(size=(224, 224), padding=4)
        >>> cropped = crop(image)
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: Optional[int] = None,
        pad_value: float = 0.0,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
        self.pad_value = pad_value

    def __call__(self, x: mx.array) -> mx.array:
        # Apply padding if specified
        if self.padding is not None:
            p = self.padding
            # Assuming HWC format
            x = mx.pad(
                x,
                [(p, p), (p, p), (0, 0)],
                constant_values=self.pad_value,
            )

        h, w = x.shape[-3], x.shape[-2]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(
                f"Input size ({h}, {w}) smaller than crop size {self.size}"
            )

        if h == th and w == tw:
            return x

        # Random crop position
        top = int(mx.random.randint(0, h - th + 1))
        left = int(mx.random.randint(0, w - tw + 1))

        return x[..., top : top + th, left : left + tw, :]

    def __repr__(self) -> str:
        return f"RandomCrop(size={self.size}, padding={self.padding})"


class CenterCrop:
    """Crop image at center to given size.

    Args:
        size: Target size as (H, W) or single int for square.
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, x: mx.array) -> mx.array:
        h, w = x.shape[-3], x.shape[-2]
        th, tw = self.size

        top = (h - th) // 2
        left = (w - tw) // 2

        return x[..., top : top + th, left : left + tw, :]

    def __repr__(self) -> str:
        return f"CenterCrop(size={self.size})"


class Resize:
    """Resize image to given size using bilinear interpolation.

    Args:
        size: Target size as (H, W) or single int (shorter edge).
        interpolation: Interpolation mode (default: 'bilinear').
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = "bilinear",
    ):
        if isinstance(size, int):
            self.size = size
            self.fixed_size = False
        else:
            self.size = size
            self.fixed_size = True

    def __call__(self, x: mx.array) -> mx.array:
        h, w = x.shape[-3], x.shape[-2]

        if self.fixed_size:
            new_h, new_w = self.size
        else:
            # Resize shorter edge to self.size
            if h < w:
                new_h = self.size
                new_w = int(w * self.size / h)
            else:
                new_w = self.size
                new_h = int(h * self.size / w)

        # Simple nearest-neighbor resize (MLX doesn't have built-in resize)
        # For production, consider implementing proper bilinear interpolation
        scale_h = h / new_h
        scale_w = w / new_w

        # Create coordinate grids
        y_coords = mx.floor(mx.arange(new_h) * scale_h).astype(mx.int32)
        x_coords = mx.floor(mx.arange(new_w) * scale_w).astype(mx.int32)

        y_coords = mx.clip(y_coords, 0, h - 1)
        x_coords = mx.clip(x_coords, 0, w - 1)

        # Index into image
        return x[..., y_coords[:, None], x_coords[None, :], :]

    def __repr__(self) -> str:
        return f"Resize(size={self.size})"


class ColorJitter:
    """Randomly adjust brightness, contrast, saturation, and hue.

    Args:
        brightness: Brightness adjustment factor (default: 0).
        contrast: Contrast adjustment factor (default: 0).
        saturation: Saturation adjustment factor (default: 0).
        hue: Hue adjustment factor (default: 0).

    Each factor should be a float in [0, 1] indicating the range of adjustment.

    Example:
        >>> jitter = ColorJitter(brightness=0.2, contrast=0.2)
        >>> augmented = jitter(image)
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x: mx.array) -> mx.array:
        # Apply transforms in random order
        transforms = []

        if self.brightness > 0:
            transforms.append(self._adjust_brightness)
        if self.contrast > 0:
            transforms.append(self._adjust_contrast)
        if self.saturation > 0:
            transforms.append(self._adjust_saturation)

        # Shuffle transform order
        perm = mx.random.permutation(len(transforms)).tolist()
        for i in perm:
            x = transforms[int(i)](x)

        return x

    def _adjust_brightness(self, x: mx.array) -> mx.array:
        factor = 1.0 + float(mx.random.uniform(-self.brightness, self.brightness))
        return mx.clip(x * factor, 0, 1)

    def _adjust_contrast(self, x: mx.array) -> mx.array:
        factor = 1.0 + float(mx.random.uniform(-self.contrast, self.contrast))
        mean = mx.mean(x, axis=(-3, -2), keepdims=True)
        return mx.clip((x - mean) * factor + mean, 0, 1)

    def _adjust_saturation(self, x: mx.array) -> mx.array:
        factor = 1.0 + float(mx.random.uniform(-self.saturation, self.saturation))
        # Convert to grayscale
        gray = mx.mean(x, axis=-1, keepdims=True)
        return mx.clip(gray + (x - gray) * factor, 0, 1)

    def __repr__(self) -> str:
        return (
            f"ColorJitter(brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue})"
        )


class GaussianNoise:
    """Add Gaussian noise to image.

    Args:
        mean: Mean of noise distribution (default: 0.0).
        std: Standard deviation of noise (default: 0.1).
        p: Probability of applying noise (default: 0.5).
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1, p: float = 0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        if mx.random.uniform() < self.p:
            noise = mx.random.normal(x.shape) * self.std + self.mean
            return mx.clip(x + noise, 0, 1)
        return x

    def __repr__(self) -> str:
        return f"GaussianNoise(mean={self.mean}, std={self.std}, p={self.p})"


class CutOut:
    """Randomly mask out rectangular regions (CutOut augmentation).

    Args:
        num_holes: Number of holes to cut out.
        hole_size: Size of each hole as (H, W) or single int.
        fill_value: Value to fill holes with (default: 0).
        p: Probability of applying (default: 0.5).

    Reference:
        "Improved Regularization of Convolutional Neural Networks with Cutout"
        https://arxiv.org/abs/1708.04552
    """

    def __init__(
        self,
        num_holes: int = 1,
        hole_size: Union[int, Tuple[int, int]] = 16,
        fill_value: float = 0.0,
        p: float = 0.5,
    ):
        self.num_holes = num_holes
        if isinstance(hole_size, int):
            self.hole_size = (hole_size, hole_size)
        else:
            self.hole_size = hole_size
        self.fill_value = fill_value
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        if mx.random.uniform() >= self.p:
            return x

        h, w = x.shape[-3], x.shape[-2]
        hh, hw = self.hole_size

        # Create mask
        mask = mx.ones_like(x)

        for _ in range(self.num_holes):
            # Random hole center
            cy = int(mx.random.randint(0, h))
            cx = int(mx.random.randint(0, w))

            # Compute hole bounds
            y1 = max(0, cy - hh // 2)
            y2 = min(h, cy + hh // 2)
            x1 = max(0, cx - hw // 2)
            x2 = min(w, cx + hw // 2)

            # Zero out the region
            # Note: MLX doesn't support direct assignment, so we use where
            y_mask = (mx.arange(h) >= y1) & (mx.arange(h) < y2)
            x_mask = (mx.arange(w) >= x1) & (mx.arange(w) < x2)
            hole_mask = y_mask[:, None] & x_mask[None, :]
            hole_mask = hole_mask[..., None]  # Add channel dim

            mask = mx.where(hole_mask, self.fill_value, mask)

        return x * mask + (1 - mask) * self.fill_value

    def __repr__(self) -> str:
        return f"CutOut(num_holes={self.num_holes}, hole_size={self.hole_size})"


def mixup(
    x1: mx.array,
    y1: mx.array,
    x2: mx.array,
    y2: mx.array,
    alpha: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """Apply MixUp augmentation to a pair of samples.

    MixUp creates virtual training examples by linearly interpolating
    both inputs and labels.

    Args:
        x1: First input tensor.
        y1: First label tensor (one-hot or soft labels).
        x2: Second input tensor.
        y2: Second label tensor.
        alpha: Beta distribution parameter (default: 1.0).

    Returns:
        Tuple of (mixed_input, mixed_label).

    Reference:
        "mixup: Beyond Empirical Risk Minimization"
        https://arxiv.org/abs/1710.09412

    Example:
        >>> x_mixed, y_mixed = mixup(x1, y1, x2, y2, alpha=0.2)
    """
    # Sample mixing coefficient from Beta distribution
    # Using the property that Beta(a,a) is symmetric
    lam = float(mx.random.uniform())
    if alpha > 0:
        # Approximate Beta(alpha, alpha) using Gamma distributions
        g1 = mx.random.uniform() ** (1.0 / alpha)
        g2 = mx.random.uniform() ** (1.0 / alpha)
        lam = float(g1 / (g1 + g2))

    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2

    return mixed_x, mixed_y


def cutmix(
    x1: mx.array,
    y1: mx.array,
    x2: mx.array,
    y2: mx.array,
    alpha: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """Apply CutMix augmentation to a pair of samples.

    CutMix cuts and pastes patches between training images,
    mixing labels proportionally to the area of the patches.

    Args:
        x1: First input tensor (H, W, C).
        y1: First label tensor.
        x2: Second input tensor.
        y2: Second label tensor.
        alpha: Beta distribution parameter (default: 1.0).

    Returns:
        Tuple of (mixed_input, mixed_label).

    Reference:
        "CutMix: Regularization Strategy to Train Strong Classifiers"
        https://arxiv.org/abs/1905.04899
    """
    h, w = x1.shape[-3], x1.shape[-2]

    # Sample lambda from Beta distribution
    lam = float(mx.random.uniform())
    if alpha > 0:
        g1 = mx.random.uniform() ** (1.0 / alpha)
        g2 = mx.random.uniform() ** (1.0 / alpha)
        lam = float(g1 / (g1 + g2))

    # Compute cut region
    cut_ratio = math.sqrt(1 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    # Random center
    cy = int(mx.random.randint(0, h))
    cx = int(mx.random.randint(0, w))

    # Bounding box
    y1_box = max(0, cy - cut_h // 2)
    y2_box = min(h, cy + cut_h // 2)
    x1_box = max(0, cx - cut_w // 2)
    x2_box = min(w, cx + cut_w // 2)

    # Create mask for the cut region
    y_mask = (mx.arange(h) >= y1_box) & (mx.arange(h) < y2_box)
    x_mask = (mx.arange(w) >= x1_box) & (mx.arange(w) < x2_box)
    mask = y_mask[:, None] & x_mask[None, :]
    mask = mask[..., None]  # Add channel dim

    # Mix images
    mixed_x = mx.where(mask, x2, x1)

    # Adjust lambda based on actual cut area
    actual_lam = 1 - ((y2_box - y1_box) * (x2_box - x1_box)) / (h * w)

    # Mix labels
    mixed_y = actual_lam * y1 + (1 - actual_lam) * y2

    return mixed_x, mixed_y


class MixUpTransform:
    """MixUp as a batch transform.

    Applies MixUp within a batch by shuffling and mixing pairs.

    Args:
        alpha: Beta distribution parameter (default: 1.0).
        p: Probability of applying (default: 1.0).
    """

    def __init__(self, alpha: float = 1.0, p: float = 1.0):
        self.alpha = alpha
        self.p = p

    def __call__(
        self, x: mx.array, y: mx.array
    ) -> Tuple[mx.array, mx.array]:
        if mx.random.uniform() >= self.p:
            return x, y

        batch_size = x.shape[0]

        # Shuffle indices
        indices = mx.random.permutation(batch_size)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Sample lambda
        lam = float(mx.random.uniform())
        if self.alpha > 0:
            g1 = mx.random.uniform() ** (1.0 / self.alpha)
            g2 = mx.random.uniform() ** (1.0 / self.alpha)
            lam = float(g1 / (g1 + g2))

        mixed_x = lam * x + (1 - lam) * x_shuffled
        mixed_y = lam * y + (1 - lam) * y_shuffled

        return mixed_x, mixed_y


class CutMixTransform:
    """CutMix as a batch transform.

    Args:
        alpha: Beta distribution parameter (default: 1.0).
        p: Probability of applying (default: 1.0).
    """

    def __init__(self, alpha: float = 1.0, p: float = 1.0):
        self.alpha = alpha
        self.p = p

    def __call__(
        self, x: mx.array, y: mx.array
    ) -> Tuple[mx.array, mx.array]:
        if mx.random.uniform() >= self.p:
            return x, y

        batch_size = x.shape[0]
        h, w = x.shape[1], x.shape[2]

        # Shuffle
        indices = mx.random.permutation(batch_size)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Sample lambda
        lam = float(mx.random.uniform())
        if self.alpha > 0:
            g1 = mx.random.uniform() ** (1.0 / self.alpha)
            g2 = mx.random.uniform() ** (1.0 / self.alpha)
            lam = float(g1 / (g1 + g2))

        # Compute cut region
        cut_ratio = math.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        cy = int(mx.random.randint(0, h))
        cx = int(mx.random.randint(0, w))

        y1_box = max(0, cy - cut_h // 2)
        y2_box = min(h, cy + cut_h // 2)
        x1_box = max(0, cx - cut_w // 2)
        x2_box = min(w, cx + cut_w // 2)

        # Create mask
        y_mask = (mx.arange(h) >= y1_box) & (mx.arange(h) < y2_box)
        x_mask = (mx.arange(w) >= x1_box) & (mx.arange(w) < x2_box)
        mask = y_mask[:, None] & x_mask[None, :]
        mask = mask[None, :, :, None]  # Add batch and channel dims

        mixed_x = mx.where(mask, x_shuffled, x)

        actual_lam = 1 - ((y2_box - y1_box) * (x2_box - x1_box)) / (h * w)
        mixed_y = actual_lam * y + (1 - actual_lam) * y_shuffled

        return mixed_x, mixed_y


class RandomAugment:
    """RandAugment: Practical automated data augmentation.

    Applies N random augmentation operations with magnitude M to each image.
    This is a simplified version of AutoAugment that works well in practice.

    Args:
        num_ops: Number of augmentation operations to apply (default: 2).
        magnitude: Magnitude of augmentation operations, 0-10 scale (default: 9).
        magnitude_std: Standard deviation for magnitude randomization (default: 0).
        num_magnitude_bins: Number of magnitude bins (default: 31).
        interpolation: Interpolation mode for geometric transforms (default: 'bilinear').
        fill: Fill value for areas outside the image (default: 0).

    Reference:
        "RandAugment: Practical automated data augmentation with a reduced search space"
        https://arxiv.org/abs/1909.13719

    Example:
        >>> augment = RandomAugment(num_ops=2, magnitude=9)
        >>> augmented = augment(image)
    """

    # Available augmentation operations
    AUGMENT_OPS = [
        'identity',
        'auto_contrast',
        'equalize',
        'rotate',
        'solarize',
        'color',
        'posterize',
        'contrast',
        'brightness',
        'sharpness',
        'shear_x',
        'shear_y',
        'translate_x',
        'translate_y',
    ]

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        magnitude_std: float = 0.0,
        num_magnitude_bins: int = 31,
        interpolation: str = 'bilinear',
        fill: float = 0.0,
    ):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.magnitude_std = magnitude_std
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, x: mx.array) -> mx.array:
        # Select random operations
        op_indices = mx.random.randint(0, len(self.AUGMENT_OPS), shape=(self.num_ops,))

        for i in range(self.num_ops):
            op_idx = int(op_indices[i])
            op_name = self.AUGMENT_OPS[op_idx]

            # Sample magnitude with optional randomization
            mag = self.magnitude
            if self.magnitude_std > 0:
                mag = float(mx.clip(
                    mx.random.normal() * self.magnitude_std + mag,
                    0,
                    self.num_magnitude_bins - 1
                ))

            x = self._apply_op(x, op_name, mag)

        return x

    def _apply_op(self, x: mx.array, op_name: str, magnitude: float) -> mx.array:
        """Apply a single augmentation operation."""
        # Normalize magnitude to [0, 1] range
        mag = magnitude / (self.num_magnitude_bins - 1)

        if op_name == 'identity':
            return x

        elif op_name == 'auto_contrast':
            return self._auto_contrast(x)

        elif op_name == 'equalize':
            return self._equalize(x)

        elif op_name == 'rotate':
            # Max rotation of 30 degrees
            degrees = mag * 30
            if mx.random.uniform() < 0.5:
                degrees = -degrees
            return self._rotate(x, degrees)

        elif op_name == 'solarize':
            # Threshold decreases with magnitude
            threshold = 1.0 - mag
            return mx.where(x >= threshold, 1.0 - x, x)

        elif op_name == 'color':
            # Adjust saturation
            factor = mag * 1.8 + 0.1  # Range [0.1, 1.9]
            gray = mx.mean(x, axis=-1, keepdims=True)
            return mx.clip(gray + (x - gray) * factor, 0, 1)

        elif op_name == 'posterize':
            # Reduce bits
            bits = int(4 - mag * 4) + 4  # 4-8 bits
            bits = max(1, min(8, bits))
            scale = 2 ** (8 - bits)
            return mx.floor(x * 255 / scale) * scale / 255

        elif op_name == 'contrast':
            factor = mag * 1.8 + 0.1
            mean = mx.mean(x, axis=(-3, -2), keepdims=True)
            return mx.clip((x - mean) * factor + mean, 0, 1)

        elif op_name == 'brightness':
            factor = mag * 1.8 + 0.1
            if mx.random.uniform() < 0.5:
                factor = 1.0 / factor
            return mx.clip(x * factor, 0, 1)

        elif op_name == 'sharpness':
            # Apply simple sharpening kernel
            return self._sharpen(x, mag)

        elif op_name == 'shear_x':
            shear = mag * 0.3  # Max 0.3 shear
            if mx.random.uniform() < 0.5:
                shear = -shear
            return self._shear_x(x, shear)

        elif op_name == 'shear_y':
            shear = mag * 0.3
            if mx.random.uniform() < 0.5:
                shear = -shear
            return self._shear_y(x, shear)

        elif op_name == 'translate_x':
            # Max translation of 0.45 * width
            pixels = int(mag * 0.45 * x.shape[-2])
            if mx.random.uniform() < 0.5:
                pixels = -pixels
            return self._translate_x(x, pixels)

        elif op_name == 'translate_y':
            pixels = int(mag * 0.45 * x.shape[-3])
            if mx.random.uniform() < 0.5:
                pixels = -pixels
            return self._translate_y(x, pixels)

        return x

    def _auto_contrast(self, x: mx.array) -> mx.array:
        """Maximize contrast by stretching pixel values."""
        for c in range(x.shape[-1]):
            channel = x[..., c]
            lo = mx.min(channel)
            hi = mx.max(channel)
            if hi > lo:
                x = mx.concatenate([
                    x[..., :c],
                    ((channel - lo) / (hi - lo))[..., None],
                    x[..., c+1:]
                ], axis=-1)
        return x

    def _equalize(self, x: mx.array) -> mx.array:
        """Histogram equalization (simplified)."""
        # Simple approximation using cumulative distribution
        return x  # Skip for now as MLX doesn't have histogram ops

    def _rotate(self, x: mx.array, degrees: float) -> mx.array:
        """Rotate image by degrees."""
        h, w = x.shape[-3], x.shape[-2]
        cx, cy = w / 2, h / 2
        angle = math.radians(degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        y_out = mx.arange(h, dtype=mx.float32) - cy
        x_out = mx.arange(w, dtype=mx.float32) - cx
        yy, xx = mx.meshgrid(y_out, x_out, indexing='ij')

        x_src = cos_a * xx + sin_a * yy + cx
        y_src = -sin_a * xx + cos_a * yy + cy

        # Nearest neighbor for simplicity
        x_src = mx.round(x_src).astype(mx.int32)
        y_src = mx.round(y_src).astype(mx.int32)

        valid = (x_src >= 0) & (x_src < w) & (y_src >= 0) & (y_src < h)
        x_src = mx.clip(x_src, 0, w - 1)
        y_src = mx.clip(y_src, 0, h - 1)

        if x.ndim == 3:
            result = x[y_src, x_src, :]
        else:
            result = x[..., y_src, x_src, :]

        return mx.where(valid[..., None], result, self.fill)

    def _sharpen(self, x: mx.array, magnitude: float) -> mx.array:
        """Apply sharpening."""
        # Simple unsharp mask approximation
        # Compute local average
        kernel_size = 3
        pad = kernel_size // 2

        padded = mx.pad(x, [(pad, pad), (pad, pad), (0, 0)], constant_values=0)

        # Simple box blur
        blur = mx.zeros_like(x)
        h, w = x.shape[-3], x.shape[-2]
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                blur = blur + padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w, :]
        blur = blur / (kernel_size * kernel_size)

        # Sharpen: original + magnitude * (original - blur)
        factor = magnitude * 2
        return mx.clip(x + factor * (x - blur), 0, 1)

    def _shear_x(self, x: mx.array, shear: float) -> mx.array:
        """Shear image along x-axis."""
        h, w = x.shape[-3], x.shape[-2]
        cy = h / 2

        y_coords = mx.arange(h, dtype=mx.float32)
        x_coords = mx.arange(w, dtype=mx.float32)

        yy, xx = mx.meshgrid(y_coords, x_coords, indexing='ij')
        x_src = xx - shear * (yy - cy)
        y_src = yy

        x_src_int = mx.round(x_src).astype(mx.int32)
        y_src_int = y_src.astype(mx.int32)

        valid = (x_src_int >= 0) & (x_src_int < w)
        x_src_int = mx.clip(x_src_int, 0, w - 1)

        if x.ndim == 3:
            result = x[y_src_int, x_src_int, :]
        else:
            result = x[..., y_src_int, x_src_int, :]

        return mx.where(valid[..., None], result, self.fill)

    def _shear_y(self, x: mx.array, shear: float) -> mx.array:
        """Shear image along y-axis."""
        h, w = x.shape[-3], x.shape[-2]
        cx = w / 2

        y_coords = mx.arange(h, dtype=mx.float32)
        x_coords = mx.arange(w, dtype=mx.float32)

        yy, xx = mx.meshgrid(y_coords, x_coords, indexing='ij')
        x_src = xx
        y_src = yy - shear * (xx - cx)

        x_src_int = x_src.astype(mx.int32)
        y_src_int = mx.round(y_src).astype(mx.int32)

        valid = (y_src_int >= 0) & (y_src_int < h)
        y_src_int = mx.clip(y_src_int, 0, h - 1)

        if x.ndim == 3:
            result = x[y_src_int, x_src_int, :]
        else:
            result = x[..., y_src_int, x_src_int, :]

        return mx.where(valid[..., None], result, self.fill)

    def _translate_x(self, x: mx.array, pixels: int) -> mx.array:
        """Translate image along x-axis."""
        if pixels == 0:
            return x

        h, w = x.shape[-3], x.shape[-2]
        result = mx.full(x.shape, self.fill, dtype=x.dtype)

        if pixels > 0:
            if x.ndim == 3:
                result[:, pixels:, :] = x[:, :-pixels, :]
            else:
                result[..., :, pixels:, :] = x[..., :, :-pixels, :]
        else:
            if x.ndim == 3:
                result[:, :pixels, :] = x[:, -pixels:, :]
            else:
                result[..., :, :pixels, :] = x[..., :, -pixels:, :]

        return result

    def _translate_y(self, x: mx.array, pixels: int) -> mx.array:
        """Translate image along y-axis."""
        if pixels == 0:
            return x

        h, w = x.shape[-3], x.shape[-2]
        result = mx.full(x.shape, self.fill, dtype=x.dtype)

        if pixels > 0:
            if x.ndim == 3:
                result[pixels:, :, :] = x[:-pixels, :, :]
            else:
                result[..., pixels:, :, :] = x[..., :-pixels, :, :]
        else:
            if x.ndim == 3:
                result[:pixels, :, :] = x[-pixels:, :, :]
            else:
                result[..., :pixels, :, :] = x[..., -pixels:, :, :]

        return result

    def __repr__(self) -> str:
        return f"RandomAugment(num_ops={self.num_ops}, magnitude={self.magnitude})"


class TrivialAugmentWide:
    """TrivialAugment Wide - simpler single-operation augmentation.

    Applies a single randomly selected augmentation with random magnitude.
    Often matches or exceeds RandAugment performance.

    Args:
        num_magnitude_bins: Number of magnitude bins (default: 31).
        interpolation: Interpolation mode (default: 'bilinear').
        fill: Fill value for areas outside the image (default: 0).

    Reference:
        "TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation"
        https://arxiv.org/abs/2103.10158

    Example:
        >>> augment = TrivialAugmentWide()
        >>> augmented = augment(image)
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: str = 'bilinear',
        fill: float = 0.0,
    ):
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        # Reuse RandomAugment operations
        self._rand_augment = RandomAugment(
            num_ops=1,
            magnitude=0,  # Will be overridden
            num_magnitude_bins=num_magnitude_bins,
            interpolation=interpolation,
            fill=fill,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Select random operation
        op_idx = int(mx.random.randint(0, len(RandomAugment.AUGMENT_OPS)))
        op_name = RandomAugment.AUGMENT_OPS[op_idx]

        # Select random magnitude
        magnitude = float(mx.random.randint(0, self.num_magnitude_bins))

        return self._rand_augment._apply_op(x, op_name, magnitude)

    def __repr__(self) -> str:
        return f"TrivialAugmentWide(num_magnitude_bins={self.num_magnitude_bins})"