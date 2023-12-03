from typing import Any, List, Tuple, Optional, Union
from random import choices
from matplotlib import colormaps
import numpy as np
from PIL import Image
from skimage import color
from skimage.measure import label

import torch
from torch import Tensor, logical_not
from torch.nn.functional import one_hot

from torchvision.transforms import RandomCrop
from torchvision.transforms.autoaugment import TrivialAugmentWide as TAWide, _apply_op
from torchvision.transforms.functional import (
    InterpolationMode,
    resize,
    rotate,
    crop,
    get_dimensions,
    to_tensor,
    normalize,
)


class GrayToRGB(torch.nn.Module):
    def forward(self, sample) -> Any:
        res = sample
        if len(sample.shape) < 3:
            res = np.expand_dims(res, axis=2)
            res = np.concatenate((res, res, res), axis=2)
        return res


class FilterOutAlphaChannel(torch.nn.Module):
    def forward(self, img) -> Any:
        return img[:3, :, :] if len(img.shape) == 3 and img.shape[0] > 3 else img


class MinMaxNormalization(torch.nn.Module):
    def forward(self, img: Tensor) -> Tensor:
        return (img - img.min()) / (img.max() - img.min())


class Blobify(torch.nn.Module):
    def forward(self, img: Union[Tensor, np.array]) -> np.ndarray:
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        if isinstance(img, Tensor):
            img = img.numpy()

        return label(img)


class OneHotEncodeFromBlobs(torch.nn.Module):
    def __init__(self, background: bool = False):
        super().__init__()
        self.background = background

    def _one_hot_numpy(self, img: np.ndarray) -> np.ndarray:
        return ((np.arange(img.max() + 1) == img[..., None])).astype(np.uint8)

    def forward(self, img: Union[Tensor, np.array]) -> Union[Tensor, np.array]:
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]

        if isinstance(img, Tensor):
            encoded = one_hot(img.long())
        elif isinstance(img, np.ndarray):
            encoded = self._one_hot_numpy(img)

        if self.background:
            return encoded
        else:
            return encoded[..., 1:]


class Colormap(torch.nn.Module):
    # Add custom colormap support (FLIR most used, etc.)
    # https://stackoverflow.com/questions/28495390/thermal-imaging-palette
    # https://www.flir.ca/discover/industrial/picking-a-thermal-color-palette/
    def __init__(self, colormap: str = "jet"):
        super().__init__()
        self._preprossing = MinMaxNormalization()
        self.cmap = colormaps.get_cmap(colormap)

    def forward(self, img: Tensor) -> Tensor:
        assert isinstance(
            img, (Tensor, np.ndarray)
        ), "Image must be a Tensor or np.ndarray."

        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]

        if img.max() > 1.0 or img.min() < 0.0:
            img = self._preprossing(img)
        if isinstance(img, Tensor):
            img = img.numpy()
            colored = self.cmap(img)
        else:
            colored = self.cmap(img)

        return colored[..., :3]


class BothRandomRotate(torch.nn.Module):
    def __init__(self, angles: Tuple[int], weights: Tuple[int] = None):
        super().__init__()
        self.angles = angles
        self.weights = weights if not weights else [1] * len(angles)

    def forward(self, img, target):
        ang = choices(self.angles, weights=self.weights, k=1)[0]
        return rotate(img, ang), rotate(target, ang)


class BothRandomCrop(torch.nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.size = crop_size

    def forward(self, img, target):
        i, j, h, w = RandomCrop.get_params(img, self.size)
        return crop(img, i, j, h, w), crop(target, i, j, h, w)


class BothToTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, target):
        return to_tensor(img), target


class BothNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img, target):
        return normalize(img, self.mean, self.std), target


class BothCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __next__(self):
        for t in self.transforms:
            yield t

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ImageResizeByCoefficient(torch.nn.Module):
    def __init__(
        self,
        coefficient,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias=None,
    ):
        super().__init__()
        self.coefficient = coefficient
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, img):
        img_size = list(img.shape)
        img_size[0] = (img_size[0] // self.coefficient) * self.coefficient
        img_size[1] = (img_size[1] // self.coefficient) * self.coefficient

        img_pil = Image.fromarray(np.uint8(img))
        res = resize(
            img_pil, img_size[:2], self.interpolation, self.max_size, self.antialias
        )
        return np.asarray(res)


class ToGrayscale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        return color.rgb2gray(img) if len(img.shape) > 2 else img


class TargetDilation(torch.nn.Module):
    def __init__(self, factor, channel: int = 1) -> None:
        super().__init__()
        self.kernel = torch.ones(
            (1, 1, factor, factor), requires_grad=False, dtype=torch.uint8
        )
        self.channel = channel

    def forward(self, img: Image):
        if img.size(0) == 2:
            img[1, ...] = torch.clamp(
                torch.nn.functional.conv2d(
                    img[1:, ...], self.kernel.to(img.dtype), padding="same"
                ),
                0,
                1,
            )
            img[0, ...] = logical_not(img[1:, ...])
            return img
        elif img.size(0) == 1:
            return torch.clamp(
                torch.nn.functional.conv2d(
                    img, self.kernel.to(img.dtype), padding="same"
                ),
                0,
                1,
            )
        else:
            raise NotImplementedError(
                "Dilation not implemented for targets with more than 2 channels."
            )


class TrivialAugmentWide(TAWide):
    """Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__(num_magnitude_bins, interpolation, fill)

    def forward(self, img: Tensor, target: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(
                magnitudes[
                    torch.randint(len(magnitudes), (1,), dtype=torch.long)
                ].item()
            )
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        img = _apply_op(
            img, op_name, magnitude, interpolation=self.interpolation, fill=fill
        )
        if op_name in {
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
            "Rotate",
            "Invert",
        }:
            target = _apply_op(
                target, op_name, magnitude, interpolation=self.interpolation, fill=fill
            )

        return img, target
