from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib.colors import Colormap as pltColormap
from matplotlib import colormaps
from scipy.ndimage import label
from torch.nn import functional as F
from torchvision import tv_tensors
from torchvision.transforms.v2 import InterpolationMode, Transform
from torchvision.transforms.v2._utils import _setup_size, query_size
from torchvision.transforms.v2.functional import pad, resize
from torchvision.transforms.v2.functional._misc import (
    _get_kernel,
    _register_kernel_internal,
)
from torchvision.utils import _log_api_usage_once


def blobify(
    inpt: torch.Tensor,
) -> torch.Tensor:
    """See :class:`Blobify` for details."""
    _log_api_usage_once(blobify)

    kernel = _get_kernel(blobify, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(blobify, tv_tensors.Mask)
def blobify_mask_tensor(inpt: tv_tensors.Mask) -> tv_tensors.Mask:
    label(inpt.numpy(), output=inpt.numpy())
    inpt = tv_tensors.Mask(inpt, device=inpt.device)
    return inpt


class Blobify(Transform):
    def _transform(
        self, inpt: tv_tensors.Mask, params: Dict[str, Any]
    ) -> tv_tensors.Mask:
        return self._call_kernel(blobify, inpt)


def one_hot(
    inpt: tv_tensors.Mask,
) -> tv_tensors.Mask:
    """See :class:`one_hot` for details."""
    if torch.jit.is_scripting():
        return one_hot_mask_tensor(inpt)

    _log_api_usage_once(one_hot)

    kernel = _get_kernel(one_hot, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(one_hot, tv_tensors.Mask)
def one_hot_mask_tensor(
    inpt: tv_tensors.Mask, background: bool = False
) -> tv_tensors.Mask:
    out = F.one_hot(inpt) if background else F.one_hot(inpt)[..., 1]
    return tv_tensors.Mask(out)


class OneHotEncodeFromBlobs(Transform):
    def __init__(self, background: bool = False):
        super().__init__()
        self.background = background

    def _transform(
        self, inpt: tv_tensors.Mask, params: Dict[str, Any]
    ) -> tv_tensors.Mask:
        return self._call_kernel(one_hot, inpt, self.background)


class Colormap(Transform):
    # Add custom colormap support (FLIR most used, etc.)
    # https://stackoverflow.com/questions/28495390/thermal-imaging-palette
    # https://www.flir.ca/discover/industrial/picking-a-thermal-color-palette/
    def __init__(self, colormap: str = "jet"):
        super().__init__()
        self.cmap: pltColormap = colormaps.get_cmap(colormap)

    def _transform(self, inpt: tv_tensors.Image) -> tv_tensors.Image:
        # FIXME: Not sure if this is okay
        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        if inpt.max() > 1.0 or inpt.min() < 0.0:
            inpt = self._call_kernel(minmax_normalize, inpt)

        colored = self.cmap(inpt)  # Will be numpy array
        return tv_tensors.Image(colored[..., :3])


class ResizeByCoefficient(Transform):
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

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        oldh, oldw = query_size(flat_inputs)
        newh = (oldh // self.coefficient) * self.coefficient
        neww = (oldw // self.coefficient) * self.coefficient

        return {"height": newh, "width": neww}

    def _transform(self, inpt):
        params = self._get_params(inpt)
        return self._call_kernel(
            resize,
            inpt,
            size=(params["heigth"], params["width"]),
            interpolation=self.interpolation,
            antialias=self.antialias,
        )


def minmax_normalize(
    inpt: tv_tensors.Image,
) -> tv_tensors.Image:
    """See :class:`MinMaxNormalization` for details."""
    if torch.jit.is_scripting():
        return minmax_scale_image(inpt)

    _log_api_usage_once(minmax_normalize)

    kernel = _get_kernel(minmax_normalize, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(minmax_normalize, tv_tensors.Image)
def minmax_scale_image(inpt: tv_tensors.Image) -> tv_tensors.Image:
    return inpt.sub(inpt.min()).div_(inpt.max() - inpt.min())


class MinMaxNormalization(Transform):
    def _transform(
        self, inpt: tv_tensors.Image, params: Dict[str, Any]
    ) -> tv_tensors.Image:
        return self._call_kernel(minmax_normalize, inpt)


class ResizeLongestSideAndPad(Transform):
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.

    Code from segmenta-anything META
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.target_size = _setup_size(
            target_size,
            error_msg="Please provide only two dimensions (h, w) for target_size.",
        )
        self.interpolation = interpolation
        self.antialias = antialias

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        h, w = params["height"], params["width"]
        resized_target = self._call_kernel(
            resize,
            inpt,
            size=min(self.target_size),
            max_size=max(self.target_size),
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        # Pad to match
        h, w = resized_target.shape[-2:]
        padh = self.target_size[0] - h
        padw = self.target_size[1] - w
        padded_target = self._call_kernel(
            pad,
            resized_target,
            # pad_left,  pad_top,  pad_right, pad_bottom
            (0, 0, padw, padh),
        )
        return padded_target

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """
        Compute the output size given input size and target long side length.
        """
        oldh, oldw = query_size(flat_inputs)
        size_idx = np.argmax([oldh, oldw])
        scale = self.target_size[size_idx] * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return {"height": newh, "width": neww}
