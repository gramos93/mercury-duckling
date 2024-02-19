from typing import Any, List, Tuple, Optional, Union
from typing import List, Optional, Tuple, Union, Any, Dict
from matplotlib import colormaps
import numpy as np
from PIL import Image
from skimage.measure import label

import torch
from torch import Tensor
from torch.nn.functional import one_hot

from torchvision import tv_tensors
from torchvision.utils import _log_api_usage_once
from torchvision.transforms.v2 import InterpolationMode, Transform
from torchvision.transforms.v2._utils import _setup_size, query_size
from torchvision.transforms.v2.functional._misc import (
    _register_kernel_internal,
    _get_kernel,
)
from torchvision.transforms.v2.functional import resize, pad


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


def minmax_normalize(
    inpt: torch.Tensor,
) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.Normalize` for details."""
    if torch.jit.is_scripting():
        return minmax_scale_image(inpt)

    _log_api_usage_once(minmax_normalize)

    kernel = _get_kernel(minmax_normalize, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(minmax_normalize, torch.Tensor)
@_register_kernel_internal(minmax_normalize, tv_tensors.Image)
def minmax_scale_image(inpt: torch.Tensor) -> torch.Tensor:
    return inpt.sub(inpt.min()).div_(inpt.max() - inpt.min())


class MinMaxNormalization(Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
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
