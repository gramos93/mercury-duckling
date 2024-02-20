import os

import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection

from .thermal import Thermal
from .thermal_dataset import ThermalDataset
from .transforms import (
    Colormap,
    MinMaxNormalization,
    ResizeByCoefficient,
    ResizeLongestSideAndPad,
)


def build_thermal(cfg):
    if cfg.mode == "train":
        transform = None
        target_transform = None
        transforms = v2.Compose(
            [
                v2.ToImage(),
                # MinMaxNormalization(),
                Colormap(colormap=cfg.colormap),
                v2.RandomZoomOut(fill={tv_tensors.Image: (0), "others": 0}),
                v2.RandomIoUCrop(),
                v2.RandomHorizontalFlip(p=0.7),
                # ResizeByCoefficient(cfg.data.coeff),
                ResizeLongestSideAndPad(target_size=cfg.model.img_size),
                v2.ClampBoundingBoxes(),
            ]
        )
    else:
        transform = None
        target_transform = None
        transforms = v2.Compose(
            [
                v2.ToImage(),
                # MinMaxNormalization(),
                Colormap(colormap=cfg.colormap),
                # ResizeByCoefficient(cfg.data.coeff),
                ResizeLongestSideAndPad(target_size=cfg.model.img_size),
            ]
        )
    return ThermalDataset(
        root=cfg.data.root,
        annFile=os.path.join(cfg.data.root, cfg.data.ann_file),
        transform=transform,
        target_transform=target_transform,
        transforms=transforms,
    )


def build_segmentation(cfg):
    if cfg.mode == "train":
        transform = None
        target_transform = None
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomZoomOut(fill={tv_tensors.Image: (0), "others": 0}),
                v2.RandomIoUCrop(),
                v2.RandomHorizontalFlip(p=0.7),
                # ResizeByCoefficient(cfg.data.coeff),
                ResizeLongestSideAndPad(target_size=cfg.model.img_size),
                v2.ClampBoundingBoxes(),
            ]
        )
    else:
        transform = None
        target_transform = None
        transforms = v2.Compose(
            [
                v2.ToImage(),
                # ResizeByCoefficient(cfg.data.coeff),
                ResizeLongestSideAndPad(target_size=cfg.model.img_size),
            ]
        )
    return CocoDetection(
        root=cfg.data.root,
        annFile=os.path.join(cfg.data.root, cfg.data.ann_file),
        transform=transform,
        target_transform=target_transform,
        transforms=transforms,
    )
