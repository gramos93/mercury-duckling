import os
from omegaconf import DictConfig

import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from .thermal import Thermal
from .thermal_dataset import ThermalDataset
from .transforms import (
    Blobify,
    OneHotEncodeFromBlobs,
    Colormap,
    Clahe,
    MinMaxNormalization,
    ResizeByCoefficient,
    ResizeLongestSideAndPad,
    StandardizeTarget
)


def build_thermal(cfg: DictConfig):
    if cfg.mode == "train":
        transform = None
        target_transform = None
        transforms = [
            v2.ToImage(),
            v2.RandomZoomOut(fill={tv_tensors.Image: (0), "others": 0}),
            v2.RandomIoUCrop(),
            v2.RandomHorizontalFlip(p=0.7),
            # ResizeByCoefficient(cfg.data.coeff),
            ResizeLongestSideAndPad(target_size=cfg.target_size),
            # MinMaxNormalization(),
            Colormap(colormap=cfg.colormap),
            # StandardizeTarget(cfg.model.classes),
        ]
    else:
        transform = None
        target_transform = None
        transforms = [
            v2.ToImage(),
            # MinMaxNormalization(),
            ResizeLongestSideAndPad(target_size=cfg.target_size),
            # ResizeByCoefficient(cfg.data.coeff),
            Colormap(colormap=cfg.colormap),
        ]
    if cfg.model.type == "interactive":
        transforms.extend([
            Blobify(),
            OneHotEncodeFromBlobs()
        ])
    transforms.append(
        v2.ToDtype(
            {tv_tensors.Image: torch.float32, "others": None}, scale=False
        )
    )
    transforms = v2.Compose(transforms)
    return wrap_dataset_for_transforms_v2(
        ThermalDataset(
            root=cfg.data.root,
            annFile=os.path.join(cfg.data.root, cfg.data.ann_file),
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        ),
        target_keys=["masks", "boxes", "image_id", "labels"],
    )


def build_segmentation(cfg: DictConfig):
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
                ResizeLongestSideAndPad(target_size=cfg.target_size),
                v2.ClampBoundingBoxes(),
                v2.ToDtype(
                    {tv_tensors.Image: torch.float32, "others": None}, scale=True
                ),
            ]
        )
    else:
        transform = None
        target_transform = None
        transforms = v2.Compose(
            [
                v2.ToImage(),
                # ResizeByCoefficient(cfg.data.coeff),
                ResizeLongestSideAndPad(target_size=cfg.target_size),
                StandardizeTarget(cfg.model.classes),
                v2.ToDtype(
                    {tv_tensors.Image: torch.float32, "others": None}, scale=True
                ),
            ]
        )
    return wrap_dataset_for_transforms_v2(
        CocoDetection(
            root=cfg.data.root,
            annFile=os.path.join(cfg.data.root, cfg.data.ann_file),
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        ),
        target_keys=["masks", "boxes", "image_id", "labels"],
    )
