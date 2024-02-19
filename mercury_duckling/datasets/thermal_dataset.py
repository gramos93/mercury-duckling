import os

from enum import Enum
from typing import Union
import exiftool
from . import Thermal
import numpy as np
from torch import load, Tensor
from torchvision.datasets import CocoDetection


class THERMAL_TYPE(Enum):
    DJI = 1
    FLIR = 2
    DYNAMIC = 3


class ThermalDataset(CocoDetection):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform,
        target_transform,
        transforms=None,
        thermal_type: Union[THERMAL_TYPE, None] = None,
    ) -> None:
        """
        ThermalDataset is a MSCOCO format dataset for thermal images.

        Args:
            root (str): Root directory of the dataset.
            annFile (str): Path to the annotation file.
            transform (nn.Module): Transforms to apply to the image.
            target_transform (nn.Module): Transforms to apply to the mask.
            both_transform (nn.Module, optional): Transforms to apply to both the
                image and mask. Defaults to None.
            thermal_type (THERMAL_TYPE, None): Type of thermal image
                to be loaded. Defaults to None.
        """
        super().__init__(
            root=root,
            annFile=os.path.join(root, "annotations", annFile),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.ann_file = annFile
        if thermal_type is not None and isinstance(thermal_type, THERMAL_TYPE):
            self.thermal_handler = Thermal(
                dirp_filename="plugins/libdirp.so",
                dirp_sub_filename="plugins/libv_dirp.so",
                iirp_filename="plugins/libv_iirp.so",
                exif_filename=None,
                dtype=np.float32,
            )
            self.exif_handler = exiftool.ExifToolHelper()

        self.thermal_type = thermal_type

    def _load_image_thermal(
        self, img_path, type, metadata, thermal_handler
    ) -> np.ndarray:
        """
        Opens a thermal image and returns a tensor.

        Args:
            img_path (str): Path to the image
            type (THERMAL): Type of thermal image

        Returns:
            torch.Tensor: Tensor of the thermal image (float32)
        """

        if type == THERMAL_TYPE.DYNAMIC:
            if (
                metadata.get("APP1:CreatorSoftware") == "ResearchIR"
                or "flir" in metadata.get("EXIF:Software").lower()
            ):
                type = THERMAL_TYPE.FLIR
            elif metadata.get("EXIF:Make") == "DJI":
                type = THERMAL_TYPE.DJI
            else:
                raise ValueError(
                    "Invalid thermal type detected."
                    " Plase provide either DJI H20T images or "
                    "Research IR FLIR processed images."
                )
        if type == THERMAL_TYPE.FLIR:
            return thermal_handler.parse_flir(
                img_path,
                emissivity=em
                if (em := metadata.get("APP1:Emissivity", 0.95) < 0.9)
                else 0.95,
                object_distance=metadata["APP1:ObjectDistance"],
                atmospheric_temperature=metadata["APP1:AtmosphericTemperature"],
                reflected_apparent_temperature=metadata[
                    "APP1:ReflectedApparentTemperature"
                ],
                ir_window_temperature=metadata["APP1:IRWindowTemperature"],
                ir_window_transmission=metadata["APP1:IRWindowTransmission"],
                relative_humidity=metadata["APP1:RelativeHumidity"] * 100,
                # planck constants
                planck_r1=metadata["APP1:PlanckR1"],
                planck_b=metadata["APP1:PlanckB"],
                planck_f=metadata["APP1:PlanckF"],
                planck_o=metadata["APP1:PlanckO"],
                planck_r2=metadata["APP1:PlanckR2"],
                # constants
                ata1=float(metadata["APP1:AtmosphericTransAlpha1"]),
                ata2=metadata["APP1:AtmosphericTransAlpha2"],
                atb1=float(metadata["APP1:AtmosphericTransBeta1"]),
                atb2=float(metadata["APP1:AtmosphericTransBeta2"]),
                atx=metadata["APP1:AtmosphericTransX"],
            )
        else:
            return thermal_handler.parse_dirp2(
                img_path,
                object_distance=metadata["XMP:LRFTargetDistance"],
                relative_humidity=metadata["APP4:RelativeHumidity"],
                emissivity=0.97,  # concrete's emissivity
                reflected_apparent_temperature=metadata["APP4:AmbientTemperature"],
            )

    def _load_image(self, id: int) -> Tensor:
        path = self.coco.loadImgs(id)[0]["file_name"].replace("jpg", "pt")
        return load(os.path.join(self.root, path), map_location="cpu").float()

    def __getitem__(self, id: int) -> tuple:
        """
        Returns a tuple of (image, mask) at the index of the dataset.

        Notes:
            1) The mask and image are both torch.Tensors when opened,
               no need to include ToTensor.
            2) The both_transform will be applied to both the image and mask,
               if provided, before individual tranformations.

        Args:
            index (int): Index of the dataset

        Returns:
            tuple: (image, mask)
        """
        id = self.ids[id]

        if self.thermal_type is not None:
            # image path is root/<img_file_name>
            img_path = os.path.join(
                self.root,
                self.coco.loadImgs(id)[0]["file_name"],
            )
            with self.exif_handler as exif_handler:
                metadata = exif_handler.get_metadata(img_path, params=["-m", "-q"])[0]

            image = self._load_image_thermal(
                img_path,
                self.thermal_type,
                metadata,
                self.thermal_handler,
            )
            image = Tensor(image)
        else:
            image = self._load_image(id)

        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
