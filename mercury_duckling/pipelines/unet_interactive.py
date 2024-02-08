from warnings import warn
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose

from .core import InteractiveTest
from ..datasets import ThermalDataset, THERMAL_TYPE, transforms

from ..models.inter_unet.networks.models import build_model
from ..models.inter_unet.networks.transforms import (
    groupnorm_normalise_image,
    trimap_transform,
)


class UnetInteractiveTest(InteractiveTest):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._current_id = None

    def _set_dataset(self):
        transform = Compose(
            [
                transforms.Colormap("jet"),
            ]
        )
        target_transform = Compose(
            [
                transforms.Blobify(),
                transforms.OneHotEncodeFromBlobs(),
            ]
        )
        both_transform = None
        self._dataset = ThermalDataset(
            root=self._config["dataset"]["root"],
            annFile=self._config["dataset"]["annFile"],
            transform=transform,
            target_transform=target_transform,
            both_transform=both_transform,
            thermal_type=THERMAL_TYPE.FLIR,
        )
        # Log dataset information
        # self.logger.log_parameters(
        #     {
        #         "transform": self._get_attribute_name(transform),
        #         "both_transform": self._get_attribute_name(both_transform),
        #         "target_transform": self._get_attribute_name(target_transform),
        #     }
        # )

    def _setup_model(self):
        class InterUnetArgs:
            encoder = "resnet50_GN_WS"
            decoder = "InteractiveSegNet"
            use_mask_input = True
            use_usr_encoder = True
            weights = self._config["model"]["checkpoint"]
            device = self.device

        args = InterUnetArgs()
        self.model = build_model(args)
        self.model.eval()

    def get_predictions(
        self,
        image_np: np.ndarray,
        trimap_np: np.ndarray,
        alpha_old_np: np.ndarray,
    ) -> np.ndarray:
        """Predict segmentation
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap/Click map, first background then foreground.
                    Dimensions: (h, w, 2)
        Returns:
        alpha: alpha matte/non-binary segmentation image between 0 and 1.
                Dimensions: (h, w)
        """
        image_np = image_np / 255.0
        alpha_old_np = remove_non_fg_connected(alpha_old_np, trimap_np[:, :, 1])

        h, w = trimap_np.shape[:2]
        image_scale_np = scale_input(image_np, cv2.INTER_LANCZOS4)
        trimap_scale_np = scale_input(trimap_np, cv2.INTER_NEAREST)
        alpha_old_scale_np = scale_input(alpha_old_np, cv2.INTER_LANCZOS4)

        with torch.no_grad():
            image_torch = np_to_torch(image_scale_np)
            trimap_torch = np_to_torch(trimap_scale_np)
            alpha_old_torch = np_to_torch(alpha_old_scale_np[:, :, None])

            trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np))
            image_transformed_torch = groupnorm_normalise_image(
                image_torch.clone(), format="nchw"
            )

            alpha = self.model(
                image_transformed_torch,
                trimap_transformed_torch,
                alpha_old_torch,
                trimap_torch,
            )
            alpha = cv2.resize(
                alpha[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4
            )
        alpha[trimap_np[:, :, 0] == 1] = 0
        alpha[trimap_np[:, :, 1] == 1] = 1

        alpha = remove_non_fg_connected(alpha, trimap_np[:, :, 1])
        return alpha

    def prepare_promtps(self, prompts, trimap):
        for prompt in prompts:
            if prompt["type"] == "point":
                # coords are in (y, x) format
                trimap[prompt["coords"][1], prompt["coords"][0], int(prompt["label"])] = 1
            else:
                warn(f"Ignoring invalid prompt type: {prompt['type']}.")
        return trimap

    def predict(self, inputs, prompts, aux, id):
        h, w, c = inputs.shape
        if id != self._current_id:
            # This model does not need to set an image. 
            # Hence we only reset the previous mask.
            self._current_id = id
        
        if aux is None:
            aux = np.zeros((h, w))

        trimap = np.zeros((h, w, 2))
        trimap = self.prepare_promtps(prompts, trimap)
        alpha = self.get_predictions(inputs, trimap, aux)
        return alpha, None


def scale_input(x: np.ndarray, scale_type) -> np.ndarray:
    """Scales so that min side length is 352 and sides are divisible by 8"""
    h, w = x.shape[:2]
    h1 = int(np.ceil(h / 32) * 32)
    w1 = int(np.ceil(w / 32) * 32)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale


def remove_non_fg_connected(alpha_np, fg_pos):
    if np.count_nonzero(fg_pos) > 0:
        ys, xs = np.where(fg_pos == 1)

        alpha_np_bin = alpha_np > 0.5
        ret, labels_con = cv2.connectedComponents((alpha_np_bin * 255).astype(np.uint8))

        labels_f = []
        for y, x in zip(ys, xs):
            if labels_con[y, x] != 0:
                labels_f.append(labels_con[y, x])
        fg_con = np.zeros_like(alpha_np)
        for lab in labels_f:
            fg_con[labels_con == lab] = 1

        alpha_np[fg_con == 0] = 0

    return alpha_np


def np_to_torch(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float()
