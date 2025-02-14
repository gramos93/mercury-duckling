from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from warnings import warn

import numpy as np
import cv2
from skimage.segmentation import slic
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
import torchvision.transforms.v2 as v2

__all__ = ["SamPredictor", "RITMPredictor"]


class BasePredictor(nn.Module, ABC):
    def __init__(self, config: DictConfig) -> None:
        """
        Common API to be used by interactive predictors in the interactive pipelines.

        Args:
            config (DictConfig): Configuration for each model in the
                OmegaConf format. All model parameters should be inside
                the config dictionary.
        """
        super().__init__()
        self._config = config
        self._current_id = None

    @abstractmethod
    def _setup_model(self, device: str) -> None:
        """This function should be overloaded to setup the model"""
        raise NotImplementedError

    @abstractmethod
    def prepare_prompts(self, prompts: List[Any]) -> Dict[str, Any]:
        """
        Prepare the prompts returned by the Sampler.
        For reference see the BaseSampler class.

        Args:
            prompts (List[Any]): List of prompts to be parser.

        Returns:
            Preprocessed list of prompts in the format needed by the model.
        """
        raise NotImplementedError

    def propose_init_mask(
            self, inpts: np.ndarray, prompts: Dict[str, Any]
        ) -> np.ndarray:
        segments_slic: np.ndarray = slic(
            inpts,
            n_segments=150,
            compactness=10,
            sigma=1,
            start_label=1
        )
        pre_sp: int = segments_slic[prompts[0]["coords"][1], prompts[0]["coords"][0]]
        segments_slic = (segments_slic == pre_sp).astype(np.int32)
        return segments_slic

    def preprocess(self, inpts: np.ndarray) -> np.ndarray:
        return inpts

    def postprocess(self, inpts: Tensor) -> Tensor:
        return inpts

    @abstractmethod
    def predict(
        self, inpts: np.ndarray, prompts: List[Any], aux: Dict[str, Any], id: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Main prediction function, usually used with no gradient.

        Args:
            inputs (Array): Input array image (H, W, 3) in RGB format.
                Currently only a batch of one is supported for
                interactive predictions.
            prompts (List[Any]): Raw prompts generated by a sampler.
            aux (Dict[Any]): Auxiliary inputs the model may need for
                prediction.
            id (int): id of the image being predicted on. If this changes
                some models may need to reset the prompt history and preprocess the new
                inputs.
        """
        assert id is not None, "id should not be None."
        raise NotImplementedError
    
    def forward(self, inpts: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        return self.predict(**inpts)


class SamPredictor(BasePredictor):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    def _setup_model(self, device: str) -> None:
        from segment_anything import SamPredictor as SP
        from segment_anything import sam_model_registry

        self.__sam_checkpoint = self._config.checkpoint
        self.__model_size = self._config.size
        sam = sam_model_registry[self.__model_size](self.__sam_checkpoint)
        self.model: SP = SP(sam.to(device))

    def propose_init_mask(
            self, inpts: np.ndarray, prompts: Dict[str, Any]
        ) -> np.ndarray:
        aux = super().propose_init_mask(inpts, prompts)
        aux = v2.functional.resize(
            torch.tensor(aux).unsqueeze(0),
            (256, 256),
            interpolation=v2.functional.InterpolationMode.NEAREST
        )
        return aux.float() + 5.
    
    def prepare_prompts(self, prompts: List[Any]) -> Dict[str, Any]:
        """
        Prepare the prompts returned by the Sampler.
        For reference see the BaseSampler class.

        Args:
            prompts (List[Any]): List of prompts to be parser.

        Returns:
            Preprocessed list of prompts in the format needed by the model.
        """
        sam_prompts = defaultdict(list)
        for prompt in prompts:
            if prompt["type"] == "point":
                sam_prompts["point_coords"].append(prompt["coords"])
                sam_prompts["point_labels"].append(prompt["label"])
            elif prompt["type"] == "bbox":
                sam_prompts["bbox"].append(prompt["coords"])
            else:
                warn(f"Ignoring invalid prompt type: {prompt['type']}.")

        for prompt, val in sam_prompts.items():
            sam_prompts[prompt] = np.array(val)
        return sam_prompts

    def predict(
        self, inpts: np.ndarray, prompts: List[Any], aux: np.ndarray, id: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Main prediction function, usually used with no gradient.

        Args:
            inputs (Tensor): Input tensor image (1, N, H, W).
                Currently only a batch of one is supported for
                interactive predictions.
            prompts (List[Any]): Raw prompts generated by a sampler.
            aux (Dict[Any]): Auxiliary inputs the model may need for
                prediction.
            id (int): id of the image being predicted on. If this changes
                some models may need to reset the prompt history and preprocess the new
                inputs.
        """
        assert id is not None, "id should not be None."

        if id != self._current_id:
            # Basically if input comes from DataLoader.
            if isinstance(inpts, Tensor):
                inpts = inpts.squeeze(0).permute(1, 2, 0).numpy()
            if inpts.max() <= 1.:
                inpts = (inpts * 255).astype(np.uint8)

            inpts = self.preprocess(inpts)
            self.model.set_image(inpts)
            self._current_id = id

        if aux is None:
            # Case were the image_id is the same but the target is different
            if isinstance(inpts, Tensor):
                inpts = inpts.squeeze(0).permute(1, 2, 0).numpy()
            if inpts.max() <= 1.:
                inpts = (inpts * 255).astype(np.uint8)

            aux = self.propose_init_mask(inpts, prompts)

        prompts = self.prepare_prompts(prompts)
        masks, scores, logits = self.model.predict(
            **prompts, mask_input=aux, multimask_output=False
        )

        # self.logger.log_metric("sam_pred_iou", scores[0])
        # Logic for returning the best mask
        self._prev_mask: np.ndarray = logits # (1, 256, 256)
        masks = self.postprocess(masks[0])
        
        return torch.tensor(masks, dtype=torch.uint8), logits


class RITMPredictor(BasePredictor):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        # Fix for numpy.int being deprecated since numpy 1.20
        np.int = np.int_
        from isegm.inference.clicker import Click, Clicker

        self._clicker = Clicker
        self._click = Click
        self._threshold = self._config.threshold

    def _setup_model(self, device: str) -> None:
        from isegm.inference import utils
        from isegm.inference.predictors import get_predictor

        model = utils.load_is_model(self._config.checkpoint, device)
        self.model = get_predictor(model, "NoBRS", device, prob_thresh=self._threshold)

    def prepare_prompts(self, prompts: List[Any]) -> Dict[str, Any]:
        """
        Prepare the prompts returned by the Sampler.
        For reference see the BaseSampler class.

        Args:
            prompts (List[Any]): List of prompts to be parser.

        Returns:
            Preprocessed list of prompts in the format needed by the model.
        """
        clicker = self._clicker(gt_mask=None, init_clicks=None)
        for prompt in prompts:
            if prompt["type"] == "point":
                points = self._click(
                    is_positive=prompt["label"],
                    coords=prompt["coords"][::-1],  # coords are in (y, x) format
                )
                clicker.add_click(points)
            else:
                warn(f"Ignoring invalid prompt type: {prompt['type']}.")
        return clicker

    def predict(
        self, inpts: np.ndarray, prompts: List[Any], aux: Any, id: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Main prediction function.

        Args:
            inputs (Tensor): Input tensor image (1, N, H, W).
                Currently only a batch of one is supported for
                interactive predictions.
            prompts (List[Any]): Raw prompts generated by a sampler.
            aux (Dict[Any]): Auxiliary inputs the model may need for
                prediction.
            id (int): id of the image being predicted on. If this changes
                some models may need to reset the prompt history and preprocess the new
                inputs.
        """
        assert id is not None, "id should not be None."

        b, c, h, w = inpts.shape
        if id != self._current_id:
            # Basically if input comes from DataLoader.
            if isinstance(inpts, Tensor):
                inpts = inpts.squeeze(0).permute(1, 2, 0).numpy()
            if inpts.max() <= 1.:
                inpts = (inpts * 255).astype(np.uint8)

            inpts = self.preprocess(inpts)
            self.model.set_input_image(inpts)
            self._current_id = id

        if aux is not None:
            aux = Tensor(aux).unsqueeze(0).unsqueeze(0).to(self.model.device)
        else:
            # Basically if input comes from DataLoader.
            if isinstance(inpts, Tensor):
                inpts = inpts.squeeze(0).permute(1, 2, 0).numpy()
            if inpts.max() <= 1.:
                inpts = (inpts * 255).astype(np.uint8)

            aux = self.propose_init_mask(inpts, prompts)
            aux = torch.tensor(aux, device=self.model.device)[None, None, ...]
            # aux = torch.zeros((1, 1, h, w), device=self.model.device)

        prompts = self.prepare_prompts(prompts)
        # By default RITM will use the previous prediction saved internally.
        logits: np.ndarray = self.model.get_prediction(prompts, aux) #(H, W)
        masks = self.postprocess(logits > self._threshold)
        return torch.tensor(masks, dtype=torch.uint8), logits


class IUnetPredictor(BasePredictor):
    def __init__(self, config: DictConfig) -> None:
        """
        IS Unet API to be used by interactive predictors in the interactive pipelines.

        Args:
            config (DictConfig): Configuration for each model in the
                OmegaConf format. All model parameters should be inside
                the config dictionary.
        """
        super().__init__(config)

    def _setup_model(self, device: str):
        from inter_unet.models import build_model
        from inter_unet.transforms import (
            groupnorm_normalise_image,
            trimap_transform,
        )
        self._groupnorm_normalise_image = groupnorm_normalise_image
        self._trimap_transform = trimap_transform
        self.device = device
        class InterUnetArgs:
            encoder = "resnet50_GN_WS"
            decoder = "InteractiveSegNet"
            use_mask_input = True
            use_usr_encoder = True
            weights = self._config.checkpoint

        args = InterUnetArgs()
        self.model = build_model(args)
        self.model.to(device)
        self.model.eval()

    def get_predictions(
        self,
        image_np: np.ndarray,
        trimap_np: np.ndarray,
        alpha_old: np.ndarray,
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
        # alpha_old_np = remove_non_fg_connected(alpha_old_np, trimap_np[:, :, 1])

        h, w = trimap_np.shape[:2]
        # image_scale_np = scale_input(image_np, cv2.INTER_LANCZOS4)
        # trimap_scale_np = scale_input(trimap_np, cv2.INTER_NEAREST)
        # alpha_old_scale_np = scale_input(alpha_old_np, cv2.INTER_LANCZOS4)

        image_torch = np_to_torch(image_np)
        trimap_torch = np_to_torch(trimap_np)
        if isinstance(alpha_old, Tensor):
            alpha_old_torch = alpha_old[None, None, ...].float()
        else:
            alpha_old_torch = np_to_torch(alpha_old[:, :, None])

        trimap_transformed_torch = np_to_torch(self._trimap_transform(trimap_np))
        image_transformed_torch = self._groupnorm_normalise_image(
            image_torch.clone(), format="nchw"
        )
        alpha = self.model(
            image_transformed_torch.to(self.device),
            trimap_transformed_torch.to(self.device),
            alpha_old_torch.to(self.device),
            trimap_torch.to(self.device),
        )
        alpha = cv2.resize(
            alpha[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4
        )
        alpha[trimap_np[:, :, 0] == 1] = 0
        alpha[trimap_np[:, :, 1] == 1] = 1
        alpha = remove_non_fg_connected(alpha, trimap_np[:, :, 1])
        return alpha

    def prepare_prompts(self, prompts, trimap):
        for prompt in prompts:
            if prompt["type"] == "point":
                # coords are in (y, x) format
                trimap[
                    prompt["coords"][1], prompt["coords"][0], int(prompt["label"])
                ] = 1
            else:
                warn(f"Ignoring invalid prompt type: {prompt['type']}.")
        return trimap

    def predict(
        self, inpts: np.ndarray, prompts: List[Any], aux: Any, id: int
    ) -> Tuple[Tensor, Tensor]:
        if id != self._current_id:
            # This model does not need to set an image.
            # Hence we only reset the previous mask.
            self._current_id = id

        b, c, h, w = inpts.shape
        if isinstance(inpts, Tensor):
            inpts = inpts.squeeze(0).permute(1, 2, 0).numpy()
        if inpts.max() <= 1.:
            inpts = (inpts * 255).astype(np.uint8)
        if aux is None:
            aux = self.propose_init_mask(inpts, prompts)
            # aux = np.zeros((h, w))

        trimap = np.zeros((h, w, 2))
        trimap = self.prepare_prompts(prompts, trimap)
        alpha = self.get_predictions(inpts, trimap, aux)

        return torch.tensor(alpha > 0.5, dtype=torch.uint8), None

# TODO: Re implement this as a v2.transform
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


# TODO : replace this with an actual transform.
def np_to_torch(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float()
