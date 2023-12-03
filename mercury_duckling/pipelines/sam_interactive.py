from warnings import warn
from collections import defaultdict
import numpy as np
from torchvision.transforms import Compose
from segment_anything import sam_model_registry, SamPredictor

from .core import InteractiveTest
from ..datasets import ThermalDataset, THERMAL_TYPE, transforms


class SamInteractiveTest(InteractiveTest):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._current_id = None

    def _set_dataset(self):
        transform = Compose([
            transforms.Colormap("jet"),
        ])
        target_transform = Compose([
            transforms.Blobify(),
            transforms.OneHotEncodeFromBlobs(),
        ])
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
        self.logger.log_parameters(
            {
                "transform": self._get_attribute_name(transform),
                "both_transform": self._get_attribute_name(both_transform),
                "target_transform": self._get_attribute_name(target_transform),
            }
        )

    def _setup_model(self):
        self.__sam_checkpoint = self._config["model"]["checkpoint"]
        self.__model_type = self._config["model"]["type"]
        sam = (
            sam_model_registry[self.__model_type](self.__sam_checkpoint)
            .to(self.device)
            .eval()
        )
        self.model = SamPredictor(sam)

    def prepare_prompts(self, prompts):
        sam_prompts = defaultdict(list)
        for prompt in prompts:
            if prompt["type"] == "point":
                sam_prompts["point_coords"].append(prompt["coords"])
                sam_prompts["point_labels"].append(prompt["label"])
            elif prompt["type"] == "bbox":
                sam_prompts["bbox"].append(prompt["coords"])
            elif prompt["type"] == "logits":
                sam_prompts["mask_input"].append(prompt["mask"])
            else:
                warn(f"Ignoring invalid prompt type: {prompt['type']}.")
        
        for prompt, val in sam_prompts.items():
            sam_prompts[prompt] = np.array(val)
        return sam_prompts

    def predict(self, inputs, prompts, id):
        assert id is not None, "id should not be None."
        if id != self._current_id:
            self.model.set_image(inputs)

        prompts = self.prepare_prompts(prompts)
        masks, scores, logits = self.model.predict(**prompts)

        # Logic for returning the best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx, :, :]  # Choose the model's best mask
        aux = {
            "predicted_iou": scores[best_idx],
            "logits": logits[best_idx, :, :],  # Used for subsequent predictions
        }
        return best_mask, aux
