from warnings import warn
import numpy as np
from torchvision.transforms import Compose

from .core import InteractiveTest
from ..datasets import ThermalDataset, THERMAL_TYPE, transforms

from ..models.isegm.inference import utils
from ..models.isegm.inference.predictors import get_predictor
from ..models.isegm.inference.clicker import Clicker, Click

# Fix for numpy.int being deprecated since numpy 1.20
np.int = np.int_


class RitmInteractiveTest(InteractiveTest):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._current_id = None
        self._threshold = self._config["model"]["threshold"]

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
        self.logger.log_parameters(
            {
                "transform": self._get_attribute_name(transform),
                "both_transform": self._get_attribute_name(both_transform),
                "target_transform": self._get_attribute_name(target_transform),
            }
        )

    def _setup_model(self):
        model = utils.load_is_model(self._config["model"]["checkpoint"], self.device)
        self.model = get_predictor(
            model, "NoBRS", self.device, prob_thresh=self._threshold
        )

    def prepare_prompts(self, prompts):
        clicker = Clicker(gt_mask=None, init_clicks=None)
        for prompt in prompts:
            if prompt["type"] == "point":
                points = Click(
                    is_positive=prompt["label"],
                    coords=prompt["coords"][::-1] # coords are in (y, x) format
                )
                clicker.add_click(points)
            else:
                warn(f"Ignoring invalid prompt type: {prompt['type']}.")
        return clicker

    def predict(self, inputs, prompts, aux, id):
        assert id is not None, "id should not be None."
        if id != self._current_id:
            self.model.set_input_image(inputs)

        prompts = self.prepare_prompts(prompts)
        # By default RITM will use the previous prediction saved internally.
        masks = self.model.get_prediction(prompts)
        return masks > self._threshold, None
