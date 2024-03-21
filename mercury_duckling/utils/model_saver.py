import os
import time
import json
from omegaconf import OmegaConf
from animus import ICallback, IExperiment
from safetensors.torch import save_file


class ModelLogger(ICallback):
    def __init__(
        self,
        metric_name: str,
        dataset_name: str = "val",
        minimise: bool = True,
        model_attr: str = "segmentor",
        topK: int = 1,  # Not Implemented for other topK
        save_root: str = "./checkpoints",
    ) -> None:
        self._topK = topK
        self._key = dataset_name
        self.metric_name = metric_name
        self.minimise = minimise
        self._model_attr = model_attr
        self._best = float("inf") if minimise else 0.0
        self._cmp_func = "__le__" if minimise else "__ge__"

        if os.path.isdir(save_root):
            self._save_path = save_root
        else:
            raise ValueError(f"Folder {save_root} not found.")

    def on_experiment_start(self, exp: "IExperiment") -> None:
        model_name = exp._cfg.selected_model
        model_data = OmegaConf.to_container(exp._cfg.model, resolve=True)
        exp_uri = (
            exp.callbacks["logger"]._online_logger.url 
            if exp._cfg.logging.is_online else self._save_path
        )
        self.metadata = {
            "performance": {},
            "epoch": exp.epoch_step,
            "model_info": {
                "name": model_name,
                "args": model_data,
                "class": str(getattr(exp, self._model_attr).__class__)
            },
            "uri": exp_uri
        }

    def on_epoch_end(self, exp: "IExperiment") -> None:
        if logger := exp.callbacks.get("logger", False):
            self._save_path = logger._logging_dir
        else:
            self._save_path = os.path.join(
                self._save_path, time.strftime("%Y-%m-%d-%H:%M:%S")
            )
        if exp.dataset_key == self._key:
            last_score = exp.dataset_metrics[self.metric_name]
            if getattr(last_score, self._cmp_func)(self._best):
                self._best = last_score
                if hasattr(exp, "engine"):
                    exp.engine.save_model(
                        getattr(exp, self._model_attr),
                        self._save_path,
                        # This will use safetensors to save the model.
                        # Otherwise we can use the normal pth pytorch way.
                        # Then the model will be saved in a pytorch_model.bin file
                        safe_serialization=True,
                    )
                else:
                    state_dict = getattr(exp, self._model_attr).state_dict()
                    save_file(
                        state_dict, os.path.join(self._save_path, "model.safetensors")
                    )

                self.metadata["performance"] = exp.dataset_metrics
                with open(os.path.join(self._save_path, "metadata.json", "w")) as fid:
                    json.dump(self.metadata, fid)
