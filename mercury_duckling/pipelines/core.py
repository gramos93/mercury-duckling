from collections import defaultdict
from typing import Dict, List

from animus import IExperiment
from omegaconf import DictConfig
from segmentation_models_pytorch.metrics import get_stats, iou_score
from torch import no_grad
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose

from ..models.predictor import BasePredictor
from ..samplers import BaseSampler, sampler_register
from ..utils import console, console_status


class InteractiveTest(IExperiment):
    """
    InteractiveTest is a base class for segmentation experiments.

    Args:
            config (Dict): Configuration dictionary for the experiment.
            engine (Engine): Engine to use for the experiment.
    """

    def __init__(self, predictor: BasePredictor, dataset: Dataset, config: DictConfig):
        super().__init__()
        self._config = config
        self.predictor = predictor
        self._dataset = dataset
        self.device = self._config.device
        self.num_epochs = 1
        self.metrics = {
            "iou_score": iou_score,
        }

    def __setup_logger(self) -> None:
        # TODO: Setup cometml logger for interactive logging.
        pass

    def _get_attribute_name(self, attribute) -> List[str]:
        if isinstance(attribute, Compose):
            return [T.__class__.__name__ for T in attribute.transforms]
        else:
            return [attribute.__class__.__name__]

    def __setup_sampler(self):
        sampler_type = self._config.sampler.type
        sampler_method = self._config.sampler.method
        self.sampler: BaseSampler = sampler_register[sampler_type][sampler_method](
            **self._config.sampler.args
        )

    def _setup_model(self):
        self.predictor.eval()
        self.predictor.to(self.device)

    def __setup_dataloaders(self) -> None:
        """Setup the dataloaders for the experiment."""
        test_loader = DataLoader(
            self._dataset, batch_size=1, shuffle=False, num_workers=4
        )
        self.datasets = {"test": test_loader}

    def _setup_callbacks(self):
        self.callbacks = {}

    def on_experiment_start(self, exp: "IExperiment") -> None:
        super().on_experiment_start(exp)
        self.__setup_sampler()
        self.__setup_dataloaders()
        self._setup_model()
        self._setup_callbacks()

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)

    def on_epoch_start(self, exp: "IExperiment") -> None:
        self.epoch_metrics: Dict = defaultdict(
            lambda: dict(zip(self.metrics.keys(), [0.0] * len(self.metrics)))
        )

    def run_epoch(self) -> None:
        self._run_event("on_dataset_start")
        self.run_dataset()
        self._run_event("on_dataset_end")

    def on_dataset_start(self, exp: "IExperiment"):
        super().on_dataset_start(exp)
        self.sampler.is_sampling = True

    def on_epoch_end(self, exp: IExperiment) -> None:
        super().on_epoch_end(exp)
        metrics_str = ", ".join(
            "{}={:.3f}".format(key.title(), val)
            for (key, val) in self.epoch_metrics.items()
        )
        console.log(
            f"[bold][red]Epoch: {self.epoch_step}[/] - "
            f"[bold cyan]{self.dataset_key}[/]"
            f" -> [magenta]metrics: {metrics_str}[/]"
        )

    @no_grad()
    def run_batch(self) -> None:
        # TODO: Move to on_batch_start
        console_status.update(
            f"[bold green]Running: {self.dataset_key} "
            f"-> Epoch: {self.epoch_step}/{self.num_epochs} "
            f"-> Batch: {self.dataset_batch_step}/{len(self.dataset)}"
        )
        inputs, targets, id = self.batch

        for target in targets:
            outputs = None
            aux = None
            for prompts in self.sampler.interact(mask=target, outs=outputs):
                outputs, aux = self.predictor.predict(
                    inputs, prompts, aux if aux is not None else outputs, id
                )
                self.sampler.set_outputs(outputs)
                stats = get_stats(outputs, target, mode="binary")
                for metric_name, metric in self.metrics.items():
                    self.batch_metrics[metric_name] = metric(*stats, reduction="micro")

                self.batch_metrics = {
                    k: float(v) * (inputs.size(0) / len(self.dataset.dataset))
                    for k, v in self.batch_metrics.items()
                }

    def on_batch_end(self, exp: IExperiment) -> None:
        for metric_name, metric_value in self.batch_metrics.items():
            self.epoch_metrics[metric_name] += metric_value

    def _run(self) -> None:
        self._run_event("on_experiment_start")
        self.run_experiment()
        self._run_event("on_experiment_end")
