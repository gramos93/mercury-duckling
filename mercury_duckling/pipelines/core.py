from abc import ABC, abstractmethod
from typing import Dict, List
from collections import defaultdict

from torch import no_grad
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from animus import IExperiment
from segmentation_models_pytorch.metrics import get_stats, iou_score

# from comet_ml import Experiment

from ..samplers import sampler_register, BaseSampler
from ..datasets.transforms import BothCompose
from ..utils import console, console_status


class InteractiveTest(IExperiment, ABC):
    """
    InteractiveTest is a base class for segmentation experiments.

    Args:
            config (Dict): Configuration dictionary for the experiment.
            engine (Engine): Engine to use for the experiment.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self._config = config
        self.device = self._config["device"]
        self.num_epochs = 1
        self.metrics = {
            "iou_score": iou_score,
        }

    def __setup_logger(self) -> None:
        # self.logger = Experiment(
        #     api_key=os.environ["COMET_API_KEY"],
        #     project_name=os.environ["COMET_PROJECT_NAME"],
        #     workspace=os.environ["COMET_WORKSPACE"],
        #     log_graph=False,
        #     parse_args=False,
        #     log_code=True,
        #     auto_log_co2=False,
        #     auto_param_logging=False,
        #     auto_output_logging="default",
        #     auto_metric_logging=False,
        #     auto_histogram_weight_logging=False,
        #     log_env_host=False,
        #     log_env_details=False,
        #     log_env_cpu=False,
        #     log_env_network=False,
        #     log_env_disk=False,
        # )
        # if tags := self._config["logging"].get("tags"):
        #     self.logger.add_tags(tags)

        # self.logger.log_parameters(
        #     json_normalize(self._config).to_dict(orient="records")[0]
        # )
        # self.logger.log_code(self._config["origin"], code_name="config_origin_file")

        # logging_dir = f"./checkpoints/{self.logger.get_name()}"
        # if not os.path.isdir(logging_dir):
        #     os.mkdir(logging_dir)
        pass

    def _get_attribute_name(self, attribute) -> List[str]:
        if isinstance(attribute, (Compose, BothCompose)):
            return [T.__class__.__name__ for T in attribute.transforms]
        else:
            return [attribute.__class__.__name__]

    def __setup_sampler(self):
        sampler_type = self._config["sampler"]["type"]
        sampler_method = self._config["sampler"]["method"]
        self.sampler: BaseSampler = sampler_register[sampler_type][sampler_method](
            **self._config["sampler"]["args"]
        )

    @abstractmethod
    def _set_dataset(self) -> None:
        raise NotImplementedError("Please implement this method.")

    @abstractmethod
    def _setup_model(self):
        raise NotImplementedError("Please implement this method.")

    @abstractmethod
    def predict(self, prompts):
        raise NotImplementedError("Please implement this method.")

    def __setup_dataloaders(self) -> None:
        """Setup the dataloaders for the experiment."""
        self._set_dataset()

        test_loader = DataLoader(
            self._dataset, batch_size=1, shuffle=False, num_workers=4
        )
        self.datasets = {"test": test_loader}

    def _setup_callbacks(self):
        self.callbacks = {}

    def on_experiment_start(self, exp: "IExperiment") -> None:
        super().on_experiment_start(exp)
        self.__setup_logger()
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
        # with self.logger.context_manager(self.dataset_key):
        #     self.logger.log_metrics(self.epoch_metrics, epoch=self.epoch_step)

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
        console_status.update(
            f"[bold green]Running: {self.dataset_key} "
            f"-> Epoch: {self.epoch_step}/{self.num_epochs} "
            f"-> Batch: {self.dataset_batch_step}/{len(self.dataset)}"
        )
        inputs, targets, id = self.batch
        assert targets.shape[1] == 1, "Individual targets should be binary."

        for target in targets:
            outputs = None
            aux = None
            for prompts in self.sampler.interact(mask=target, outs=outputs):
                outputs, aux = self.predict(
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
