from collections import defaultdict
from typing import Dict

from accelerate import Accelerator
from animus import IExperiment
from omegaconf import DictConfig
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.metrics import f1_score, get_stats, iou_score
from torch import Generator, no_grad, set_grad_enabled, zeros, stack, Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from ..models.predictor import BasePredictor
from ..samplers import BaseSampler, sampler_register
from ..utils import ConsoleLogger


def collate_fn(batch):
    samples = []
    targets = defaultdict(list)
    for sample, target in batch:
        samples.append(sample)
        for k, v in target.items():
            targets[k].append(v)
    samples = stack(samples)
    for k, v in targets.items():
        targets[k] = stack(v) if isinstance(v[0], Tensor) else v
    return samples, dict(targets)


class SegmentationExp(IExperiment):
    """
    Segmentation Exp. is a base class for training segmentation models.

    Args:
            config (Dict): Configuration dictionary for the experiment.
            engine (Engine): Engine to use for the experiment.
    """

    def __init__(
            self, 
            segmentor: Module, 
            dataset: Dataset, 
            config: DictConfig, 
            engine: Accelerator = Accelerator(),
        ):
        super().__init__()
        self._cfg = config
        self._dataset = dataset
        self.segmentor = segmentor
        self.device = self._cfg.device
        self.num_epochs = self._cfg.num_epochs
        self.metrics = {
            "iou_score": iou_score,
            "f1_score": f1_score,
        }
        self.engine = engine

    def _setup_model(self):
        self.criterion = DiceLoss(**self._cfg.loss)
        self.optimizer = AdamW(self.segmentor.parameters(), **self._cfg.optimizer)

        self.segmentor, self.optimizer = self.engine.prepare(
            self.segmentor,
            self.optimizer,
        )

    def __setup_dataloaders(self) -> None:
        """Setup the dataloaders for the experiment."""
        split_train = self._cfg.split
        generator = Generator().manual_seed(self._cfg.seed)
        train_data, val_data = random_split(
            self._dataset, (split_train, 1 - split_train), generator
        )
        train_loader = DataLoader(
            train_data,
            batch_size=self._cfg.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self._cfg.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
        train_loader, val_loader = self.engine.prepare(
            train_loader,
            val_loader,
        )
        self.datasets = {"train": train_loader, "val": val_loader}

    def _setup_callbacks(self):
        self.callbacks = {"console": ConsoleLogger(self._cfg)}

    def on_experiment_start(self, exp: "IExperiment") -> None:
        super().on_experiment_start(exp)
        self.__setup_dataloaders()
        self._setup_model()
        self._setup_callbacks()
        self._valid_done = True

    def on_epoch_start(self, exp: "IExperiment") -> None:
        if not (self._valid_done) and self.epoch_step % self._cfg.val_interval == 0:
            self.dataset_key = "val"
            self._valid_done = True
        else:
            self.dataset_key = "train"
            self._valid_done = False
            self.epoch_step += 1

        self.dataset = self.datasets[self.dataset_key]
        self.epoch_metrics: Dict = defaultdict(
            lambda: dict(
                zip(
                    [*self.metrics.keys()] + ["loss"],
                    [0.0] * (len(self.metrics)) + [999],
                )
            )
        )

    def on_dataset_start(self, exp: "IExperiment"):
        super().on_dataset_start(exp)
        self.dataset_metrics: Dict = defaultdict(lambda: 0.0)
        if not self.is_train_dataset:
            self.segmentor.eval()

    def run_batch(self) -> None:
        with set_grad_enabled(self.is_train_dataset):
            inputs, target_info = self.batch
            targets = target_info["masks"]

            outputs = self.segmentor(inputs)
            loss = self.criterion(outputs, targets.long())
            if self.is_train_dataset:
                self.engine.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # TODO: Put this into a callback object.
                # Log learning rate every batch
                # self.logger.log_metrics(
                #     {"lr": self.optimizer.param_groups[0]["lr"]}, step=self.batch_step
                # )
                # for scheduler in self.schedulers:
                #     scheduler.step(
                #         self.epoch_step + self.dataset_batch_step / len(self.dataset)
                #     )
        with no_grad():
            self.batch_metrics["loss"] = loss.sum().item()
            stats = get_stats(
                outputs.argmax(dim=1),
                targets.long(),
                mode=self._cfg.loss.mode,
                num_classes=self._cfg.model.classes
                # threshold=0.5,
            )
            for metric_name, metric in self.metrics.items():
                scores = self.engine.reduce(
                    metric(*stats, reduction="macro").cuda(),
                    reduction="mean"
                )
                self.batch_metrics[metric_name] = scores

            # self.batch_metrics = self.mean_reduce_ddp_metrics(self.batch_metrics)
            self.batch_metrics = {
                k: float(v) * (inputs.size(0) / len(self.dataset))
                for k, v in self.batch_metrics.items()
            }

    def run_epoch(self) -> None:
        self._run_event("on_dataset_start")
        self.run_dataset()
        self._run_event("on_dataset_end")

    def on_batch_end(self, exp: IExperiment) -> None:
        for metric_name, metric_value in self.batch_metrics.items():
            self.dataset_metrics[metric_name] += metric_value

    def _run_local(self, local_rank: int = -1, world_size: int = 1) -> None:
        self._local_rank, self._world_size = local_rank, world_size
        self._run_event("on_experiment_start")
        self.run_experiment()
        self._run_event("on_experiment_end")

    def _run(self) -> None:
        self._run_local()


class InteractiveTest(IExperiment):
    """
    InteractiveTest is a base class for interactive segmentation experiments.

    Args:
            config (Dict): Configuration dictionary for the experiment.
            engine (Engine): Engine to use for the experiment.
    """

    def __init__(self, predictor: BasePredictor, dataset: Dataset, config: DictConfig):
        super().__init__()
        self._cfg = config
        self._dataset = dataset
        self.predictor = predictor
        self.device = self._cfg.device
        self.num_epochs = 1
        self.metrics = {
            "iou_score": iou_score,
        }

    def __setup_sampler(self):
        sampler_type = self._cfg.sampler.type
        sampler_method = self._cfg.sampler.method
        self.sampler: BaseSampler = sampler_register[sampler_type][sampler_method](
            **self._cfg.sampler.args
        )

    def _setup_model(self):
        self.predictor._setup_model()
        self.predictor.eval()
        self.predictor.to(self.device)

    def __setup_dataloaders(self) -> None:
        """Setup the dataloaders for the experiment."""
        test_loader = DataLoader(
            self._dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
        self.datasets = {"test": test_loader}

    def _setup_callbacks(self):
        self.callbacks = {ConsoleLogger(self._cfg)}

    def on_experiment_start(self, exp: "IExperiment") -> None:
        super().on_experiment_start(exp)
        self.__setup_sampler()
        self.__setup_dataloaders()
        self._setup_model()
        self._setup_callbacks()

    def on_epoch_start(self, exp: "IExperiment") -> None:
        self.epoch_metrics: Dict = defaultdict(
            lambda: zeros((1, self._cfg.sampler.args.click_limit - 1))
        )

    def on_dataset_start(self, exp: "IExperiment"):
        super().on_dataset_start(exp)
        self.sampler.is_sampling = True

    # on_batch_start
    def on_batch_start(self, exp: IExperiment):
        super().on_batch_start(exp)
        self.batch_metrics: Dict = defaultdict(
            lambda: zeros((1, self._cfg.sampler.args.click_limit - 1))
        )

    @no_grad()
    def run_batch(self) -> None:
        inputs, targets = self.batch
        id = targets["image_id"]

        for target in targets["masks"]:
            outputs = None
            aux = None
            for click_step, prompts in enumerate(
                self.sampler.interact(mask=target, outs=outputs)
            ):
                outputs, aux = self.predictor.predict(
                    inputs, prompts, aux if aux is not None else outputs, id
                )
                self.sampler.set_outputs(outputs)
                stats = get_stats(outputs, target, mode="binary")
                # NOTE: This metric calculation is not the same as NOCs.
                for metric_name, metric in self.metrics.items():
                    score = metric(*stats, reduction="micro")
                    self.batch_metrics[metric_name][click_step](
                        float(score.item()) / len(self.dataset)
                    )

    def on_batch_end(self, exp: IExperiment) -> None:
        for metric_name, metric_value in self.batch_metrics.items():
            self.epoch_metrics[metric_name] += metric_value

    # on_dataset_end

    def run_epoch(self) -> None:
        self._run_event("on_dataset_start")
        self.run_dataset()
        self._run_event("on_dataset_end")

    def _run(self) -> None:
        self._run_event("on_experiment_start")
        self.run_experiment()
        self._run_event("on_experiment_end")
