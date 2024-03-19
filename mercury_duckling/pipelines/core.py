from collections import defaultdict
from typing import Dict

from accelerate import Accelerator, DistributedDataParallelKwargs
from animus import IExperiment
from omegaconf import DictConfig
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.metrics import f1_score, get_stats, iou_score
from torch import (
    Generator, 
    no_grad, 
    set_grad_enabled, 
    zeros,
    zeros_like,
    stack, 
    Tensor, 
    isnan,    
    argwhere as tch_argwhere,
    cuda
)
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from ..models.predictor import BasePredictor
from ..models import SymmetricUnifiedFocalLoss, softIOU
from ..samplers import BaseSampler, sampler_register
from ..utils import ConsoleLogger, ModelLogger


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
        engine: Accelerator = Accelerator(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        ),
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
        self.criterion = softIOU()#(**self._cfg.loss)
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
        if self.engine.is_local_main_process:
            self.callbacks = {
                "logger": ConsoleLogger(self._cfg, self),
                "model_save": ModelLogger(
                    metric_name="iou_score",
                    minimise=False,
                    model_attr="segmentor",
                )
            }
        else:
            self.callbacks = {}

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
        else:
            self.segmentor.train()

    def run_batch(self) -> None:
        with set_grad_enabled(self.is_train_dataset):
            inputs, target_info = self.batch
            targets = target_info["masks"]

            outputs = self.segmentor(inputs)
            loss = self.criterion(outputs, targets.long())
            if self.is_train_dataset:
                if isnan(loss).any():
                    loss = zeros_like(loss, device=loss.device, requires_grad=True)
                self.engine.backward(loss)
                # clip_grad_norm_(self.segmentor.parameters(), max_norm=20.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
        with no_grad():
            self.batch_metrics["loss"] = loss
            stats = get_stats(
                outputs.sigmoid(),
                targets.long(),
                mode=self._cfg.loss.mode,
                threshold=0.5,
            )
            for metric_name, metric in self.metrics.items():
                self.batch_metrics[metric_name] = metric(
                    *stats, reduction="micro"
                ).cuda()

            self.batch_metrics = {
                k: self.engine.reduce(
                    v * (inputs.size(0) / len(self.dataset.dataset)), "sum"
                ).item()
                for k, v in self.batch_metrics.items()
            }
        cuda.empty_cache()

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
        # TODO: pass device here to the construction of the model.
        self.predictor._setup_model(self.device)
        self.predictor.eval()
        self.predictor.to(self.device)

    def __setup_dataloaders(self) -> None:
        """Setup the dataloaders for the experiment."""
        # split_train = 0.99
        # generator = Generator().manual_seed(self._cfg.seed)
        # train_data, val_data = random_split(
        #     self._dataset, (split_train, 1 - split_train), generator
        # )
        test_loader = DataLoader(
            # val_data,
            self._dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
        self.dataset = test_loader

    def _setup_callbacks(self):
        self.callbacks = {"logger": ConsoleLogger(self._cfg)}

    def on_experiment_start(self, exp: "IExperiment") -> None:
        super().on_experiment_start(exp)
        self.__setup_sampler()
        self.__setup_dataloaders()
        self._setup_model()
        self._setup_callbacks()

    def on_epoch_start(self, exp: "IExperiment") -> None:
        self.epoch_step += 1
        self.dataset_key = "test"

    def on_dataset_start(self, exp: "IExperiment"):
        self.dataset_metrics: Dict = defaultdict(lambda : [])
        self.sampler.is_sampling = True

    # on_batch_start
    def on_batch_start(self, exp: IExperiment):
        super().on_batch_start(exp)
        self.batch_metrics: Dict = defaultdict(
            lambda: zeros((1, self._cfg.sampler.args.click_limit))
        )

    @no_grad()
    def run_batch(self) -> None:
        inputs, targets = self.batch
        id = targets["image_id"]
        # The targets["masks"] shape is [B, Blobs, H, W] and we need [Blobs, C, H, W]
        targets["masks"] = targets["masks"].permute(1, 0, 2, 3).squeeze(1)
        print(f"Image ID: {id} with {len(targets['masks'])} defects.")
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
                for metric_name, metric in self.metrics.items():
                    score = metric(*stats, reduction="micro")
                    self.batch_metrics[metric_name][0, click_step] += float(
                        score.item()
                    ) / len(targets["masks"])

    def on_batch_end(self, exp: IExperiment) -> None:
        for metric_name, metric_value in self.batch_metrics.items():
            self.dataset_metrics[metric_name].append(metric_value)

    def on_dataset_end(self, exp: IExperiment):
        for metric_name, metric_value in self.dataset_metrics.items():
            scores = stack(metric_value).mean(dim=0)
            noc_score = NOCS(scores, 0.75, 20)[0] # The batch is already averaged here.
            self.dataset_metrics[metric_name] = noc_score

            scores = ', '.join((f"{i:.3f}" for i in scores[0].tolist()))
            self.callbacks["logger"]._console.log(
                "[bold][red]Dataset IOU per click: [/]"
                f" -> [magenta]metrics: {scores}[/]"
            )
        super().on_dataset_end(exp)
        
    # on_epoch_end
            
    def run_epoch(self) -> None:
        self._run_event("on_dataset_start")
        self.run_dataset()
        self._run_event("on_dataset_end")

    def _run(self) -> None:
        self._run_event("on_experiment_start")
        self.run_experiment()
        self._run_event("on_experiment_end")


def NOCS(ious, thresh, max_clicks=20):
    """Number of clicks to reach threshold"""
    nocs = []
    for i in ious:
        iou_above_thresh = tch_argwhere(i > thresh)
        if len(iou_above_thresh) == 0:
            nocs.append(max_clicks)
        else:
            nocs.append(
                min(
                    iou_above_thresh[0].item(), max_clicks
                )
            )
    return nocs
