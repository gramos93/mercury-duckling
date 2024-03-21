import os
import time
from accelerate import accelerator
from animus import ICallback, IExperiment
from comet_ml import Experiment
from omegaconf import DictConfig, OmegaConf
from pandas import json_normalize
from rich.console import Console
from rich.status import Status
from rich.table import Table


class ConsoleLogger(ICallback):
    def __init__(self, cfg: DictConfig, exp: "IExperiment") -> None:
        super().__init__()
        self._cfg = cfg
        self._console = Console(color_system="truecolor")
        self._console_status = Status("[bold green]Running", spinner="point")
        
        if self._cfg.logging.is_online:
            self._online_logger = Experiment(
                api_key=os.environ["COMET_API_KEY"],
                project_name=os.environ["COMET_PROJECT_NAME"],
                workspace=os.environ["COMET_WORKSPACE"],
                log_graph=False,
                parse_args=False,
                log_code=False,
                auto_log_co2=False,
                auto_param_logging=False,
                auto_output_logging="default",
                auto_metric_logging=False,
                auto_histogram_weight_logging=False,
                log_env_host=False,
                log_env_details=False,
                log_env_cpu=False,
                log_env_network=False,
                log_env_disk=False,
            )
            if tags := self._cfg.logging.get("tags", False):
                self._online_logger.add_tags(tags)

        self._exp_name = (
            self._online_logger.get_name()
            if self._cfg.logging.is_online
            else time.strftime("%Y-%m-%d-%H:%M:%S")
        )
        self._logging_dir = f"./checkpoints/{self._exp_name}"
        if not os.path.isdir(self._logging_dir):
            os.makedirs(self._logging_dir, exist_ok=True)
    
    def on_experiment_start(self, exp: "IExperiment") -> None:
        normed_config = json_normalize(
            OmegaConf.to_container(self._cfg, resolve=True),
            sep=" "
        )
        table_config = config2table(normed_config)
        self._console.log(table_config)
        if self._cfg.logging.is_online:
            self._online_logger.log_parameters(self._cfg)

    def on_batch_start(self, exp: IExperiment) -> None:
        self._console_status.update(
            f"[bold green]Running: {exp.dataset_key} "
            f"-> Epoch: {exp.epoch_step}/{exp.num_epochs} "
            f"-> Batch: {exp.dataset_batch_step}/{len(exp.dataset)}"
        )

    def on_epoch_end(self, exp: IExperiment) -> None:
        if self._cfg.logging.is_online:
            with self._online_logger.context_manager(exp.dataset_key):
                self._online_logger.log_metrics(
                    exp.dataset_metrics,
                    epoch=exp.epoch_step
                )

        metrics_str = ", ".join(
            "{}={:.3f}".format(key.title(), val)
            for (key, val) in exp.dataset_metrics.items()
        )
        self._console.log(
            f"[bold][red]Epoch: {exp.epoch_step}[/] - "
            f"[bold cyan]{exp.dataset_key.upper()}[/]"
            f" -> [magenta]metrics: {metrics_str}[/]"
        )

    def on_experiment_end(self, exp: "IExperiment") -> None:
        # TODO: Better track best metrics for experiment.
        # Now it's mostly used for the interactive testing.
        metrics_str = ", ".join(
            "{}={:.3f}".format(key.title(), val)
            for (key, val) in 
            exp.experiment_metrics[exp.epoch_step][exp.dataset_key].items()
        )
        self._console.log(
            f"[bold][red]Experiment Metrics: [/]"
            f"[magenta] {metrics_str}[/]"
        )


def config2table(config):
    config_table = Table(title="Experiment Configuration")
    config_table.add_column(
        "[cyan]Parameter",
        justify="left",
        style="cyan",
        no_wrap=True,
        header_style="bold",
    )
    config_table.add_column(
        "[magenta]Value",
        justify="center",
        style="magenta",
        no_wrap=True,
        header_style="bold",
    )
    for param, value in config.items():
        if param.startswith(('models', 'datasets')):
            continue
        config_table.add_row(param.title(), str(value[0]))

    return config_table
