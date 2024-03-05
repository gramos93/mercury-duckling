from animus import ICallback, IExperiment
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.status import Status
from rich.table import Table
from pandas import json_normalize


class ConsoleLogger(ICallback):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._console = Console(color_system="truecolor")
        self._console_status = Status("[bold green]Running", spinner="point")

    def on_experiment_start(self, exp: "IExperiment") -> None:
        # table_config = config2table(OmegaConf.to_container(self._cfg))
        # self._console.log(table_config)
        pass

    def on_batch_start(self, exp: IExperiment):
        # This function will run AFTER the exp.on_batch_start
        if hasattr(exp, "engine"):
            if not exp.engine.is_local_main_process:
                return

        self._console_status.update(
            f"[bold green]Running: {exp.dataset_key} "
            f"-> Epoch: {exp.epoch_step}/{exp.num_epochs} "
            f"-> Batch: {exp.dataset_batch_step}/{len(exp.dataset)}"
        )

    def on_epoch_end(self, exp: IExperiment) -> None:
        if hasattr(exp, "engine"):
            if not exp.engine.is_local_main_process:
                return
            
        # This function will run BEFORE the exp.on_batch_start
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
        # TODO: print summary of metrics from the exp.
        pass


def config2table(config):
    config_table = Table(title="Experiment Configuration")
    normed_config = json_normalize(config, sep=" ")
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

    for param, value in normed_config.items():
        config_table.add_row(param.title(), str(value[0]))

    return config_table
