from rich.console import Console
from rich.table import Table
from rich.status import Status
from pandas import json_normalize

console_status = Status("[bold green]Running", spinner="point")
console = Console(color_system="truecolor")

def config2table(config):
    config_table = Table(title="Experiment Configuration")
    # normed_config = json_normalize(config, sep=" ")
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
        config_table.add_row(param.title(),str(value[0]))

    return config_table
