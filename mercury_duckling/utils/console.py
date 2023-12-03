from rich.console import Console
from rich.status import Status

console_status = Status("[bold green]Running", spinner="point")
console = Console(color_system="truecolor")

__all__ = ["console", "console_status"]
