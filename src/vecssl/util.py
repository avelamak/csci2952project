"""Utils for vecssl"""

import logging
from logging import FileHandler
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

_CONSOLE: Optional[Console] = None


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    reset: bool = True,
    rich_tracebacks: bool = True,
    show_level: bool = False,
    show_path: bool = False,
) -> Console:
    """
    Configure root logging with a single RichHandler (console) and optional FileHandler.
    Returns the Console so Progress can share it.
    """
    global _CONSOLE
    if _CONSOLE is not None and not reset:
        return _CONSOLE
    # Single console for both logs and progress
    console = Console(stderr=True)
    # No duplicate handler unless `reset==True`
    root = logging.getLogger()
    if reset:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Use rich handler
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=rich_tracebacks,
        show_level=show_level,
        show_path=show_path,
        markup=True,
    )

    root.addHandler(rich_handler)

    # File handler if logging to file
    if log_file:
        fh = FileHandler(log_file, mode="w")
        fh.setLevel(root.level)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(fh)

    _CONSOLE = console
    return console


def make_progress(console: Optional[Console] = None) -> Progress:
    """Progress that shares the same Console as the RichHandler."""
    console = console or get_console()
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,  # same console as logger
        transient=False,
    )


def get_console() -> Console:
    if _CONSOLE is None:
        # Use default if we forget to do setup at entry point
        return setup_logging()
    return _CONSOLE
