"""Utils for vecssl"""

import logging
import os
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

# We set a global Console variable so we
# never double format
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
    The log level can be overridden by setting the LOG_LEVEL environment variable.
    """
    global _CONSOLE
    if _CONSOLE is not None and not reset:
        return _CONSOLE

    # Check environment variable for log level (overrides parameter)
    env_level = os.environ.get("LOG_LEVEL", level).upper()

    # Single console for both logs and progress
    console = Console(stderr=True)
    # No duplicate handler unless `reset==True`
    root = logging.getLogger()
    if reset:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(getattr(logging, env_level, logging.INFO))

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


def linear(a, b, x, min_x, max_x):
    """
    b             ___________
                /|
               / |
    a  _______/  |
              |  |
           min_x max_x
    """
    return a + min(max((x - min_x) / (max_x - min_x), 0), 1) * (b - a)


def batchify(data, device):
    return (d.unsqueeze(0).to(device) for d in data)


def _make_seq_first(*args):
    # N, G, S, ... -> S, G, N, ...
    if len(args) == 1:
        (arg,) = args
        return arg.permute(2, 1, 0, *range(3, arg.dim())) if arg is not None else None
    return (
        *(arg.permute(2, 1, 0, *range(3, arg.dim())) if arg is not None else None for arg in args),
    )


def _make_batch_first(*args):
    # S, G, N, ... -> N, G, S, ...
    if len(args) == 1:
        (arg,) = args
        return arg.permute(2, 1, 0, *range(3, arg.dim())) if arg is not None else None
    return (
        *(arg.permute(2, 1, 0, *range(3, arg.dim())) if arg is not None else None for arg in args),
    )


def _pack_group_batch(*args):
    # S, G, N, ... -> S, G * N, ...
    if len(args) == 1:
        (arg,) = args
        return (
            arg.reshape(arg.size(0), arg.size(1) * arg.size(2), *arg.shape[3:])
            if arg is not None
            else None
        )
    return (
        *(
            arg.reshape(arg.size(0), arg.size(1) * arg.size(2), *arg.shape[3:])
            if arg is not None
            else None
            for arg in args
        ),
    )


def _unpack_group_batch(N, *args):
    # S, G * N, ... -> S, G, N, ...
    if len(args) == 1:
        (arg,) = args
        return arg.reshape(arg.size(0), -1, N, *arg.shape[2:]) if arg is not None else None
    return (
        *(
            arg.reshape(arg.size(0), -1, N, *arg.shape[2:]) if arg is not None else None
            for arg in args
        ),
    )
