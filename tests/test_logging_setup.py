import logging
from pathlib import Path
import pytest
from vecssl.util import setup_logging, make_progress
from rich.logging import RichHandler
import time


def test_setup_logging_attaches_richhandler(tmp_path: Path):
    console = setup_logging(level="INFO", log_file=str(tmp_path / "train.log"), reset=True)
    root = logging.getLogger()
    handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
    assert len(handlers) == 1
    assert console is handlers[0].console  # same console for logs


def test_progress_uses_same_console(capsys):
    console = setup_logging(level="INFO", reset=True)
    with make_progress(console) as progress:
        t = progress.add_task("Train", total=3)
        progress.advance(t)
        progress.advance(t)
        progress.advance(t)
    # Just ensure something was printed and no exceptions
    out = capsys.readouterr().err  # Rich writes to stderr by default
    assert "Train" in out


def test_no_duplicate_handlers_on_second_call():
    _ = setup_logging(level="INFO", reset=True)
    before = len(logging.getLogger().handlers)
    _ = setup_logging(level="INFO", reset=False)  # should not add again
    after = len(logging.getLogger().handlers)
    assert before == after


def test_file_logging_written(tmp_path: Path):
    log_file = tmp_path / "train.log"
    _ = setup_logging(level="INFO", log_file=str(log_file), reset=True)
    logging.getLogger(__name__).info("hello file")
    # flush file handlers
    for h in logging.getLogger().handlers:
        if hasattr(h, "flush"):
            h.flush()
    assert log_file.exists()
    assert "hello file" in log_file.read_text()


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING"])
def test_levels_work(level):
    _ = setup_logging(level=level, reset=True)
    logger = logging.getLogger("vecssl.test")
    logger.log(getattr(logging, level), "msg")  # should not raise


def test_visual():
    console = setup_logging(level="INFO", log_file=None, reset=True)
    log = logging.getLogger("demo")
    with make_progress(console) as prog:
        t = prog.add_task("Train", total=10)
        for i in range(10):
            time.sleep(0.05)
            prog.advance(t)
            log.info("step %d done", i + 1)
