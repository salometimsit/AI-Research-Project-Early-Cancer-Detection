"""
Centralized logging for the HCC pipeline (text + JSONL + helpers).

Pipeline stage: cross-cutting; used by every module in ``src/`` and every
script in ``scripts/``.

Inputs
------
* Phase name (``"hcc"`` or ``"hcc.<phase>"``).
* Optional log file under ``logs/``; the same path is reused as
  ``<basename>.jsonl`` for a structured record.
* Optional console level (``DEBUG``/``INFO``/``WARNING``/``ERROR``).
  Falls back to ``HCC_LOG_LEVEL`` env var when not provided.

Outputs
-------
* ``logs/<phase>.log`` -- human-readable text log (``DEBUG`` and above).
* ``logs/<phase>.jsonl`` -- one JSON event per line, easy to ``jq``.
* Console output -- coloured when ``sys.stdout.isatty()``.

Side effects
------------
* Creates the ``logs/`` directory and parent paths on demand.
* Suppresses log propagation so messages do not double-print.

Failure modes
-------------
* If ``log_file`` cannot be written, only console handlers stay active and
  a warning is emitted via ``logging.basicConfig``.

Helpers
-------
* :func:`log_section`   -- visual separator banner.
* :func:`log_dict`      -- pretty JSON dump at the chosen level.
* :func:`log_exception` -- traceback + structured context.
* :func:`tqdm_logger`   -- iterator wrapper that drops a DEBUG message
  every N items, useful for long non-tty runs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Iterator

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


# ANSI colour codes, applied only when stdout is a tty.
_LEVEL_COLOURS = {
    "DEBUG": "\033[36m",      # cyan
    "INFO": "\033[32m",       # green
    "WARNING": "\033[33m",    # yellow
    "ERROR": "\033[31m",      # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """Drop-in formatter that wraps the level name in ANSI colour codes."""

    def __init__(self, fmt: str, datefmt: str) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        original = record.levelname
        padded = original.ljust(8)
        record.levelname = f"{colour}{padded}{_RESET}" if colour else padded
        try:
            return super().format(record)
        finally:
            record.levelname = original


class _JSONLFormatter(logging.Formatter):
    """Emit one JSON object per line for structured grep/jq workflows."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exc_type"] = exc_type.__name__ if exc_type else None
            payload["exc_msg"] = str(exc_value)
            payload["exc_info"] = self.formatException(record.exc_info)
        extras = getattr(record, "_extra", None)
        if extras:
            payload["extra"] = extras
        return json.dumps(payload, ensure_ascii=False, default=str)


def _resolve_console_level(console_level: str | int | None) -> int:
    if console_level is None:
        console_level = os.environ.get("HCC_LOG_LEVEL", "INFO")
    if isinstance(console_level, str):
        return logging.getLevelName(console_level.upper())  # type: ignore[return-value]
    return int(console_level)


def setup_logger(
    name: str = "hcc",
    log_file: Path | str | None = None,
    level: int = logging.DEBUG,
    console_level: str | int | None = None,
) -> logging.Logger:
    """Configure the root pipeline logger with console + text + JSONL handlers.

    Parameters
    ----------
    name:
        Logger namespace, typically ``"hcc"`` or ``"hcc.<phase>"``.
    log_file:
        Path to the text log file. The companion JSONL file is written
        next to it with a ``.jsonl`` suffix. Parent directories are
        created on demand.
    level:
        File handler minimum level (default ``DEBUG``).
    console_level:
        Console handler minimum level. Defaults to ``HCC_LOG_LEVEL`` env
        var, then ``INFO``.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    text_formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)
    console_formatter: logging.Formatter
    if sys.stdout.isatty() and os.name != "nt":
        console_formatter = _ColourFormatter(_DEFAULT_FORMAT, _DATE_FORMAT)
    else:
        console_formatter = text_formatter

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(_resolve_console_level(console_level))
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(text_formatter)
            logger.addHandler(file_handler)

            jsonl_path = log_path.with_suffix(log_path.suffix + ".jsonl") if log_path.suffix else log_path.with_suffix(".jsonl")
            jsonl_handler = logging.FileHandler(jsonl_path, mode="a", encoding="utf-8")
            jsonl_handler.setLevel(level)
            jsonl_handler.setFormatter(_JSONLFormatter())
            logger.addHandler(jsonl_handler)
        except OSError as exc:
            logging.basicConfig()
            logging.getLogger("hcc").warning(
                "Could not attach file handlers (%s); console-only logging.",
                exc,
            )

    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger for a module, ensuring the root logger is configured."""
    if not _CONFIGURED:
        setup_logger("hcc", log_file=None)
    if name.startswith("hcc"):
        return logging.getLogger(name)
    return logging.getLogger(f"hcc.{name}")


# ---------------------------------------------------------------------------
# Helpers for easy debugging
# ---------------------------------------------------------------------------


def log_section(logger: logging.Logger, title: str, char: str = "=") -> None:
    """Print a visually obvious banner so phases are easy to spot in logs."""
    line = char * max(8, min(72, len(title) + 6))
    logger.info(line)
    logger.info("%s %s %s", char * 2, title, char * 2)
    logger.info(line)


def log_dict(
    logger: logging.Logger,
    label: str,
    data: Any,
    level: int = logging.DEBUG,
) -> None:
    """Pretty-print a dict (or any JSON-serialisable object) at ``level``."""
    try:
        payload = json.dumps(data, indent=2, default=str, sort_keys=True)
    except TypeError:
        payload = repr(data)
    for line in payload.splitlines() or [""]:
        logger.log(level, "%s | %s", label, line)


def log_exception(
    logger: logging.Logger,
    exc: BaseException,
    context: dict[str, Any] | None = None,
) -> None:
    """Log a traceback plus contextual key/values for fast triage."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if context:
        log_dict(logger, "exception_context", context, level=logging.ERROR)
    logger.error("Exception:\n%s", tb)


def tqdm_logger(
    iterable: Iterable[Any],
    logger: logging.Logger,
    desc: str = "progress",
    every: int = 1,
) -> Iterator[Any]:
    """Iterate while emitting a DEBUG progress line every ``every`` items.

    Useful when running pipelines under nohup / CI where ``tqdm``'s
    in-place updates do not render. The wrapped iterator still yields
    the original items, so callers can drop it in transparently.
    """
    start = time.time()
    count = 0
    for item in iterable:
        count += 1
        if every > 0 and (count % every == 0):
            elapsed = time.time() - start
            gpu_mem_suffix = ""
            try:
                import torch  # local import — torch is optional

                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.memory_reserved() / 1e9
                    gpu_mem_suffix = f" | GPU mem reserved: {gpu_mem_gb:.2f} GB"
            except Exception:  # noqa: BLE001
                pass
            logger.debug(
                "%s | %d items in %.1fs (%.2f items/s)%s",
                desc,
                count,
                elapsed,
                count / elapsed if elapsed > 0 else 0.0,
                gpu_mem_suffix,
            )
        yield item
    elapsed = time.time() - start
    logger.debug("%s | done | %d items in %.1fs", desc, count, elapsed)
