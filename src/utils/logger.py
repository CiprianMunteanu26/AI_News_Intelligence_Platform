"""
Structured logging for the AI News Intelligence Platform.

Provides colored console output and optional file logging
with consistent formatting across all pipeline stages.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI color codes for console output
# ---------------------------------------------------------------------------
class _Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"

    LEVEL_MAP = {
        "DEBUG": GRAY,
        "INFO": GREEN,
        "WARNING": YELLOW,
        "ERROR": RED,
        "CRITICAL": f"{BOLD}{RED}",
    }


# ---------------------------------------------------------------------------
# Custom Formatter
# ---------------------------------------------------------------------------
class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes based on log level."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color = _Colors.LEVEL_MAP.get(record.levelname, _Colors.RESET)
        fmt = f"{color}{self.FMT}{_Colors.RESET}"
        formatter = logging.Formatter(fmt, datefmt=self.DATE_FMT)
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes)."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(self.FMT, datefmt=self.DATE_FMT)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------
_initialized: set[str] = set()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """
    Create or retrieve a named logger with colored console output.

    Parameters
    ----------
    name : str
        Logger name (usually ``__name__`` of the calling module).
    level : int
        Logging level (default: INFO).
    log_file : optional path
        If provided, also logs to this file with plain formatting.

    Returns
    -------
    logging.Logger
    """
    if name in _initialized:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ColorFormatter())
    logger.addHandler(console)

    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(_PlainFormatter())
        logger.addHandler(file_handler)

    _initialized.add(name)
    return logger
