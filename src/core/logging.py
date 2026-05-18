"""Logging configuration for CLI entry points.

Configured once at startup per `.claude/rules/logging.md`. The caller passes a
full path so the log file can live inside the per-run output directory.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"


def setup_logging(log_file: Path | str, level: int = logging.INFO) -> Path:
    """Stream INFO+ to stdout and to `log_file`. Returns the log file path."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    return log_file
