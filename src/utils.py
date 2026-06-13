"""
Utility helpers for logging, date handling, and I/O operations.
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configure and return a logger with console and optional file handler.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level (e.g. ``logging.INFO``).
        log_file: Optional path to a log file.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def text_hash(text: str) -> str:
    """Return SHA-256 hex digest of ``text`` for deduplication.

    Args:
        text: Input string.

    Returns:
        Hexadecimal hash string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_date_parse(
    date_str: Optional[str],
    fmt: str = "%Y-%m-%dT%H:%M:%S%z",
) -> Optional[datetime]:
    """Parse an ISO datetime string safely, returning ``None`` on failure.

    Args:
        date_str: Datetime string.
        fmt: Expected format.

    Returns:
        Parsed :class:`datetime` or ``None``.
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, fmt)
    except (ValueError, TypeError):
        try:
            return datetime.fromisoformat(date_str)
        except (ValueError, TypeError):
            return None


def now_iso() -> str:
    """Return current UTC datetime as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    format: str = "parquet",
    **kwargs: Any,
) -> None:
    """Save a DataFrame to disk in Parquet or CSV format.

    Args:
        df: Data to persist.
        path: Destination path (extension is optional when ``format`` is set).
        format: ``"parquet"`` (default) or ``"csv"``.
        **kwargs: Additional arguments passed to the writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    target = path.with_suffix(f".{format}")
    if format == "parquet":
        df.to_parquet(target, index=False, **kwargs)
    elif format == "csv":
        df.to_csv(target, index=False, encoding="utf-8-sig", **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'.")


def load_dataframe(
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a DataFrame from disk, inferring format from extension if needed.

    Args:
        path: File path.
        format: ``"parquet"`` or ``"csv"``. If ``None``, inferred from suffix.
        **kwargs: Additional arguments passed to the reader.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If format cannot be inferred or is unsupported.
    """
    path = Path(path)
    if format is None:
        suffix = path.suffix.lstrip(".")
        if suffix in ("parquet", "pq"):
            format = "parquet"
        elif suffix in ("csv", "tsv"):
            format = "csv"
        else:
            raise ValueError(
                f"Cannot infer format from {path.suffix}. "
                "Specify the `format` argument explicitly."
            )

    if format == "parquet":
        return pd.read_parquet(path, **kwargs)
    elif format == "csv":
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Serialize ``data`` to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
