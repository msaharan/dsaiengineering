"""Dataset abstractions and loader registry for tabular sources."""

from __future__ import annotations

from typing import Any

from .base import DataSource, load_dataset
from .loaders import (
    DataFormat,
    DatasetLoader,
    detect_format_from_path,
    get_loader_for_format,
    register_loader,
)

__all__ = [
    "DataFormat",
    "DatasetLoader",
    "DataSource",
    "DatasetCatalog",
    "detect_format_from_path",
    "get_loader_for_format",
    "load_dataset",
    "load_catalog",
    "register_loader",
]

_LAZY_ATTRS = {"DatasetCatalog", "load_catalog"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        from .catalog import DatasetCatalog, load_catalog

        globals().update(
            {
                "DatasetCatalog": DatasetCatalog,
                "load_catalog": load_catalog,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'src.datasets' has no attribute '{name}'")

