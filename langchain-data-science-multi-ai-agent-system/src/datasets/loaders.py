"""Format-specific dataset loaders built on pandas."""

from __future__ import annotations

import sqlite3
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import DataSource
    import pandas as pd


class DataFormat(str, Enum):
    """Enumeration of supported dataset storage formats."""

    CSV = "csv"
    PARQUET = "parquet"
    SQLITE = "sqlite"


class DatasetLoader(Protocol):
    """Protocol for objects capable of loading a dataset into a DataFrame."""

    def load(
        self,
        *,
        source: "DataSource",
        path: Path,
        options: Mapping[str, Any],
    ) -> "pd.DataFrame":  # pragma: no cover - exercised via concrete loaders
        ...


def _load_pandas():
    return import_module("pandas")


class CSVLoader:
    """Load comma-separated files using pandas."""

    def load(
        self,
        *,
        source: "DataSource",
        path: Path,
        options: Mapping[str, Any],
    ) -> "pd.DataFrame":
        pandas = _load_pandas()
        return pandas.read_csv(path, **options)


class ParquetLoader:
    """Load parquet datasets using pandas/pyarrow."""

    def load(
        self,
        *,
        source: "DataSource",
        path: Path,
        options: Mapping[str, Any],
    ) -> "pd.DataFrame":
        pandas = _load_pandas()
        return pandas.read_parquet(path, **options)


class SQLiteLoader:
    """Load tables or queries from a SQLite database."""

    def load(
        self,
        *,
        source: "DataSource",
        path: Path,
        options: Mapping[str, Any],
    ) -> "pd.DataFrame":
        query: Optional[str] = options.get("query") if isinstance(options, dict) else None
        table: Optional[str] = options.get("table") if isinstance(options, dict) else None
        limit: Optional[int] = options.get("limit") if isinstance(options, dict) else None

        if query is None and table is None:
            raise ValueError(
                "SQLite loader requires either a 'query' or 'table' option to be provided."
            )

        with sqlite3.connect(path) as conn:
            if query is not None:
                effective_query = query
            else:
                effective_query = f"SELECT * FROM {table}"
            if limit is not None and "limit" not in effective_query.lower():
                effective_query = f"{effective_query} LIMIT {int(limit)}"
            pandas = _load_pandas()
            return pandas.read_sql_query(effective_query, conn)


_DEFAULT_LOADERS: Dict[DataFormat, DatasetLoader] = {
    DataFormat.CSV: CSVLoader(),
    DataFormat.PARQUET: ParquetLoader(),
    DataFormat.SQLITE: SQLiteLoader(),
}


def register_loader(format_: DataFormat, loader: DatasetLoader) -> None:
    """Register or overwrite a loader for a given data format."""

    _DEFAULT_LOADERS[format_] = loader


def get_loader_for_format(format_: DataFormat) -> DatasetLoader:
    """Retrieve the loader registered for the provided data format."""

    try:
        return _DEFAULT_LOADERS[format_]
    except KeyError as exc:  # pragma: no cover - protection against misconfiguration
        raise ValueError(f"No dataset loader registered for format '{format_}'.") from exc


def detect_format_from_path(path: Path) -> DataFormat | None:
    """Infer a dataset format from a file path using its suffix."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return DataFormat.CSV
    if suffix in {".parquet", ".pq"}:
        return DataFormat.PARQUET
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return DataFormat.SQLITE
    return None


__all__ = [
    "CSVLoader",
    "DatasetLoader",
    "DataFormat",
    "ParquetLoader",
    "SQLiteLoader",
    "detect_format_from_path",
    "get_loader_for_format",
    "register_loader",
]

