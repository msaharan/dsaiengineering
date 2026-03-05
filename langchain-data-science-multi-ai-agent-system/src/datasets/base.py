"""Core dataset abstractions built atop pandas DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    import pandas as pd

from .loaders import (
    DataFormat,
    DatasetLoader,
    detect_format_from_path,
    get_loader_for_format,
)


@dataclass
class DataSource:
    """Declarative description of a tabular dataset."""

    name: str
    uri: str | Path
    format: DataFormat | None = None
    description: str | None = None
    read_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, base_path: Path | None = None) -> Path:
        """Return an absolute path to the underlying resource."""

        raw_path = Path(self.uri)
        if raw_path.is_absolute():
            return raw_path

        if base_path is None:
            base_path = Path.cwd()

        return (base_path / raw_path).expanduser().resolve()

    def infer_format(self, *, resolved_path: Path | None = None) -> DataFormat:
        """Infer the dataset format using explicit configuration or heuristics."""

        if self.format is not None:
            return self.format

        path = resolved_path or self.resolve_path()
        detected = detect_format_from_path(path)
        if detected is None:
            raise ValueError(
                f"Unable to infer dataset format for '{self.name}' from path '{path}'."
            )
        self.format = detected
        return detected

    def load(
        self,
        *,
        base_path: Path | None = None,
        loader: DatasetLoader | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "pd.DataFrame":
        """Load the dataset into a pandas DataFrame."""

        resolved_path = self.resolve_path(base_path)
        data_format = self.infer_format(resolved_path=resolved_path)
        active_loader = loader or get_loader_for_format(data_format)
        merged_options: dict[str, Any] = dict(self.read_options)
        if overrides:
            merged_options.update(overrides)
        return active_loader.load(
            source=self,
            path=resolved_path,
            options=merged_options,
        )


def load_dataset(
    source: DataSource,
    *,
    base_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> "pd.DataFrame":
    """Helper for loading a `DataSource` without instantiating the class consumer."""

    return source.load(base_path=base_path, overrides=overrides)


__all__ = ["DataSource", "load_dataset"]

