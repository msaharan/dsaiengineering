"""Dataset catalog manifest handling and CLI utilities."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping

from .base import DataSource
from .loaders import DataFormat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"


@dataclass
class CatalogEntry:
    """Lightweight description of a dataset as stored in the manifest."""

    name: str
    config: Mapping[str, object]

    @property
    def format(self) -> DataFormat | None:
        fmt = self.config.get("format")
        if fmt is None:
            return None
        return DataFormat(fmt)

    @property
    def description(self) -> str | None:
        value = self.config.get("description")
        return value if isinstance(value, str) else None


class DatasetCatalog:
    """Load and manage a collection of `DataSource` definitions."""

    def __init__(
        self,
        manifest_path: Path | None = None,
        *,
        base_path: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path or DEFAULT_CATALOG_PATH)
        self.base_path = base_path or PROJECT_ROOT
        self._raw_manifest = self._load_manifest()
        self._datasets: dict[str, DataSource] = {}

    def _load_manifest(self) -> MutableMapping[str, object]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset catalog not found at {self.manifest_path}."
                " Create it or pass --catalog when invoking the CLI."
            )
        with self.manifest_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _entry_configs(self) -> Mapping[str, Mapping[str, object]]:
        datasets = self._raw_manifest.get("datasets", {})
        if not isinstance(datasets, Mapping):
            raise ValueError("Catalog manifest must contain a 'datasets' mapping.")
        return datasets  # type: ignore[return-value]

    def entries(self) -> Iterator[CatalogEntry]:
        for name, config in sorted(self._entry_configs().items()):
            yield CatalogEntry(name=name, config=config)

    def get(self, name: str) -> DataSource:
        key = name.strip()
        if not key:
            raise ValueError("Dataset name must not be blank.")
        if key not in self._datasets:
            config = self._entry_configs().get(key)
            if config is None:
                raise KeyError(f"Dataset '{name}' not found in catalog.")
            self._datasets[key] = self._build_source(key, config)
        return self._datasets[key]

    def load(self, name: str, **overrides: object):
        source = self.get(name)
        options = overrides.pop("overrides", None)
        if options is None and overrides:
            options = overrides
        return source.load(base_path=self.base_path, overrides=options)

    def _build_source(self, name: str, config: Mapping[str, object]) -> DataSource:
        uri = config.get("uri") or config.get("path")
        if uri is None:
            raise ValueError(f"Dataset '{name}' is missing a 'uri' field.")
        format_value = config.get("format")
        data_format = DataFormat(format_value) if format_value else None
        read_options = config.get("read_options") or {}
        metadata = config.get("metadata") or {}
        description = config.get("description")
        if not isinstance(read_options, Mapping):
            raise ValueError(f"Dataset '{name}' has non-mapping read_options.")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Dataset '{name}' has non-mapping metadata.")
        return DataSource(
            name=name,
            uri=str(uri),
            format=data_format,
            description=description if isinstance(description, str) else None,
            read_options=dict(read_options),
            metadata=dict(metadata),
        )


def load_catalog(
    manifest_path: Path | None = None,
    *,
    base_path: Path | None = None,
) -> DatasetCatalog:
    """Convenience helper mirroring `DatasetCatalog` construction."""

    return DatasetCatalog(manifest_path=manifest_path, base_path=base_path)


def _cmd_list(catalog: DatasetCatalog, verbose: bool = False) -> None:
    for entry in catalog.entries():
        summary = f"{entry.name}"
        if entry.format is not None:
            summary += f" ({entry.format.value})"
        if verbose and entry.description:
            summary += f": {entry.description}"
        print(summary)


def _cmd_describe(catalog: DatasetCatalog, name: str, show_options: bool = False) -> None:
    source = catalog.get(name)
    print(f"Name: {source.name}")
    print(f"URI: {source.uri}")
    print(f"Format: {source.format.value if source.format else 'auto'}")
    print(f"Description: {source.description or '(none)'}")
    if source.metadata:
        print("Metadata:")
        for key, value in source.metadata.items():
            print(f"  - {key}: {value}")
    if show_options and source.read_options:
        print("Read options:")
        for key, value in source.read_options.items():
            print(f"  - {key}: {value}")


def _cmd_load(catalog: DatasetCatalog, name: str, limit: int | None) -> None:
    overrides = {"overrides": {}}
    if limit is not None:
        overrides["overrides"]["nrows"] = limit
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    frame = catalog.load(name, **overrides)
    print(frame.head())


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dataset catalog utilities.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="Path to the catalog manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=PROJECT_ROOT,
        help="Base directory used to resolve relative dataset URIs.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List datasets in the catalog.")
    list_parser.add_argument("--verbose", action="store_true", help="Show descriptions.")

    describe_parser = subparsers.add_parser(
        "describe", help="Show manifest details for a dataset."
    )
    describe_parser.add_argument("name", help="Dataset identifier to describe.")
    describe_parser.add_argument(
        "--show-options",
        action="store_true",
        help="Include read options in the output.",
    )

    load_parser = subparsers.add_parser(
        "load", help="Load a dataset and print the first few rows."
    )
    load_parser.add_argument("name", help="Dataset identifier to load.")
    load_parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of rows to read (maps to pandas nrows).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    catalog = DatasetCatalog(
        manifest_path=args.catalog,
        base_path=args.base_path,
    )

    if args.command == "list":
        _cmd_list(catalog, verbose=args.verbose)
    elif args.command == "describe":
        _cmd_describe(catalog, args.name, show_options=args.show_options)
    elif args.command == "load":
        _cmd_load(catalog, args.name, args.limit)
    else:  # pragma: no cover - safeguard for future extensions
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - exercised via manual CLI use
    main()

