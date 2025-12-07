# Copyright 2025 Mohit Saharan (github.com/msaharan)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset profiling helpers for exploratory data analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.datasets import DataSource, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "data" / "profiles"
PROFILE_CACHE_VERSION = 1


def _ensure_iterable(values: Iterable[Any], limit: int) -> list[Any]:
    result: list[Any] = []
    for value in values:
        result.append(value)
        if len(result) >= limit:
            break
    return result


@dataclass
class ProfileConfig:
    """Tunable parameters controlling profiling behaviour."""

    sample_size: int = 5
    top_k: int = 5
    cache_dir: Path = DEFAULT_PROFILE_DIR


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    unique_values: int | None = None
    sample_values: list[Any] = field(default_factory=list)
    top_frequencies: list[tuple[Any, int]] = field(default_factory=list)
    summary_stats: Mapping[str, Any] | None = None


@dataclass
class DatasetProfile:
    dataset: str
    row_count: int
    columns: list[ColumnProfile]
    profile_version: int = PROFILE_CACHE_VERSION
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Convert tuples to lists for JSON compatibility
        for column in payload.get("columns", []):
            if "top_frequencies" in column:
                column["top_frequencies"] = [list(pair) for pair in column["top_frequencies"]]
        return payload


def _compute_summary_stats(series) -> Mapping[str, Any]:  # type: ignore[no-untyped-def]
    pandas = series.__class__.__module__.startswith("pandas")
    if not pandas:
        return {}
    if series.dtype.kind in {"i", "u", "f"}:
        desc = series.describe()
        return {
            "mean": float(desc.get("mean", 0.0)),
            "std": float(desc.get("std", 0.0)),
            "min": float(desc.get("min", 0.0)),
            "max": float(desc.get("max", 0.0)),
        }
    if series.dtype.kind == "M":
        desc = series.describe(datetime_is_numeric=True)
        min_value = desc.get("min")
        max_value = desc.get("max")
        if hasattr(min_value, "isoformat"):
            min_value = min_value.isoformat()
        if hasattr(max_value, "isoformat"):
            max_value = max_value.isoformat()
        return {
            "min": min_value,
            "max": max_value,
        }
    return {}


def _profile_column(series, config: ProfileConfig) -> ColumnProfile:  # type: ignore[no-untyped-def]
    non_null = series.count()
    null_count = int(series.shape[0] - non_null)
    unique_values = None
    try:
        unique_values = int(series.nunique(dropna=True))
    except Exception:  # pragma: no cover - fallback for unsupported dtypes
        unique_values = None

    sample_values = _ensure_iterable(series.dropna().head(config.sample_size), config.sample_size)
    try:
        top_freq_series = series.value_counts(dropna=True).head(config.top_k)
        top_frequencies = []
        for index, count in top_freq_series.items():
            if hasattr(index, "isoformat"):
                display_index = index.isoformat()
            else:
                display_index = index
            top_frequencies.append((display_index, int(count)))
    except Exception:  # pragma: no cover - fallback when value_counts unsupported
        top_frequencies = []

    summary_stats = _compute_summary_stats(series)

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        non_null_count=int(non_null),
        null_count=null_count,
        unique_values=unique_values,
        sample_values=[
            value.isoformat() if hasattr(value, "isoformat") else value
            for value in sample_values
        ],
        top_frequencies=top_frequencies,
        summary_stats=summary_stats,
    )


def profile_dataset(
    source: DataSource,
    *,
    config: ProfileConfig | None = None,
    base_path: Path | None = None,
    use_cache: bool = True,
) -> DatasetProfile:
    config = config or ProfileConfig()
    if base_path is None:
        base_path = PROJECT_ROOT
    cache_path = config.cache_dir / f"{source.name}.json"
    if use_cache and cache_path.exists():
        cached = load_profile_from_cache(cache_path)
        if cached is not None:
            return cached

    dataframe = load_dataset(source, base_path=base_path)
    columns: list[ColumnProfile] = []
    for column in dataframe.columns:
        series = dataframe[column]
        columns.append(_profile_column(series, config))

    profile = DatasetProfile(
        dataset=source.name,
        row_count=int(dataframe.shape[0]),
        columns=columns,
        metadata=source.metadata,
    )

    if use_cache:
        save_profile_to_cache(profile, cache_path)

    return profile


def save_profile_to_cache(profile: DatasetProfile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(profile.to_dict(), handle, indent=2, default=str)


def load_profile_from_cache(path: Path) -> DatasetProfile | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("profile_version") != PROFILE_CACHE_VERSION:
        return None
    columns_payload: Sequence[Mapping[str, Any]] = payload.get("columns", [])
    columns = [
        ColumnProfile(
            name=column.get("name", ""),
            dtype=column.get("dtype", ""),
            non_null_count=column.get("non_null_count", 0),
            null_count=column.get("null_count", 0),
            unique_values=column.get("unique_values"),
            sample_values=column.get("sample_values", []),
            top_frequencies=[tuple(item) for item in column.get("top_frequencies", [])],
            summary_stats=column.get("summary_stats"),
        )
        for column in columns_payload
    ]
    return DatasetProfile(
        dataset=payload.get("dataset", ""),
        row_count=payload.get("row_count", 0),
        columns=columns,
        profile_version=payload.get("profile_version", PROFILE_CACHE_VERSION),
        generated_at=payload.get("generated_at", ""),
        metadata=payload.get("metadata"),
    )


def main(dataset_name: str | None = None) -> None:
    from src.datasets import load_catalog

    catalog = load_catalog()
    if dataset_name is None:
        print("Available datasets:")
        for entry in catalog.entries():
            print(f"- {entry.name}")
        return

    source = catalog.get(dataset_name)
    profile = profile_dataset(source)
    print(json.dumps(profile.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover - manual execution only
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset profiles.")
    parser.add_argument("name", nargs="?", help="Dataset name to profile.")
    args = parser.parse_args()
    main(args.name)

