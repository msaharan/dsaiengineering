"""Utilities for preparing and profiling datasets."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ColumnProfile",
    "DatasetProfile",
    "ProfileConfig",
    "load_profile_from_cache",
    "profile_dataset",
    "save_profile_to_cache",
]

_PROFILE_EXPORTS = tuple(__all__)


def __getattr__(name: str) -> Any:
    if name in _PROFILE_EXPORTS:
        from .profile import (  # Local import to avoid eager module execution
            ColumnProfile,
            DatasetProfile,
            ProfileConfig,
            load_profile_from_cache,
            profile_dataset,
            save_profile_to_cache,
        )

        globals().update(
            {
                "ColumnProfile": ColumnProfile,
                "DatasetProfile": DatasetProfile,
                "ProfileConfig": ProfileConfig,
                "load_profile_from_cache": load_profile_from_cache,
                "profile_dataset": profile_dataset,
                "save_profile_to_cache": save_profile_to_cache,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'src.data_prep' has no attribute '{name}'")


def __dir__() -> list[str]:  # pragma: no cover - introspection helper only
    return sorted(set(globals()) | set(_PROFILE_EXPORTS))

