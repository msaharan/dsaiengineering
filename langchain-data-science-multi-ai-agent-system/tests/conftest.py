import json

import pandas as pd
import pytest

from src.datasets import DatasetCatalog
from src.ds_agent_tools import DataScienceContext


@pytest.fixture
def sample_catalog(tmp_path):
    left_frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "country": ["us", "fr", "de"],
        }
    )
    right_frame = pd.DataFrame(
        {
            "identifier": [1, 2, 4],
            "sales": [10.0, 12.5, 7.2],
        }
    )

    csv_path = tmp_path / "left.csv"
    parquet_path = tmp_path / "right.parquet"

    left_frame.to_csv(csv_path, index=False)
    right_frame.to_parquet(parquet_path, index=False)

    manifest_path = tmp_path / "catalog.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": {
                    "left_dataset": {
                        "uri": str(csv_path),
                        "format": "csv",
                    },
                    "right_dataset": {
                        "uri": str(parquet_path),
                        "format": "parquet",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    return DatasetCatalog(manifest_path=manifest_path, base_path=tmp_path)


@pytest.fixture
def sample_context(sample_catalog, monkeypatch):
    context = DataScienceContext(catalog=sample_catalog)
    monkeypatch.setattr("src.ds_agent_tools._get_context", lambda: context)
    return context

