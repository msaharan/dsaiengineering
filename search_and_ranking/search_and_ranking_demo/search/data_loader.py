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

from pathlib import Path
import pandas as pd

from .ontology import extract_attributes


def load_catalog(data_dir: Path) -> pd.DataFrame:
    """Load catalog and create a combined text field."""
    catalog = pd.read_csv(data_dir / "catalog.csv")
    catalog["ontology_attrs"] = catalog.apply(
        lambda row: extract_attributes(
            description=row.get("description", ""),
            cuisine=row.get("cuisine", ""),
            price_range=row.get("price_range", ""),
        ),
        axis=1,
    )
    catalog["text"] = (
        catalog["name"].fillna("")
        + " "
        + catalog["description"].fillna("")
        + " "
        + catalog["cuisine"].fillna("")
    ).str.lower()
    return catalog


def load_query_doc_labels(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "query_doc_labels.csv")


def load_query_intents(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "query_intents.csv")


def get_data_dir(base_dir: Path | None = None) -> Path:
    """Return the data directory relative to the project root."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "data"
