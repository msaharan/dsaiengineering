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

import os
from pathlib import Path

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "raw.csv"
CLEAN_PATH = DATA_DIR / "clean.csv"


def generate_raw_if_missing():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        return
    rng = np.random.default_rng(42)
    n = 100
    ages = rng.integers(18, 70, size=n)
    incomes = rng.normal(50000, 15000, size=n)
    churn = (ages > 50).astype(int)
    df = pd.DataFrame({"age": ages, "income": incomes, "churn": churn})
    # introduce some missing values
    df.loc[rng.choice(n, size=5, replace=False), "income"] = np.nan
    df.to_csv(RAW_PATH, index=False)
    print(f"Generated synthetic raw dataset at {RAW_PATH}")


def clean():
    generate_raw_if_missing()
    print(f"Reading raw data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # Simple cleaning: drop rows with missing income, clip extreme values
    before = len(df)
    df = df.dropna(subset=["income"])
    after = len(df)
    print(f"Dropped {before - after} rows with missing income")

    df["income"] = df["income"].clip(lower=0, upper=200000)
    df["age_bucket"] = pd.cut(df["age"], bins=[17, 30, 45, 60, 80],
                              labels=["18-30", "31-45", "46-60", "61+"])

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_PATH}")


if __name__ == "__main__":
    clean()
