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

import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.joblib")


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found at {MODEL_PATH}. Did you run the training Job?")

    model = joblib.load(MODEL_PATH)

    X = np.random.uniform(low=0.0, high=8.0, size=(5, 4))
    preds = model.predict(X)

    print("Batch predictions for 5 synthetic samples:")
    for i, pred in enumerate(preds, start=1):
        print(f"Sample {i}: class_index={int(pred)}")


if __name__ == "__main__":
    main()
