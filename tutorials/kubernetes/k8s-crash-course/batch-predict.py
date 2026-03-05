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
