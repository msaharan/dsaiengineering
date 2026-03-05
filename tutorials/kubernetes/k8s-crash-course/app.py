import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.joblib")


class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class Prediction(BaseModel):
    species_index: int
    species_name: str


app = FastAPI(title="Iris Classifier API")

iris_dataset = load_iris()


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"No model found at {MODEL_PATH}, training a new one...")
        X, y = iris_dataset.data, iris_dataset.target
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Trained and saved model to {MODEL_PATH}")
    return model


model = load_or_train_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: Features):
    try:
        x = np.array(
            [
                [
                    features.sepal_length,
                    features.sepal_width,
                    features.petal_length,
                    features.petal_width,
                ]
            ]
        )
        pred = model.predict(x)[0]
        species_name = iris_dataset.target_names[pred]
        return Prediction(species_index=int(pred), species_name=str(species_name))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
