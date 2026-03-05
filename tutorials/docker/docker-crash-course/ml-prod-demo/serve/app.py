from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = Path("/models/iris_logreg.joblib")

app = FastAPI(title="Iris Classifier API")

model = None
target_names = None


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.on_event("startup")
def load_model():
    global model, target_names
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Did you run the training container?"
        )
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    target_names = bundle["target_names"]
    print(f"Loaded model from {MODEL_PATH}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    pred_idx = int(model.predict(X)[0])
    return {
        "predicted_class_index": pred_idx,
        "predicted_class_name": str(target_names[pred_idx]),
        "input": features.dict(),
    }
