import os

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.joblib")


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
