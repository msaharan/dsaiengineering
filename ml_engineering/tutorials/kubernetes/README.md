# Kubernetes Crash Course

## 0. How to use this

We’ll build one small ML service and run it three ways:

1) Spin up a local K8s cluster.  
2) Build a single Docker image that has:
   - FastAPI inference
   - a training script
   - a batch prediction script  
3) Run:
   - a **Job** to train and save a model to a volume
   - a **Deployment + Service** to expose the API
   - a **CronJob** to run batch predictions on a schedule

Skip modules if you want; they still work independently, but together they feel closer to “real-ish” ML plumbing.

------

## 1. Prereqs & setup

### You should already know

- Python (basic scripting + virtualenvs)
- Docker basics (Dockerfile, `docker build`, `docker run`)

### You need installed

- **Docker**
- **kubectl**
- A local Kubernetes cluster: either **minikube** or **kind**. I’ll show `minikube`; swap commands if you use something else.

Start your cluster:

```bash
minikube start --cpus=4 --memory=8192
kubectl get nodes
```

Expect at least one `Ready` node.

------

## 2. Kubernetes mental model (DS/ML edition)

Picture it like this:

- **Cluster** – your compute farm.
- **Node** – a machine in that farm.
- **Pod** – the thing that actually runs; 1+ containers that always schedule together.
- **Deployment** – keeps Pods for stateless apps alive (replicas, rolling updates).
- **Service** – stable IP/DNS in front of Pods (load balancing, discovery).
- **Job** – run-to-completion workload (training).
- **CronJob** – scheduled Jobs (nightly scoring).
- **Volume / PVC** – storage that outlives Pods (models, data).
- **ConfigMap / Secret** – config and credentials.

------

## 3. Create a namespace for your ML stuff

Namespaces are like folders/projects.

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-course
```

Apply it:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl get ns
```

------

## 4. Build a simple Iris ML service image

I use one image with:

- `train.py` – trains a scikit-learn model and writes `model.joblib`
- `app.py` – FastAPI service that loads the model and serves `/predict`
- `batch_predict.py` – script for batch predictions (for the CronJob later)

### 4.1 Project structure

Create a folder, e.g.:

```bash
mkdir k8s-ml-course
cd k8s-ml-course
mkdir k8s
```

Place these files in the project root.

#### `requirements.txt`

```txt
fastapi
uvicorn[standard]
scikit-learn
joblib
pydantic
numpy
```

#### `app.py` – inference API

```python
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
    """Load the model if present, otherwise train a fresh one."""
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
```

#### `train.py` – training job

```python
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
```

#### `batch_predict.py` – batch scoring

```python
import os

import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.joblib")


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found at {MODEL_PATH}. Did you run the training Job?")

    model = joblib.load(MODEL_PATH)

    # Generate synthetic inputs shaped like iris features
    X = np.random.uniform(low=0.0, high=8.0, size=(5, 4))
    preds = model.predict(X)

    print("Batch predictions for 5 synthetic samples:")
    for i, pred in enumerate(preds, start=1):
        print(f"Sample {i}: class_index={int(pred)}")


if __name__ == "__main__":
    main()
```

#### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py train.py batch_predict.py .

# Where the model will be stored
ENV MODEL_PATH=/models/model.joblib

# Create non-root user and model directory
RUN useradd -m appuser && mkdir -p /models && chown -R appuser /models /app
USER appuser

# Default command: run the API (Jobs/CronJobs will override this)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 Build the image (minikube)

Option A – build directly in minikube’s Docker daemon:

```bash
eval "$(minikube docker-env)"     # point Docker at minikube
docker build -t iris-ml:latest .
```

(If you’re on a remote cluster, push to a registry like Docker Hub and use `your-username/iris-ml:latest` instead.)

------

## 5. Add persistent storage for the model

Use a **PersistentVolumeClaim** (PVC) the Job and API share.

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-model-pvc
  namespace: ml-course
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

Apply:

```bash
kubectl apply -f k8s/pvc.yaml
kubectl -n ml-course get pvc
```

------

## 6. Run a training Job

This simulates a “training pipeline step” that writes `model.joblib` into the PVC.

```yaml
# k8s/train-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: iris-train-job
  namespace: ml-course
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: iris-train
          image: iris-ml:latest      # or your-registry/iris-ml:latest
          imagePullPolicy: IfNotPresent
          command: ["python", "train.py"]
          env:
            - name: MODEL_PATH
              value: /models/model.joblib
          volumeMounts:
            - name: model-storage
              mountPath: /models
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: ml-model-pvc
  backoffLimit: 2
```

Apply & check:

```bash
kubectl -n ml-course apply -f k8s/train-job.yaml
kubectl -n ml-course get jobs
kubectl -n ml-course get pods
kubectl -n ml-course logs job/iris-train-job
```

You’re looking for `Model trained and saved to /models/model.joblib`.

------

## 7. Deploy the inference API (Deployment + Service)

### 7.1 Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-api
  namespace: ml-course
spec:
  replicas: 2
  selector:
    matchLabels:
      app: iris-api
  template:
    metadata:
      labels:
        app: iris-api
    spec:
      containers:
        - name: iris-api
          image: iris-ml:latest       # same image
          imagePullPolicy: IfNotPresent
          env:
            - name: MODEL_PATH
              value: /models/model.joblib
          ports:
            - containerPort: 8000
          volumeMounts:
            - name: model-storage
              mountPath: /models
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: ml-model-pvc
```

### 7.2 Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: iris-api-svc
  namespace: ml-course
spec:
  selector:
    app: iris-api
  type: NodePort          # great for minikube; use LoadBalancer in cloud
  ports:
    - port: 80
      targetPort: 8000
      nodePort: 30080     # must be between 30000–32767
```

Apply:

```bash
kubectl -n ml-course apply \
  -f k8s/deployment.yaml \
  -f k8s/service.yaml

kubectl -n ml-course get pods
kubectl -n ml-course get svc
```

### 7.3 Call the API

With minikube you can do:

```bash
minikube service iris-api-svc -n ml-course --url
```

Or using the NodePort:

```bash
NODE_IP=$(minikube ip)
curl http://$NODE_IP:30080/health
```

Prediction example:

```bash
curl -X POST "http://$NODE_IP:30080/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

You should get something like:

```json
{"species_index":0,"species_name":"setosa"}
```

You can also visit `http://$NODE_IP:30080/docs` in a browser for the FastAPI Swagger UI.

------

## 8. Batch predictions with a CronJob

Now we’ll simulate nightly/batch scoring that reuses the same model.

```yaml
# k8s/cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: iris-batch-predict
  namespace: ml-course
spec:
  schedule: "*/5 * * * *"   # every 5 minutes (for demo)
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: iris-batch-predict
              image: iris-ml:latest
              imagePullPolicy: IfNotPresent
              command: ["python", "batch_predict.py"]
              env:
                - name: MODEL_PATH
                  value: /models/model.joblib
              volumeMounts:
                - name: model-storage
                  mountPath: /models
          volumes:
            - name: model-storage
              persistentVolumeClaim:
                claimName: ml-model-pvc
```

Apply:

```bash
kubectl -n ml-course apply -f k8s/cronjob.yaml
kubectl -n ml-course get cronjobs
```

After a few minutes:

```bash
kubectl -n ml-course get jobs
kubectl -n ml-course get pods
# Replace <pod-name> with the latest pod from the CronJob
kubectl -n ml-course logs <pod-name>
```

Typical logs look like:

```text
Batch predictions for 5 synthetic samples:
Sample 1: class_index=2
...
```

Real-world analogy: this could be “nightly scoring job” writing to S3 / warehouse instead of printing.

------

## 9. ConfigMaps & Secrets (12‑factor ML services)

Instead of hardcoding env vars, use ConfigMaps/Secrets.

### 9.1 ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: iris-api-config
  namespace: ml-course
data:
  MODEL_PATH: "/models/model.joblib"
  LOG_LEVEL: "info"
```

### 9.2 Secret (e.g., API key)

> NOTE: values are base64‑encoded.

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: iris-api-secret
  namespace: ml-course
type: Opaque
data:
  API_KEY: c3VwZXItc2VjcmV0LWFwaS1rZXk=  # "super-secret-api-key" base64
```

Use them in the Deployment:

```yaml
# snippet inside spec.template.spec.containers[0] in deployment.yaml

        - name: iris-api
          image: iris-ml:latest
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: iris-api-config
            - secretRef:
                name: iris-api-secret
          ports:
            - containerPort: 8000
          volumeMounts:
            - name: model-storage
              mountPath: /models
```

Then in `app.py` you’d read `os.getenv("API_KEY")` or `LOG_LEVEL`.

Apply:

```bash
kubectl -n ml-course apply -f k8s/configmap.yaml -f k8s/secret.yaml
kubectl -n ml-course rollout restart deployment/iris-api
```

------

## 10. Scaling & autoscaling

### 10.1 Resource requests/limits

We already added:

```yaml
resources:
  requests:
    cpu: "250m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"
```

This helps Kubernetes pack Pods onto nodes and avoid noisy neighbors.

### 10.2 HorizontalPodAutoscaler (optional)

You’ll need a metrics server (for minikube: `minikube addons enable metrics-server`).

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iris-api-hpa
  namespace: ml-course
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iris-api
  minReplicas: 2
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

Apply:

```bash
kubectl -n ml-course apply -f k8s/hpa.yaml
kubectl -n ml-course get hpa
```

Under load, Kubernetes will scale `iris-api` Pods between 2 and 5 replicas.

------

## 11. Debugging essentials for ML workloads

You’ll use these commands constantly:

```bash
# What's running?
kubectl -n ml-course get pods,svc,jobs,deploy,cronjobs

# Describe a failing object
kubectl -n ml-course describe pod <pod-name>
kubectl -n ml-course describe job iris-train-job

# Logs (single pod)
kubectl -n ml-course logs <pod-name>

# Logs for all pods in Deployment
kubectl -n ml-course logs deploy/iris-api

# Shell into a container
kubectl -n ml-course exec -it <pod-name> -- /bin/bash

# Port-forward to debug directly
kubectl -n ml-course port-forward svc/iris-api-svc 8080:80
curl http://localhost:8080/health
```

Common issues:

- **ImagePullBackOff** – image name/tag wrong or registry auth issues.
- **CrashLoopBackOff** – usually Python exception; check `kubectl logs`.
- **Pending PVC** – no storage class; check `kubectl get pvc` and cluster storage settings.

------

## 12. GPU training example (bonus snippet)

If you have GPU nodes with NVIDIA plugin installed, a Job might look like:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-train-job
  namespace: ml-course
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: gpu-train
          image: your-registry/deep-learning-train:latest
          resources:
            limits:
              nvidia.com/gpu: 1
```

The pattern is identical to the CPU training Job – just different image and GPU resource limit.

------

## 13. Troubleshooting and stopping

- **Image can’t be pulled** for Job/Deployment: build inside minikube’s Docker (`eval "$(minikube docker-env)" && docker build -t iris-ml:latest .`), then rerun the Job:
  ```bash
  kubectl -n ml-course delete job/iris-train-job --ignore-not-found
  kubectl -n ml-course apply -f k8s/train-job.yaml
  ```
  Or push to a registry and update `image:` in the manifests.
- **minikube start fails on memory**: reduce `--memory` to what Docker Desktop allows (e.g., 6000–7000 MB) or increase Docker Desktop’s memory limit.
- **Stop/cleanup**:
  ```bash
  kubectl delete namespace ml-course
  minikube stop         # or: minikube delete
  eval "$(minikube docker-env -u)"   # reset Docker to host
  docker system prune -f             # optional: remove unused containers/networks
  docker system prune -a -f          # optional: also remove unused images
  ```
