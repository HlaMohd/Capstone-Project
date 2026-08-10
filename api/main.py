from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATH = BASE_DIR / "models" / "final_random_forest.pkl"
SCALER_PATH = BASE_DIR / "models" / "final_random_forest_scaler.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "final_random_forest_threshold.pkl"

# Create FastAPI application
app = FastAPI(
    title="SWaT Attack Detection API",
    description="REST API for detecting attacks in the SWaT industrial control system dataset.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: list[float]


# Load final model
model = joblib.load(MODEL_PATH)

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Load classification threshold
threshold = joblib.load(THRESHOLD_PATH)


@app.get("/")
def root():
    return {
        "message": "SWaT Attack Detection API is running",
        "model": "Tuned Random Forest",
        "threshold": float(threshold)
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    X = np.array(request.features).reshape(1, -1)

    if X.shape[1] != 44:
        return {
            "error": f"Expected 44 features, received {X.shape[1]}"
        }

    X_scaled = scaler.transform(X)

    probability = model.predict_proba(X_scaled)[0, 1]

    prediction = int(probability >= threshold)

    label = "Attack" if prediction == 1 else "Normal"

    return {
        "prediction": prediction,
        "label": label,
        "attack_probability": float(probability),
        "threshold": float(threshold)
    }