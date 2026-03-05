"""
FastAPI Inference Service
Healthcare AI - Hospital Readmission Prediction
REST API with prediction logging and Prometheus metrics
"""

import time
import logging
import joblib
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Prometheus Metrics ───────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["model_version", "prediction_label"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to make a prediction",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)
READMISSION_PROBABILITY = Histogram(
    "readmission_probability",
    "Distribution of readmission probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
HIGH_RISK_GAUGE = Gauge(
    "high_risk_patients_last_hour",
    "Number of high risk patients in last hour"
)
REQUEST_COUNTER = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"]
)

# ─── App Initialization ───────────────────────────────────────────
app = FastAPI(
    title="Healthcare AI - Readmission Prediction API",
    description="Predicts 30-day hospital readmission risk for diabetes patients",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model & Config ──────────────────────────────────────────
with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

model = None
feature_pipeline = None

def load_artifacts():
    global model, feature_pipeline
    try:
        model = joblib.load("data/processed/best_model.pkl")
        feature_pipeline = joblib.load("data/processed/feature_pipeline.pkl")
        logger.info("Model and feature pipeline loaded successfully")
    except FileNotFoundError:
        logger.warning("Model artifacts not found. Run training pipeline first.")


# ─── Request/Response Schemas ─────────────────────────────────────
class PatientData(BaseModel):
    race: Optional[str] = Field(default="Caucasian", example="Caucasian")
    gender: str = Field(example="Male")
    age: str = Field(example="[60-70)")
    admission_type_id: int = Field(example=1)
    discharge_disposition_id: int = Field(example=1)
    admission_source_id: int = Field(example=7)
    time_in_hospital: int = Field(ge=1, le=14, example=5)
    num_lab_procedures: int = Field(ge=0, example=44)
    num_procedures: int = Field(ge=0, example=1)
    num_medications: int = Field(ge=0, example=15)
    number_outpatient: int = Field(ge=0, example=0)
    number_emergency: int = Field(ge=0, example=0)
    number_inpatient: int = Field(ge=0, example=1)
    number_diagnoses: int = Field(ge=1, example=9)
    max_glu_serum: Optional[str] = Field(default="None", example="None")
    A1Cresult: Optional[str] = Field(default="None", example=">7")
    insulin: Optional[str] = Field(default="No", example="Steady")
    change: Optional[str] = Field(default="No", example="Ch")
    diabetesMed: Optional[str] = Field(default="Yes", example="Yes")

    class Config:
        schema_extra = {
            "example": {
                "race": "Caucasian", "gender": "Female", "age": "[70-80)",
                "admission_type_id": 1, "discharge_disposition_id": 1,
                "admission_source_id": 7, "time_in_hospital": 8,
                "num_lab_procedures": 72, "num_procedures": 2,
                "num_medications": 21, "number_outpatient": 0,
                "number_emergency": 1, "number_inpatient": 2,
                "number_diagnoses": 9, "max_glu_serum": "None",
                "A1Cresult": ">8", "insulin": "Up",
                "change": "Ch", "diabetesMed": "Yes"
            }
        }


class PredictionResponse(BaseModel):
    patient_id: str
    readmission_risk: str
    probability: float
    risk_score: int
    recommendation: str
    model_version: str
    timestamp: str


# ─── Helper Functions ─────────────────────────────────────────────
def get_risk_label(probability: float) -> tuple:
    if probability >= 0.7:
        return "HIGH", 3, "Immediate follow-up required. Schedule discharge planning and post-care coordination."
    elif probability >= 0.4:
        return "MEDIUM", 2, "Schedule follow-up appointment within 7 days. Review medication compliance."
    else:
        return "LOW", 1, "Standard discharge protocol. Provide patient education materials."


def log_prediction(patient_data: dict, response: dict):
    """Log prediction to file for audit trail."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": patient_data,
        "output": response
    }
    log_path = Path("data/processed/prediction_logs.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# ─── API Endpoints ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_artifacts()
    logger.info("Healthcare AI API started successfully")


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Healthcare AI - Readmission Prediction",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_readmission(patient: PatientData):
    """
    Predict 30-day hospital readmission risk for a diabetes patient.
    Returns risk level (LOW/MEDIUM/HIGH), probability, and clinical recommendation.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")

    start_time = time.time()

    try:
        # Convert to DataFrame
        patient_dict = patient.dict()
        df = pd.DataFrame([patient_dict])

        # Feature engineering
        df_processed = feature_pipeline.transform(df)
        target = CONFIG["data"]["target_column"]
        if target in df_processed.columns:
            df_processed = df_processed.drop(columns=[target])

        # Predict
        probability = float(model.predict_proba(df_processed)[0][1])
        risk_label, risk_score, recommendation = get_risk_label(probability)

        # Build response
        patient_id = f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:16]}"
        response = {
            "patient_id": patient_id,
            "readmission_risk": risk_label,
            "probability": round(probability, 4),
            "risk_score": risk_score,
            "recommendation": recommendation,
            "model_version": "champion-v1",
            "timestamp": datetime.now().isoformat()
        }

        # Prometheus metrics
        latency = time.time() - start_time
        PREDICTION_COUNTER.labels(model_version="champion-v1", prediction_label=risk_label).inc()
        PREDICTION_LATENCY.observe(latency)
        READMISSION_PROBABILITY.observe(probability)

        # Log prediction
        log_prediction(patient_dict, response)

        REQUEST_COUNTER.labels(endpoint="/predict", method="POST", status="200").inc()
        logger.info(f"Prediction: {patient_id} | Risk: {risk_label} | Prob: {probability:.4f} | Latency: {latency:.3f}s")

        return response

    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="/predict", method="POST", status="500").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(patients: list[PatientData]):
    """Batch prediction endpoint for multiple patients."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100 patients.")

    results = []
    for patient in patients:
        result = await predict_readmission(patient)
        results.append(result)

    high_risk_count = sum(1 for r in results if r["readmission_risk"] == "HIGH")
    HIGH_RISK_GAUGE.set(high_risk_count)

    return {"total_patients": len(results), "high_risk_count": high_risk_count, "predictions": results}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return current model information."""
    return {
        "model_name": CONFIG["mlflow"]["model_registry_name"],
        "version": "champion-v1",
        "model_type": type(model).__name__ if model else None,
        "features": len(CONFIG["features"]["numeric_columns"]) + len(CONFIG["features"]["categorical_columns"]),
        "target": CONFIG["data"]["target_column"]
    }


# ─── Run Server ───────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=CONFIG["api"]["host"],
        port=CONFIG["api"]["port"],
        reload=True,
        log_level="info"
    )
