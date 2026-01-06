# serving/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from src.monitoring.log_predictions import predict_and_log, PREDICTION_THRESHOLD

app = FastAPI(title="Credit Approval API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify actual origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class CreditApplication(BaseModel):
    CODE_GENDER: str
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    AMT_INCOME_TOTAL: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: int
    AGE: int
    YEARS_EMPLOYED: int

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "credit-approval-api"}

@app.post("/predict")
def predict(applications: List[CreditApplication]):
    """
    Predict credit risk for applicants.
    """
    # Convert to list of dicts
    apps_list = [app.dict() for app in applications]
    
    # Single call that handles everything
    predictions, log_metadata = predict_and_log(apps_list, log_to_mlflow=True)
    
    # Prepare response
    results = []
    for i, (app, pred) in enumerate(zip(applications, predictions)):
        results.append({
            "applicant_id": i,
            "prediction": float(pred),
            "risk_category": "bad" if pred > PREDICTION_THRESHOLD else "good",
            "probability": float(pred)
        })
    
    return {
        "predictions": results,
        "batch_metadata": {
            "size": log_metadata["batch_size"],
            "timestamp": log_metadata["timestamp"],
            "latency_ms": log_metadata["latency_ms"],
            "model_used": log_metadata["model_uri"]
        }
    }