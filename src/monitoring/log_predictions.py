import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time
from typing import Tuple
import logging
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utils
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config("src/config/config.yaml")

# MLflow setup
MODEL_URI = config["api"]["production_model_uri"]
PRODUCTION_DATA_PATH = config["api"]["production_data_path"]
PREDICTION_THRESHOLD = config["api"]["prediction_threshold"]

# Global model instance to avoid reloading
_production_model = None

def load_production_model():
    """Load the production model once and cache it"""
    global _production_model
    if _production_model is None:
        try:
            _production_model = mlflow.pyfunc.load_model(MODEL_URI)
            logger.info(f"Loaded production model: {MODEL_URI}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    return _production_model

def predict_and_log(applications: list, log_to_mlflow: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Single function that handles everything: model loading, prediction, logging
    Returns predictions and logging metadata
    """
    start_time = time.time()
    
    # Load model
    model = load_production_model()
    
    # Convert to DataFrame
    df = pd.DataFrame(applications)
    
    # Make predictions
    predictions = model.predict(df)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Prepare logging data
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "batch_size": len(applications),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
        "model_uri": MODEL_URI,
        "latency_ms": latency_ms,
        "predictions_count": len(predictions)
    }
    
    # Log to CSV (production data)
    save_to_production_csv(df, predictions, latency_ms)
    
    # Log to MLflow (metrics only, no run creation)
    if log_to_mlflow:
        log_to_mlflow_metrics(log_data)
    
    return predictions, log_data

def save_to_production_csv(df: pd.DataFrame, predictions: np.ndarray, latency_ms: float):
    """Append production data with predictions to CSV"""
    try:
        # Create production directory if it doesn't exist
        os.makedirs(os.path.dirname(PRODUCTION_DATA_PATH), exist_ok=True)
        
        # Add predictions and metadata
        df_with_predictions = df.copy()
        df_with_predictions['PREDICTION'] = predictions
        df_with_predictions['PREDICTION_BINARY'] = (predictions > PREDICTION_THRESHOLD).astype(int)
        df_with_predictions['TIMESTAMP'] = datetime.utcnow().isoformat()
        df_with_predictions['MODEL_VERSION'] = MODEL_URI
        df_with_predictions['LATENCY_MS'] = latency_ms
        
        # Append to CSV (or create if doesn't exist)
        if os.path.exists(PRODUCTION_DATA_PATH):
            df_with_predictions.to_csv(PRODUCTION_DATA_PATH, mode='a', header=False, index=False)
        else:
            df_with_predictions.to_csv(PRODUCTION_DATA_PATH, index=False)
        
        logger.info(f"Saved {len(df)} records to {PRODUCTION_DATA_PATH} (latency: {latency_ms:.2f}ms)")
        
    except Exception as e:
        logger.error(f"Failed to save to CSV: {str(e)}")

def log_to_mlflow_metrics(log_data: dict):
    """Log metrics to MLflow using a single, active run"""
    try:
        # Set MLflow tracking URI from config if available
        if "mlflow" in config and "tracking_uri" in config["mlflow"]:
            mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        
        # Get or create experiment
        experiment_name = "production_monitoring"
        
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Create experiment
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow experiment: {experiment_name} with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.debug(f"Using existing experiment: {experiment_name} with ID: {experiment_id}")
        
        # Set experiment before starting run
        mlflow.set_experiment(experiment_id=experiment_id)
        
        # Start a run with nested=True
        with mlflow.start_run(run_name="production_predictions", nested=True, experiment_id=experiment_id) as run:
            # Log batch metrics
            mlflow.log_metrics({
                "prediction_mean": log_data["prediction_mean"],
                "prediction_std": log_data["prediction_std"],
                "batch_size": log_data["batch_size"],
                "latency_ms": log_data["latency_ms"]
            })
            
            # Add tags
            mlflow.set_tags({
                "last_prediction_time": log_data["timestamp"],
                "model_uri": log_data["model_uri"],
                "run_type": "production_monitoring"
            })
        
        logger.debug(f"Logged metrics to MLflow experiment {experiment_id}, run: {run.info.run_id}")
        
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {str(e)}")