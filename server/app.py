import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Fix import path so src/ is discoverable
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger
from src.data_ingestion import ingest_data
from src.data_validation import validate_data
from src.data_transformation import transform_data
from src.model_trainer import train_model
from src.model_evaluate import evaluate_model

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API")
logger = get_logger()

# Pydantic input model
class HeartInput(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str


MODEL_PATH = "artifacts/model/best_model.pkl"


def run_pipeline():
    """
    Runs the full ML pipeline:
    1. Data ingestion
    2. Validation
    3. Transformation
    4. Model training
    5. Model evaluation
    """
    try:
        logger.info("Step 1: Data Ingestion")
        ingest_data()

        logger.info("Step 2: Data Validation")
        valid = validate_data()
        if not valid:
            return {"status": "Pipeline stopped", "reason": "Validation failed"}

        logger.info("Step 3: Data Transformation")
        transformed = transform_data()
        if not transformed:
            return {"status": "Pipeline stopped", "reason": "Data transformation failed"}

        logger.info("Step 4: Model Training")
        acc = train_model()

        logger.info("Step 5: Model Evaluation")
        evaluate_model()

        return {"status": "Pipeline completed", "accuracy": acc}

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return {"status": "Pipeline failed", "error": str(e)}


@app.on_event("startup")
def startup_event():
    """
    Automatically run ML pipeline on startup.
    """
    logger.info(" Starting ML pipeline automatically on app startup...")
    result = run_pipeline()
    logger.info(f" Pipeline result: {result}")


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running!"}


@app.post("/predict")
def predict(input_data: HeartInput):
    try:
        # Ensure model is loaded
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model not found at {MODEL_PATH}")
            return {"error": "Model not trained yet. Please rerun the pipeline."}

        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully for prediction.")

        # Convert input to DataFrame
        data_dict = input_data.dict()
        X = pd.DataFrame([data_dict])

        # Predict
        prediction = model.predict(X)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e)}
