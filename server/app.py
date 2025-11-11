
import joblib
import pandas as pd
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Assuming your pipeline functions and logger are set up correctly
# from utils.logger import get_logger
# from src.data_ingestion import ingest_data
# from src.data_validation import validate_data
# from src.data_transformation import transform_data
# from src.model_trainer import train_model
# from src.model_evaluate import evaluate_model

# --- Placeholder/Mock Functions for Pipeline & Logger ---
# You MUST ensure your actual imported functions work correctly.
def get_logger():
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    return MockLogger()

def ingest_data(): pass
def validate_data(): return True
def transform_data(): return True
def train_model(): return 0.85 
def evaluate_model(): pass

logger = get_logger()
# --------------------------------------------------------


# -------------------------------
# Use Pathlib for robust paths
# -------------------------------
# BASE_DIR is the directory where this 'app.py' file is located
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "artifacts" / "model" / "best_model.pkl"
ENCODER_PATH = BASE_DIR / "artifacts" / "transformation" / "encoder.pkl"
SCALER_PATH = BASE_DIR / "artifacts" / "transformation" / "scaler.pkl"

app = FastAPI(title="Heart Disease Prediction API")


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


def run_pipeline():
    """Run full ML pipeline to generate artifacts."""
    logger.info("Running ML pipeline...")
    try:
        ingest_data()
        if not validate_data():
            raise Exception("Data validation failed.")
        if not transform_data():
            raise Exception("Data transformation failed.")
        acc = train_model()
        evaluate_model()
        logger.info(f"Pipeline completed successfully with accuracy: {acc}")
        return acc
    except Exception as e:
        logger.error(f"ML Pipeline failed: {e}")
        # Reraise the exception to stop prediction attempt
        raise HTTPException(status_code=500, detail=f"ML Pipeline initialization failed: {str(e)}")


@app.post("/predict")
def predict(input_data: HeartInput):
    try:
        # Check if artifacts exist
        artifact_paths = [MODEL_PATH, ENCODER_PATH, SCALER_PATH]
        artifacts_missing = False
        for path in artifact_paths:
            if not Path(path).exists():
                artifacts_missing = True
                break
        
        # Auto-run pipeline if artifacts are missing
        if artifacts_missing:
            logger.warning("Artifacts missing. Running pipeline...")
            run_pipeline() # This function will raise HTTPException on failure

        # Load artifacts
        # We use .resolve() just to ensure the path is fully resolved before loading
        model = joblib.load(MODEL_PATH.resolve())
        encoder = joblib.load(ENCODER_PATH.resolve())
        scaler = joblib.load(SCALER_PATH.resolve())

        # Convert input to DataFrame
        # Using model_dump_json() then reading it back ensures Pydantic validation handles
        # the conversion correctly for FastAPI's POST request body.
        df = pd.DataFrame([input_data.model_dump()])

        # Define categorical & numeric columns (Must match your training columns!)
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        numeric_cols = [col for col in df.columns if col not in categorical_cols]

        # Preprocess
        encoded_df = pd.DataFrame(
            encoder.transform(df[categorical_cols]).toarray(),
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        scaled_df = pd.DataFrame(
            scaler.transform(df[numeric_cols]),
            columns=numeric_cols
        )

        final_df = pd.concat([scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Predict
        prediction = model.predict(final_df)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return {"prediction": int(prediction), "result": result}

    except HTTPException:
        # Re-raise HTTPExceptions raised by run_pipeline
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction processing failed: {str(e)}")