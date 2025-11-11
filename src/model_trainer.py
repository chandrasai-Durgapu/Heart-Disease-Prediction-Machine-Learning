import pandas as pd
import os
import joblib
import time
import yaml
from utils.logger import get_logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import mlflow
import dagshub

logger = get_logger()

# Initialize Dagshub + MLflow
dagshub.init(
    repo_owner='chandrasekharcse522',
    repo_name='Heart-Disease-Prediction-Machine-Learning',
    mlflow=True
)

def train_model():
    try:
        logger.info(" Model training begins...")
        start_time = time.perf_counter()

        # Ensure directories exist
        os.makedirs("artifacts/model", exist_ok=True)
        os.makedirs("artifacts/encoder", exist_ok=True)
        os.makedirs("artifacts/scaler", exist_ok=True)

        # Load train/test data
        X_train = pd.read_csv("artifacts/transformation/X_train.csv")
        X_test = pd.read_csv("artifacts/transformation/X_test.csv")
        y_train = pd.read_csv("artifacts/transformation/y_train.csv").values.ravel()
        y_test = pd.read_csv("artifacts/transformation/y_test.csv").values.ravel()

        # Load params
        with open("config/params.yaml", "r") as f:
            params = yaml.safe_load(f)
        model_name = params["model"]["name"]

        # Train model
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f" Model Accuracy: {acc:.3f}")

        # Save model artifact
        joblib.dump(model, "artifacts/model/best_model.pkl")
        logger.info(" Model saved at artifacts/model/best_model.pkl")

        # MLflow Logging
        mlflow.start_run()
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.end_run()

        end_time = time.perf_counter()
        logger.info(f" Training completed in {end_time - start_time:.2f} seconds")

        return acc

    except Exception as e:
        logger.error(f" Error during model training: {e}")
        raise e


if __name__ == "__main__":
    train_model()
