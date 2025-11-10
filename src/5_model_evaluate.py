import os
import pandas as pd
import joblib
from logger import get_logger
import yaml
import mlflow
import dagshub
from sklearn.metrics import accuracy_score

logger = get_logger()

# Initialize DagsHub + MLflow
dagshub.init(
    repo_owner='chandrasekharcse522',
    repo_name='Heart-Disease-Prediction-Machine-Learning',
    mlflow=True
)

def evaluate_model():
    try:
        logger.info("Model Evaluation begins")

        # Load model saved with joblib
        model_path = "artifacts/model/best_model.pkl"
        model = joblib.load(model_path)

        # Load test data
        X_test = pd.read_csv("artifacts/transformation/X_test.csv")
        y_test = pd.read_csv("artifacts/transformation/y_test.csv").values.ravel()

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {acc:.4f}")

        # Log metrics to MLflow
        mlflow.start_run()
        mlflow.log_metric("accuracy", acc)
        mlflow.end_run()

        logger.info("Model Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during Model Evaluation: {e}")
        raise e


if __name__ == "__main__":
    evaluate_model()
