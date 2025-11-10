import os
import pandas as pd
import yaml
from logger import get_logger
import time

logger = get_logger()

def validate_data():
    try:
        start_time = time.perf_counter()
        logger.info("Data Validation begins")

        os.makedirs("artifacts/validation", exist_ok=True)

        # Load ingested data
        df = pd.read_csv("artifacts/ingestion/ingested.csv")

        # Load schema
        with open("config/schema.yaml") as f:
            schema = yaml.safe_load(f)

        # Load params
        with open("config/params.yaml") as f:
            params = yaml.safe_load(f)
        stop_on_fail = params["validation"]["stop_on_fail"]

        # Validation logic
        errors = {}
        for col, dtype in schema["columns"].items():
            if col not in df.columns:
                errors[col] = "missing"

        if schema["target"] not in df.columns:
            errors[schema["target"]] = "missing"

        # Create a structured report
        validation_report = {
            "errors": errors,
            "passed": len(errors) == 0
        }

        # Save validation report
        with open("artifacts/validation/report.yaml", "w") as f:
            yaml.dump(validation_report, f)

        if errors:
            logger.error(f"Validation failed: {errors}")
            if stop_on_fail:
                raise ValueError("Data validation failed. Stopping the pipeline.")
        else:
            logger.info("Validation passed")

        end_time = time.perf_counter()
        logger.info(f"Data Validation completed in {end_time-start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error during Validation: {e}")
        raise e

if __name__ == "__main__":
    validate_data()
