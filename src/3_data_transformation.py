import os
from logger import get_logger
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import time
import yaml

logger = get_logger()

def transform_data() -> bool:
    try:
        start_time = time.perf_counter()
        logger.info("Data Transformation begins")

        # Check if validation passed
        validation_file = "artifacts/validation/report.yaml"
        if os.path.exists(validation_file):
            with open(validation_file, "r") as f:
                report = yaml.safe_load(f)
            errors = report.get("errors", {})  # <-- only consider actual errors
            if errors:
                logger.error(f"Validation failed with errors: {errors}. Stopping transformation.")
                return False
        else:
            logger.warning("Validation report not found! Proceeding with caution...")

        os.makedirs("artifacts/transformation", exist_ok=True)

        # Load ingested data
        df = pd.read_csv("artifacts/ingestion/ingested.csv")

        # Load schema and params
        with open("config/schema.yaml", "r") as f:
            schema = yaml.safe_load(f)
        with open("config/params.yaml", "r") as f:
            params = yaml.safe_load(f)["data"]

        target = schema["target"]

        # Identify categorical and numerical columns
        cat_cols = [col for col, dtype in schema["columns"].items() if dtype == "str"]
        num_cols = [col for col, dtype in schema["columns"].items() if dtype != "str" and col != target]

        # Encode categorical columns
        if cat_cols:
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            cat_encoded = encoder.fit_transform(df[cat_cols])
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
        else:
            cat_encoded_df = pd.DataFrame()

        # Scale numerical columns
        if num_cols:
            scaler = StandardScaler()
            num_scaled = scaler.fit_transform(df[num_cols])
            num_scaled_df = pd.DataFrame(num_scaled, columns=num_cols)
        else:
            num_scaled_df = pd.DataFrame()

        # Combine numerical and categorical features
        X = pd.concat([num_scaled_df, cat_encoded_df], axis=1)
        y = df[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params.get("test_size", 0.2), random_state=42
        )

        # Save transformed artifacts
        X_train.to_csv("artifacts/transformation/X_train.csv", index=False)
        X_test.to_csv("artifacts/transformation/X_test.csv", index=False)
        y_train.to_csv("artifacts/transformation/y_train.csv", index=False)
        y_test.to_csv("artifacts/transformation/y_test.csv", index=False)

        end_time = time.perf_counter()
        logger.info(f"Data Transformation completed in {end_time - start_time:.2f} seconds")
        return True

    except Exception as e:
        logger.error(f"Error during Data Transformation: {e}")
        return False

if __name__ == "__main__":
    transform_data()
