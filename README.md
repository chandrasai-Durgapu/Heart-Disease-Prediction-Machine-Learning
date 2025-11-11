# ❤️ Heart Disease Prediction: A complete project with Full-Stack Machine-Learning Application

This project implements a complete Machine Learning (ML) pipeline—from data ingestion to deployment—to predict the likelihood of heart disease in patients based on clinical features.

The application is deployed using a **FastAPI** backend for serving predictions and a **Streamlit** frontend for a user-friendly interface. It includes MLflow and etc...

## 🚀 Key Features

* **Complete ML Pipeline:** Scripts for data ingestion, validation, transformation, model training, and evaluation.
* **Model Agnostic:** Designed to easily swap out different classification algorithms (Logistic Regression, Random Forest, XGBoost, etc.).
* **Production Ready:** Uses FastAPI for creating a robust, scalable prediction API.
* **Interactive UI:** A Streamlit web application for real-time, user-friendly prediction.
* **Automatic Retraining:** The FastAPI endpoint automatically triggers the full ML pipeline if the model artifacts are missing.

---
🧠 Model & Algorithm Details
The core of this application is a Machine Learning model trained for binary classification.

The specific algorithm used to train the final model (artifacts/model/best_model.pkl) is the Gradient Boosting Classifier (GBC).

Gradient Boosting Classifier (GBC)
GBC is a powerful ensemble learning technique that builds a strong predictive model by combining the predictions of several weaker models (typically decision trees) in a sequential manner.

Sequential Learning: Each new tree is specifically trained to correct the errors (residuals) made by the collective predictions of all previously trained trees.

Optimization: The method uses a gradient descent approach to minimize the loss (error), effectively forcing subsequent trees to focus their learning on the most difficult-to-classify data points.

Performance: GBC and its variants (like XGBoost or LightGBM) are widely known for delivering high predictive accuracy on structured, tabular datasets like the one used for heart disease prediction.

---
## Jupyter Notebook
Jupyter notebook contains all test cases and experiments before building complete application
---

## Clone the repository
```bash
git clone https://github.com/chandrasai-Durgapu/Heart-Disease-Prediction-Machine-Learning.git
```
```bash
cd Heart-Disease-Prediction-Machine-Learning
```

## Create Virtual Environment
```bash
python -m venv heart-ml-05
```

## Activate Virtual Enviroment
```bash
.\heart-ml-05\Scripts\activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## mlflow url
open the mlflow
```bash
https://dagshub.com/chandrasekharcse522/Heart-Disease-Prediction-Machine-Learning.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
```
---

## run the fastapi 
```bash
uvicorn server.app:app --reload
```
---
## run the frontend
```bash
cd frontend
```
```bash
streamlit run app.py
```
---
## Visaulisations
images were present in images folder

