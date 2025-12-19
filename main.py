from fastapi import FastAPI
import pickle
import pandas as pd
import shap
import lime
import lime.lime_tabular
import numpy as np

# -------------------------
# App Initialization
# -------------------------
app = FastAPI(title="Explainable Loan Approval API")

# -------------------------
# Load Trained Model
# -------------------------
with open("backend/model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------
# Feature Names (MUST MATCH TRAINING)
# -------------------------
FEATURE_NAMES = [
    'person_age',
    'person_gender',
    'person_education',
    'person_income',
    'person_emp_exp',
    'person_home_ownership',
    'loan_amnt',
    'loan_intent',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score',
    'previous_loan_defaults_on_file'
]

# -------------------------
# SHAP Explainer
# -------------------------
shap_explainer = shap.TreeExplainer(model)

# ======================================================
# 1️⃣ PREDICTION + SHAP ENDPOINT
# ======================================================
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Ensure correct column order
    df = df[FEATURE_NAMES]

    prediction = int(model.predict(df)[0])
    confidence = model.predict_proba(df)[0].max()

    # -------- SAFE SHAP HANDLING --------
    try:
        shap_values = shap_explainer.shap_values(df)

        # Case 1: Binary classification (list of arrays)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_array = shap_values[1][0]   # Approved class
        else:
            shap_array = shap_values[0][0]   # Single-output case

        shap_output = dict(zip(FEATURE_NAMES, shap_array))

    except Exception as e:
        shap_output = {"error": str(e)}

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 2),
        "shap_values": shap_output
    }

# ======================================================
# 2️⃣ LIME EXPLANATION ENDPOINT
# ======================================================
@app.post("/lime")
def lime_explain(data: dict):
    df = pd.DataFrame([data])

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(df),
        feature_names=FEATURE_NAMES,
        class_names=["Rejected", "Approved"],
        mode="classification"
    )

    explanation = lime_explainer.explain_instance(
        df.iloc[0].values,
        model.predict_proba
    )

    return {
        "lime_explanation": explanation.as_list()
    }
