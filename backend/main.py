from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Credit Scoring API",
    description="Predicts loan default risk using a Random Forest model trained on the Credit Risk dataset.",
    version="1.0.0"
)

# Allow the Streamlit frontend (or any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

model = joblib.load(os.path.join(MODEL_DIR, "credit_scoring_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))


class ClientData(BaseModel):
    person_age: int = Field(..., ge=18, le=100, example=27)
    person_income: float = Field(..., gt=0, example=45000)
    person_home_ownership: Literal["MORTGAGE", "OTHER", "OWN", "RENT"] = Field(..., example="RENT")
    person_emp_length: float = Field(..., ge=0, example=3.0)
    loan_intent: Literal["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"] = Field(..., example="EDUCATION")
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(..., example="B")
    loan_amnt: float = Field(..., gt=0, example=10000)
    loan_int_rate: float = Field(..., ge=0, example=11.5)
    loan_percent_income: float = Field(..., ge=0, le=1, example=0.22)
    cb_person_default_on_file: Literal["N", "Y"] = Field(..., example="N")
    cb_person_cred_hist_length: int = Field(..., ge=0, example=4)


@app.get("/")
def root():
    return {"message": "Credit Scoring API is running. See /docs for usage."}


@app.post("/predict")
def predict(data: ClientData):
    try:
        input_dict = data.dict()

        # Encode categorical columns using the same encoders used at training time
        for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
            le = label_encoders[col]
            input_dict[col] = int(le.transform([input_dict[col]])[0])

        df_input = pd.DataFrame([input_dict])[feature_columns]
        scaled_input = scaler.transform(df_input)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        return {
            "default_risk": bool(prediction == 1),
            "probability_default": round(float(probability), 4),
            "recommendation": "Refuser le prêt" if prediction == 1 else "Accorder le prêt"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
