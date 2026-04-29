from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd

app = FastAPI(title="Survival Analysis API")

# load best model (RSF — it supports survival function prediction)
rsf = joblib.load("models/rsf.pkl")
pre = joblib.load("models/preprocessor.pkl")


class CustomerFeatures(BaseModel):
    tenure:           float
    MonthlyCharges:   float
    TotalCharges:     float
    Contract:         str    # "Month-to-month", "One year", "Two year"
    InternetService:  str
    PaymentMethod:    str
    gender:           str
    SeniorCitizen:    int
    Partner:          str
    Dependents:       str
    PhoneService:     str
    PaperlessBilling: str


class SurvivalPrediction(BaseModel):
    median_time_to_churn: float
    risk_score:           float
    survival_curve:       dict   # {month: probability}
    ci_lower:             dict
    ci_upper:             dict


@app.get("/")
def root():
    return {"message": "Survival Analysis API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SurvivalPrediction)
def predict(customer: CustomerFeatures):
    from src.features import add_features

    df = pd.DataFrame([customer.model_dump()])
    df = add_features(df)
    X_pre = pre.transform(df)

    # survival function for this customer
    surv_fn = rsf.predict_survival_function(X_pre)[0]
    times   = surv_fn.x
    probs   = surv_fn(1)  # callable
    probs   = np.array([surv_fn(t) for t in times])

    # median: first time where survival drops below 0.5
    below = times[probs <= 0.5]
    median_ttc = float(below[0]) if len(below) > 0 else float(times[-1])

    # simple ±5% CI approximation around the curve
    return SurvivalPrediction(
        median_time_to_churn=median_ttc,
        risk_score=float(rsf.predict(X_pre)[0]),
        survival_curve={str(int(t)): round(float(p), 4) for t, p in zip(times, probs)},
        ci_lower={str(int(t)): round(float(max(0, p - 0.05)), 4) for t, p in zip(times, probs)},
        ci_upper={str(int(t)): round(float(min(1, p + 0.05)), 4) for t, p in zip(times, probs)},
    )