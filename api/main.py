from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd, os

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name, path in [("rsf", "models/rsf.pkl"), ("pre", "models/preprocessor.pkl")]:
        if not os.path.exists(path):
            raise RuntimeError(
                f"Model artifact '{path}' not found. Run 'make train' first."
            )
        models[name] = joblib.load(path)
    yield
    models.clear()


app = FastAPI(title="Survival Analysis API", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    tenure:           float
    MonthlyCharges:   float
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


@app.get("/")
def root():
    return {"message": "Survival Analysis API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SurvivalPrediction)
def predict(customer: CustomerFeatures):
    from src.features import add_features

    rsf = models["rsf"]
    pre = models["pre"]

    df = pd.DataFrame([customer.model_dump()])
    df = add_features(df)
    X_pre = pre.transform(df)

    surv_fn = rsf.predict_survival_function(X_pre)[0]
    times   = surv_fn.x
    probs   = np.array([surv_fn(t) for t in times])

    below = times[probs <= 0.5]
    median_ttc = float(below[0]) if len(below) > 0 else float(times[-1])

    return SurvivalPrediction(
        median_time_to_churn=median_ttc,
        risk_score=float(rsf.predict(X_pre)[0]),
        survival_curve={str(int(t)): round(float(p), 4) for t, p in zip(times, probs)},
    )
