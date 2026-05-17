import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rsf.pkl")
MODELS_EXIST = os.path.exists(MODEL_PATH)

SAMPLE_CUSTOMER = {
    "tenure": 12,
    "MonthlyCharges": 65.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
}


@pytest.fixture(scope="module")
def client():
    if not MODELS_EXIST:
        pytest.skip("models not trained — run make train")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_expected_fields(client):
    r = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert r.status_code == 200
    data = r.json()
    assert "median_time_to_churn" in data
    assert "risk_score" in data
    assert "survival_curve" in data


def test_predict_values_are_valid(client):
    r = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = r.json()
    assert data["median_time_to_churn"] > 0
    assert isinstance(data["survival_curve"], dict)
    probs = list(data["survival_curve"].values())
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predict_missing_field_returns_422(client):
    bad_payload = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "Contract"}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422
