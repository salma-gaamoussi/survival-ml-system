import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd


@pytest.fixture
def sample_X():
    return pd.DataFrame({
        "tenure": [1, 12, 24],
        "MonthlyCharges": [50.0, 70.0, 100.0],
        "TotalCharges": [50.0, 840.0, 2400.0],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "PhoneService": ["Yes", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer"
        ],
        "PaperlessBilling": ["Yes", "No", "Yes"],
    })