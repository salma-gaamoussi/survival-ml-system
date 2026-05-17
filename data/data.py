import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TARGET_DURATION = "tenure"          # months as customer
TARGET_EVENT    = "Churn"            # 1 = churned, 0 = censored

DROP_COLS = ["customerID"]          # identifier, not a feature

NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges"
]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "InternetService", "Contract",
    "PaymentMethod", "PaperlessBilling"
]


def load_data(path: str):
    df = pd.read_csv(path)
    # TotalCharges has stray spaces — fix before casting
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # encode event: Yes → 1, No → 0
    df[TARGET_EVENT] = (df[TARGET_EVENT] == "Yes").astype(int)
    df = df.drop(columns=DROP_COLS)

    # survival target: structured array required by scikit-survival
    y = np.array(
        list(zip(df[TARGET_EVENT].astype(bool), df[TARGET_DURATION])),
        dtype=[("event", bool), ("duration", float)]
    )
    X = df.drop(columns=[TARGET_EVENT])
    return X, y


def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed)