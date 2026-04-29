import pandas as pd
import numpy as np


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # 1. charge per month of tenure — high ratio = potentially at-risk
    X["charge_per_tenure"] = (
        X["MonthlyCharges"] / (X["tenure"] + 1)  # +1 avoids div-by-zero
    )

    # 2. tenure bucket — early / mid / long-term customer
    X["tenure_bucket"] = pd.cut(
        X["tenure"],
        bins=[0, 12, 36, np.inf],
        labels=["early", "mid", "long"]
    ).astype(str)

    # 3. total vs monthly ratio — signals plan stability
    X["total_monthly_ratio"] = (
        X["TotalCharges"] / (X["MonthlyCharges"] + 1)
    )

    X = X.drop(columns=["tenure"])
    print("dropped")
    return X