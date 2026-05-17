import pandas as pd
import numpy as np


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # charge per month of tenure — high ratio = potentially at-risk
    X["charge_per_tenure"] = (
        X["MonthlyCharges"] / (X["tenure"] + 1)  # +1 avoids div-by-zero
    )

    # tenure bucket — early / mid / long-term customer
    X["tenure_bucket"] = pd.cut(
        X["tenure"],
        bins=[0, 12, 36, np.inf],
        labels=["early", "mid", "long"]
    ).astype(str)

    X = X.drop(columns=["tenure"])
    return X