from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
)
import numpy as np


def fit_rsf(X_train_pre, y_train):
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=42,
    )
    rsf.fit(X_train_pre, y_train)
    return rsf


def integrated_brier(model, X_pre, y_train, y_test):
    """
    IBS measures calibration across all time points.
    Lower is better. 0.25 = uninformative model (random).
    """
    # initial time grid
    times = np.percentile(
        y_test["duration"], np.linspace(10, 80, 50)
    )

    # ⚠️ restrict times to valid range (critical fix)
    t_min = max(y_train["duration"].min(), y_test["duration"].min())
    t_max = min(y_train["duration"].max(), y_test["duration"].max())
    times = times[(times >= t_min) & (times <= t_max)]

    # remove duplicates (can happen with percentiles)
    times = np.unique(times)

    # compute survival predictions
    survs = model.predict_survival_function(X_pre)
    preds = np.row_stack([fn(times) for fn in survs])

    # compute IBS
    ibs = integrated_brier_score(y_train, y_test, preds, times)
    return ibs