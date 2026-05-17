from sksurv.metrics import concordance_index_censored, integrated_brier_score
import numpy as np


def score_model(name, risk_scores, y_test):
    c = concordance_index_censored(
        y_test["event"], y_test["duration"], risk_scores
    )[0]
    print(f"{name:25s} | C-index: {c:.3f}")
    return c


def compute_ibs(model, X_pre, y_train, y_test):
    """Integrated Brier Score — calibration across all time points. Lower is better, 0.25 = uninformative."""
    times = np.percentile(y_test["duration"], np.linspace(10, 80, 50))

    t_min = max(y_train["duration"].min(), y_test["duration"].min())
    t_max = min(y_train["duration"].max(), y_test["duration"].max())
    times = times[(times >= t_min) & (times <= t_max)]
    times = np.unique(times)

    survs = model.predict_survival_function(X_pre)
    preds = np.row_stack([fn(times) for fn in survs])

    return integrated_brier_score(y_train, y_test, preds, times)
