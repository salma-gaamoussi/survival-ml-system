from sksurv.metrics import concordance_index_censored
import numpy as np


def score_model(name, risk_scores, y_test):
    c = concordance_index_censored(
        y_test["event"], y_test["duration"], risk_scores
    )[0]
    print(f"{name:25s} | C-index: {c:.3f}")
    return c