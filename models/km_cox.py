import numpy as np
from lifelines import KaplanMeierFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt


def fit_kaplan_meier(y_train, save_path="km_curve.png"):
    """Baseline: population-level survival curve, no features."""
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=y_train["duration"],
        event_observed=y_train["event"],
        label="All customers"
    )
    ax = kmf.plot_survival_function(ci_show=True)
    ax.set_title("Kaplan-Meier survival curve")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("P(still a customer)")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"KM median survival: {kmf.median_survival_time_:.1f} months")
    return kmf


def fit_cox(X_train_pre, y_train):
    """Cox PH: log-partial-likelihood, interpretable hazard ratios."""
    cox = CoxPHSurvivalAnalysis(alpha=0.1)   # L2 regularisation
    cox.fit(X_train_pre, y_train)
    return cox


def c_index(model, X_pre, y):
    """Concordance index — higher is better, 0.5 = random."""
    risk_scores = model.predict(X_pre)
    result = concordance_index_censored(
        y["event"], y["duration"], risk_scores
    )
    return result[0]   # concordance index value