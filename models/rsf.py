from sksurv.ensemble import RandomSurvivalForest


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
