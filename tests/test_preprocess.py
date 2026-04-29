import numpy as np
from src.preprocess import build_preprocessor
from src.features import add_features


def test_preprocessor_output_shape(sample_X):
    pre = build_preprocessor()

    # apply feature engineering first (VERY IMPORTANT)
    X_feat = add_features(sample_X)

    X_trans = pre.fit_transform(X_feat)

    # same number of rows
    assert X_trans.shape[0] == X_feat.shape[0]

    # at least something was generated
    assert X_trans.shape[1] > 0

def test_no_nans_after_preprocessing(sample_X):
    pre = build_preprocessor()
    X_feat = add_features(sample_X)

    X_trans = pre.fit_transform(X_feat)

    assert not np.isnan(X_trans).any()

def test_unknown_category_does_not_crash(sample_X):
    pre = build_preprocessor()

    X_train = add_features(sample_X)
    pre.fit(X_train)

    # create new data with unseen category
    X_new = sample_X.copy()
    X_new.loc[0, "Contract"] = "Super-Long-Term"  # unseen value

    X_new_feat = add_features(X_new)

    # should NOT raise error
    X_trans = pre.transform(X_new_feat)

    # number of input rows (before preprocessing) should match output rows (after preprocessing)
    assert X_trans.shape[0] == X_new_feat.shape[0]

def test_output_shape_consistency(sample_X):
    pre = build_preprocessor()

    X_train = add_features(sample_X)
    pre.fit(X_train)

    X_test = add_features(sample_X.copy())

    X_train_trans = pre.transform(X_train)
    X_test_trans = pre.transform(X_test)

    assert X_train_trans.shape[1] == X_test_trans.shape[1]


# Critical detail (many people forget this)
# Always do:
# X_feat = add_features(X)
# BEFORE preprocessing.