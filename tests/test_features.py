import numpy as np
import os
import sys

sys.path.append(os.path.abspath('..'))

from src.features import add_features


def test_new_columns_exist(sample_X):
    X_new = add_features(sample_X)

    assert "charge_per_tenure" in X_new.columns
    assert "tenure_bucket" in X_new.columns


def test_no_nulls_in_new_features(sample_X):
    X_new = add_features(sample_X)

    assert X_new["charge_per_tenure"].isnull().sum() == 0
    assert X_new["tenure_bucket"].isnull().sum() == 0


def test_charge_per_tenure_values(sample_X):
    X_new = add_features(sample_X)

    expected = sample_X["MonthlyCharges"] / (sample_X["tenure"] + 1)

    assert np.allclose(X_new["charge_per_tenure"], expected)


def test_tenure_bucket_values(sample_X):
    X_new = add_features(sample_X)

    # Expected buckets based on:
    # bins=[0, 12, 36, inf]
    expected = ["early", "early", "mid"]

    assert list(X_new["tenure_bucket"]) == expected

