from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUMERIC_COLS = [
    "MonthlyCharges", "TotalCharges",
    "charge_per_tenure", "total_monthly_ratio"    # engineered
]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "InternetService", "Contract",
    "PaymentMethod", "PaperlessBilling", "tenure_bucket"  # engineered
]


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe,     NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])