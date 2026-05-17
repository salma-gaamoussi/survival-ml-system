# Survival Analysis ML System

Customer churn modelled as a **time-to-event problem** using the Telco dataset.  
Instead of a binary classifier (will this customer churn?), survival analysis answers:  
*"When* will this customer churn, and what is the probability they are still active at month N?"

---

## Why survival analysis over classification?

| Concern | Classification | Survival analysis |
|---|---|---|
| Censored customers (still active) | Treated as "no churn" — information lost | Handled correctly as censored observations |
| Time dimension | Ignored | Explicitly modelled |
| Output | Single probability | Full survival curve over time |
| Business use | Segment customers | Prioritise outreach by predicted churn month |

---

## Models

| Model | Type | Key metric |
|---|---|---|
| Kaplan-Meier | Non-parametric baseline (no features) | Median survival time |
| Cox PH | Semi-parametric, interpretable hazard ratios | C-index |
| Random Survival Forest | Non-parametric, captures non-linear interactions | C-index + IBS |
| DeepSurv | Neural Cox model (PyTorch) | C-index |

**Metrics**
- **C-index** (concordance index): fraction of pairs correctly ranked by risk. 0.5 = random, 1.0 = perfect.
- **IBS** (integrated Brier score): calibration across all time points. Lower is better; 0.25 = uninformative.

---

## Results

| Model | C-index | IBS |
|---|---|---|
| Cox PH | 0.957 | — |
| RSF | 0.913 | 0.036 |
| DeepSurv | 0.941 | — |

RSF is selected as the production model for the API — it natively outputs a full survival curve per customer, which the API endpoint requires.

> **Note on feature engineering:** `tenure_bucket` and `charge_per_tenure` encode the customer's current tenure, which is also the survival duration. This is appropriate for a **landmark prediction** (scoring a customer's churn risk given their current state), but inflates C-index compared to a purely baseline-feature model. Removing `TotalCharges` (≈ `MonthlyCharges × tenure`) and the `total_monthly_ratio` derived from it eliminates the most direct form of leakage.

---

## Project structure

```
survival-ml-system/
├── data/
│   ├── data.py                  # load + split, builds sksurv structured array
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── features.py              # feature engineering (3 derived features)
│   ├── preprocess.py            # sklearn ColumnTransformer pipeline
│   ├── evaluate.py              # concordance index helper
│   └── train.py                 # end-to-end training script
├── models/
│   ├── km_cox.py                # Kaplan-Meier + Cox PH
│   ├── rsf.py                   # Random Survival Forest + IBS
│   └── deepsurv.py              # PyTorch DeepSurv
├── api/
│   └── main.py                  # FastAPI — /predict returns full survival curve
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   └── test_preprocess.py
├── notebooks/
│   └── 01_eda.ipynb             # EDA + KM curves
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (saves artifacts to models/)
python src/train.py --data-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Run tests
pytest tests/ -v

# 4. Start the API locally
uvicorn api.main:app --reload --port 8000
```

Or use Make:
```bash
make install
make train
make test
make serve
```

---

## API

`POST /predict`

```json
{
  "tenure": 6,
  "MonthlyCharges": 75.0,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "PaperlessBilling": "Yes"
}
```

Response:
```json
{
  "median_time_to_churn": 18.0,
  "risk_score": 1.42,
  "survival_curve": {"1": 0.97, "6": 0.82, "12": 0.61, "24": 0.38}
}
```

Interactive docs available at `http://localhost:8000/docs`.

---

## Docker

```bash
make build   # docker build -t survival-api .
make run     # docker run -p 8000:8000 survival-api
```

---

## Engineered features

| Feature | Formula | Intuition |
|---|---|---|
| `charge_per_tenure` | `MonthlyCharges / (tenure + 1)` | High ratio early → potentially price-sensitive |
| `tenure_bucket` | bins: 0–12 / 12–36 / 36+ | Captures non-linear loyalty effect |

`TotalCharges` is excluded — it is approximately `MonthlyCharges × tenure`, making it a near-direct proxy for the survival duration and a source of leakage.
