# Helios — Wind Power Forecasting

## The Problem

The trading desk runs two production models:

- **Zephyr** — conservative and reliable in normal conditions, but blind to disruptions
- **Boreas** — aggressive and catches spikes, but fails confidently and expensively

The current approach is manual switching between them. The goal is to build a meta-forecaster that automatically combines both — capturing Boreas's upside while protecting against its failures.

---

## Our Approach

### Why Stacking?

A simple average of Zephyr and Boreas doesn't work well because their errors are not symmetric — Boreas is either much better or much worse than Zephyr depending on conditions. What we need is a model that **learns when to trust each one**.

Stacking lets us do exactly that:

1. **Layer 1 — Base models** learn the relationship between the 20 weather features and actual MW output independently
2. **Layer 2 — Meta-model** (Ridge regression) learns how to weight each base model's prediction given the situation

By including `forecast_zephyr` and `forecast_boreas` as input features, the base models can also learn to **correct the errors** of the existing forecasts rather than starting from scratch.

### Feature Engineering

Beyond the raw weather features, we added time-based features because wind output has a strong daily cycle (peaks ~05:00–07:00, drops ~17:00–18:00):

- `hour`, `dayofweek`, `month`, `dayofyear` — raw time components
- `hour_sin`, `hour_cos`, `month_sin`, `month_cos` — **cyclical encoding** so the model understands that hour 23 and hour 0 are 1 hour apart, not 23 hours apart

### Model Architecture

```
Input: 20 weather features + forecast_zephyr + forecast_boreas + 8 time features
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
       XGBoost           LightGBM       Random Forest
    (n=500, lr=0.05)  (n=500, lr=0.05)  (n=200, depth=10)
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                     Ridge Regression
                     (meta-model learns
                      optimal weighting)
                            │
                            ▼
                      predicted_mw
```

Training uses **5-fold cross-validation** (no shuffle, respecting time order) to generate out-of-fold predictions for the meta-model — this prevents leakage.

---

## Results

| Model | MAE (MW) | RMSE (MW) | Bias (MBE) |
|---|---|---|---|
| Zephyr (baseline) | 35.23 | 46.76 | +28.31 (over-predicts) |
| Boreas (baseline) | 24.04 | 33.52 | +5.60 |
| Naive average | ~29.00 | ~39.00 | — |
| **Our stack** | **8.17** | — | — |

## Architecture

```
helios/
├── api/
│   ├── src/
│   │   ├── features.py     # feature engineering, validation
│   │   ├── train.py        # PowerForecastModel class, training logic
│   │   └── serve.py        # FastAPI app
│   ├── models/             # saved model weights
│   ├── requirements.txt
│   └── Dockerfile
├── streamlit/
│   ├── app.py              # dashboard (EDA + predictions)
│   ├── requirements.txt
│   └── Dockerfile
├── notebooks/
│   └── Untitled.ipynb      # exploration and EDA
├── data/
│   ├── forecast_train.csv
│   └── forecast_test.csv
└── docker-compose.yml
```

---

## Running It

### Train the model

```bash
cd api
python -m src.train
```

### Run with Docker

```bash
docker compose up --build
```

- Streamlit dashboard → `http://localhost:8501`
- FastAPI docs → `http://localhost:8000/docs`

### API usage

```bash
POST /predict
Content-Type: application/json

{
  "rows": [
    {
      "timestamp": "2025-05-20 19:00:00",
      "forecast_zephyr": 348.3,
      "forecast_boreas": 335.3,
      "feature_0": -50.9,
      ...
    }
  ]
}
```

## What We Would Do Next

- **Monitoring endpoint** — `/metrics` returning rolling MAE over the last 24/168 hours

- **Automated retraining pipeline** — triggered when rolling MAE exceeds a threshold

- **Model versioning** — tag each model with training date and OOF MAE

- **CI/CD** — run tests and rebuild Docker image on every push to main
