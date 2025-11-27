# AI Powered Real-Time Fraud Detection

Streamlit application + REST API for real-time fraud scoring combining
ML models (supervised + Isolation Forest) and a deterministic rule engine.

## 1. Architecture Overview

- **UI layer**: `app.py` (Streamlit)
  - Captures channel-specific inputs (Bank, ATM, Credit Card, Mobile App, POS, Online Purchase, NetBanking)
  - Computes date/time features (hour, day_of_week, month)
  - Sends structured `payload` into ML + rules
  - Displays:
    - Final risk label
    - ML fraud probability & anomaly score
    - Rules highest severity
    - Response time
    - Human-readable justification
    - Debug JSON payload
    - Example transactions for KT

- **ML layer**:
  - `models/supervised_lgbm_pipeline.joblib`
  - `models/iforest_pipeline.joblib`
  - Loaded once using `@st.cache_resource`
  - `score_transaction_ml(...)` wraps model invocation

- **Rule engine**:
  - `evaluate_rules(payload, currency)` in `app.py`
  - Uses:
    - Amount thresholds
    - Home vs transaction city/country
    - Time-of-day (midnight/high amount)
    - Velocity (1h, 24h, 7d)
    - ATM distance
    - New beneficiaries
    - IP intel flag
    - Card issuing country vs home country
  - Returns triggered rules + highest severity

- **API layer**:
  - `api.py` (FastAPI)
  - `/score` endpoint:
    - Input: JSON payload
    - Output: ML probabilities, anomaly score, rules, final risk

## 2. Running Locally

### Prerequisites

- Python 3.9+
- `pip` or `conda`
- Model artifacts in `models/`:
  - `supervised_lgbm_pipeline.joblib`
  - `iforest_pipeline.joblib`

### Install dependencies

```bash
pip install -r requirements.txt
