# app_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

from fraud_logic import (
    load_artifacts,
    score_transaction_ml,
    evaluate_rules,
    combine_final_risk,
)

app = FastAPI(title="Fraud Detection API")

# ===========================
# Load ML artifacts once at startup
# ===========================
supervised_pipeline, iforest_pipeline = load_artifacts()


# ===========================
# Pydantic model for incoming payload
# (aligned with the main app.py payload fields)
# ===========================
class TransactionPayload(BaseModel):
    Amount: float
    Currency: str = "INR"
    TransactionType: str
    Channel: str

    hour: Optional[int] = 0
    day_of_week: Optional[int] = 0
    month: Optional[int] = 0

    # Device info
    DeviceID: Optional[str] = ""
    device_last_seen: Optional[str] = ""

    # Telemetry
    monthly_avg: Optional[float] = 0.0
    rolling_avg_7d: Optional[float] = 0.0
    txns_last_1h: Optional[int] = 0
    txns_last_24h: Optional[int] = 0
    txns_last_7d: Optional[int] = 0
    beneficiaries_added_24h: Optional[int] = 0
    beneficiary_added_minutes: Optional[int] = 9999
    failed_login_attempts: Optional[int] = 0

    # Beneficiary flags
    new_beneficiary: Optional[bool] = False
    known_beneficiary: Optional[bool] = False

    # IP & Geo
    client_ip: Optional[str] = ""
    ip_country: Optional[str] = ""
    txn_location_ip: Optional[str] = ""
    txn_city: Optional[str] = ""
    txn_country: Optional[str] = ""
    home_city: Optional[str] = ""
    home_country: Optional[str] = ""
    suspicious_ip_flag: Optional[bool] = False
    last_known_lat: Optional[float] = None
    last_known_lon: Optional[float] = None
    txn_lat: Optional[float] = None
    txn_lon: Optional[float] = None

    # ATM / POS / Card
    atm_distance_km: Optional[float] = 0.0
    card_country: Optional[str] = ""
    cvv_provided: Optional[bool] = True
    card_small_attempts_in_5min: Optional[int] = 0
    pos_repeat_count: Optional[int] = 0
    card_masked: Optional[str] = ""

    # Branch Identity
    id_type: Optional[str] = ""
    id_number: Optional[str] = ""
    branch: Optional[str] = ""
    teller_id: Optional[str] = ""

    # Shipping / Billing
    shipping_address: Optional[str] = ""
    billing_address: Optional[str] = ""

    # NetBanking
    username: Optional[str] = ""
    beneficiary: Optional[str] = ""


# ===========================
# FastAPI endpoint
# ===========================
@app.post("/predict")
def predict(payload: TransactionPayload) -> Dict[str, Any]:
    """
    Accepts a transaction payload and returns:
    - Raw ML fraud probability
    - Raw anomaly score
    - Normalized fraud/anomaly scores (0–100)
    - ML risk label
    - Triggered deterministic rules
    - Highest severity from rules
    - Final combined risk
    """
    payload_dict = payload.dict()

    # Treat home_country as declared_country for rules
    payload_dict["declared_country"] = payload_dict.get("home_country", "")

    # -----------------------------
    # ML scoring (raw + normalized)
    # -----------------------------
    (
        fraud_prob_raw,
        anomaly_raw,
        fraud_score_norm,
        anomaly_score_norm,
        ml_label,
    ) = score_transaction_ml(
        supervised_pipeline,
        iforest_pipeline,
        payload_dict,
    )

    # -----------------------------
    # Deterministic rules
    # -----------------------------
    rules_triggered, rules_highest = evaluate_rules(payload_dict, payload.Currency)

    # -----------------------------
    # Final risk
    # -----------------------------
    final_risk = combine_final_risk(ml_label, rules_highest)

    # -----------------------------
    # Structured result
    # -----------------------------
    return {
        "ML": {
            "FraudProbabilityRaw": fraud_prob_raw,
            "AnomalyScoreRaw": anomaly_raw,
            "FraudRiskScore_0_100": fraud_score_norm,
            "AnomalyRiskScore_0_100": anomaly_score_norm,
            "MLRiskLabel": ml_label,
        },
        "Rules": {
            "TriggeredRules": rules_triggered,
            "HighestSeverity": rules_highest,
        },
        "FinalRisk": final_risk,
    }
