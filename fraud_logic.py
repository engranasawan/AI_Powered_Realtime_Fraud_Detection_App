# fraud_logic.py
"""
fraud_logic.py

Fully self-contained fraud logic for both Streamlit-like flows and FastAPI.
Includes:
- Currency conversion
- ML scoring (with normalized 0–100 scores)
- Beneficiary impact (Balanced Option B)
- Deterministic rules (aligned with app.py)
- Final risk aggregation
"""

from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd

# ===========================
# Currency configuration (INR base)
# ===========================
INR_PER_UNIT = {
    "INR": 1.0,
    "USD": 83.2,
    "EUR": 90.5,
    "GBP": 105.3,
    "AED": 22.7,
    "AUD": 61.0,
    "SGD": 61.5,
}


def inr_to_currency(amount_in_inr: float, currency: str) -> float:
    if currency not in INR_PER_UNIT or INR_PER_UNIT[currency] == 0:
        return amount_in_inr
    return amount_in_inr / INR_PER_UNIT[currency]


def normalize_score(x: float, min_val: float = 0.0, max_val: float = 0.02) -> float:
    """Normalize ML score into 0–100."""
    if x is None:
        return 0.0
    try:
        val = float(x)
    except Exception:
        return 0.0
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    if max_val == min_val:
        return 0.0
    return (val - min_val) / (max_val - min_val) * 100.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Return distance in km between two lat/lon points."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def escalate(a: str, b: str) -> str:
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b


# ===========================
# Load ML artifacts
# ===========================
def load_artifacts(models_dir: str = "models"):
    models_dir = Path(models_dir)

    def _load(name: str):
        return joblib.load(models_dir / name)

    try:
        supervised_pipeline = _load("supervised_lgbm_pipeline.joblib")
    except Exception:
        supervised_pipeline = None

    try:
        iforest_pipeline = _load("iforest_pipeline.joblib")
    except Exception:
        iforest_pipeline = None

    return supervised_pipeline, iforest_pipeline


# ===========================
# ML risk label based on 0–100 scores
# ===========================
def ml_risk_label(fraud_score: float, anomaly_score: float) -> str:
    """
    Business mapping:
      0–30   => LOW
      30–60  => MEDIUM
      60–90  => HIGH
      90–100 => CRITICAL
    """
    agg = max(fraud_score, anomaly_score)
    if agg >= 90.0:
        return "CRITICAL"
    if agg >= 60.0:
        return "HIGH"
    if agg >= 30.0:
        return "MEDIUM"
    return "LOW"


# ===========================
# Base thresholds (INR)
# ===========================
BASE_THRESHOLDS_INR = {
    "absolute_crit_amount": 10_000_000,
    "high_amount_threshold": 2_000_000,
    "medium_amount_threshold": 100_000,
    "atm_high_withdrawal": 300_000,
    "card_test_small_amount_inr": 200,
}


# ===========================
# Rule engine (aligned with app.py)
# ===========================
def evaluate_rules(payload: Dict, currency: str) -> Tuple[List[Dict], str]:
    ABS_CRIT = inr_to_currency(BASE_THRESHOLDS_INR["absolute_crit_amount"], currency)
    HIGH_AMT = inr_to_currency(BASE_THRESHOLDS_INR["high_amount_threshold"], currency)
    MED_AMT = inr_to_currency(BASE_THRESHOLDS_INR["medium_amount_threshold"], currency)
    ATM_HIGH = inr_to_currency(BASE_THRESHOLDS_INR["atm_high_withdrawal"], currency)
    CARD_TEST_SMALL = inr_to_currency(BASE_THRESHOLDS_INR["card_test_small_amount_inr"], currency)

    rules: List[Dict] = []

    amt = float(payload.get("Amount", 0.0) or 0.0)
    channel_raw = str(payload.get("Channel", "") or "")
    channel = channel_raw.lower()
    hour = int(payload.get("hour", 0) or 0)
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    rolling_avg_7d = float(payload.get("rolling_avg_7d", 0.0) or 0.0)
    txns_1h = int(payload.get("txns_last_1h", 0) or 0)
    txns_24h = int(payload.get("txns_last_24h", 0) or 0)
    txns_7d = int(payload.get("txns_last_7d", 0) or 0)
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)
    new_benef = bool(payload.get("new_beneficiary", False))

    ip_country = str(payload.get("ip_country", "") or "").lower()
    declared_country = str(payload.get("declared_country", "") or "").lower()

    home_city = str(payload.get("home_city", "") or "").lower()
    home_country = str(payload.get("home_country", "") or "").lower()
    txn_city = str(payload.get("txn_city", "") or "").lower()
    txn_country = str(payload.get("txn_country", "") or "").lower()

    last_device = str(payload.get("device_last_seen", "") or "").lower()
    curr_device = str(payload.get("DeviceID", "") or "").lower()

    last_lat = payload.get("last_known_lat")
    last_lon = payload.get("last_known_lon")
    txn_lat = payload.get("txn_lat")
    txn_lon = payload.get("txn_lon")
    atm_distance_km = float(payload.get("atm_distance_km", 0.0) or 0.0)

    card_country = str(payload.get("card_country", "") or "").lower()
    cvv_provided = payload.get("cvv_provided", True)
    shipping_addr = payload.get("shipping_address", "")
    billing_addr = payload.get("billing_address", "")
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0) or 0)
    suspicious_ip_flag = payload.get("suspicious_ip_flag", False)
    card_small_attempts = int(payload.get("card_small_attempts_in_5min", 0) or 0)
    pos_repeat_count = int(payload.get("pos_repeat_count", 0) or 0)
    beneficiary_added_minutes = int(payload.get("beneficiary_added_minutes", 9999) or 9999)

    id_type = str(payload.get("id_type", "") or "").strip()
    id_number = str(payload.get("id_number", "") or "").strip()

    def add_rule(name: str, severity: str, detail: str):
        rules.append({"name": name, "severity": severity, "detail": detail})

    # Absolute amount
    if amt >= ABS_CRIT:
        add_rule(
            "Absolute very large amount",
            "CRITICAL",
            f"Amount {amt:.2f} {currency} >= critical {ABS_CRIT:.2f} {currency}.",
        )

    # Impossible travel
    impossible_travel_distance = None
    if last_lat is not None and last_lon is not None and txn_lat is not None and txn_lon is not None:
        impossible_travel_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)

    device_checks_enabled = channel not in ("onsite branch transaction", "atm")

    if device_checks_enabled:
        device_new = (not last_device) or last_device == "" or (curr_device and curr_device != last_device)
        location_changed = impossible_travel_distance is not None and impossible_travel_distance > 500
        if device_new and location_changed and amt > MED_AMT:
            add_rule(
                "New device + Impossible travel + High amount",
                "CRITICAL",
                f"New device + travel {impossible_travel_distance:.1f} km; amount {amt:.2f} {currency}.",
            )

    # Beneficiaries + high transfer
    if beneficiaries_added_24h >= 3 and amt > HIGH_AMT:
        add_rule(
            "Multiple beneficiaries added + high transfer",
            "CRITICAL",
            f"{beneficiaries_added_24h} beneficiaries added and amount {amt:.2f} {currency}.",
        )

    # Velocity
    if txns_1h >= 10:
        add_rule("High velocity (1h)", "HIGH", f"{txns_1h} txns in last 1 hour.")
    if txns_24h >= 50:
        add_rule("Very high velocity (24h)", "HIGH", f"{txns_24h} txns in last 24h.")

    # IP vs declared
    if ip_country and declared_country and ip_country != declared_country:
        sev = "HIGH" if amt > HIGH_AMT else "MEDIUM"
        add_rule(
            "IP / Declared country mismatch",
            sev,
            f"IP country '{ip_country}' differs from declared '{declared_country}'.",
        )

    # Login security
    if failed_logins >= 5:
        add_rule("Multiple failed login attempts", "HIGH", f"{failed_logins} failed auth attempts.")

    # New beneficiary + amount
    if new_benef and amt >= MED_AMT:
        add_rule(
            "New beneficiary + significant amount",
            "HIGH",
            "Transfer to newly added beneficiary with amount above threshold.",
        )

    # IP flagged
    if suspicious_ip_flag and amt > (MED_AMT / 4):
        add_rule("IP flagged by threat intelligence", "HIGH", "IP flagged and non-trivial amount.")

    # ATM distance
    if channel == "atm" and atm_distance_km and atm_distance_km > 300:
        add_rule("ATM distance from last location", "HIGH", f"ATM is {atm_distance_km:.1f} km away.")

    # Card country vs home
    if card_country and home_country and card_country != home_country and amt > MED_AMT:
        add_rule(
            "Card country mismatch vs home country",
            "HIGH",
            f"Card country {card_country} != home country {home_country}.",
        )

    # Amount vs historical patterns
    if monthly_avg > 0 and amt >= 5 * monthly_avg and amt > MED_AMT:
        add_rule(
            "Large spike vs monthly avg",
            "HIGH",
            f"Amount {amt:.2f} >= 5x monthly avg {monthly_avg:.2f}.",
        )
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > (MED_AMT / 2):
        add_rule(
            "Spike vs 7-day avg",
            "MEDIUM",
            f"Amount {amt:.2f} >= 3x 7-day avg {rolling_avg_7d:.2f}.",
        )
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > (MED_AMT / 2):
        add_rule(
            "Above monthly usual",
            "MEDIUM",
            f"Amount {amt:.2f} >= 2x monthly avg {monthly_avg:.2f}.",
        )

    # Additional velocity
    if txns_1h >= 5:
        add_rule("Elevated velocity (1h)", "MEDIUM", f"{txns_1h} in last 1 hour.")
    if 10 <= txns_24h < 50:
        add_rule("Elevated velocity (24h)", "MEDIUM", f"{txns_24h} in last 24h.")

    # Time-of-day
    if 0 <= hour <= 5 and monthly_avg < (MED_AMT * 2) and amt > (MED_AMT / 10):
        add_rule(
            "Late-night txn for low-activity customer",
            "MEDIUM",
            f"Txn at hour {hour} for low-activity customer; amt {amt:.2f}.",
        )

    if 0 <= hour <= 4 and amt >= HIGH_AMT:
        add_rule(
            "Very high amount during unusual time",
            "HIGH",
            f"Txn at hour {hour} with amount {amt:.2f} {currency} >= high threshold {HIGH_AMT:.2f}.",
        )

    # ATMs & POS
    if channel == "atm" and amt >= ATM_HIGH:
        add_rule(
            "Large ATM withdrawal",
            "HIGH",
            f"ATM withdrawal {amt:.2f} {currency} >= {ATM_HIGH:.2f}",
        )

    if channel == "pos" and pos_repeat_count >= 15:
        add_rule("POS heavy repeat transactions", "CRITICAL", f"{pos_repeat_count} rapid transactions at same POS.")
    elif channel == "pos" and pos_repeat_count >= 10:
        add_rule("POS repeat transactions", "HIGH", f"{pos_repeat_count} rapid transactions at same POS.")

    # Branch / Netbanking immediate beneficiary transfers
    if channel in ("netbanking", "onsite branch transaction") and beneficiary_added_minutes < 10 and amt >= MED_AMT:
        add_rule(
            "Immediate transfer to newly added beneficiary",
            "HIGH",
            f"Beneficiary added {beneficiary_added_minutes} minutes ago and transfer amount {amt:.2f} {currency}.",
        )

    # Higher-risk countries
    high_risk_countries = {"nigeria", "romania", "ukraine", "russia"}
    if txn_country and txn_country in high_risk_countries:
        add_rule(
            "Transaction in higher-risk country",
            "MEDIUM",
            f"Transaction country flagged as higher-risk: {txn_country}.",
        )

    # Card testing
    if card_small_attempts >= 6 and CARD_TEST_SMALL > 0:
        add_rule(
            "Card testing / micro-charges detected",
            "HIGH",
            f"{card_small_attempts} small attempts; micro amount {CARD_TEST_SMALL:.2f} {currency}.",
        )

    # Home vs txn location
    if home_country and txn_country and home_country != txn_country:
        sev = "HIGH" if amt >= MED_AMT else "MEDIUM"
        add_rule(
            "Txn country differs from home country",
            sev,
            f"Home country '{home_country}' vs transaction country '{txn_country}'.",
        )

    if home_city and txn_city and home_city != txn_city and amt >= (MED_AMT / 2):
        add_rule(
            "Txn city differs from home city",
            "MEDIUM",
            f"Home city '{home_city}' vs transaction city '{txn_city}'.",
        )

    # Branch-specific: missing identity
    if channel == "onsite branch transaction":
        if not id_type or not id_number:
            add_rule(
                "Missing customer identity at branch",
                "CRITICAL",
                "Onsite branch transaction without identity document details.",
            )

    # Card channels: CVV and address mismatch
    if channel in ("credit card", "debit card", "online purchase"):
        if not cvv_provided:
            add_rule(
                "Card-not-present without CVV",
                "HIGH",
                "Card used without CVV in a channel where CVV is expected.",
            )
        if shipping_addr and billing_addr and shipping_addr != billing_addr and amt > MED_AMT:
            add_rule(
                "Shipping / billing address mismatch",
                "HIGH",
                "Shipping and billing addresses differ for a high-value card transaction.",
            )

    # Other channel
    if channel == "other":
        if amt >= HIGH_AMT:
            add_rule("Large transaction on 'Other' channel", "HIGH", f"High amount {amt:.2f} {currency}.")
        if 0 <= hour <= 5 and amt > (MED_AMT / 2):
            add_rule("Unusual time for 'Other' channel", "MEDIUM", f"Transaction at unusual hour {hour}.")

    # Transfer-specific structural checks
    if str(payload.get("TransactionType", "")).upper() == "TRANSFER":
        from_acc = payload.get("from_account_number")
        to_acc = payload.get("to_account_number")
        if not from_acc or not to_acc:
            add_rule(
                "Missing transfer account data",
                "HIGH",
                "Transfer missing source or destination account details.",
            )

    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest


# ===========================
# Combine ML + Rules
# ===========================
def combine_final_risk(ml_risk: str, rule_highest: str) -> str:
    return escalate(ml_risk, rule_highest)


# ===========================
# ML scoring
# ===========================
def score_transaction_ml(
    supervised_pipeline,
    iforest_pipeline,
    model_payload: Dict,
) -> Tuple[float, float, float, float, str]:
    """
    Returns:
      fraud_prob_raw (0–1)
      anomaly_raw
      fraud_score (0–100)
      anomaly_score (0–100)
      ml_label
    """
    model_df = pd.DataFrame(
        [
            {
                "Amount": model_payload.get("Amount", 0.0),
                "TransactionType": model_payload.get("TransactionType", "PAYMENT"),
                "Location": model_payload.get("txn_city", "Unknown"),
                "DeviceID": model_payload.get("DeviceID", "Unknown"),
                "Channel": model_payload.get("Channel", "Other"),
                "hour": model_payload.get("hour", 0),
                "day_of_week": model_payload.get("day_of_week", 0),
                "month": model_payload.get("month", 0),
            }
        ]
    )

    fraud_prob_raw = 0.0
    anomaly_raw = 0.0

    if supervised_pipeline is not None:
        try:
            fraud_prob_raw = float(supervised_pipeline.predict_proba(model_df)[0, 1])
        except Exception:
            fraud_prob_raw = 0.0

    if iforest_pipeline is not None:
        try:
            anomaly_raw = -float(iforest_pipeline.decision_function(model_df)[0])
        except Exception:
            anomaly_raw = 0.0

    # Normalize
    fraud_score = normalize_score(fraud_prob_raw, min_val=0.0, max_val=0.02)
    anomaly_score = normalize_score(anomaly_raw, min_val=0.0, max_val=0.10)

    # Beneficiary impact (Balanced Option B)
    new_beneficiary = bool(model_payload.get("new_beneficiary", False))
    known_beneficiary = bool(model_payload.get("known_beneficiary", False))

    if new_beneficiary:
        fraud_score += 10.0
    if known_beneficiary:
        fraud_score -= 5.0

    fraud_score = max(0.0, min(100.0, fraud_score))

    ml_label = ml_risk_label(fraud_score, anomaly_score)
    return fraud_prob_raw, anomaly_raw, fraud_score, anomaly_score, ml_label
