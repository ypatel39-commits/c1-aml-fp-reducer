"""Synthetic AML alert generator.

Produces a realistic distribution mirroring industry pain point: ~92% false
positives, ~8% true positives. Each alert includes customer context so the
triage LLM has enough information to make a defensible decision.

Random state is fixed at 42 for full reproducibility.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

RANDOM_SEED = 42

CUSTOMER_SEGMENTS = ["retail", "private", "business"]
SEGMENT_WEIGHTS = [0.70, 0.10, 0.20]

# A small list of countries chosen to span low-risk and high-risk jurisdictions.
LOW_RISK_COUNTRIES = ["US", "CA", "GB", "DE", "FR", "JP", "AU", "SG"]
HIGH_RISK_COUNTRIES = ["IR", "KP", "SY", "RU", "VE", "MM"]
MED_RISK_COUNTRIES = ["AE", "TR", "MX", "BR", "ZA", "NG"]

TXN_TYPES = ["wire", "ach", "cash", "card"]
TXN_TYPE_WEIGHTS = [0.30, 0.40, 0.10, 0.20]

ALERT_REASONS = [
    "sanctions_match",      # name fuzzy-matches OFAC list
    "structuring",          # multiple sub-$10k cash deposits
    "velocity",             # unusual txn frequency vs baseline
    "round_amount",         # suspiciously round dollar amounts
    "high_risk_geo",        # counterparty in high-risk jurisdiction
    "dormant_reactivation", # dormant account suddenly active
    "pep_match",            # politically exposed person screening
]

# Per-reason base true-positive probability. Tuned so overall TP rate sits
# around 8% with the segment / geography modifiers below.
REASON_TP_BASE = {
    "sanctions_match": 0.18,
    "structuring": 0.22,
    "velocity": 0.04,
    "round_amount": 0.02,
    "high_risk_geo": 0.10,
    "dormant_reactivation": 0.05,
    "pep_match": 0.08,
}


@dataclass
class Alert:
    alert_id: str
    customer_id: str
    customer_segment: str
    customer_tenure_years: int
    country: str
    counterparty_country: str
    txn_amount: float
    txn_type: str
    counterparty: str
    alert_reason: str
    prior_alerts_90d: int
    is_true_positive: int  # 1 / 0 — ground truth label

    def to_dict(self) -> dict:
        return asdict(self)


def _pick(rng: random.Random, choices: list[str], weights: list[float]) -> str:
    return rng.choices(choices, weights=weights, k=1)[0]


def _sample_country(rng: random.Random, segment: str) -> str:
    """Retail / business mostly low-risk. Private banking has tail risk."""
    if segment == "private":
        bucket = rng.choices(["low", "med", "high"], weights=[0.55, 0.30, 0.15])[0]
    else:
        bucket = rng.choices(["low", "med", "high"], weights=[0.80, 0.17, 0.03])[0]
    if bucket == "low":
        return rng.choice(LOW_RISK_COUNTRIES)
    if bucket == "med":
        return rng.choice(MED_RISK_COUNTRIES)
    return rng.choice(HIGH_RISK_COUNTRIES)


def _sample_amount(rng: random.Random, reason: str, segment: str) -> float:
    """Amount distributions vary by alert reason — structuring clusters near
    $9.5k, sanctions are full-spectrum, velocity tends toward smaller txns."""
    if reason == "structuring":
        return round(rng.uniform(7800, 9990), 2)
    if reason == "round_amount":
        return float(rng.choice([5000, 10000, 25000, 50000, 100000]))
    if reason == "sanctions_match":
        return round(rng.uniform(500, 250000), 2)
    if reason == "velocity":
        return round(rng.uniform(50, 5000), 2)
    if reason == "high_risk_geo":
        base = 50000 if segment == "private" else 5000
        return round(rng.uniform(base, base * 5), 2)
    return round(rng.uniform(200, 50000), 2)


def _label(rng: random.Random, reason: str, country: str, segment: str,
           prior_alerts: int) -> int:
    """Stochastic label using reason + risk modifiers."""
    p = REASON_TP_BASE[reason]
    if country in HIGH_RISK_COUNTRIES:
        p *= 2.0
    elif country in MED_RISK_COUNTRIES:
        p *= 1.3
    if segment == "private":
        p *= 1.2
    if prior_alerts >= 3:
        p *= 1.5
    p = min(p, 0.85)
    return 1 if rng.random() < p else 0


def generate_alerts(n: int = 10_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate `n` synthetic alerts. Returns a pandas DataFrame."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(n):
        segment = _pick(rng, CUSTOMER_SEGMENTS, SEGMENT_WEIGHTS)
        country = _sample_country(rng, segment)
        counterparty_country = _sample_country(rng, segment)
        reason = rng.choice(ALERT_REASONS)
        amount = _sample_amount(rng, reason, segment)
        txn_type = _pick(rng, TXN_TYPES, TXN_TYPE_WEIGHTS)
        prior_alerts = rng.choices([0, 1, 2, 3, 5, 8], weights=[0.55, 0.20, 0.12, 0.08, 0.03, 0.02])[0]
        tenure = rng.randint(0, 25)
        label = _label(rng, reason, country, segment, prior_alerts)
        rows.append(Alert(
            alert_id=f"AL{i:06d}",
            customer_id=f"C{rng.randint(10_000, 99_999)}",
            customer_segment=segment,
            customer_tenure_years=tenure,
            country=country,
            counterparty_country=counterparty_country,
            txn_amount=amount,
            txn_type=txn_type,
            counterparty=f"CP{rng.randint(1000, 9999)}",
            alert_reason=reason,
            prior_alerts_90d=prior_alerts,
            is_true_positive=label,
        ).to_dict())
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def summary(df: pd.DataFrame) -> dict:
    """Quick distributional summary for sanity checks."""
    return {
        "n_alerts": int(len(df)),
        "tp_count": int(df["is_true_positive"].sum()),
        "tp_rate": float(df["is_true_positive"].mean()),
        "fp_rate": float(1 - df["is_true_positive"].mean()),
        "by_reason": df.groupby("alert_reason")["is_true_positive"].mean().to_dict(),
        "by_segment": df.groupby("customer_segment").size().to_dict(),
    }


if __name__ == "__main__":
    df = generate_alerts()
    out = write_csv(df, "data/alerts.csv")
    print(f"Wrote {len(df):,} alerts to {out}")
    print(summary(df))
