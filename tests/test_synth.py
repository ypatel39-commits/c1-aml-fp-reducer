"""Synthetic data generation tests."""
import pandas as pd

from c1_aml_fp_reducer import synth


def test_generate_alerts_shape_and_columns():
    df = synth.generate_alerts(n=500)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    expected = {
        "alert_id", "customer_id", "customer_segment", "country",
        "txn_amount", "txn_type", "counterparty", "alert_reason",
        "is_true_positive", "prior_alerts_90d",
    }
    assert expected.issubset(df.columns)


def test_generate_alerts_is_reproducible():
    a = synth.generate_alerts(n=200, seed=42)
    b = synth.generate_alerts(n=200, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_alert_distribution_matches_industry():
    df = synth.generate_alerts(n=10_000, seed=42)
    tp_rate = df["is_true_positive"].mean()
    # industry pain point: ~5-10% TP rate. Allow generous bounds.
    assert 0.04 <= tp_rate <= 0.15, f"TP rate {tp_rate} outside expected band"
    # alert_id uniqueness
    assert df["alert_id"].is_unique
