"""Evaluation metric tests."""
import pandas as pd

from c1_aml_fp_reducer import baseline, eval as eval_mod, synth


def test_baseline_metrics_full_recall():
    df = synth.generate_alerts(n=1_000)
    m = eval_mod.evaluate(df, baseline.baseline_label_kept(df), name="baseline")
    assert m.recall == 1.0
    assert m.volume_reduction_pct == 0.0
    assert m.tp_missed == 0
    assert m.n_kept == 1_000


def test_evaluate_perfect_oracle():
    df = synth.generate_alerts(n=500)
    # oracle keeps only true positives — perfect precision and recall
    oracle_keep = df["is_true_positive"].astype(bool)
    m = eval_mod.evaluate(df, oracle_keep, name="oracle")
    assert m.recall == 1.0
    assert m.precision == 1.0
    assert m.f1 == 1.0
    assert m.tp_missed == 0
    assert m.fp_kept == 0


def test_evaluate_dismiss_all():
    df = synth.generate_alerts(n=200)
    keep_none = pd.Series([False] * len(df), index=df.index)
    m = eval_mod.evaluate(df, keep_none, name="dismiss_all")
    assert m.volume_reduction_pct == 100.0
    assert m.recall == 0.0  # every TP missed — disaster scenario
    assert m.n_kept == 0
