"""Smoke test — package imports cleanly."""
from c1_aml_fp_reducer import baseline, eval as eval_mod, synth, triage


def test_imports():
    assert hasattr(synth, "generate_alerts")
    assert hasattr(baseline, "baseline_decide")
    assert hasattr(triage, "build_graph")
    assert hasattr(eval_mod, "evaluate")
