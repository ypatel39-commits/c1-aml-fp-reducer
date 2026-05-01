"""LangGraph triage agent tests — uses an injected fake LLM, no Ollama needed."""
import json

from c1_aml_fp_reducer import synth, triage


def _fake_llm_high_fp(_prompt: str) -> str:
    return json.dumps({"fp_confidence": 0.95, "rationale": "looks routine"})


def _fake_llm_low_fp(_prompt: str) -> str:
    return json.dumps({"fp_confidence": 0.10, "rationale": "looks suspicious"})


def test_graph_dismisses_when_confidence_high():
    df = synth.generate_alerts(n=20).head(5)
    out = triage.triage_dataframe(df, llm_call=_fake_llm_high_fp)
    assert (out["decision"] == "dismiss").all()
    assert (out["fp_confidence"] >= 0.75).all()


def test_graph_keeps_when_confidence_low():
    df = synth.generate_alerts(n=20).head(5)
    out = triage.triage_dataframe(df, llm_call=_fake_llm_low_fp)
    assert (out["decision"] == "keep").all()


def test_parse_response_handles_garbage():
    conf, rat = triage._parse_response("model went off the rails, no json here")
    assert 0.0 <= conf <= 1.0
    assert isinstance(rat, str)


def test_rule_fallback_is_deterministic():
    df = synth.generate_alerts(n=30)
    # Use empty-string llm_call -> _parse_response returns (0.5, _) which is below threshold
    # so we must invoke fallback explicitly via Exception path:
    def boom(_p: str) -> str:
        raise RuntimeError("ollama down")

    a = triage.triage_dataframe(df, llm_call=boom)
    b = triage.triage_dataframe(df, llm_call=boom)
    # same inputs + deterministic fallback -> identical decisions
    assert (a["decision"].to_numpy() == b["decision"].to_numpy()).all()
