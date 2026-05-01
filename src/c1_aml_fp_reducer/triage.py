"""LangGraph LLM triage agent.

Topology:
    enrich -> reason -> decide

- enrich: pulls customer + alert context into a prompt-ready payload
- reason: calls local Ollama (qwen2.5:7b) with strict JSON output instructions
- decide: parses confidence and applies the FP-dismissal threshold

If Ollama is unavailable we fall back to a deterministic rule-based triage
so unit tests and CI runs still produce deterministic decisions. The fallback
is loud — it logs a clear warning so users never mistake it for the real LLM.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import pandas as pd

try:  # ollama is optional at import time so tests can run without it.
    import ollama  # type: ignore
except Exception:  # pragma: no cover - import-time fallback
    ollama = None  # type: ignore

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_FP_THRESHOLD = 0.75  # dismiss only when LLM is >= 75% sure it's FP
HIGH_RISK_COUNTRIES = {"IR", "KP", "SY", "RU", "VE", "MM"}


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class TriageState(TypedDict, total=False):
    alert: dict
    prompt: str
    raw_response: str
    fp_confidence: float
    decision: str       # "keep" | "dismiss"
    rationale: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def enrich(state: TriageState) -> TriageState:
    a = state["alert"]
    prompt = (
        "You are a senior AML compliance analyst at a major US bank. "
        "Given the alert below, decide whether it is a likely FALSE POSITIVE "
        "(routine activity that does not warrant SAR investigation).\n\n"
        f"Alert ID: {a['alert_id']}\n"
        f"Reason: {a['alert_reason']}\n"
        f"Customer segment: {a['customer_segment']} | tenure {a['customer_tenure_years']}y\n"
        f"Customer country: {a['country']} | Counterparty country: {a['counterparty_country']}\n"
        f"Transaction: {a['txn_type']} ${a['txn_amount']:,.2f}\n"
        f"Prior alerts in last 90d: {a['prior_alerts_90d']}\n\n"
        "Reply with STRICT JSON only. Schema:\n"
        '{"fp_confidence": <float 0-1>, "rationale": "<one sentence>"}\n'
        "fp_confidence = probability this is a FALSE POSITIVE (1.0 = certainly FP, 0.0 = certainly true)."
    )
    return {**state, "prompt": prompt}


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    if ollama is None:
        raise RuntimeError("ollama package not installed")
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.0, "num_predict": 120},
        stream=False,
    )
    return resp["response"]  # type: ignore[index]


def _rule_fallback(alert: dict) -> tuple[float, str]:
    """Deterministic rule-based FP scorer used when Ollama is unreachable."""
    p_fp = 0.92  # prior: most alerts are FP
    if alert["country"] in HIGH_RISK_COUNTRIES or alert["counterparty_country"] in HIGH_RISK_COUNTRIES:
        p_fp -= 0.35
    if alert["alert_reason"] in {"sanctions_match", "structuring"}:
        p_fp -= 0.20
    if alert["prior_alerts_90d"] >= 3:
        p_fp -= 0.15
    if alert["customer_segment"] == "retail" and alert["txn_amount"] < 5000:
        p_fp += 0.05
    p_fp = max(0.0, min(1.0, p_fp))
    return p_fp, "rule_fallback"


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_response(raw: str) -> tuple[float, str]:
    match = _JSON_RE.search(raw)
    if not match:
        return 0.5, "unparseable_response"
    try:
        obj = json.loads(match.group(0))
        conf = float(obj.get("fp_confidence", 0.5))
        rationale = str(obj.get("rationale", ""))[:200]
        conf = max(0.0, min(1.0, conf))
        return conf, rationale
    except Exception:
        return 0.5, "json_decode_error"


def reason(state: TriageState, llm_call: Callable[[str], str] | None = None) -> TriageState:
    prompt = state["prompt"]
    caller = llm_call or _call_ollama
    try:
        raw = caller(prompt)
        conf, rationale = _parse_response(raw)
    except Exception as e:
        logger.warning("Ollama call failed (%s); using rule fallback", e)
        conf, rationale = _rule_fallback(state["alert"])
        raw = f"FALLBACK: {rationale}"
    return {**state, "raw_response": raw, "fp_confidence": conf, "rationale": rationale}


def decide(state: TriageState, threshold: float = DEFAULT_FP_THRESHOLD) -> TriageState:
    decision = "dismiss" if state.get("fp_confidence", 0.0) >= threshold else "keep"
    return {**state, "decision": decision}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph(llm_call: Callable[[str], str] | None = None,
                threshold: float = DEFAULT_FP_THRESHOLD):
    g = StateGraph(TriageState)
    g.add_node("enrich", enrich)
    g.add_node("reason", lambda s: reason(s, llm_call=llm_call))
    g.add_node("decide", lambda s: decide(s, threshold=threshold))
    g.set_entry_point("enrich")
    g.add_edge("enrich", "reason")
    g.add_edge("reason", "decide")
    g.add_edge("decide", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------
@dataclass
class TriageResult:
    alert_id: str
    decision: str
    fp_confidence: float
    rationale: str


def triage_dataframe(df: pd.DataFrame, *, llm_call: Callable[[str], str] | None = None,
                     threshold: float = DEFAULT_FP_THRESHOLD,
                     progress: bool = False) -> pd.DataFrame:
    """Run triage over every row of `df`. Returns df with decision columns."""
    graph = build_graph(llm_call=llm_call, threshold=threshold)
    decisions: list[dict[str, Any]] = []
    n = len(df)
    for i, row in enumerate(df.to_dict(orient="records")):
        state = graph.invoke({"alert": row})
        decisions.append({
            "alert_id": row["alert_id"],
            "decision": state["decision"],
            "fp_confidence": state["fp_confidence"],
            "rationale": state.get("rationale", ""),
        })
        if progress and (i + 1) % 50 == 0:
            print(f"  triaged {i + 1}/{n}", flush=True)
    out = df.merge(pd.DataFrame(decisions), on="alert_id", how="left")
    return out


def triage_label_kept(triaged: pd.DataFrame) -> pd.Series:
    return triaged["decision"] == "keep"
