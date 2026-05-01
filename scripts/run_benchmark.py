"""End-to-end benchmark: generate -> baseline -> LLM triage -> metrics.

Examples
--------
    # full 10k baseline + 500-alert LLM sample (recommended for first run)
    python scripts/run_benchmark.py --sample 500

    # full 10k LLM triage (slow)
    python scripts/run_benchmark.py

    # CI / no-Ollama path
    python scripts/run_benchmark.py --sample 200 --no-llm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from c1_aml_fp_reducer import baseline, eval as eval_mod, synth, triage  # noqa: E402


def _ollama_up() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10_000, help="synthetic dataset size")
    ap.add_argument("--sample", type=int, default=500,
                    help="LLM-triage subsample (0 = all). Baseline always runs on full set.")
    ap.add_argument("--no-llm", action="store_true", help="force deterministic fallback")
    ap.add_argument("--out", default="data/metrics.json")
    ap.add_argument("--alerts-out", default="data/alerts.csv")
    ap.add_argument("--triaged-out", default="data/triaged.csv")
    args = ap.parse_args()

    t0 = time.time()

    print(f"[1/4] Generating {args.n:,} synthetic alerts ...")
    df = synth.generate_alerts(n=args.n)
    synth.write_csv(df, args.alerts_out)
    summary = synth.summary(df)
    print(f"      TP rate = {summary['tp_rate']:.3%}  ({summary['tp_count']:,} TPs)")

    print("[2/4] Baseline (keep all) ...")
    base_metrics = eval_mod.evaluate(df, baseline.baseline_label_kept(df), name="baseline")

    print("[3/4] LLM triage agent ...")
    use_llm = not args.no_llm and _ollama_up()
    if not use_llm:
        print("      Ollama unreachable or --no-llm set — using deterministic rule fallback.")
    llm_call = (lambda _p: "") if not use_llm else None

    sample_df = df.head(args.sample).copy() if args.sample else df
    triaged = triage.triage_dataframe(sample_df, llm_call=llm_call, progress=True)
    Path(args.triaged_out).parent.mkdir(parents=True, exist_ok=True)
    triaged.to_csv(args.triaged_out, index=False)
    tri_metrics = eval_mod.evaluate(sample_df, triage.triage_label_kept(triaged), name="llm_triage")

    print("[4/4] Writing metrics ...")
    payload = {
        "config": {
            "n_alerts": args.n,
            "triage_sample_size": len(sample_df),
            "llm_used": use_llm,
            "model": triage.OLLAMA_MODEL if use_llm else "rule_fallback",
            "fp_threshold": triage.DEFAULT_FP_THRESHOLD,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "synth_summary": summary,
        "baseline": base_metrics.to_dict(),
        "llm_triage": tri_metrics.to_dict(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print()
    print(eval_mod.format_report([base_metrics, tri_metrics]))
    print()
    print(f"metrics  -> {args.out}")
    print(f"alerts   -> {args.alerts_out}")
    print(f"triaged  -> {args.triaged_out}")
    print(f"elapsed  -> {payload['config']['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
