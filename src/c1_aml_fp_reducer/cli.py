"""Click CLI: c1-aml-bench."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import baseline, eval as eval_mod, synth, triage


@click.group()
def main() -> None:
    """C1 AML False-Positive Reducer CLI."""


@main.command("gen")
@click.option("--n", default=10_000, type=int)
@click.option("--out", default="data/alerts.csv", type=click.Path())
def gen_cmd(n: int, out: str) -> None:
    df = synth.generate_alerts(n=n)
    p = synth.write_csv(df, out)
    click.echo(f"wrote {len(df):,} alerts -> {p}")
    click.echo(json.dumps(synth.summary(df), indent=2, default=str))


@main.command("bench")
@click.option("--alerts", default="data/alerts.csv", type=click.Path(exists=True))
@click.option("--sample", default=0, type=int, help="run triage on first N rows (0 = all)")
@click.option("--out", default="data/metrics.json", type=click.Path())
@click.option("--no-llm", is_flag=True, help="force rule-based fallback (no Ollama)")
def bench_cmd(alerts: str, sample: int, out: str, no_llm: bool) -> None:
    import pandas as pd

    df = pd.read_csv(alerts)
    if sample:
        df = df.head(sample).copy()

    base = baseline.baseline_decide(df)
    base_metrics = eval_mod.evaluate(df, baseline.baseline_label_kept(df), name="baseline")

    llm_call = (lambda _p: "") if no_llm else None
    triaged = triage.triage_dataframe(df, llm_call=llm_call, progress=True)
    tri_metrics = eval_mod.evaluate(df, triage.triage_label_kept(triaged), name="llm_triage")

    payload = {"baseline": base_metrics.to_dict(), "llm_triage": tri_metrics.to_dict()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    click.echo(eval_mod.format_report([base_metrics, tri_metrics]))
    click.echo(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
