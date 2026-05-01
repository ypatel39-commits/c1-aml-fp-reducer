"""Evaluation metrics for AML alert triage.

The relevant business metrics differ from typical ML classification:

- **Volume reduction**: % of alerts the system DISMISSES. Higher = more
  analyst hours saved.
- **Recall on true positives**: of all real SAR-worthy alerts, what fraction
  did the system KEEP? Must be 99%+ for compliance comfort.
- **Precision on kept alerts**: of the alerts the system KEPT, what fraction
  were actually true positives?
- **F1**: harmonic mean of precision and recall on the kept set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Metrics:
    name: str
    n_alerts: int
    n_kept: int
    n_dismissed: int
    volume_reduction_pct: float
    recall: float
    precision: float
    f1: float
    tp_kept: int
    tp_missed: int   # FALSE NEGATIVES — the dangerous ones
    fp_kept: int
    fp_dismissed: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(df: pd.DataFrame, kept_mask: pd.Series, *, name: str) -> Metrics:
    """Compute Metrics given the labelled frame and a boolean mask of kept rows."""
    if len(df) != len(kept_mask):
        raise ValueError("df and kept_mask must align")
    y_true = df["is_true_positive"].astype(int)
    kept = kept_mask.astype(bool).to_numpy()

    tp_kept = int(((y_true == 1) & kept).sum())
    tp_missed = int(((y_true == 1) & ~kept).sum())
    fp_kept = int(((y_true == 0) & kept).sum())
    fp_dismissed = int(((y_true == 0) & ~kept).sum())

    n = len(df)
    n_kept = int(kept.sum())
    n_dismissed = n - n_kept

    total_tp = tp_kept + tp_missed
    recall = tp_kept / total_tp if total_tp else 1.0
    precision = tp_kept / n_kept if n_kept else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    volume_reduction = n_dismissed / n if n else 0.0

    return Metrics(
        name=name,
        n_alerts=n,
        n_kept=n_kept,
        n_dismissed=n_dismissed,
        volume_reduction_pct=round(volume_reduction * 100, 2),
        recall=round(recall, 4),
        precision=round(precision, 4),
        f1=round(f1, 4),
        tp_kept=tp_kept,
        tp_missed=tp_missed,
        fp_kept=fp_kept,
        fp_dismissed=fp_dismissed,
    )


def format_report(metrics: list[Metrics]) -> str:
    """Pretty-printed text table comparing one or more strategies."""
    header = (
        f"{'strategy':<14} {'n':>6} {'kept':>6} {'dismiss':>7} "
        f"{'vol_red%':>8} {'recall':>7} {'prec':>6} {'f1':>6} {'TP_miss':>8}"
    )
    lines = [header, "-" * len(header)]
    for m in metrics:
        lines.append(
            f"{m.name:<14} {m.n_alerts:>6} {m.n_kept:>6} {m.n_dismissed:>7} "
            f"{m.volume_reduction_pct:>8.2f} {m.recall:>7.4f} "
            f"{m.precision:>6.4f} {m.f1:>6.4f} {m.tp_missed:>8}"
        )
    return "\n".join(lines)
