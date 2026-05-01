# C1 AML FP Reducer — State

## Status: v0.1 — initial benchmark complete

| Component                         | State    | Notes                                              |
| --------------------------------- | -------- | -------------------------------------------------- |
| pyproject + deps                  | DONE     | pandas, ollama, langgraph, langchain-core, click, pytest, pytest-cov, matplotlib, jupyter |
| `synth.py` — 10k synthetic alerts | DONE     | seed=42, ~92/8 FP/TP split, segment + geo modifiers |
| `baseline.py` — keep-all baseline | DONE     | 100% recall, 0% volume reduction                   |
| `triage.py` — LangGraph + Ollama  | DONE     | qwen2.5:7b, FP-confidence threshold 0.75, rule fallback |
| `eval.py` — metrics               | DONE     | volume reduction, recall, precision, F1, TP_missed |
| `cli.py` + `scripts/run_benchmark.py` | DONE | `python scripts/run_benchmark.py --sample 500`     |
| pytest suite                      | DONE     | 11 tests passing (synth, eval, triage, smoke)      |
| `notebooks/01_demo.ipynb`         | DONE     | end-to-end inline demo                             |
| `docs/whitepaper.md`              | DONE     | 5-page writeup with ASCII architecture             |
| README                            | DONE     | results, architecture, run instructions            |
| Pushed to GitHub main             | DONE     | github.com/ypatel39-commits/c1-aml-fp-reducer      |

## Key results (500-alert LLM sample, qwen2.5:7b @ FP-threshold 0.75)

See `data/metrics.json` for the live numbers; README headline is auto-updated
from that file on each benchmark run.

## Open follow-ups

- Run full 10k LLM benchmark (currently using 500-sample for speed; baseline runs on full 10k).
- Tune FP threshold per alert reason (e.g., 0.85 for sanctions_match — false-negative cost is much higher).
- Add a self-consistency vote (3 samples per alert) for high-stakes reasons.
- Replace synthetic data with public AML challenge data (Synapse / IBM AMLSim).
- Calibrate confidence with isotonic regression once we have a labelled holdout.

## Decisions

- **Local LLM (Ollama qwen2.5:7b)**: zero API cost, zero data egress.
- **LangGraph state machine**: explicit `enrich -> reason -> decide` topology
  makes the agent auditable for compliance (every node logs state).
- **Conservative dismissal threshold (0.75)**: skews recall over precision —
  the asymmetric cost of missing a SAR-worthy alert dominates analyst-hour
  savings.
- **Deterministic rule fallback**: keeps tests + CI green when Ollama is
  unavailable. Loud warning prevents accidental misuse in production.
