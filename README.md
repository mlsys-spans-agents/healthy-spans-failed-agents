# Healthy Spans, Failed Agents: reproduction code

```
export HF_TOKEN=...      # TRAIL is gated on Hugging Face
./reproduce.sh           # fetch data, run everything (~20 s after download)
```

| File | Contents |
|---|---|
| `hs/data.py` | τ-bench and Who&When converters (messages → GenAI spans, ERROR-status rules) |
| `hs/trail.py` | TRAIL loader (real OTLP spans, depth-first order, decisive = earliest high-impact error) |
| `hs/sampling.py` | trace-level policies, weighted equal-budget admission with fractional tie groups, bootstrap, classifier, anomaly samplers |
| `hs/spans.py` | span-level trimming policies and diagnosis-study views |
| `hs/run_all.py` | all experiments → `results/` |
| `PAPER_FILLS.md` | text to paste for each TODO, open items, and the reproduction check |

The LLM judge, the LLM investigator and the collector experiments aren't included. Their outputs and harness weren't available here.
