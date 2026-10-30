# Healthy Spans, Failed Agents — reproduction code

Reproduces the checkable numbers in the paper from the raw public datasets. No GPU
and no API key are needed for the core experiments (only to download the gated TRAIL
dataset). Everything runs in about 20 seconds once the data is in place.

## Requirements

- Python 3.10+
- The packages in `requirements.txt` (`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`)

```powershell
pip install -r requirements.txt
```

## 1. Get the data

The three datasets go under `data/`. If `data/trail/`, `data/taubench/` and
`data/whowhen/` are already populated (they ship with this checkout), skip to step 2.

TRAIL is gated on Hugging Face: accept its terms at
https://huggingface.co/datasets/PatronusAI/TRAIL and get a token, then:

```powershell
$env:HF_TOKEN = "hf_..."
bash ./fetch_data.sh        # Git Bash or WSL
```

`fetch_data.sh` downloads:

- **τ-bench** trajectories (GPT-4o and Sonnet-3.5, airline + retail) → `data/taubench/`
- **Who&When** (Algorithm-Generated, Hand-Crafted) → `data/whowhen/`
- **TRAIL** (GAIA, SWE-bench) → `data/trail/` *(requires `HF_TOKEN`)*

## 2. Run the experiments

```powershell
python -m hs.run_all
```

Writes `results/results.json` (every computed number) and `results/report.txt`
(the same numbers as a readable log). This rebuilds Tables 1, 2, 4 and the §5, §6
and §10 figures from the raw data.

To do both steps at once on Git Bash / WSL:

```bash
export HF_TOKEN=...
./reproduce.sh              # fetch data (if missing), install deps, run everything
```

## 3. Validate the paper's claims

```powershell
python -m hs.validate
```

Recomputes each numeric claim where the data allows it and checks the internal
arithmetic of the rest. Writes `results/validation.json` and prints a claim-by-claim
table (`OK` / `DIFF` / `INFO`). See [VALIDATION.md](VALIDATION.md) for the summary of
what reproduces and what to check before submission.

## Optional: collector replay (needs Docker)

The `hs.collector` experiments replay traces through the stock OpenTelemetry
Collector (`otelcol-contrib`) in Docker. These need Docker plus the extra packages
`requests` and `opentelemetry-proto`, and take longer to run.

```powershell
python -m hs.collector equiv      # τ-bench: ERROR vs outcome vs probabilistic, kept-set equivalence
python -m hs.collector window     # TRAIL: kept share per decision window (60/120/300/600 s)
python -m hs.collector memory     # TRAIL: collector RSS / CPU / held traces at different rates
```

Each writes `results/collector_<exp>.json`. Redraw Figure 5 from those results
(needs `matplotlib`):

```powershell
python -m hs.fig5                  # -> results/fig5.png
```

## Layout

| Path | Contents |
|---|---|
| `hs/data.py` | τ-bench and Who&When converters (messages → GenAI spans, ERROR-status rules) |
| `hs/trail.py` | TRAIL loader (real OTLP spans, depth-first order, decisive = earliest high-impact error) |
| `hs/sampling.py` | trace-level policies, weighted equal-budget admission, bootstrap, classifier, anomaly samplers |
| `hs/spans.py` | span-level trimming policies and diagnosis-study views |
| `hs/run_all.py` | all experiments → `results/` |
| `hs/validate.py` | claim-by-claim validation → `results/validation.json` |
| `hs/collector.py` | OpenTelemetry Collector replay harness (Docker) |
| `hs/fig5.py` | redraw Figure 5 from the collector results |
| `data/` | the three raw datasets |
| `results/` | computed outputs |

The LLM judge, the LLM investigator and the collector experiments' original harness
aren't included — their outputs weren't available here.
