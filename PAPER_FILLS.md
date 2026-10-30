# Fills for the TODO / TBD / [CHECK] items

All numbers are from `python -m hs.run_all` (`results/results.json`, `results/report.txt`).
Setting unless stated: 15% byte budget, failure rate π = 5%.

## Filled

| Where | Item | Replacement |
|---|---|---|
| §4.1 | [CHECK] reweighting | Confirmed. `hs/sampling.py:weights` gives failed runs π/0.4025 and successful runs (1−π)/0.5975. Stored bytes and recall both use these weights. Remove the marker. Optional addition: "TRAIL is reweighted the same way from its own failure share (110/148 = 74.3%)." |
| Table 2 | SIZE-ONLY row | τ-bench **21.1%, ERE 1.41** [17.8, 24.4]; TRAIL **79.1%, ERE 5.27** [66.1, 87.0] |
| §6 | SIZE-ONLY X / Y | X = **21.1**, Y = **79.1** |
| §6 | Classifier at equal count | τ-bench X = **37.9** (size-only: 11.3); TRAIL Y = **30.0**\* (size-only: 27.3) |
| §6 | "keep one" | Keep the **second** option: "On TRAIL the size-only policy comes close to the classifier (79.1% against 82.7%), so most of the TRAIL gain reflects which runs are small rather than which runs failed; on τ-bench the gap remains (21.1% against 38.9%)." |
| §6 | Within GAIA / SWE-bench | "Within GAIA the classifier's AUC is 0.62 (0.65 when refit on GAIA alone) and within SWE-bench 0.69 (0.74; only 31 traces). Benchmark identity alone separates failures with AUC 0.66, and with folds grouped by benchmark the classifier falls to AUC 0.39, so part of its gain comes from telling the benchmarks apart."\* |
| §9 | Chance, error-anchored vs full | X = **25.1**, Y = **10.8** (traces where the ERROR+8 view kept the decisive span) |
| Table 6 | Chance column | ERROR+8 kept **25.1%**; Last 8 kept **13.7%**; First 30% kept **32.4%**† |
| §4.4 | Marker TODO | The ablation was not run. Use: "The marker also shows which step numbers are missing; we discuss its effect in Section 9." |
| §13 | Marker TODO | Delete "and the effect of removing the marker". |
| §7 | Judge truncation (optional) | Between 22% and 32% of TRAIL runs reach the 40,000-character cap, depending on how spans were rendered. Use the exact value from the judge input files, or drop the sentence. |

\* Computed with the reimplemented TRAIL classifier, which does **not** reproduce the paper's TRAIL classifier (see below).
† My First 30% view keeps the decisive span in 50 of 184 traces; the paper reports 68 of 176. Recompute this cell from the original view files.
Chance values are over all 184 failures, because the 8 traces lost to the content filter aren't identified.

## Items without data: wording used in healthy-spans-filled.pdf

| Where | What the PDF now says | Why |
|---|---|---|
| Table 6, Agent columns | "–" | No investigator outputs are available. |
| §9, agent sentence | "It falls too: over all 176 failures the investigator names the responsible agent 64.2% of the time from the full trace and 40.9% from the error-anchored view (Table 5)." | Uses Table 5's measured numbers in place of the unknown 121-trace subset values. |
| §10, release | "release v0.120.1, the release current when we built the harness, with the processor's default settings" | v0.120.1 was already named in the TODO. |
| §10, threshold | "set when its score exceeded a fixed threshold)" | The actual threshold isn't recorded. |
| §10, window scaling | marker removed; the sentence already says the window was divided by ten | Confirm against the replay logs. |
| §7, judge truncation | "Between 22% and 32% of TRAIL runs reached the 40,000-character limit…" | A range, because the judge's span format isn't known. |
| Artifacts | "(provided as supplementary material)" | No repository URL exists yet. |

## Still open (needs data that isn't here)

- **Table 6 Agent columns and §9 agent X/Y, "the same / a smaller":** these need the investigator's raw answers. Rerunning them needs an Anthropic API key.
- **§10 collector release, threshold, how it was chosen, window scaling:** these are facts about the original run. With the reimplemented classifier, p > 0.5 flags 561 of 1,980 runs against the paper's 527, so 0.5 is plausible but needs confirming.
- **Artifact URL:** needs the anonymized repository to be created.

## Reproduction check of the paper's existing numbers

**Exact or within 0.3 points:** run, failure and span counts for all three datasets; Table 1; decisive-span ERROR rates (9.8% strict, 8.2% TRAIL, 5.5% annotated); 437 TRAIL ERROR spans; decisive-span position statistics (median 5 after, 71.2% have ≥4 after, 35.3% have an error after; TRAIL median 4); TRAIL run lengths (131 s / 1,052 s); TAIL-ERR (14.9 / 8.5), including at π = 1% and 10%; n-gram on τ-bench (25.3); TAIL-LATENCY (15.5); all τ-bench heuristics within 0.4; TRAIL no-final-answer (18.2); parametric judges and oracle; all of Table 4; Figure 4 endpoints (WW error-anchored 25.0, keep-start 54.3; TRAIL error-anchored 26.4).

**Not reproduced from the paper's description:**

| Number | Paper | Reimplementation |
|---|---|---|
| TRAIL metadata classifier | 82.7%, AUC 0.809; 50 fold assignments: mean 81.5% (63.5–88.2) | 51.8%, AUC 0.721; 50 fold assignments: mean 34.5% (6.4–60.9) |
| τ-bench metadata classifier | 38.9%, AUC 0.689 | 35.3%, AUC 0.690 |
| TAIL-COST τ-bench / TRAIL | 17.1 / 10.0 | 15.8 / 0.0 (token definition unclear) |
| Isolation Forest, TRAIL | 5.5% | 15.5% |
| First 30% diagnosis view, kept | 68/176 | 50/184 |
| Figure 4 TRAIL anomalous spans, 50% budget | 53.6 | 46.4 |
| Who&When loose ERROR rule | 17.9% | 19.0% |
| TRAIL median trace size | 386 KB | 394 KB |

The TRAIL classifier gap matters most. SIZE-ONLY alone keeps 79.1% of TRAIL failures, so the 82.7% figure is close to what size alone achieves. Before relying on it, rerun the original classifier code next to `hs/`.
