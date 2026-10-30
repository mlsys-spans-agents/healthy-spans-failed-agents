# Validation of the paper's experiments and results

Full claim-by-claim output: `python -m hs.validate` → `results/validation.json`.

## Correction applied to healthy-spans-filled.pdf

| Where | Was | Now | Why |
|---|---|---|---|
| §6, TRAIL classifier paragraph | "Even the low end is well above every rule that ignores the outcome" | "…well above the error, latency, cost and anomaly rules (though not the size-only policy below)" | SIZE-ONLY ignores the outcome and keeps 79.1% [66.1, 87.0] of TRAIL failures, which is above the classifier's low end of 63.5%. |

## Reproduced from the raw data

- **Dataset facts:** run and failure counts, spans (54,612 / 4,092), TRAIL errors (841 / 839 / 577 / 110), and 437 ERROR spans.
- **Table 1:** every row, plus decisive-span ERROR rates of 9.8% and 8.2%.
- **§5:** AUCs 0.518, 0.509 and 0.475; median 5 spans after the decisive one; 71.2%; TRAIL median 4; span kinds 135 / 36 / 13; Figure 3 histogram.
- **§6:** TAIL-ERR 14.8–14.9% and 8.5%, also at π = 1% and 10%; within-benchmark TAIL-ERR 8.9% and 13.2%; GAIA ERROR shares 44.8% and 28.6%; 21 / 17 / 14 runs; 86% of weighted bytes in ERROR traces; latency ERE range 0.55–1.55; long-run rule fires 2.0× more often on successful runs (counting agent steps).
- **Table 2:** HEAD, TAIL-ERR, TAIL-LATENCY, τ-bench n-gram, Isolation Forest (τ-bench), heuristics, measured-judge row (34.2%), parametric judges and oracle; every ERE equals recall / 15.
- **Table 4:** all 8 rows. **Figure 4:** error-anchored 25.0 / 26.4, keep-start 54.3, and which rule is best per dataset.
- **§10:** run lengths 131 s / 1,052 s; predicted keep rates 10.1 / 45.9 / 79.7 / 89.9%; the measured rates (12.2 / 46.6 / 81.8 / 89.9%) fall within one decision tick of these, as stated; 628 traces kept by the ERROR collector.

## Internally consistent (can't be recomputed without the LLM or collector runs)

- **Tables 5 and 6:** the n values sum to 176; kept shares match; full and view step accuracies rebuilt from Table 6 give Table 5 exactly; Bonferroni reasoning is correct (0.029 > 0.0167).
- **Table 3:** Wilson CIs for TPR and FPR; per-domain averages; 60.5% agreement; cost of $3.85 ≈ $4 at Haiku 4.5's $1/MTok; Who&When flag rate of 72.8% = 134 / 184.
- **§10 arithmetic:** 186 is within 1 SD of 198; memory ratios 2.2× and 3.3×; extrapolated 4.4 GB; 2.2 MB per held trace.
- **External facts:** the collector options `decision_wait_after_root_received` and `tail_storage` exist; references [6], [11], [15], [16], [17], [20], [21] and [22] match arXiv (titles and authors).

## Check before submission (not changed in the PDF)

1. **Collector version.** The official `otelcol-contrib` binary release repo has v0.120.0 but **no v0.120.1**, which exists only as a source tag. Confirm which binary was run.
2. **TRAIL classifier (82.7%, AUC 0.809) doesn't reproduce.** The reimplementation gets 51.8% (AUC 0.72), and 34.5% on average over 50 fold assignments. SIZE-ONLY alone gets 79.1%. Rerun the original code. The §6 GAIA/SWE-bench AUCs and the 30.0% TRAIL equal-count value come from the reimplementation.
3. **Not reproduced, probably implementation details:**

   | Item | Paper | Reproduced |
   |---|---|---|
   | τ-bench classifier | 38.9% | 35.3% (AUC matches: 0.690 vs 0.689) |
   | TAIL-COST | 17.1 / 10.0 | 15.8 / 0.0 |
   | TRAIL n-gram | 21.8% (AUC 0.452) | 20.0% (AUC 0.63) |
   | TRAIL Isolation Forest | 5.5% | 15.5% |
   | Figure 4 TRAIL anomalous, 50% budget | 53.6 | 46.4 |
   | Loose ERROR rule | 17.9% | 19.0% |
   | Any ERROR after decisive span | 35.3% | 35.9% (1 trace) |
   | Metadata share of bytes | 42.6% | depends on encoding |
   | Protobuf/byte ratio | 0.97 / 0.99 | 1.08 / 1.04 under our encoding; Spearman 0.9996 / 0.9999 matches |

4. **Narrative point.** SIZE-ONLY keeps 79.1% of TRAIL failures without any outcome signal (ERE 5.27, the best non-outcome policy). The abstract and introduction don't mention this. Consider adding it, since it bears on the claim that outcome signals are what help on TRAIL.
