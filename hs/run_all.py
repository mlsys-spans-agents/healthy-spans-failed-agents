"""Reproduce the paper's checkable numbers and compute the open (TODO/TBD) items.

Usage: python -m hs.run_all   ->  results/results.json, results/report.md
"""
import json
import os
from collections import Counter

import numpy as np

from . import sampling as S
from . import spans as V
from .data import load_taubench, load_trail, load_whowhen

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
R = {}
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def pct(x):
    return round(100 * float(x), 1)


def trace_level(name, T):
    b, y = S.arrays(T)
    P = {"HEAD": S.p_head(T), "SIZE-ONLY": S.p_size(T), "TAIL-ERR": S.p_err(T), "TAIL-COST": S.p_cost(T),
         "TAIL-ERR+COST": S.p_err_cost(T)}
    if name == "trail":
        P["TAIL-LATENCY"] = S.p_latency(T)
    P["Anomaly: Isol. Forest"] = S.iforest_scores(T)
    P["Anomaly: n-gram"] = S.p_ngram(T)
    P.update(S.heuristics(T))
    P["Metadata classifier"] = S.classifier_scores(T)
    P["Oracle"] = y.astype(float)
    res = {}
    log(f"\n## Trace-level, {name} (15% byte budget, pi=5%)")
    log(f"{'policy':35s} {'recall':>7s} {'ERE':>5s} {'AUC':>6s} {'eq-count':>8s}  95% CI (bytes)")
    for k, v in P.items():
        r = S.recall_at(v, b=b, y=y)
        rc = S.recall_at(v, b=b, y=y, by="count")
        a = S.auc(v, y) if len(set(v)) > 1 else None
        ci = S.bootstrap_ci(v, b, y, n=1000) if k in ("SIZE-ONLY", "Metadata classifier", "TAIL-ERR") else None
        res[k] = dict(recall=pct(r), ere=round(r / 0.15, 2), auc=None if a is None else round(a, 3),
                      recall_equal_count=pct(rc), ci=None if ci is None else [pct(c) for c in ci])
        log(f"{k:35s} {pct(r):7.1f} {r / .15:5.2f} {'' if a is None else f'{a:.3f}':>6s} {pct(rc):8.1f}"
            + (f"  [{pct(ci[0])}, {pct(ci[1])}]" if ci is not None else ""))
    for tpr, fpr in [(.6, .05), (.8, .1), (.95, .05)]:
        r = S.recall_at(None, b=b, y=y, split=np.where(y, tpr, fpr))
        res[f"Judge TPR {tpr}/FPR {fpr}"] = dict(recall=pct(r), ere=round(r / .15, 2))
        log(f"{'Judge TPR %.2f / FPR %.2f' % (tpr, fpr):35s} {pct(r):7.1f} {r / .15:5.2f}")
    # size-only and classifier across budgets
    res["size_only_curve"] = {f"{int(B * 100)}%": pct(S.recall_at(P["SIZE-ONLY"], b=b, y=y, budget=B))
                              for B in (0.05, 0.10, 0.15, 0.20, 0.30)}
    log("SIZE-ONLY across budgets:", res["size_only_curve"])
    # other failure rates
    res["tail_err_other_pi"] = {pi: pct(S.recall_at(P["TAIL-ERR"], b=b, y=y, pi=pi)) for pi in (0.01, 0.10)}
    res["size_only_other_pi"] = {pi: pct(S.recall_at(P["SIZE-ONLY"], b=b, y=y, pi=pi)) for pi in (0.01, 0.10)}
    log("TAIL-ERR at pi=1%/10%:", res["tail_err_other_pi"], " SIZE-ONLY at pi=1%/10%:", res["size_only_other_pi"])
    # size in failed vs other runs
    res["median_bytes_failed_vs_ok"] = [int(np.median(b[y])), int(np.median(b[~y]))]
    log("median bytes failed / not failed:", res["median_bytes_failed_vs_ok"])
    return res, P, b, y


def trail_by_benchmark(T, P, b, y):
    log("\n## TRAIL classifier within each benchmark")
    bench = np.array([t["meta"]["bench"] for t in T])
    p = P["Metadata classifier"]
    res = {}
    for bm in ("GAIA", "SWE"):
        m = bench == bm
        res[f"pooled_oof_auc_{bm}"] = round(S.auc(p[m], y[m]), 3)
        res[f"size_only_auc_{bm}"] = round(S.auc(-b[m], y[m]), 3)
        res[f"n_{bm}"] = [int(m.sum()), int(y[m].sum())]
        # refit within the benchmark only
        sub = [t for t, k in zip(T, m) if k]
        q = S.classifier_scores(sub)
        res[f"within_refit_auc_{bm}"] = round(S.auc(q, y[m]), 3)
        log(f"{bm}: n={m.sum()} failed={y[m].sum()}  pooled-OOF AUC={res[f'pooled_oof_auc_{bm}']}  "
            f"refit-within AUC={res[f'within_refit_auc_{bm}']}  size-only AUC={res[f'size_only_auc_{bm}']}")
    # benchmark-only predictor: AUC of "is SWE" as failure score, and folds grouped by benchmark
    res["auc_benchmark_identity"] = round(S.auc((bench == "SWE").astype(float), y), 3)
    q = S.classifier_scores(T, groups=bench, n_splits=2)
    res["benchmark_grouped_auc"] = round(S.auc(q, y), 3)
    res["benchmark_grouped_recall"] = pct(S.recall_at(q, b=b, y=y))
    log("AUC of benchmark identity alone:", res["auc_benchmark_identity"],
        " classifier with folds grouped by benchmark: AUC", res["benchmark_grouped_auc"],
        "recall", res["benchmark_grouped_recall"])
    # fold-assignment spread (paper: 50 assignments)
    rec = [S.recall_at(S.classifier_scores(T, seed=s), b=b, y=y) for s in range(1, 51)]
    res["fold_spread_recall"] = dict(mean=pct(np.mean(rec)), min=pct(min(rec)), max=pct(max(rec)))
    log("classifier recall over 50 fold assignments:", res["fold_spread_recall"])
    return res


def tau_transfer(T, b, y):
    log("\n## tau-bench classifier transfer")
    res = {}
    for key in ("model", "domain"):
        vals = sorted({t["meta"][key] for t in T})
        for a in vals:
            tr = np.array([t["meta"][key] == a for t in T])
            te = ~tr
            p = S.classifier_scores(T, train_idx=np.where(tr)[0], test_idx=np.where(te)[0])
            r = S.recall_at(p, b=b[te], y=y[te])
            res[f"train_{a}"] = dict(auc=round(S.auc(p, y[te]), 3), recall=pct(r))
            log(f"train on {a}, test on the rest: AUC {res[f'train_{a}']['auc']}  recall {pct(r)}")
    return res


def diagnosis_chance(W):
    log("\n## Diagnosis views (Who&When, all 184 failures) and chance baselines")
    res = {}
    full = V.DIAG_VIEWS["Full"]
    for name, f in V.DIAG_VIEWS.items():
        vis = [f(t) for t in W]
        kept = np.array([t["decisive"] in v for t, v in zip(W, vis)])
        chance_view = np.array([1 / len(v) if (t["decisive"] in v) else 0.0 for t, v in zip(W, vis)])
        chance_full = np.array([1 / len(t["spans"]) for t in W])
        pooled = sum(map(len, vis)) / sum(len(t["spans"]) for t in W)
        res[name] = dict(
            spans_pooled=pct(pooled), kept=int(kept.sum()), dropped=int((~kept).sum()),
            chance_view_overall=pct(chance_view.mean()),
            chance_view_on_kept=pct(chance_view[kept].mean()) if kept.any() else None,
            chance_full_on_kept=pct(chance_full[kept].mean()) if kept.any() else None,
            chance_full_on_dropped=pct(chance_full[~kept].mean()) if (~kept).any() else None)
        log(f"{name:10s} spans {pct(pooled):5.1f}%  kept {kept.sum():3d}/{len(W)}  "
            f"chance(view|kept)={res[name]['chance_view_on_kept']}%  chance(full|kept)={res[name]['chance_full_on_kept']}%")
    return res


def judge_truncation(T, per_span=1200, per_run=40000):
    log("\n## TRAIL judge input truncation")
    res = {}
    for label, render in [("span JSON", lambda s: s["text"]),
                          ("name + input/output", lambda s: f"[{s['name']}] " + s.get("io", ""))]:
        n = np.array([sum(min(len(render(s)), per_span) for s in t["spans"]) for t in T])
        res[label] = dict(share_at_limit=pct((n >= per_run).mean()), median_chars=int(np.median(n)))
        log(f"{label}: {res[label]['share_at_limit']}% of runs reach {per_run} chars "
            f"(median {res[label]['median_chars']} chars before the run cap)")
    return res


def collector_threshold(T, p):
    log("\n## Outcome-collector flag counts for candidate thresholds (paper: 527 of 1,980 kept)")
    res = {}
    for th in (0.4, 0.45, 0.5, 0.55):
        res[str(th)] = int((p > th).sum())
    b, y = S.arrays(T)
    k = S.admit(p, S.weights(y, 0.05) * b, 0.15)
    res["n_admitted_at_15pct_budget"] = int((k > 0).sum())
    log(res)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    tau = load_taubench()
    trail = load_trail()
    ww = load_whowhen()

    R["checks"] = dict(
        tau_runs=len(tau), tau_failed=int(sum(t["failed"] for t in tau)), tau_spans=sum(len(t["spans"]) for t in tau),
        ww_runs=len(ww), ww_spans=sum(len(t["spans"]) for t in ww),
        ww_decisive_error_strict=pct(np.mean([t["spans"][t["decisive"]]["status"] == "ERROR" for t in ww])),
        ww_decisive_error_loose=pct(np.mean([t["spans"][t["decisive"]]["status"] == "ERROR" for t in load_whowhen(True)])),
        trail_runs=len(trail), trail_failed=int(sum(t["failed"] for t in trail)),
        trail_decisive_error=pct(np.mean([t["spans"][t["decisive"]]["status"] == "ERROR" for t in trail if t["failed"]])),
        trail_error_spans=sum(s["status"] == "ERROR" for t in trail for s in t["spans"]),
        trail_median_trace_kb=round(np.median([sum(s["bytes"] for s in t["spans"]) for t in trail]) / 1e3, 1),
    )
    log("## Data checks", json.dumps(R["checks"], indent=1))

    R["tau"], Ptau, btau, ytau = trace_level("tau", tau)
    R["trail"], Ptr, btr, ytr = trace_level("trail", trail)
    R["trail_by_benchmark"] = trail_by_benchmark(trail, Ptr, btr, ytr)
    R["tau_transfer"] = tau_transfer(tau, btau, ytau)

    log("\n## Table 4 (Who&When fixed span policies): bytes (mean per trace), recall, Alg, Hand")
    R["table4"] = {}
    for k, f in V.FIXED.items():
        rec = lambda g: pct(np.mean([t["decisive"] in f(t) for t in g]))
        A = [t for t in ww if t["meta"]["system"] == "Alg"]
        H = [t for t in ww if t["meta"]["system"] == "Hand"]
        R["table4"][k] = [pct(np.mean([V.byte_share(t, f(t)) for t in ww])), rec(ww), rec(A), rec(H)]
        log(f"{k:20s}", R["table4"][k])
    R["fig4"] = {"ww": V.budget_curve(ww), "trail": V.budget_curve(trail)}
    R["fig4"] = {d: {k: [pct(x) for x in v] for k, v in c.items()} for d, c in R["fig4"].items()}
    log("\n## Figure 4 (10/20/30/50% per-trace budget)", json.dumps(R["fig4"]))

    R["diagnosis"] = diagnosis_chance(ww)
    R["judge_truncation"] = judge_truncation(trail)
    R["collector_threshold"] = collector_threshold(tau, Ptau["Metadata classifier"])

    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()
