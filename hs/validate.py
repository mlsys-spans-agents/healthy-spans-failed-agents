"""Check the paper's numeric claims: recompute what the data allows, and check the internal arithmetic
of what it doesn't. Writes results/validation.json and prints a claim-by-claim table."""
import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr, binomtest
from statsmodels.stats.proportion import proportion_confint

from . import sampling as S
from . import spans as V
from .data import load_taubench, load_trail, load_whowhen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []


def claim(sec, what, paper, ours, tol=0.15, note=""):
    if isinstance(paper, (int, float)) and isinstance(ours, (int, float)):
        ok = abs(paper - ours) <= tol
        status = "OK" if ok else "DIFF"
    else:
        status = "INFO"
    rows.append(dict(sec=sec, what=what, paper=paper, ours=ours, status=status, note=note))


def pct(x):
    return round(100 * float(x), 1)


def otlp_sizes(T):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status
    out = []
    for n, t in enumerate(T):
        req = ExportTraceServiceRequest()
        ss = ScopeSpans()
        for i, s in enumerate(t["spans"]):
            sp = Span(trace_id=n.to_bytes(16, "big"), span_id=(i + 1).to_bytes(8, "big"), name=s["kind"],
                      start_time_unix_nano=i * 10 ** 9, end_time_unix_nano=i * 10 ** 9 + 1,
                      status=Status(code=2 if s["status"] == "ERROR" else 1))
            for k, v in [("gen_ai.operation.name", s["kind"]), ("gen_ai.agent.name", s["agent"]),
                         ("gen_ai.tool.name", s["tool"]), ("gen_ai.content", s["text"])]:
                sp.attributes.append(KeyValue(key=k, value=AnyValue(string_value=v)))
            sp.attributes.append(KeyValue(key="gen_ai.usage.tokens", value=AnyValue(int_value=s["tokens"])))
            ss.spans.append(sp)
        req.resource_spans.append(ResourceSpans(scope_spans=[ss]))
        out.append(req.ByteSize())
    return np.array(out, float)


def main():
    tau, trail, ww = load_taubench(), load_trail(), load_whowhen()
    btau, ytau = S.arrays(tau)
    btr, ytr = S.arrays(trail)

    # ---------------- Section 4 / Table 1
    claim("4.1", "tau-bench runs / failed", "1,980 / 797", f"{len(tau)} / {ytau.sum()}")
    claim("4.1", "tau-bench failure share %", 40.3, pct(ytau.mean()), 0.05)
    claim("4.2", "spans tau / WW", "54,612 / 4,092", f"{sum(len(t['spans']) for t in tau)} / {sum(len(t['spans']) for t in ww)}")
    he = lambda t: any(s["status"] == "ERROR" for s in t["spans"])
    claim("T1", "tau all traces with ERROR: failed %", 33.9, pct(np.mean([he(t) for t in tau if t["failed"]])))
    claim("T1", "tau all traces with ERROR: OK %", 30.3, pct(np.mean([he(t) for t in tau if not t["failed"]])))
    claim("T1", "WW decisive ERROR strict %", 9.8, pct(np.mean([t["spans"][t["decisive"]]["status"] == "ERROR" for t in ww])))
    wl = load_whowhen(loose=True)
    claim("T1", "WW decisive ERROR loose %", 17.9, pct(np.mean([t["spans"][t["decisive"]]["status"] == "ERROR" for t in wl])),
          note="depends on the exact loose regex, which the paper does not give")
    F = [t for t in trail if t["failed"]]
    claim("T1", "TRAIL decisive ERROR %", 8.2, pct(np.mean([t["spans"][t["decisive"]]["status"] == "ERROR" for t in F])))
    claim("1", "TRAIL median trace KB", 386, round(np.median(btr) / 1e3, 1), 3,
          note="byte accounting (JSON of each OTLP span) differs slightly")

    # ---------------- Section 5
    claim("5", "AUC ERROR presence tau", 0.518, round(S.auc(S.p_err(tau), ytau), 3), 0.002)
    claim("5", "AUC ERROR presence TRAIL", 0.509, round(S.auc(S.p_err(trail), ytr), 3), 0.002)
    claim("5", "AUC latency TRAIL", 0.475, round(S.auc(S.p_latency(trail), ytr), 3), 0.002)
    claim("5", "AUC token count tau", 0.594, round(S.auc(S.p_cost(tau), ytau), 3), 0.005)
    claim("5", "TRAIL ERROR spans total", 437, sum(s["status"] == "ERROR" for t in trail for s in t["spans"]), 0)
    after = np.array([len(t["spans"]) - 1 - t["decisive"] for t in ww])
    rel = np.array([t["decisive"] / len(t["spans"]) for t in ww])
    claim("5", "WW median spans after decisive", 5, float(np.median(after)), 0)
    claim("5", "WW median relative position (paper: one third)", 0.33, round(float(np.median(rel)), 2), 0.05)
    claim("5", "WW share with >=4 spans after %", 71.2, pct((after >= 4).mean()))
    claim("5", "WW share with ERROR after decisive %", 35.3,
          pct(np.mean([any(s["status"] == "ERROR" for s in t["spans"][t["decisive"] + 1:]) for t in ww])))
    claim("5", "TRAIL median spans after decisive", 4, float(np.median([len(t["spans"]) - 1 - t["decisive"] for t in F])), 0)
    bins = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 19), (20, 39), (40, 10 ** 9)]
    claim("F3", "Figure 3 histogram (0,1,2,3,4-5,6-8,9-19,20-39,40+)", "bars ~8,18,7,20,49,36,17,8,20",
          str([int(((after >= a) & (after <= b)).sum()) for a, b in bins]))

    # ---------------- Section 6
    err = S.p_err(trail).astype(bool)
    w = S.weights(ytr, 0.05)
    claim("6", "TRAIL median size with/without ERROR (MB)", "1.37 / 0.131",
          f"{np.median(btr[err]) / 1e6:.2f} / {np.median(btr[~err]) / 1e6:.3f}")
    claim("6", "TRAIL weighted bytes held by ERROR traces %", 86, pct((w * btr)[err].sum() / (w * btr).sum()), 2)
    bench = np.array([t["meta"]["bench"] for t in trail])
    claim("6", "TRAIL median SWE / GAIA (MB)", "1.56 / 0.21",
          f"{np.median(btr[bench == 'SWE']) / 1e6:.2f} / {np.median(btr[bench == 'GAIA']) / 1e6:.2f}")
    for bm, ptxt in [("GAIA", "1.15 / 0.10"), ("SWE", "1.93 / 0.83")]:
        m = bench == bm
        claim("6", f"{bm}: median size with/without ERROR (MB)", ptxt,
              f"{np.median(btr[m & err]) / 1e6:.2f} / {np.median(btr[m & ~err]) / 1e6:.2f}")
    for bm, pv in [("GAIA", 8.9), ("SWE", 13.2)]:
        m = bench == bm
        sub = [t for t, k in zip(trail, m) if k]
        claim("6", f"TAIL-ERR recall within {bm} %", pv, pct(S.recall_at(S.p_err(sub), sub)))
    g = bench == "GAIA"
    claim("6", "GAIA: ERROR share failed / others %", "44.8 / 28.6",
          f"{pct(err[g & ytr].mean())} / {pct(err[g & ~ytr].mean())}")
    claim("6", "non-failed runs GAIA / SWE; SWE failures", "21 / 17; 14",
          f"{(g & ~ytr).sum()} / {(~g & ~ytr).sum()}; {(~g & ytr).sum()}")
    for pi, pv in [(0.01, 14.9), (0.10, 14.7)]:
        claim("6", f"TAIL-ERR tau recall at pi={pi}", pv, pct(S.recall_at(S.p_err(tau), b=btau, y=ytau, pi=pi)), 0.15)
    cost = S.p_cost(tau)
    claim("6", "TAIL-COST tau ERE at 5% budget", 1.66, round(S.recall_at(cost, b=btau, y=ytau, budget=.05) / .05, 2), 0.05,
          note="token definition not fully specified")
    lat = S.p_latency(trail)
    eres = [S.recall_at(lat, b=btr, y=ytr, budget=B) / B for B in (0.05, 0.10, 0.15, 0.20, 0.30)]
    claim("6", "TAIL-LATENCY ERE range over budgets", "0.55-1.55", f"{min(eres):.2f}-{max(eres):.2f}")
    claim("T2", "TAIL-ERR tau recall", 14.8, pct(S.recall_at(S.p_err(tau), b=btau, y=ytau)))
    claim("T2", "TAIL-ERR TRAIL recall", 8.5, pct(S.recall_at(S.p_err(trail), b=btr, y=ytr)))
    claim("T2", "TAIL-ERR+COST tau recall", 15.6, pct(S.recall_at(S.p_err_cost(tau), b=btau, y=ytau)), 0.5)
    claim("T2", "TAIL-LATENCY TRAIL recall", 15.5, pct(S.recall_at(lat, b=btr, y=ytr)))
    ng_t, ng_r = S.p_ngram(tau), S.p_ngram(trail)
    claim("T2", "n-gram tau recall", 25.2, pct(S.recall_at(ng_t, b=btau, y=ytau)))
    claim("T2", "n-gram TRAIL recall", 21.8, pct(S.recall_at(ng_r, b=btr, y=ytr)), 2)
    claim("6", "n-gram AUC tau / TRAIL (paper order: n-gram, IF)", "0.563 / 0.452",
          f"{S.auc(ng_t, ytau):.3f} / {S.auc(ng_r, ytr):.3f}")
    H = S.heuristics(trail)
    lr = H["Long run (top quarter by steps)"].astype(bool)
    claim("T2", "TRAIL long-run rule fires on OK vs failed %", "≈2x",
          f"{pct(lr[~ytr].mean())} / {pct(lr[ytr].mean())}")
    jr = S.recall_at(None, b=btau, y=ytau, split=np.where(ytau, 0.32, 0.11))
    claim("T2", "Judge measured TPR .32/FPR .11 applied to all runs", 34.2, pct(jr), 0.5,
          note="paper used unrounded TPR/FPR")
    for (tp, fp), pv in [((.6, .05), 63.0), ((.8, .1), 80.2), ((.95, .05), 95.3)]:
        claim("T2", f"parametric judge {tp}/{fp}", pv, pct(S.recall_at(None, b=btau, y=ytau, split=np.where(ytau, tp, fp))))
    claim("T2", "oracle ERE at 15%", 6.67, round(1 / 0.15, 2), 0.005)
    claim("6", "TRAIL weighted bytes of the 38 non-failed runs %", "most", pct((w * btr)[~ytr].sum() / (w * btr).sum()))

    # ---------------- Section 7 / Table 3 internal arithmetic
    lo, hi = proportion_confint(64, 200, method="wilson")
    claim("T3", "TPR 0.32 (n=200) Wilson 95% CI", "0.26-0.39", f"{lo:.2f}-{hi:.2f}")
    lo, hi = proportion_confint(22, 200, method="wilson")
    claim("T3", "FPR 0.11 (n=200) Wilson 95% CI", "0.07-0.16", f"{lo:.2f}-{hi:.2f}")
    claim("T3", "TPR = mean(airline .43, retail .21)", 0.32, (0.43 + 0.21) / 2, 0.005)
    claim("T3", "FPR = mean(airline .15, retail .07)", 0.11, (0.15 + 0.07) / 2, 0.005)
    claim("6", "judge agreement on balanced sample = (TPR+1-FPR)/2 %", 60.5, pct((0.32 + 0.89) / 2), 0.05)
    claim("T3", "cost per 1,000 runs at $1/MTok input (Haiku 4.5)", 4, round(3850 * 1000 / 1e6 * 1.0, 2), 0.2)
    claim("7", "WW judge flag rate from per-system rates %", 72.8, pct((0.675 * 126 + 0.845 * 58) / 184), 0.1)

    # ---------------- Table 4 / Figure 4
    for k, f in V.FIXED.items():
        claim("T4", f"{k}: recall", None, pct(np.mean([t["decisive"] in f(t) for t in ww])))
    rows[-8:] = [dict(r, paper=pv, status="OK" if abs(pv - r["ours"]) < 0.15 else "DIFF") for r, pv in
                 zip(rows[-8:], [9.8, 25.5, 31.5, 28.8, 66.8, 19.6, 73.4, 92.9])]
    fig = {"ww": V.budget_curve(ww), "trail": V.budget_curve(trail)}
    claim("8", "WW error-anchored at 50%", 25.0, pct(fig["ww"]["Error-anchored"][-1]))
    claim("8", "TRAIL error-anchored at 50%", 26.4, pct(fig["trail"]["Error-anchored"][-1]))
    claim("8", "WW keep-the-start at 50%", 54.3, pct(fig["ww"]["Keep the start"][-1]))
    claim("8", "TRAIL anomalous at 50%", 53.6, pct(fig["trail"]["Anomalous spans"][-1]), 1.0,
          note="anomaly score details not fully specified")
    claim("8", "Who beats random at 50%: WW / TRAIL", "start / anomalous",
          f"{max((v[-1], k) for k, v in fig['ww'].items() if k != 'Random')[1]} / "
          f"{max((v[-1], k) for k, v in fig['trail'].items() if k != 'Random')[1]} "
          f"(random {pct(fig['ww']['Random'][-1])} / {pct(fig['trail']['Random'][-1])})")

    # ---------------- Section 9 internal consistency (Tables 5 and 6)
    t6 = {"ERROR+8": (121, 40.5, 7.4, 55, 38.2, 56.4), "Last 8": (60, 28.3, 6.7, 116, 45.7, 45.7),
          "First 30%": (108, 39.8, 20.4, 68, 39.7, 47.1)}
    t5 = {"Full": 39.8, "Last 8": 32.4, "First 30%": 30.7, "ERROR+8": 22.7}
    kept5 = {"Last 8": 65.9, "First 30%": 38.6, "ERROR+8": 31.3}
    for v, (nd, fd, vd, nk, fk, vk) in t6.items():
        claim("T5/T6", f"{v}: n dropped + kept = 176", 176, nd + nk, 0)
        claim("T5/T6", f"{v}: kept share = Table 5 Kept", kept5[v], pct(nk / 176))
        full = round(nd * fd / 100) + round(nk * fk / 100)
        view = round(nd * vd / 100) + round(nk * vk / 100)
        claim("T5/T6", f"{v}: full-trace step accuracy from Table 6", t5["Full"], pct(full / 176))
        claim("T5/T6", f"{v}: view step accuracy from Table 6", t5[v], pct(view / 176))
    claim("9", "Bonferroni threshold for 3 tests vs p=.029", "not significant", "0.0167 < 0.029")

    # ---------------- Section 10
    d = np.array([t["duration"] for t in trail])
    claim("10", "TRAIL run length median / p90 (s)", "131 / 1,052", f"{np.median(d):.0f} / {np.percentile(d, 90):.0f}")
    for W, pv in [(60, 10.1), (120, 45.9), (300, 79.7), (600, 89.9)]:
        claim("10", f"predicted kept (run+5s < {W}s) %", pv, pct(np.mean(d + 5 < W)))
    claim("10", "ERROR collector kept (traces with an ERROR span)", 628, int(S.p_err(tau).sum()), 0)
    sd = math.sqrt(1980 * 0.1 * 0.9)
    claim("10", "probabilistic 186 within 1 SD of 198", "yes", f"|186-198|=12 < SD {sd:.1f}")
    claim("10", "memory ratio 60 s ((484-162)/(311-162))", 2.2, round((484 - 162) / (311 - 162), 2), 0.05)
    claim("10", "memory ratio 300 s ((2292-162)/(814-162))", 3.3, round((2292 - 162) / (814 - 162), 2), 0.05)
    claim("10", "600 s at 3/s extrapolated total GB", 4.4, round((2 * (2292 - 162) + 162) / 1000, 2), 0.05)
    claim("10", "MB per held trace (2292-162)/984", "1.1-2.2", round((2292 - 162) / 984, 2))
    claim("10", "TRAIL mean MB per trace", 1.14, round(btr.mean() / 1e6, 2), 0.05)
    attr = sum(s["bytes"] - len(s["text"].encode()) for t in tau for s in t["spans"])
    claim("10", "metadata share of tau span bytes %", 42.6, pct(attr / btau.sum()), 1.0,
          note="depends on attribute encoding")

    # ---------------- Section 4.2 fidelity: OTLP protobuf sizes
    for name, T, b in [("tau", tau, btau), ("WW", ww, S.arrays(ww)[0])]:
        pb = otlp_sizes(T)
        claim("4.2", f"{name}: Spearman(protobuf, our bytes)", 0.9999, round(spearmanr(pb, b).statistic, 4), 0.0002)
        claim("4.2", f"{name}: median protobuf/our ratio", 0.97 if name == "tau" else 0.99, round(np.median(pb / b), 2), 0.03)
        if name == "tau":
            diffs = []
            for B in (0.05, 0.10, 0.15, 0.20, 0.30):
                for sc in (S.p_err(tau), S.p_cost(tau), S.p_err_cost(tau), ng_t, S.p_size(tau)):
                    diffs.append(abs(S.recall_at(sc, b=b, y=ytau, budget=B) - S.recall_at(sc, b=pb, y=ytau, budget=B)))
            claim("4.2", "max recall change with protobuf sizes (pp)", 0.27, round(100 * max(diffs), 2), 0.3,
                  note="our policy set; SIZE-ONLY included")

    json.dump(rows, open(os.path.join(ROOT, "results", "validation.json"), "w"), indent=1, default=str)
    for r in rows:
        print(f"{r['status']:4s} [{r['sec']}] {r['what']}: paper={r['paper']} ours={r['ours']}"
              + (f"  ({r['note']})" if r["note"] else ""))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
