"""Span-level trimming policies and the diagnosis-study views."""
import numpy as np


# ---------------------------------------------------------------- fixed views (Table 4 / Section 9)
def v_full(t):
    return list(range(len(t["spans"])))


def v_error_plus(t, k):
    keep = set()
    for i, s in enumerate(t["spans"]):
        if s["status"] == "ERROR":
            keep.update(range(max(0, i - k), i + 1))
    return sorted(keep)


def v_last(t, k):
    n = len(t["spans"])
    return list(range(max(0, n - k), n))


def v_first_bytes(t, frac):
    """Spans from the start while cumulative bytes fit in frac of the trace's bytes."""
    b = [s["bytes"] for s in t["spans"]]
    cap, used, keep = frac * sum(b), 0, []
    for i, x in enumerate(b):
        if used + x > cap:
            break
        keep.append(i)
        used += x
    return keep


def v_kind(t, kinds):
    return [i for i, s in enumerate(t["spans"]) if s["kind"] in kinds]


FIXED = {
    "ERROR spans only": lambda t: v_error_plus(t, 0),
    "ERROR + 1 before": lambda t: v_error_plus(t, 1),
    "ERROR + 8 before": lambda t: v_error_plus(t, 8),
    "Last 4 spans": lambda t: v_last(t, 4),
    "Last 8 spans": lambda t: v_last(t, 8),
    "Tool spans only": lambda t: v_kind(t, {"tool"}),
    "Agent spans only": lambda t: v_kind(t, {"agent"}),
    "Drop reasoning": lambda t: v_kind(t, {"input", "agent", "tool"}),
}

DIAG_VIEWS = {
    "Full": v_full,
    "Last 8": lambda t: v_last(t, 8),
    "First 30%": lambda t: v_first_bytes(t, 0.30),
    "ERROR+8": lambda t: v_error_plus(t, 8),
}


def byte_share(t, idx):
    b = np.array([s["bytes"] for s in t["spans"]], float)
    return b[idx].sum() / b.sum() if len(idx) else 0.0


# ---------------------------------------------------------------- budgeted trimming (Figure 4)
def order_error_anchored(t, rng):
    sp = t["spans"]
    errs = [i for i, s in enumerate(sp) if s["status"] == "ERROR"]
    out = list(errs)
    for d in range(1, len(sp)):  # nearest-first before each error
        for e in errs:
            if e - d >= 0:
                out.append(e - d)
    seen, res = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            res.append(i)
    return res  # nothing else


def order_end(t, rng):
    return list(range(len(t["spans"])))[::-1]


def order_start(t, rng):
    return list(range(len(t["spans"])))


def order_random(t, rng):
    return list(rng.permutation(len(t["spans"])))


def order_anomalous(t, rng, stats):
    sp = t["spans"]
    errs = [i for i, s in enumerate(sp) if s["status"] == "ERROR"]

    def z(s):
        mu, sd = stats[s["kind"]]
        v = [np.log1p(s["bytes"])] + ([np.log1p(s["dur"])] if s.get("dur") is not None else [])
        return max(abs(a - m) / (d + 1e-9) for a, m, d in zip(v, mu, sd))

    rest = sorted((i for i in range(len(sp)) if i not in set(errs)), key=lambda i: -z(sp[i]))
    return errs + rest


def kind_stats(traces):
    from collections import defaultdict
    acc = defaultdict(list)
    for t in traces:
        for s in t["spans"]:
            acc[s["kind"]].append([np.log1p(s["bytes"])] + ([np.log1p(s["dur"])] if s.get("dur") is not None else []))
    return {k: (np.mean(v, 0), np.std(v, 0)) for k, v in acc.items()}


def trim(t, order, budget):
    """Admit spans in order while they fit in budget * trace bytes; input spans always kept."""
    sp = t["spans"]
    cap = budget * sum(s["bytes"] for s in sp)
    keep = {i for i, s in enumerate(sp) if s["kind"] == "input"}
    used = sum(sp[i]["bytes"] for i in keep)
    for i in order:
        if i in keep:
            continue
        if used + sp[i]["bytes"] <= cap:
            keep.add(i)
            used += sp[i]["bytes"]
    return keep


def budget_curve(traces, budgets=(0.1, 0.2, 0.3, 0.5), n_random=50, seed=0):
    F = [t for t in traces if t["failed"] and t["decisive"] is not None]
    stats = kind_stats(traces)
    rng = np.random.default_rng(seed)
    pol = {
        "Error-anchored": order_error_anchored, "Keep the end": order_end,
        "Keep the start": order_start, "Anomalous spans": lambda t, r: order_anomalous(t, r, stats),
    }
    out = {}
    for name, fn in pol.items():
        out[name] = [np.mean([t["decisive"] in trim(t, fn(t, rng), B) for t in F]) for B in budgets]
    out["Random"] = [np.mean([np.mean([t["decisive"] in trim(t, order_random(t, rng), B) for _ in range(n_random)])
                              for t in F]) for B in budgets]
    return out
