"""Trace-level policies and the equal-budget evaluation."""
import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import TAU_WRITE_TOOLS


# ---------------------------------------------------------------- basic arrays
def arrays(traces):
    b = np.array([sum(s["bytes"] for s in t["spans"]) for t in traces], float)
    y = np.array([t["failed"] for t in traces], bool)
    return b, y


def weights(y, pi):
    """Reweight so failed runs make up a share pi of the traffic."""
    pf = y.mean()
    return np.where(y, pi / pf, (1 - pi) / (1 - pf))


# ---------------------------------------------------------------- evaluation
def admit(score, cost, budget):
    """Admit items in descending score until weighted cost reaches budget (a fraction of total).
    The tie group at the boundary is admitted fractionally. Returns keep fractions in [0,1]."""
    score = np.asarray(score, float)
    keep = np.zeros(len(score))
    left = budget * cost.sum()
    for v in np.unique(score)[::-1]:
        g = score == v
        c = cost[g].sum()
        if c <= left + 1e-12:
            keep[g] = 1.0
            left -= c
        else:
            keep[g] = left / c if c > 0 else 0.0
            break
    return keep


def recall_at(score, traces=None, b=None, y=None, pi=0.05, budget=0.15, by="bytes", split=None):
    """Failure recall at an equal byte (or trace-count) budget.
    split: optional (p_flag) array for a parametric judge — each trace is split into a flagged part
    (prob p_flag, score 1) and an unflagged part (score 0); the result is the exact expectation."""
    if b is None:
        b, y = arrays(traces)
    w = weights(y, pi)
    unit = b if by == "bytes" else np.ones_like(b)
    if split is not None:
        p = np.asarray(split, float)
        score2 = np.concatenate([np.ones(len(p)), np.zeros(len(p))])
        cost = np.concatenate([w * unit * p, w * unit * (1 - p)])
        keep = admit(score2, cost, budget)
        k = keep[: len(p)] * p + keep[len(p):] * (1 - p)
    else:
        k = admit(score, w * unit, budget)
    return (k * w)[y].sum() / w[y].sum()


def bootstrap_ci(score, b, y, n=1000, seed=0, **kw):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(b), len(b))
        if y[i].all() or not y[i].any():
            continue
        out.append(recall_at(np.asarray(score)[i], b=b[i], y=y[i], **kw))
    return np.percentile(out, [2.5, 97.5])


# ---------------------------------------------------------------- simple policies
def has_error(t):
    return float(any(s["status"] == "ERROR" for s in t["spans"]))


def tokens(t):
    return float(sum(s["tokens"] for s in t["spans"]))


def p_head(T):
    return np.zeros(len(T))


def p_err(T):
    return np.array([has_error(t) for t in T])


def p_cost(T):
    return np.array([tokens(t) for t in T])


def p_err_cost(T):
    tok = p_cost(T)
    top = tok >= np.quantile(tok, 0.9)
    return np.maximum(p_err(T), top.astype(float))


def p_latency(T):
    return np.array([t.get("duration", 0.0) for t in T])


def p_size(T):
    b, _ = arrays(T)
    return -b


# ---------------------------------------------------------------- span tokens for sequence models
def sym(s):
    return f"{s['kind']}:{s['tool']}" if s["tool"] else s["kind"]


def p_ngram(T, k=0.1):
    """Sifter-style: smoothed trigram over span kinds/tool names; score = mean NLL."""
    tri, bi = Counter(), Counter()
    vocab = set()
    seqs = [["<s>", "<s>"] + [sym(s) for s in t["spans"]] + ["</s>"] for t in T]
    for q in seqs:
        vocab.update(q)
        for i in range(2, len(q)):
            tri[tuple(q[i - 2:i + 1])] += 1
            bi[tuple(q[i - 2:i])] += 1
    V = len(vocab)
    out = []
    for q in seqs:
        nll = [-math.log((tri[tuple(q[i - 2:i + 1])] + k) / (bi[tuple(q[i - 2:i])] + k * V))
               for i in range(2, len(q))]
        out.append(np.mean(nll))
    return np.array(out)


# ---------------------------------------------------------------- tool calls
def calls(t):
    """(tool, args) pairs for every tool call in the trace."""
    out = []
    for s in t["spans"]:
        if "calls" in s:
            out += s["calls"]
        elif t["dataset"] == "trail" and s["kind"] == "tool":
            out.append((s["tool"], hashlib.md5(s.get("io", "").split("\n")[0].encode()).hexdigest()))
        elif t["dataset"] == "ww" and s["kind"] == "tool":
            out.append((s["tool"], ""))
    return out


def repeated_call(t):
    c = Counter(calls(t))
    return float(any(v > 1 for v in c.values()))


# ---------------------------------------------------------------- heuristics
def h_escalated(t):
    return float(any(n == "transfer_to_human_agents" for n, _ in calls(t)))


def h_no_write(t):
    return float(not any(n in TAU_WRITE_TOOLS for n, _ in calls(t)))


def h_no_final(t):
    return float(not any("final_answer" in (s["tool"] or "") for s in t["spans"]))


def h_long(T):
    n = np.array([len(t["spans"]) for t in T])
    return (n >= np.quantile(n, 0.75)).astype(float)


def heuristics(T):
    ds = T[0]["dataset"]
    H = {"Repeated identical tool call": np.array([repeated_call(t) for t in T])}
    if ds == "tau":
        H["Escalated to a human"] = np.array([h_escalated(t) for t in T])
        H["No state-changing action"] = np.array([h_no_write(t) for t in T])
    else:
        H["No final answer"] = np.array([h_no_final(t) for t in T])
        H["Long run (top quarter by steps)"] = h_long(T)
    H["Any heuristic"] = np.max(np.vstack(list(H.values())), axis=0)
    return H


# ---------------------------------------------------------------- features
def meta_features(t):
    """Features a collector can compute without parsing payload content:
    span counts and token counts by kind, error counts by kind, tool-name counts,
    trace duration (when recorded) and a hash-equality flag for repeated tool calls."""
    f = Counter()
    sp = t["spans"]
    f["n_spans"] = len(sp)
    for s in sp:
        f["kind=" + s["kind"]] += 1
        f["tok_kind=" + s["kind"]] += len(s["text"]) // 4
        if s["tool"]:
            f["tool=" + s["tool"]] += 1
        if s["status"] == "ERROR":
            f["err_kind=" + s["kind"]] += 1
    f["n_error"] = sum(s["status"] == "ERROR" for s in sp)
    f["tokens"] = sum(len(s["text"]) // 4 for s in sp)
    if "duration" in t:
        f["duration"] = t["duration"]
    c = Counter(calls(t))
    f["repeated"] = float(any(v > 1 for v in c.values()))
    f["n_repeated"] = sum(v - 1 for v in c.values() if v > 1)
    return {k: math.log1p(v) for k, v in f.items()}


def classifier_scores(T, groups=None, n_splits=5, seed=0, labels=None, train_idx=None, test_idx=None):
    """Out-of-fold failure probabilities from a logistic regression on metadata features."""
    X = DictVectorizer(sparse=False).fit_transform([meta_features(t) for t in T])
    y = np.array([t["failed"] for t in T]) if labels is None else np.asarray(labels)
    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    if train_idx is not None:
        m = model().fit(X[train_idx], y[train_idx])
        return m.predict_proba(X[test_idx])[:, 1]
    groups = np.array([t["group"] for t in T]) if groups is None else groups
    if seed:  # shuffle group -> fold assignment
        rng = np.random.default_rng(seed)
        ug = np.unique(groups)
        perm = dict(zip(ug, rng.permutation(len(ug))))
        groups = np.array([perm[g] for g in groups])
    p = np.zeros(len(T))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        p[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def iforest_scores(T, seed=0):
    rows = []
    for t in T:
        f = {k: v for k, v in meta_features(t).items() if not k.startswith("tool=")}
        seq = [sym(s) for s in t["spans"] if s["kind"] == "tool"]
        for a, b2 in zip(seq, seq[1:]):
            f[f"bi={a}>{b2}"] = f.get(f"bi={a}>{b2}", 0) + 1
        for s in t["spans"]:
            if s["agent"]:
                f["agent=" + s["agent"]] = 1
        rows.append(f)
    X = DictVectorizer(sparse=False).fit_transform(rows)
    text = [" ".join((s.get("io") or s["text"])[:2000] for s in t["spans"])[:200000] for t in T]
    tf = TfidfVectorizer(max_features=20000, sublinear_tf=True).fit_transform(text)
    emb = TruncatedSVD(32, random_state=seed).fit_transform(tf)
    X = np.hstack([StandardScaler().fit_transform(X), StandardScaler().fit_transform(emb)])
    m = IsolationForest(n_estimators=300, random_state=seed).fit(X)
    return -m.score_samples(X)


def auc(score, y):
    return roc_auc_score(y, score)
