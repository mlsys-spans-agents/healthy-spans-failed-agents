"""Load the three datasets and convert them to span lists.

Every trace is a dict:
  id, dataset, failed (bool), group (for grouped CV), meta (model/domain/system),
  spans: list of dicts with kind, agent, tool, status, bytes, tokens, text, start, dur
  decisive: index of the decisive span (None if unlabeled)
"""
import json
import os
import re

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

TAU_FILES = {
    "gpt-4o-airline": ("GPT-4o", "airline"),
    "gpt-4o-retail": ("GPT-4o", "retail"),
    "sonnet-35-new-airline": ("Claude 3.5 Sonnet", "airline"),
    "sonnet-35-new-retail": ("Claude 3.5 Sonnet", "retail"),
}

# tau-bench tools that change the database
TAU_WRITE_TOOLS = {
    "book_reservation", "cancel_reservation", "update_reservation_baggages",
    "update_reservation_flights", "update_reservation_passengers", "send_certificate",
    "cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
}


def _span(kind, text, agent="", tool="", status="OK", start=None, dur=None, extra=None):
    text = text if isinstance(text, str) else ("" if text is None else json.dumps(text))
    attrs = {
        "gen_ai.operation.name": {"input": "input", "agent": "chat", "reasoning": "chat",
                                  "tool": "execute_tool"}[kind],
        "gen_ai.agent.name": agent,
        "gen_ai.tool.name": tool,
        "gen_ai.usage.tokens": len(text) // 4,
        "span.kind": kind,
        "status": status,
    }
    nbytes = len(json.dumps(attrs).encode()) + len(text.encode("utf-8"))
    s = dict(kind=kind, agent=agent, tool=tool, status=status, bytes=nbytes,
             tokens=len(text) // 4, text=text, start=start, dur=dur)
    if extra:
        s.update(extra)
    return s


def _args(a):
    """Canonical string form of tool-call arguments (some files store dicts, some JSON strings)."""
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return a
    return json.dumps(a, sort_keys=True)


def load_taubench():
    traces = []
    for fname, (model, domain) in TAU_FILES.items():
        runs = json.load(open(os.path.join(ROOT, "taubench", fname + ".json"), encoding="utf-8"))
        for i, r in enumerate(runs):
            spans = []
            for m in r["traj"]:
                role = m["role"]
                if role == "system":
                    continue
                if role == "user":
                    spans.append(_span("input", m.get("content"), agent="user"))
                elif role == "assistant":
                    calls = m.get("tool_calls") or []
                    text = m.get("content") or ""
                    if calls:
                        text = (text + json.dumps([c["function"] for c in calls])).strip()
                    spans.append(_span("agent", text, agent="assistant",
                                       extra={"calls": [(c["function"]["name"], _args(c["function"]["arguments"]))
                                                        for c in calls]}))
                elif role == "tool":
                    c = m.get("content") or ""
                    st = "ERROR" if c.startswith("Error") else "OK"
                    spans.append(_span("tool", c, agent="assistant", tool=m.get("name", ""), status=st))
            traces.append(dict(
                id=f"{fname}/{i}", dataset="tau", failed=r["reward"] < 1 - 1e-9,
                group=f"{domain}/{r['task_id']}", meta=dict(model=model, domain=domain, file=fname),
                spans=spans, decisive=None))
    return traces


# Who&When status rules
_STRICT = re.compile(
    r"(Traceback \(most recent call last\)|exitcode: [1-9]\d*|exit code[: ]+[1-9]\d*"
    r"|^\s*\w*(Error|Exception)\b)", re.M)
_LOOSE = re.compile(r"(\berror\b|\bexception\b|\bfailed\b|\bfailure\b|unable to|could not|cannot)", re.I)

WW_TOOL_AGENTS = {"Computer_terminal", "WebSurfer", "FileSurfer", "ComputerTerminal"}


def _ww_kind(name, idx):
    if name == "human":
        return "input"
    if name in WW_TOOL_AGENTS:
        return "tool"
    if name.startswith("Orchestrator (thought)") or name.startswith("Orchestrator (termination"):
        return "reasoning"
    return "agent"


def load_whowhen(loose=False):
    traces = []
    for system, fname in [("Alg", "Algorithm-Generated"), ("Hand", "Hand-Crafted")]:
        df = pd.read_parquet(os.path.join(ROOT, "whowhen", fname + ".parquet"))
        for i, r in df.iterrows():
            spans = []
            for j, m in enumerate(r["history"]):
                name = m.get("name") if system == "Alg" else m.get("role")
                text = m.get("content") or ""
                kind = _ww_kind(name, j)
                rule = _LOOSE if loose else _STRICT
                st = "ERROR" if rule.search(text) else "OK"
                agent = name.split(" (")[0] if system == "Hand" else name
                spans.append(_span(kind, text, agent=agent, tool=name if kind == "tool" else "", status=st))
            step = int(r["mistake_step"])
            if step >= len(spans):  # out of range: last span of the responsible agent
                idx = [k for k, s in enumerate(spans) if s["agent"] == r["mistake_agent"]]
                step = idx[-1] if idx else len(spans) - 1
            traces.append(dict(
                id=f"{system}/{i}", dataset="ww", failed=True, group=f"{system}/{i}",
                meta=dict(system=system, mistake_agent=r["mistake_agent"]),
                spans=spans, decisive=step))
    return traces


def load_trail():
    """TRAIL: expects data/trail/*.parquet (HF PatronusAI/TRAIL) or the GitHub JSON layout."""
    from . import trail
    return trail.load(os.path.join(ROOT, "trail"))
