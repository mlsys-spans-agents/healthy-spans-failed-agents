"""TRAIL loader: real OpenTelemetry/OpenInference spans, ordered depth-first."""
import json
import os
import re
from datetime import datetime

import pandas as pd

_DUR = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?")
KIND = {"TOOL": "tool", "LLM": "reasoning", "AGENT": "agent", "CHAIN": "agent", None: "agent"}


def _dur(s):
    m = _DUR.fullmatch(s or "")
    if not m:
        return 0.0
    h, mi, se = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(se or 0)


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _loads(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:  # a few label files have trailing commas
        return json.loads(re.sub(r",(?=\s*[}\]])", "", s))


def _walk(s, out):
    out.append(s)
    for c in sorted(s.get("child_spans", []), key=lambda c: c["timestamp"]):
        _walk(c, out)
    return out


def load(root):
    traces = []
    for bench, fname in [("GAIA", "gaia.parquet"), ("SWE", "swe_bench.parquet")]:
        df = pd.read_parquet(os.path.join(root, fname))
        for _, row in df.iterrows():
            tr = json.loads(row["trace"])
            labels = _loads(row["labels"])
            raw = []
            for r in sorted(tr["spans"], key=lambda s: s["timestamp"]):
                _walk(r, raw)
            t0 = min(_ts(s["timestamp"]) for s in raw)
            spans = []
            for s in raw:
                body = {k: v for k, v in s.items() if k != "child_spans"}
                text = json.dumps(body)
                a = s.get("span_attributes", {})
                tool = a.get("tool.name", "") or (s["span_name"] if a.get("openinference.span.kind") == "TOOL" else "")
                spans.append(dict(
                    span_id=s["span_id"], name=s["span_name"],
                    kind=KIND.get(a.get("openinference.span.kind"), "agent"),
                    agent=s["span_name"].split(".")[0] if "Agent" in s["span_name"] else "",
                    tool=tool,
                    status="ERROR" if s["status_code"] == "Error" else "OK",
                    bytes=len(text.encode("utf-8")),
                    tokens=int(a.get("llm.token_count.total", 0) or 0),
                    text=text, start=_ts(s["timestamp"]) - t0, dur=_dur(s["duration"]),
                    io=str(a.get("input.value", ""))[:4000] + "\n" + str(a.get("output.value", ""))[:4000]))
            pos = {s["span_id"]: i for i, s in enumerate(spans)}
            errs = labels.get("errors", [])
            for e in errs:
                e["_idx"] = pos.get(e.get("location"))
            high = [e["_idx"] for e in errs if e.get("impact") == "HIGH" and e["_idx"] is not None]
            for i in {e["_idx"] for e in errs if e["_idx"] is not None}:
                spans[i]["annotated"] = True
            duration = max(s["start"] + s["dur"] for s in spans)
            traces.append(dict(
                id=f"{bench}/{tr['trace_id']}", dataset="trail", failed=len(high) > 0,
                group=f"{bench}/{tr['trace_id']}", meta=dict(bench=bench, n_errors=len(errs),
                                                              n_located=sum(e["_idx"] is not None for e in errs)),
                spans=spans, decisive=min(high) if high else None, duration=duration))
    return traces
