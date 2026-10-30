"""Replay harness for the stock OpenTelemetry Collector (otelcol-contrib in Docker).

Experiments (python -m hs.collector <exp>):
  equiv   tau-bench, 1,980 traces at 40/s into three collectors (ERROR, outcome, 10% probabilistic);
          compare kept sets with the simulator.
  window  TRAIL runs with real durations (time compressed 10x) and every run flagged; one collector per
          decision window (60/120/300/600 s at original scale); share of flagged runs kept.
  memory  TRAIL traces with real span payloads at 1 and 3 traces/s, 60 s and 300 s windows (real time);
          collector RSS, CPU, traces held and late spans.
Results go to results/collector_<exp>.json.
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import requests
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "collector")
IMAGE = "otel/opentelemetry-collector-contrib:latest"
TICK = 1.0  # collector decision tick (s)


# =========================================================================== sink
class Sink:
    """Receives exported spans from a collector: records (arrival time, trace id, n spans)."""

    def __init__(self, port):
        self.events = []
        lock = threading.Lock()
        sink = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                req = ExportTraceServiceRequest()
                req.ParseFromString(body)
                now = time.time()
                per = {}
                for rs in req.resource_spans:
                    for ss in rs.scope_spans:
                        for sp in ss.spans:
                            k = sp.trace_id.hex()
                            per.setdefault(k, []).append(sp.name)
                with lock:
                    for k, names in per.items():
                        sink.events.append((now, k, names))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.end_headers()

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def traces(self):
        return {k for _, k, _ in self.events}

    def close(self):
        self.server.shutdown()


# =========================================================================== collector
def start_collector(name, port, sink_port, decision_wait, policies, cpus=None, mem=None):
    d = os.path.join(WORK, name)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "receivers": {"otlp": {"protocols": {"http": {"endpoint": "0.0.0.0:4318"}}}},
        "processors": {"tail_sampling": {
            "decision_wait": f"{decision_wait}s", "num_traces": 200000,
            "decision_cache": {"sampled_cache_size": 200000, "non_sampled_cache_size": 200000},
            "policies": policies}},
        "exporters": {"otlphttp": {"endpoint": f"http://host.docker.internal:{sink_port}",
                                   "compression": "none", "encoding": "proto"}},
        "service": {
            "telemetry": {"metrics": {"level": "detailed", "readers": [
                {"pull": {"exporter": {"prometheus": {"host": "0.0.0.0", "port": 8888}}}}]}},
            "pipelines": {"traces": {"receivers": ["otlp"], "processors": ["tail_sampling"],
                                     "exporters": ["otlphttp"]}}},
    }
    with open(os.path.join(d, "config.yaml"), "w") as f:
        json.dump(cfg, f, indent=1)  # JSON is valid YAML
    subprocess.run(["docker", "rm", "-f", f"hs-{name}"], capture_output=True)
    cmd = ["docker", "run", "-d", "--name", f"hs-{name}", "-p", f"{port}:4318", "-p", f"{port + 1}:8888",
           "--add-host", "host.docker.internal:host-gateway",
           "-v", f"{os.path.join(d, 'config.yaml')}:/etc/otelcol-contrib/config.yaml"]
    if cpus:
        cmd += ["--cpus", str(cpus)]
    if mem:
        cmd += ["--memory", mem]
    cmd.append(IMAGE)
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    subprocess.run(cmd, check=True, capture_output=True, env=env)
    for _ in range(60):
        try:
            requests.get(f"http://localhost:{port + 1}/metrics", timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"collector {name} did not start")


def stop_collector(name):
    subprocess.run(["docker", "rm", "-f", f"hs-{name}"], capture_output=True)


def collector_version():
    out = subprocess.run(["docker", "run", "--rm", IMAGE, "--version"], capture_output=True, text=True).stdout
    return out.strip().split()[-1]


class Scraper:
    """Scrape the collector's own metrics every second."""

    KEYS = {"otelcol_process_memory_rss": "rss", "otelcol_process_cpu_seconds": "cpu",
            "otelcol_processor_tail_sampling_sampling_traces_on_memory": "held"}

    def __init__(self, port):
        self.rows, self.stop = [], False
        self.t = threading.Thread(target=self.run, args=(port,), daemon=True)
        self.t.start()

    def run(self, port):
        while not self.stop:
            t0 = time.time()
            try:
                txt = requests.get(f"http://localhost:{port + 1}/metrics", timeout=2).text
                row = {"t": t0}
                for line in txt.splitlines():
                    for k, v in self.KEYS.items():
                        if line.startswith(k + " ") or line.startswith(k + "{"):
                            row[v] = row.get(v, 0) + float(line.rsplit(" ", 1)[1])
                self.rows.append(row)
            except Exception:
                pass
            time.sleep(max(0, 1 - (time.time() - t0)))


# =========================================================================== OTLP encoding
def kv(k, v):
    if isinstance(v, bool):
        return KeyValue(key=k, value=AnyValue(bool_value=v))
    if isinstance(v, int):
        return KeyValue(key=k, value=AnyValue(int_value=v))
    if isinstance(v, float):
        return KeyValue(key=k, value=AnyValue(double_value=v))
    return KeyValue(key=k, value=AnyValue(string_value=v if isinstance(v, str) else json.dumps(v)))


def request(spans):
    r = ExportTraceServiceRequest()
    r.resource_spans.append(ResourceSpans(scope_spans=[ScopeSpans(spans=spans)]))
    return r.SerializeToString()


def mkspan(trace_id, span_id, name, t_ns, dur_ns=1, error=False, attrs=(), parent=b""):
    sp = Span(trace_id=trace_id, span_id=span_id, parent_span_id=parent, name=name,
              start_time_unix_nano=t_ns, end_time_unix_nano=t_ns + max(1, dur_ns),
              status=Status(code=2 if error else 1))
    sp.attributes.extend(kv(k, v) for k, v in attrs)
    return sp


def tid(*parts):
    return hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:16]


def sid(*parts):
    return hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8]


# =========================================================================== sending
class Sender:
    def __init__(self, workers=32):
        self.pool = ThreadPoolExecutor(workers)
        self.local = threading.local()
        self.errors = 0
        self.wire = 0
        self.sent = {}   # tag -> time the request was accepted by the collector
        self.lags = []   # accepted time minus scheduled time, per request

    def _post(self, port, body, sched=None, tag=None):
        s = getattr(self.local, "s", None) or requests.Session()
        self.local.s = s
        for _ in range(5):
            try:
                r = s.post(f"http://localhost:{port}/v1/traces", data=body,
                           headers={"Content-Type": "application/x-protobuf"}, timeout=30)
                if r.status_code == 200:
                    self.wire += len(body)
                    done = time.time()
                    if sched is not None:
                        self.lags.append(done - sched)
                    if tag is not None:
                        self.sent[tag] = done
                    return
            except Exception:
                pass
            time.sleep(0.2)
        self.errors += 1

    def run(self, schedule, t0):
        """schedule: list of (offset_s, port, body_fn[, tag]) sorted by offset."""
        futs = []
        for item in schedule:
            off, port, fn = item[:3]
            tag = item[3] if len(item) > 3 else None
            delay = t0 + off - time.time()
            if delay > 0:
                time.sleep(delay)
            futs.append(self.pool.submit(lambda p=port, f=fn, a=t0 + off, g=tag: self._post(p, f(), a, g)))
        for f in futs:
            f.result()


# =========================================================================== experiment 1: tau-bench equivalence
def exp_equiv():
    from . import sampling as S
    from .data import load_taubench
    T = load_taubench()
    flag = S.classifier_scores(T) > 0.5
    err = np.array([any(s["status"] == "ERROR" for s in t["spans"]) for t in T])
    cols = {"err": ([{"name": "error", "type": "status_code", "status_code": {"status_codes": ["ERROR"]}}], 24300),
            "outcome": ([{"name": "outcome", "type": "string_attribute",
                          "string_attribute": {"key": "app.outcome", "values": ["fail"]}}], 24310),
            "prob": ([{"name": "p10", "type": "probabilistic", "probabilistic": {"sampling_percentage": 10}}], 24320)}
    sinks = {}
    for name, (pol, port) in cols.items():
        sinks[name] = Sink(port + 5)
        start_collector(f"equiv-{name}", port, port + 5, 5, pol)
    ids = [tid("tau", i) for i in range(len(T))]

    def body(i):
        t, now = T[i], time.time_ns()
        spans = [mkspan(ids[i], sid("tau", i, j), s["kind"], now + j, error=s["status"] == "ERROR",
                        attrs=[("gen_ai.operation.name", s["kind"]), ("gen_ai.agent.name", s["agent"]),
                               ("gen_ai.tool.name", s["tool"]), ("gen_ai.content", s["text"])])
                 for j, s in enumerate(t["spans"])]
        spans.append(mkspan(ids[i], sid("tau", i, "outcome"), "outcome", now + len(spans),
                            attrs=[("app.outcome", "fail" if flag[i] else "ok")]))
        return request(spans)

    cache = {}
    sched = []
    for i in range(len(T)):
        for name, (_, port) in cols.items():
            sched.append((i / 40.0, port, lambda i=i: cache[i] if i in cache else cache.setdefault(i, body(i))))
    snd = Sender()
    snd.run(sched, time.time() + 1)
    time.sleep(5 + 15)
    res = {"version": collector_version(), "send_errors": snd.errors, "threshold": 0.5,
           "n_flagged": int(flag.sum())}
    hexids = [x.hex() for x in ids]
    for name, pred in [("err", err), ("outcome", flag)]:
        kept = sinks[name].traces()
        p = {hexids[i] for i in range(len(T)) if pred[i]}
        res[name] = dict(kept=len(kept), predicted=len(p), extra=len(kept - p), missed=len(p - kept))
    res["prob"] = dict(kept=len(sinks["prob"].traces()))
    # completeness: every kept trace has all its spans including the outcome span
    full = {hexids[i]: len(T[i]["spans"]) + 1 for i in range(len(T))}
    for name in cols:
        cnt = {}
        for _, k, names in sinks[name].events:
            cnt[k] = cnt.get(k, 0) + len(names)
        res[name]["incomplete"] = sum(1 for k, c in cnt.items() if c != full[k])
    for name in cols:
        stop_collector(f"equiv-{name}")
        sinks[name].close()
    return res


# =========================================================================== experiment 2: decision window
def exp_window(windows=(60, 120, 300, 600), delay=5.0, compress=10.0, stagger=0.5, bucket=0.1):
    from .data import load_trail
    T = load_trail()
    pol = [{"name": "outcome", "type": "string_attribute",
            "string_attribute": {"key": "app.outcome", "values": ["fail"]}}]
    ports = {W: 24400 + 10 * n for n, W in enumerate(windows)}
    sinks = {}
    for W, port in ports.items():
        sinks[W] = Sink(port + 5)
        start_collector(f"win-{W}", port, port + 5, W / compress, pol)
    sched = []
    for i, t in enumerate(T):
        t_start = i * stagger  # staggered starts keep the client from building a send backlog
        for W, port in ports.items():
            trace_id = tid("win", W, i)
            groups = {}
            for j, s in enumerate(t["spans"]):
                groups.setdefault(int(s["start"] / compress / bucket), []).append((j, s))
            for g, (key, members) in enumerate(sorted(groups.items())):
                off = t_start + members[0][1]["start"] / compress
                sched.append((off, port, lambda tr=trace_id, i=i, m=members:
                              request([mkspan(tr, sid("win", i, j), s["name"], time.time_ns(),
                                              int(s["dur"] * 1e9 / compress), s["status"] == "ERROR")
                                       for j, s in m]), ("first", W, i) if g == 0 else None))
            sched.append((t_start + (t["duration"] + delay) / compress, port, lambda tr=trace_id, i=i:
                          request([mkspan(tr, sid("win", i, "o"), "outcome", time.time_ns(),
                                          attrs=[("app.outcome", "fail")])]), ("outcome", W, i)))
    sched.sort(key=lambda x: x[0])
    snd = Sender(64)
    snd.run(sched, time.time() + 1)
    time.sleep(max(windows) / compress + 10)
    d = np.array([t["duration"] for t in T])
    lags = np.array(snd.lags)
    res = {"version": collector_version(), "send_errors": snd.errors, "compress": compress, "delay": delay,
           "send_lag_s": dict(p50=round(float(np.median(lags)), 3), p99=round(float(np.percentile(lags, 99)), 3),
                              max=round(float(lags.max()), 3))}
    for W in windows:
        got = sinks[W].traces()
        kept = np.array([tid("win", W, i).hex() in got for i in range(len(T))])
        # prediction from the timings actually achieved on the wire (original time scale)
        gap = np.array([(snd.sent[("outcome", W, i)] - snd.sent[("first", W, i)]) * compress for i in range(len(T))])
        res[str(W)] = dict(kept=round(100 * kept.mean(), 1), predicted=round(100 * np.mean(d + delay < W), 1),
                           predicted_from_wire=round(100 * np.mean(gap < W), 1),
                           predicted_plus_tick=round(100 * np.mean(d + delay < W + TICK * compress), 1),
                           kept_but_late=int(np.sum(kept & (gap >= W + TICK * compress))),
                           missed_but_in_time=int(np.sum(~kept & (gap < W))))
        stop_collector(f"win-{W}")
        sinks[W].close()
    return res


# =========================================================================== experiment 3: memory
def trail_raw():
    """Real TRAIL spans (depth-first) with their attributes, for payload replay."""
    from .trail import _walk
    out = []
    for f in ("gaia.parquet", "swe_bench.parquet"):
        for tr in pd.read_parquet(os.path.join(ROOT, "data", "trail", f))["trace"]:
            t = json.loads(tr)
            raw = []
            for r in sorted(t["spans"], key=lambda s: s["timestamp"]):
                _walk(r, raw)
            from .trail import _ts, _dur
            t0 = min(_ts(s["timestamp"]) for s in raw)
            out.append([dict(name=s["span_name"], start=_ts(s["timestamp"]) - t0, dur=_dur(s["duration"]),
                             error=s["status_code"] == "Error",
                             attrs=list((s.get("span_attributes") or {}).items()),
                             sid=bytes.fromhex(s["span_id"]),
                             parent=bytes.fromhex(s["parent_span_id"]) if s.get("parent_span_id") else b"")
                        for s in raw])
    return out


PH = bytes([0xAB]) * 16  # placeholder trace id, replaced byte-wise at send time


def templates(compress=10.0, cap=60.0, bucket=0.1):
    """Pre-encode every TRAIL trace once: list of (offset_s, n_spans, body) with placeholder trace ids."""
    out = []
    for spans in trail_raw():
        groups = {}
        for s in spans:
            groups.setdefault(int(min(s["start"] / compress, cap) / bucket), []).append(s)
        reqs = []
        for key in sorted(groups):
            m = groups[key]
            body = request([mkspan(PH, s["sid"], s["name"], 1_700_000_000_000_000_000 + int(s["start"] * 1e9),
                                   int(s["dur"] * 1e9), s["error"], s["attrs"], s["parent"]) for s in m])
            reqs.append((min(m[0]["start"] / compress, cap), len(m), body))
        end = max(s["start"] + s["dur"] for s in spans)
        reqs.append((min((end + 5) / compress, cap), 1,
                     request([mkspan(PH, bytes([1]) * 8, "outcome", 1_700_000_000_000_000_000,
                                     attrs=[("app.outcome", "ok")])])))
        out.append(reqs)
    return out


def exp_memory_one(W, rate, extra=120, port=24500):
    """One memory configuration: window W (s, real time), `rate` new traces per second for W + extra s.
    Every trace is dropped at its decision, so the collector holds each trace for the full window
    (as it would for any trace) without export load on the replay host."""
    tpl = templates()
    pol = [{"name": "outcome", "type": "string_attribute",
            "string_attribute": {"key": "app.outcome", "values": ["fail"]}}]
    name = f"mem-{W}-{rate}"
    sink = Sink(port + 5)
    start_collector(name, port, port + 5, W, pol, cpus=2, mem="7g")
    time.sleep(15)
    scr = Scraper(port)
    time.sleep(10)
    base = np.median([r["rss"] for r in scr.rows if "rss" in r])
    n = int((W + extra) * rate)
    rng = random.Random(W * 10 + rate)
    sched, nspans = [], {}
    for k in range(n):
        reqs = tpl[rng.randrange(len(tpl))]
        trace_id = tid("mem", W, rate, k)
        for g, (off, cnt, body) in enumerate(reqs):
            sched.append((k / rate + off, port, lambda b=body, tr=trace_id: b.replace(PH, tr), (k, g)))
            nspans[(k, g)] = cnt
    sched.sort(key=lambda x: x[0])
    snd = Sender(32)
    t0 = time.time() + 1
    snd.run(sched, t0)
    time.sleep(W + 30)
    scr.stop = True
    rows = scr.rows
    rss = np.array([r.get("rss", np.nan) for r in rows]) / 1e6
    cpu = np.array([r.get("cpu", np.nan) for r in rows])
    tt = np.array([r["t"] for r in rows])
    busy = (tt >= t0) & (tt <= t0 + W + extra)
    cores = np.diff(cpu) / np.maximum(np.diff(tt), 1e-6)
    cores_busy = cores[busy[1:]]
    # late spans: accepted after the trace's decision (first span accepted + window + one tick)
    late = sum(nspans[(k, g)] for (k, g), t in snd.sent.items()
               if g > 0 and t > snd.sent[(k, 0)] + W + TICK)
    lags = np.array(snd.lags)
    res = dict(window=W, rate=rate, version=collector_version(), limits="2 CPUs, 7 GB",
               baseline_mb=round(float(base) / 1e6), peak_mb=round(float(np.nanmax(rss))),
               max_held=int(max(r.get("held", 0) for r in rows)),
               cpu_mean=round(float(np.nanmean(cores_busy)), 3),
               cpu_p95=round(float(np.nanpercentile(cores_busy, 95)), 3),
               late_spans=int(late), traces_sent=n, send_errors=snd.errors,
               send_lag_p50=round(float(np.median(lags)), 3), send_lag_p99=round(float(np.percentile(lags, 99)), 3),
               send_lag_max=round(float(lags.max()), 3),
               wire_mb_per_trace=round(snd.wire / n / 1e6, 2), exported_spans=len(sink.events),
               series=[(round(r["t"] - t0, 1), round(r.get("rss", 0) / 1e6, 1)) for r in rows])
    stop_collector(name)
    sink.close()
    return res


def exp_memory(configs=((60, 1), (60, 3), (300, 1), (300, 3))):
    """Run each configuration in its own process (one Python client per collector)."""
    procs = []
    for n, (W, rate) in enumerate(configs):
        procs.append(subprocess.Popen([sys.executable, "-m", "hs.collector", "memory_one", str(W), str(rate),
                                       str(24500 + 10 * n)], cwd=ROOT))
    for p in procs:
        p.wait()
    res = {}
    for W, rate in configs:
        f = os.path.join(ROOT, "results", f"collector_memory_{W}_{rate}.json")
        res[f"{W}s_{rate}ps"] = json.load(open(f))
    return res


if __name__ == "__main__":
    exp = sys.argv[1]
    t = time.time()
    if os.name == "nt":  # keep the host awake while an experiment runs (released when the process exits)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if exp == "memory_one":
        W, rate, port = map(int, sys.argv[2:5])
        res = exp_memory_one(W, rate, port=port)
        exp = f"memory_{W}_{rate}"
    else:
        res = {"equiv": exp_equiv, "window": exp_window, "memory": exp_memory}[exp]()
    res["wall_minutes"] = round((time.time() - t) / 60, 1)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", f"collector_{exp}.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "series"}, default=str)[:3000])
