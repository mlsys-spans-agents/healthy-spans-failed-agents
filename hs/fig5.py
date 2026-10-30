"""Redraw Figure 5 from results/collector_window.json and results/collector_memory.json."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = json.load(open(os.path.join(ROOT, "results", "collector_window.json")))
M = json.load(open(os.path.join(ROOT, "results", "collector_memory.json")))

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))

wins = [60, 120, 300, 600]
x = range(len(wins))
pred = [W[str(w)]["predicted"] for w in wins]
kept = [W[str(w)]["kept"] for w in wins]
a.plot(x, pred, ":", marker="o", color="#9aa0a6", ms=4, label="Predicted: run + judge < window")
a.plot(x, kept, "-", marker="s", color="#2a6fdb", ms=5, label="Measured on the collector")
for i, v in zip(x, kept):
    a.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8,
               color="#2a6fdb")
a.set_xticks(list(x), [f"{w}s" for w in wins])
a.set_ylim(0, 105)
a.set_xlabel("Decision window (decision_wait)")
a.set_ylabel("Flagged failures kept (%)")
a.set_title("(a) Outcomes arriving before the decision", fontweight="bold")
a.grid(axis="y", alpha=0.3)
a.legend(loc="lower right", frameon=False, fontsize=8)

colors = {"60_1": "#1b9e77", "60_3": "#e6ab02", "300_1": "#2a6fdb", "300_3": "#e8590c"}
for key, c in colors.items():
    s = M[key]["series"]
    w, r = key.split("_")
    b.plot([p[0] for p in s], [p[1] for p in s], color=c, lw=1.4,
           label=f"window {w}s, {r} new trace{'s' if r != '1' else ''}/s")
b.set_xlabel("Time since replay start (s)")
b.set_ylabel("Collector RSS (MB)")
b.set_title("(b) Collector memory, real TRAIL payloads", fontweight="bold")
b.grid(alpha=0.3)
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8)

fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "fig5.png")
fig.savefig(out, dpi=200)
print("wrote", out)
