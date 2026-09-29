"""Reproduce Figure 5 (heterogeneous.pdf): honest defection for the four honest/deceiver model pairings.

heterogeneous_by_trial.csv has one row per trial (question, difficulty bucket, composition, initially correct honest
agents, defections) for the 1,711 trials of the heterogeneous-model experiment. Each line connects the pooled rates
at the two tested proportions (4+1 and 8+2 at 1/5; 4+3 and 8+6 at 3/7) for one pairing. Error bars are 95%
percentile intervals from a question-level bootstrap (2,000 resamples, stratified by difficulty bucket; see
bootstrap_ci.py). Pairings are offset slightly on the horizontal axis so their intervals do not overlap.
Run with a Python environment containing matplotlib, numpy, and pandas.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from bootstrap_ci import ClusterBootstrap, rate, bars

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "lines.markersize": 5, "pdf.fonttype": 42, "legend.frameon": False,
})
# Rates reported in the manuscript at 1/5 and 3/7; the script checks that the data reproduce them.
pairings = [
    ("Gemini", "DeepSeek", [20.5, 24.3]),
    ("Gemini", "Grok", [14.5, 19.1]),
    ("Glimmer", "DeepSeek", [32.1, 48.3]),
    ("Glimmer", "Grok", [35.1, 35.4]),
]
HONEST_LABEL = {"Gemini": "Gemini", "Glimmer": "Muse Glimmer"}
colors = {"Gemini": "#1b76d2", "Glimmer": "#2f8f6b"}
styles = {"DeepSeek": ("-", "^"), "Grok": ("--", "s")}
PROPS = [0.2, 0.43]
positions = np.array([1/5, 3/7])
DODGE = [-0.012, -0.004, 0.004, 0.012]

frame = pd.read_csv(OUT / "heterogeneous_by_trial.csv")
frame["prop"] = (frame.k / frame.N).round(2)

fig, ax = plt.subplots(figsize=(4.8, 3.0))
bottom, top = 10, 55
for (honest, deceiver, expected), shift in zip(pairings, DODGE):
    d = frame[(frame.honest_model == honest) & (frame.deceiver_model == deceiver)]
    boot = ClusterBootstrap(d)
    rows = [(rate(sub),) + boot.ci(sub) for sub in (d[np.isclose(d.prop, p, atol=0.02)] for p in PROPS)]
    y, lo, hi = np.array(rows).T
    assert np.allclose(y, expected, atol=0.051), (honest, deceiver, y)
    line, marker = styles[deceiver]
    label = f"{HONEST_LABEL[honest]} / {deceiver}"
    ax.plot(positions + shift, y, color=colors[honest], linestyle=line, marker=marker, label=label, zorder=3)
    bars(ax, positions + shift, y, lo, hi, colors[honest])
    bottom, top = min(bottom, 5 * np.floor((lo.min() - 2) / 5)), max(top, 5 * np.ceil((hi.max() + 2) / 5))
    for p, r, l, h in zip(PROPS, y, lo, hi):
        print(f"{label:24s} k/N={p:.2f}  {r:5.1f}  [{l:5.1f}, {h:5.1f}]  ({len(d[np.isclose(d.prop, p, atol=0.02)])} trials)")
ax.set_xlim(0.17, 0.46)
ax.set_ylim(max(0, bottom), top)
ax.set_xticks([1/5, 3/7], [r"$1/5$", r"$3/7$"])
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.set_xlabel("adversarial proportion $k/N$")
ax.set_ylabel("honest defection (%)")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.legend(loc="upper center", bbox_to_anchor=(0.54, 1.01), ncol=2,
           title="Honest model / deceiver model", title_fontsize=9,
           handlelength=2.4, columnspacing=1.7)
fig.tight_layout(rect=(0, 0, 1, 0.78))
fig.savefig(OUT / "heterogeneous.pdf", bbox_inches="tight")
