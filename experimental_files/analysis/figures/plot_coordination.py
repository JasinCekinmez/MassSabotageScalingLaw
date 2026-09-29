"""Reproduce Figure 4 (coordination.pdf): honest defection with independent versus coordinated deceivers.

coordination_by_trial.csv has one row per trial: the matched independent and coordinated trials (same question and
composition, at least two deceivers; 856 Gemini and 757 Grok pairs) plus every independent trial without deceivers,
which gives the shared k/N = 0 baseline. Points are pooled defection rates; error bars are 95% percentile intervals
from a question-level bootstrap (2,000 resamples, stratified by difficulty bucket; see bootstrap_ci.py). The two
conditions are offset slightly on the horizontal axis so their intervals do not overlap.
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
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.linewidth": 0.8,
                     "lines.linewidth": 1.4, "lines.markersize": 4.5, "pdf.fonttype": 42, "legend.frameon": False})
FAM_COLOR = {"Gemini 3.8 Flash": "#1b76d2", "Grok 4.3": "#483e8c"}
FAM_MARK = {"Gemini 3.8 Flash": "o", "Grok 4.3": "s"}
PROP = [0.0, 0.2, 0.33, 0.43]
PROP_LABEL = {0.0: "0", 0.2: "1/5", 0.33: "1/3", 0.43: "3/7"}
positions = np.array([0, 1/5, 1/3, 3/7])
DODGE = 0.007

frame = pd.read_csv(OUT / "coordination_by_trial.csv")
frame["prop"] = (frame.k / frame.N).round(2)

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
bottom, top = 5, 45
for ax, n in zip(axes, FAM_COLOR):
    d = frame[frame.model == n]
    boot = ClusterBootstrap(d)
    independent = d[(d.condition == "independent") & (d.k >= 2)]
    coordinated = d[d.condition == "coordinated"]
    base = d[(d.condition == "independent") & (d.k == 0)]
    assert len(independent) == len(coordinated)
    b_rate, (b_lo, b_hi) = rate(base), boot.ci(base)
    print(f"{n:18s} baseline          {b_rate:5.1f}  [{b_lo:5.1f}, {b_hi:5.1f}]  ({len(base)} trials)")
    series = [("non-coordinated deceivers", independent, -DODGE, "-", FAM_COLOR[n]),
              ("coordinated deceivers", coordinated, DODGE, "--", "white")]
    for label, s, shift, linestyle, face in series:
        rows = [(rate(sub),) + boot.ci(sub) for sub in (s[np.isclose(s.prop, p, atol=0.02)] for p in PROP[1:])]
        y, lo, hi = np.array(rows).T
        x = positions[1:] + shift
        ax.plot(np.r_[0, x], np.r_[b_rate, y], marker=FAM_MARK[n], color=FAM_COLOR[n], linestyle=linestyle,
                markerfacecolor=face, label=label, zorder=3)
        bars(ax, x, y, lo, hi, FAM_COLOR[n])
        bottom, top = min(bottom, 5 * np.floor((lo.min() - 2) / 5)), max(top, 5 * np.ceil((hi.max() + 2) / 5))
        for p, r, l, h in zip(PROP[1:], y, lo, hi):
            print(f"{n:18s} {label:22s} k/N={PROP_LABEL[p]:>3s}  {r:5.1f}  [{l:5.1f}, {h:5.1f}]  ({len(s)} trials)")
    ax.plot([0], [b_rate], marker=FAM_MARK[n], color=FAM_COLOR[n], linestyle="none", zorder=4)
    bars(ax, [0], [b_rate], [b_lo], [b_hi], FAM_COLOR[n])
    bottom = min(bottom, 5 * np.floor((b_lo - 2) / 5))
    ax.set_xticks(positions); ax.set_xticklabels([PROP_LABEL[p] for p in PROP]); ax.set_title(n)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("adversarial proportion k/N")
for ax in axes:
    ax.yaxis.set_major_locator(MultipleLocator(10)); ax.set_ylim(max(0, bottom), top)
axes[0].set_ylabel("honest defection (%)")
axes[0].legend(loc="upper left", handlelength=2.0)
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT / "coordination.pdf", bbox_inches="tight")
