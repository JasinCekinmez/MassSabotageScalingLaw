"""Reproduce Figures 2 and 3 (proportion.pdf, size.pdf) from the trial-level data used in the paper's analysis.

defection_by_trial.csv has one row per trial of the 4,788-trial main grid (all completed games; Muse Glimmer's
102-question run completed 2026-09-22): question,
difficulty bucket (correct answers in four selection attempts), composition, initially correct honest agents, and
defections. defection_by_composition.csv is its per-composition aggregate; the script checks that the two agree.
Points are pooled defection rates; error bars are 95% percentile intervals from a question-level bootstrap
(2,000 resamples, stratified by difficulty bucket; see bootstrap_ci.py). Ongoing experiments do not change these
figures. Run with a Python environment containing matplotlib, numpy, and pandas.
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

def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.linewidth": 0.8,
                     "lines.linewidth": 1.4, "lines.markersize": 4.5, "pdf.fonttype": 42, "legend.frameon": False})
FAM_COLOR = {"Gemini 3.8 Flash": "#1b76d2", "Grok 4.3": "#483e8c", "DeepSeek V4.1 Flash": "#dc8969", "Muse Glimmer": "#2f8f6b"}
FAM_MARK = {"Gemini 3.8 Flash": "o", "Grok 4.3": "s", "DeepSeek V4.1 Flash": "^", "Muse Glimmer": "D"}
PROP = [0.0, 0.2, 0.33, 0.43]
PROP_LABEL = {0.0: "0", 0.2: "1/5", 0.33: "1/3", 0.43: "3/7"}
PCT_LABEL = {0.0: "0%", 0.2: "20%", 0.33: "33%", 0.43: "43%"}

frame = pd.read_csv(OUT / "defection_by_trial.csv")
frame["prop"] = (frame.k / frame.N).round(2)
fixed = pd.read_csv(OUT / "defection_by_composition.csv")
agg = frame.groupby(["model", "N", "k"]).agg(trials=("question_id", "size"), initially_correct=("initially_correct", "sum"),
                                             defections=("defections", "sum")).reset_index()
check = agg.merge(fixed, on=["model", "N", "k"], suffixes=("", "_fixed"))
assert len(check) == len(fixed) and all((check[c] == check[c + "_fixed"]).all() for c in ("trials", "initially_correct", "defections")), \
    "defection_by_trial.csv does not aggregate to defection_by_composition.csv"
data = {model: frame[frame.model == model] for model in FAM_COLOR}
boot = {model: ClusterBootstrap(d) for model, d in data.items()}


def estimate(model, d):
    (lo, hi) = boot[model].ci(d)
    return rate(d), lo, hi


def style(ax, ylab=None, xlab=None):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ylab: ax.set_ylabel(ylab)
    if xlab: ax.set_xlabel(xlab)


# ---------------------------------------------------------------- Figure 2: defection vs proportion, one panel per model
fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.15), sharex=True)
positions = [0, 1/5, 1/3, 3/7]
for ax, (n, d) in zip(axes, data.items()):
    rates, lo, hi = np.array([estimate(n, d[np.isclose(d.prop, p, atol=0.02)]) for p in PROP]).T
    ax.plot(positions, rates, marker=FAM_MARK[n], color=FAM_COLOR[n], zorder=3)
    bars(ax, positions, rates, lo, hi, FAM_COLOR[n])
    ax.set_xticks(positions); ax.set_xticklabels([PCT_LABEL[p].rstrip("%") for p in PROP])
    ax.set_xlim(-0.025, 0.455)
    bottom = max(0, 5 * np.floor((lo.min() - 2) / 5))
    top = min(100, 5 * np.ceil((hi.max() + 2) / 5))
    ax.set_ylim(bottom, top)
    ax.yaxis.set_major_locator(MultipleLocator(10 if top - bottom >= 30 else 5))
    ax.set_title(n)
    style(ax)
    for p, r, l, h in zip(PROP, rates, lo, hi):
        print(f"proportion  {n:20s} k/N={PROP_LABEL[p]:>3s}  {r:5.1f}  [{l:5.1f}, {h:5.1f}]")
axes[0].set_ylabel("honest defection (%)")
fig.supxlabel("% deceivers", fontsize=9, y=0.1)
fig.tight_layout(w_pad=0.9)
save(fig, "proportion")

# ---------------------------------------------------------------- Figure 3: Gemini at fixed proportions, different group sizes
fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.05), sharey=True)
d = data["Gemini 3.8 Flash"]
top = 45
for ax, p in zip(axes, PROP):
    s = d[np.isclose(d.prop, p, atol=0.02)]
    sizes = np.array(sorted(s.N.unique()))
    rates, lo, hi = np.array([estimate("Gemini 3.8 Flash", s[s.N == n]) for n in sizes]).T
    ax.plot(sizes, rates, marker="o", color=FAM_COLOR["Gemini 3.8 Flash"], zorder=3)
    bars(ax, sizes, rates, lo, hi, FAM_COLOR["Gemini 3.8 Flash"])
    ax.set_xscale("log", base=2)
    lo2, hi2 = np.log2([sizes.min(), sizes.max()])
    pad = 0.12 * (hi2 - lo2)
    ax.set_xlim(2 ** (lo2 - pad), 2 ** (hi2 + pad))
    ax.set_xticks(sizes); ax.set_xticklabels([str(n) for n in sizes])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title(f"{PCT_LABEL[p]} deceivers")
    style(ax)
    top = max(top, 5 * np.ceil((hi.max() + 2) / 5))
    for n, r, l, h in zip(sizes, rates, lo, hi):
        print(f"size        Gemini k/N={PROP_LABEL[p]:>3s} N={n:2d}  {r:5.1f}  [{l:5.1f}, {h:5.1f}]")
for ax in axes:
    ax.yaxis.set_major_locator(MultipleLocator(10)); ax.set_ylim(0, top)
axes[0].set_ylabel("honest defection (%)")
fig.supxlabel("# agents", fontsize=9, y=0.1)
fig.tight_layout(w_pad=0.8)
fig.suptitle("Gemini 3.8 Flash", fontsize=10, y=1.05)
save(fig, "size")
