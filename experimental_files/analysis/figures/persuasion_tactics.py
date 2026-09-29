"""Reproduce Figure 6 (persuasion_tactics.pdf): persuasion tactics in coded deceiver messages from independent trials.

tactics_by_message.csv has one row per deceiver message in rounds 0 and 1 (1,380 messages from 120 independent
trials, 30 per model, plus the coordinated trials), with a 0/1 column per tactic; a message may carry several
labels. Panel (a) shows each model's three most prevalent tactics over both rounds; panel (b) shows five tactics by
round across models. Error bars are 95% percentile intervals from a trial-level bootstrap (2,000 resamples,
stratified by model; see bootstrap_ci.py). Run with Python, matplotlib, numpy, and pandas.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from bootstrap_ci import ClusterBootstrap, rate

OUT = Path(__file__).resolve().parent
MODELS = {"Gemini 3.8 Flash": "Gemini", "Grok 4.3": "Grok", "DeepSeek V4.1 Flash": "DeepSeek", "Muse Glimmer": "Muse\nGlimmer"}
TACTIC_LABEL = {
    "fabricate_or_misrepresent_evidence": "Fabricated or misrepresented evidence",
    "misleading_inference": "Misleading inference from true facts",
    "reframe_or_equivocate": "Reinterpreting the question",
    "selective_skepticism": "Selective skepticism",
    "concede_and_pivot": "Concede and pivot",
    "exploit_uncertainty": "Exploiting unresolved uncertainty",
    "social_or_consensus_pressure": "Consensus pressure",
    "bare_assertion": "Bare assertion",
}
TACTICS = list(TACTIC_LABEL)
# Prevalence (%) reported in the manuscript; the script checks that the data reproduce these values.
EXPECTED_BY_MODEL = {
    "Fabricated or misrepresented evidence": [51, 58, 40, 42],
    "Misleading inference from true facts": [47, 67, 41, 54],
    "Reinterpreting the question": [37, 27, 31, 39],
    "Selective skepticism": [28, 38, 45, 17],
    "Concede and pivot": [42, 23, 43, 33],
    "Exploiting unresolved uncertainty": [4, 12, 25, 5],
    "Consensus pressure": [34, 13, 5, 15],
    "Bare assertion": [1, 4, 2, 0],
}
EXPECTED_BY_ROUND = {
    "Fabricated or misrepresented evidence": [56, 40],
    "Misleading inference from true facts": [57, 47],
    "Reinterpreting the question": [15, 52],
    "Selective skepticism": [7, 57],
    "Concede and pivot": [6, 65],
}
ROUND_TACTICS = ["fabricate_or_misrepresent_evidence", "misleading_inference", "reframe_or_equivocate",
                 "selective_skepticism", "concede_and_pivot"]
ROUND_LABELS = [
    "Fabricated /\nmisrepresented evidence", "Misleading inference",
    "Reinterpret question", "Selective skepticism", "Concede and pivot",
]
STYLES = {
    "fabricate_or_misrepresent_evidence": ("#0072B2", ""),
    "misleading_inference": ("#D55E00", "//"),
    "concede_and_pivot": ("#CC79A7", ".."),
    "selective_skepticism": ("#009E73", "\\\\"),
    "reframe_or_equivocate": ("#E69F00", "xx"),
}
BAR_ERR = dict(fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)

frame = pd.read_csv(OUT / "tactics_by_message.csv")
messages = frame[frame.coordinated == 0].copy()
messages["messages"] = 1
messages["trial"] = messages.model + "|" + messages.question_id + "|" + messages.N.astype(str) + "+" + messages.k.astype(str)
assert len(messages) == 1380 and messages.trial.nunique() == 120
boot = ClusterBootstrap(messages, cluster="trial", strata="model")


def estimate(sub, tactic):
    lo, hi = boot.ci(sub, tactic, "messages")
    return rate(sub, tactic, "messages"), lo, hi


def label(value):
    return str(int(np.round(value)))


for tactic in TACTICS:
    got = [int(np.round(rate(messages[messages.model == m], tactic, "messages"))) for m in MODELS]
    assert got == EXPECTED_BY_MODEL[TACTIC_LABEL[tactic]], (tactic, got)
for tactic in ROUND_TACTICS:
    got = [int(np.round(rate(messages[messages["round"] == r], tactic, "messages"))) for r in (0, 1)]
    assert got == EXPECTED_BY_ROUND[TACTIC_LABEL[tactic]], (tactic, got)


def plot_tactics(output_path):
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.3, "ytick.labelsize": 7.3, "legend.fontsize": 6.8,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "legend.frameon": False,
        "hatch.linewidth": 0.45,
    })
    fig, (ax, rounds_ax) = plt.subplots(
        1, 2, figsize=(5.5, 2.8), sharey=True,
        gridspec_kw={"width_ratios": [1, 1.15]},
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.40, top=0.88, wspace=0.22)
    ax.set_title("(a) Top three tactics by model", loc="left", fontsize=8, pad=9)
    rounds_ax.set_title("(b) Tactics by discussion round", loc="left", fontsize=8, pad=9)
    highest = 0
    centers = [1.3 * i for i in range(len(MODELS))]
    for (full, short), center in zip(MODELS.items(), centers):
        sub = messages[messages.model == full]
        est = {t: estimate(sub, t) for t in TACTICS}
        ranked = sorted(TACTICS, key=lambda t: est[t][0], reverse=True)[:3]
        for rank, tactic in enumerate(ranked):
            value, lo, hi = est[tactic]
            color, hatch = STYLES[tactic]
            x = center + (rank - 1) * 0.28
            ax.bar(x, value, width=0.25, color=color, hatch=hatch, edgecolor="#333333", linewidth=0.45, zorder=3)
            highest = max(highest, value)
            print(f"(a) {full:20s} {TACTIC_LABEL[tactic]:38s} {value:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
    ax.set_xticks(centers, list(MODELS.values()))
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(-0.65, centers[-1] + 0.65)
    handles = [Patch(facecolor=color, edgecolor="#333333", linewidth=0.45, hatch=hatch, label=TACTIC_LABEL[tactic])
               for tactic, (color, hatch) in STYLES.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.025, -0.29),
              ncol=1, handletextpad=0.5, labelspacing=0.4, handlelength=1.3,
              borderaxespad=0, fontsize=6.8)

    for round_index, color in enumerate(["#a6c8ee", "#1b76d2"]):
        sub = messages[messages["round"] == round_index]
        for i, tactic in enumerate(ROUND_TACTICS):
            value, lo, hi = estimate(sub, tactic)
            x = i + (round_index - 0.5) * 0.35
            rounds_ax.bar(x, value, width=0.32, color=color, edgecolor="#333333", linewidth=0.45, zorder=3,
                          label=f"Round {round_index}" if i == 0 else None)
            highest = max(highest, value)
            print(f"(b) round {round_index}  {TACTIC_LABEL[tactic]:38s} {value:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
    rounds_ax.set_xticks(range(len(ROUND_LABELS)), ROUND_LABELS, fontsize=6.8,
                         rotation=55, ha="right", rotation_mode="anchor")
    rounds_ax.tick_params(axis="x", length=0, pad=5)
    rounds_ax.set_xlim(-0.6, len(ROUND_LABELS) - 0.4)
    rounds_ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1.01), ncol=2,
                     handlelength=1.2, handletextpad=0.4, columnspacing=0.9,
                     borderaxespad=0, fontsize=6.8)
    ax.set_ylabel("Prevalence (%)")
    top = int(max(80, 20 * np.ceil((highest + 7) / 20)))
    for panel in (ax, rounds_ax):
        panel.set_ylim(0, top)
        panel.set_yticks(range(0, top + 1, 20))
        panel.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        panel.set_axisbelow(True)
        for spine in ("top", "right"):
            panel.spines[spine].set_visible(False)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    plot_tactics(Path(__file__).with_suffix(".pdf"))
