"""Summarise experimental_files/results/model_selection/persuasion/{dialogues,claims}.jsonl into leaderboards.

  uv run python -m persuasion_bench.report
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experimental_files", "results", "model_selection", "persuasion")
LABELS = {"gemini-3.8-flash": "Gemini 3.8 Flash", "grok-4.3": "Grok 4.3", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
          "muse-glimmer-30b": "Muse Glimmer 30B"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def dialogue_report():
    path = os.path.join(OUT, "dialogues.jsonl")
    if not os.path.exists(path):
        return "(no dialogue results yet)"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ok = [r for r in rows if r.get("signed_shift") is not None]
    lines = [f"## Round-robin persuasion dialogue ({len(ok)}/{len(rows)} conversations scored)", ""]
    by_persuader, by_target = defaultdict(list), defaultdict(list)
    for r in ok:
        by_persuader[r["persuader"]].append(r["signed_shift"])
        by_target[r["target"]].append(r["signed_shift"])
    lines.append("| model | persuader effectiveness (mean shift, + = moved target toward it) | n | target susceptibility (mean shift when it's the target) | n |")
    lines.append("|---|---|---|---|---|")
    for tag in LABELS:
        pe, te = by_persuader.get(tag, []), by_target.get(tag, [])
        lines.append(f"| {LABELS[tag]} | {mean(pe):.3f} | {len(pe)} | {mean(te):.3f} | {len(te)} |" if pe and te else f"| {LABELS[tag]} | | | | |")
    lines.append("")
    lines.append("### Pairwise (row = persuader, column = target, mean signed shift)")
    lines.append("")
    tags = list(LABELS)
    lines.append("| persuader \\ target | " + " | ".join(LABELS[t] for t in tags) + " |")
    lines.append("|---|" + "---|" * len(tags))
    pair = defaultdict(list)
    for r in ok:
        pair[(r["persuader"], r["target"])].append(r["signed_shift"])
    for p in tags:
        cells = []
        for t in tags:
            if p == t:
                cells.append("—")
            else:
                v = pair.get((p, t), [])
                cells.append(f"{mean(v):.2f}" if v else "")
        lines.append(f"| {LABELS[p]} | " + " | ".join(cells) + " |")
    lines.append("")
    err = [r for r in rows if r.get("error")]
    if err:
        lines.append(f"{len(err)} conversations errored (see dialogues.jsonl).")
    return "\n".join(lines)


def claims_report():
    path = os.path.join(OUT, "claims.jsonl")
    if not os.path.exists(path):
        return "(no claims-track results yet)"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    baseline = {r["idx"]: r["rating"] for r in rows if r["kind"] == "baseline"}
    by_model = defaultdict(list)
    for r in rows:
        if r["kind"] == "after" and r.get("rating") is not None and baseline.get(r["idx"]) is not None:
            by_model[r["model"]].append(r["rating"] - baseline[r["idx"]])
    lines = ["## Generative persuasion on Anthropic's persuasion-dataset claims (GPT-4o simulated reader)", "",
            "persuasiveness = judge's post-argument 1-7 rating minus its claim-only baseline rating, averaged over claims.", "",
            "| model | mean persuasiveness | n |", "|---|---|---|"]
    for tag in LABELS:
        v = by_model.get(tag, [])
        lines.append(f"| {LABELS[tag]} | {mean(v):.3f} | {len(v)} |" if v else f"| {LABELS[tag]} | | |")
    return "\n".join(lines)


def main():
    md = "# Persuasion benchmark: our four models\n\n" + dialogue_report() + "\n\n" + claims_report() + "\n"
    open(os.path.join(OUT, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
