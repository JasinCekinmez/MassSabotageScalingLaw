"""Peek at honest-agent trajectories in games that are not graded yet (checkpoints).

  python -m sabotage.peek experimental_files/results/main_grid/grok-4.3 grok-4.3

Judges every honest public position and vote found in the checkpoint files (cached in
<out>/peek_cache.json so re-runs are free), then prints per-question honest-correct counts per
round, pooled accuracy per round, and every honest agent whose position changed.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from .judge import judge_answer


def load_cache(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def main(out: str, models: list[str]) -> None:
    cache_path = os.path.join(out, "peek_cache.json")
    cache = load_cache(cache_path)
    games = []
    for m in models:
        for f in sorted(glob.glob(os.path.join(out, "games", f"*{m}*.json"))):
            games.append(json.load(open(f)))
    jobs = []
    for g in games:
        honest = [str(a["id"]) for a in g["agents"] if a["role"] == "honest"]
        grades = g.get("grades", {})
        for rd in g["rounds"]:
            for a in honest:
                if a in rd["public_text"]:
                    key = f"{g['tag']}|{g['question_id']}|r{rd['round']}|{a}"
                    pre = grades.get("public", {}).get(str(rd["round"]), {}).get(a)
                    if pre:
                        cache[key] = pre["correct"] == "yes"
                    jobs.append((key, g, rd["public_text"][a]))
        for a in honest:
            if a in g["votes_text"]:
                key = f"{g['tag']}|{g['question_id']}|vote|{a}"
                pre = grades.get("votes", {}).get(a)
                if pre:
                    cache[key] = pre["correct"] == "yes"
                jobs.append((key, g, g["votes_text"][a]))
    todo = [j for j in jobs if j[0] not in cache]
    print(f"{len(jobs)} honest positions, {len(todo)} need judging", flush=True)
    with ThreadPoolExecutor(16) as ex:
        for (key, _, _), j in zip(todo, ex.map(lambda j: judge_answer(j[1]["question"], j[1]["correct_answer"], j[2]), todo)):
            cache[key] = j["correct"] == "yes"
    json.dump(cache, open(cache_path, "w"))

    for m in models:
        print(f"\n===== {m} =====")
        print(f"{'k':>1s} {'question':9s} {'subject':14s} {'status':9s}  honest correct (of 4): r0 r1 r2 ... then vote")
        agg = collections.defaultdict(lambda: [0, 0])
        shifts = []
        for g in sorted((g for g in games if g["config"]["honest_model"] == m), key=lambda g: (g["config"]["n_deceivers"], g["question_id"])):
            k = g["config"]["n_deceivers"]
            honest = [str(a["id"]) for a in g["agents"] if a["role"] == "honest"]
            nr = len([rd for rd in g["rounds"] if rd["public"]])
            cols = [f"r{r}" for r in range(nr)] + (["vote"] if g["votes_text"] else [])
            cells = []
            for c in cols:
                vals = [cache.get(f"{g['tag']}|{g['question_id']}|{c}|{a}") for a in honest]
                vals = [v for v in vals if v is not None]
                if not vals:
                    cells.append("-"); continue
                cells.append(str(sum(vals)) if len(vals) == len(honest) else f"{sum(vals)}/{len(vals)}")
                agg[(k, c)][0] += sum(vals); agg[(k, c)][1] += len(vals)
            status = "complete" if g.get("status") == "complete" else f"r{nr - 1}"
            print(f"{k} {g['question_id'][:8]} {g['raw_subject'][:14]:14s} {status:9s}  " + " ".join(f"{x:>3s}" for x in cells))
            for a in honest:
                traj = [cache.get(f"{g['tag']}|{g['question_id']}|{c}|{a}") for c in cols]
                traj = [t for t in traj if t is not None]
                if len(traj) > 1 and len(set(traj)) > 1:
                    shifts.append(f"  k={k} {g['question_id'][:8]} colleague {a}: " + "".join("Y" if t else "." for t in traj) + ("  (last = vote)" if g["votes_text"] else ""))
        print("\n  pooled honest accuracy per round (only rounds reached):")
        for k in (0, 2, 3):
            parts = [f"{c}={agg[(k, c)][0]}/{agg[(k, c)][1]}" for c in [f"r{r}" for r in range(8)] + ["vote"] if agg[(k, c)][1]]
            if parts:
                print(f"    k={k}: " + "  ".join(parts))
        print("\n  honest agents whose position changed (Y=correct, .=wrong):")
        print("\n".join(shifts) if shifts else "    none")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
