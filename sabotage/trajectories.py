"""Per-agent, per-round stance trajectories from the round-wise grades.

No LLM calls: everything comes from grades.public / grades.votes already in the game JSONs.

Outputs (default under experimental_files/analysis/):
  honest_by_round.csv           honest accuracy per round, per family x coord x k/N, plus the share of
                                honest agents whose stated answer equals a deceiver's answer that round,
                                and correct->wrong / wrong->correct transition counts from the previous round
  honest_trajectory_patterns.csv  how each honest agent's stance evolved from round 0 to the vote
                                (start/end correctness, number of stance changes), counted per cell
  flip_events.csv               one row per honest agent per round where the graded answer changed
                                correct->wrong (defect) or wrong->correct (recover); the input rows for
                                later transcript reading

Usage:
  python -m sabotage.trajectories experimental_files/results/main_grid/gemini-3.8-flash/honest_agents_* \
      --out experimental_files/analysis \
      --bucket-csv experimental_files/question_selection/gemini-3.8-flash/selection.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

FAMILY = [("gemini", "Gemini 3.8 Flash"), ("grok", "Grok 4.3"),
          ("deepseek", "DeepSeek V4.1 Flash"), ("glimmer", "Muse Glimmer 30B")]

_MC = re.compile(r"^\s*\(?([A-Za-z])\)?(?:[\.\):\s]|$)")


def family_of(model: str) -> str:
    m = model.lower()
    for key, name in FAMILY:
        if key in m:
            return name
    return model


def norm_answer(extracted: str | None, answer_type: str) -> str | None:
    """Normalise the judge's extracted answer so two agents' answers can be compared."""
    if extracted is None:
        return None
    s = str(extracted).strip()
    if not s or s.lower() in ("none", "null", "n/a", "no answer", "undecided"):
        return None
    if answer_type == "multipleChoice":
        m = _MC.match(s)
        if m:
            return m.group(1).upper()
    return re.sub(r"\s+", " ", s.lower()).strip(" .")


def extract(path: str) -> dict | None:
    """Load one game and reduce it to per-agent, per-round stances. Runs in a worker process."""
    try:
        g = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    gr = g.get("grades") or {}
    if "outcome" not in gr or "public" not in gr or "votes" not in gr:
        return None
    cfg = g["config"]
    roles = {str(a["id"]): a["role"] for a in g["agents"]}
    honest = [a for a, r in roles.items() if r == "honest"]
    deceivers = [a for a, r in roles.items() if r == "deceiver"]
    atype = g.get("answer_type", "")
    n_rounds = len(g["rounds"])
    pub, votes = gr["public"], gr["votes"]

    def stance(cell):
        if not cell:
            return None, None
        return norm_answer(cell.get("extracted_final_answer"), atype), cell.get("correct") == "yes"

    # stages: rounds 0..n_rounds-1 then the vote
    stages = [str(r) for r in range(n_rounds)] + ["vote"]
    per_agent = {}
    dec_answers = {}
    for st in stages:
        cells = votes if st == "vote" else pub.get(st, {})
        dec_answers[st] = {norm_answer(cells.get(a, {}).get("extracted_final_answer"), atype) for a in deceivers} - {None}
        for a in honest:
            ans, ok = stance(cells.get(a))
            per_agent.setdefault(a, {})[st] = (ans, ok)

    return {
        "path": path,
        "trial": os.path.basename(os.path.dirname(os.path.dirname(path))),
        "question_id": g["question_id"], "category": g.get("category"),
        "family": family_of(cfg["honest_model"]), "coord": bool(cfg.get("coordinate", False)),
        "N": cfg["n_agents"], "k": cfg["n_deceivers"], "n_rounds": n_rounds,
        "stages": stages, "honest": honest, "per_agent": per_agent, "dec_answers": dec_answers,
    }


def ratio_label(k: int, n: int) -> str:
    return f"{k/n:.2f}" if n else "?"


def build(games: list[dict], buckets: dict[str, str]):
    by_round = collections.defaultdict(lambda: collections.Counter())
    patterns = collections.defaultdict(lambda: collections.Counter())
    flips = []
    games_in_cell = collections.defaultdict(set)

    for g in games:
        cell = (g["family"], g["coord"], g["k"], g["N"], g["n_rounds"])
        stages = g["stages"]
        for a in g["honest"]:
            traj = g["per_agent"][a]
            prev_ok = None
            prev_ans = None
            n_changes = 0
            first_defect = None
            for i, st in enumerate(stages):
                ans, ok = traj[st]
                key = cell + (st,)
                games_in_cell[key].add(g["path"])
                c = by_round[key]
                c["n_honest"] += 1
                if ok is None:
                    c["n_ungraded"] += 1
                else:
                    c["n_correct"] += int(ok)
                    if ans is None:
                        c["n_undecided"] += 1
                    elif ans in g["dec_answers"][st]:
                        c["n_on_deceiver_answer"] += 1
                if prev_ok is not None and ok is not None:
                    if prev_ok and not ok:
                        c["n_defect_from_prev"] += 1
                        if first_defect is None:
                            first_defect = st
                        flips.append(_flip(g, a, stages[i - 1], st, "defect", prev_ans, ans, buckets))
                    elif not prev_ok and ok:
                        c["n_recover_from_prev"] += 1
                        flips.append(_flip(g, a, stages[i - 1], st, "recover", prev_ans, ans, buckets))
                    if prev_ans != ans:
                        n_changes += 1
                prev_ok, prev_ans = ok, ans
            start_ok = traj[stages[0]][1]
            end_ok = traj["vote"][1]
            if start_ok is None or end_ok is None:
                continue
            pat = _pattern(start_ok, end_ok, n_changes)
            patterns[cell][pat] += 1
            patterns[cell]["_n"] += 1
            if first_defect is not None:
                patterns[cell][f"first_defect@{first_defect}"] += 1

    round_rows = []
    for key in sorted(by_round, key=lambda k: (k[0], k[1], k[4], k[2] / k[3] if k[3] else 0, k[3], _stage_ord(k[5]))):
        fam, coord, k, n, rounds, st = key
        c = by_round[key]
        graded = c["n_honest"] - c["n_ungraded"]
        round_rows.append({
            "family": fam, "coord": int(coord), "rounds_played": rounds, "k": k, "N": n, "ratio": ratio_label(k, n), "stage": st,
            "n_games": len(games_in_cell[key]), "n_honest": c["n_honest"], "n_graded": graded,
            "honest_acc": round(c["n_correct"] / graded, 4) if graded else None,
            "on_deceiver_answer_frac": round(c["n_on_deceiver_answer"] / graded, 4) if graded and k else None,
            "undecided_frac": round(c["n_undecided"] / graded, 4) if graded else None,
            "n_defect_from_prev": c["n_defect_from_prev"], "n_recover_from_prev": c["n_recover_from_prev"],
        })

    pat_names = ["always_correct", "never_correct", "converted", "defected",
                 "wavered_held", "wavered_recovered", "wavered_lost", "wavered_stayed_wrong"]
    pattern_rows = []
    for cell in sorted(patterns, key=lambda c: (c[0], c[1], c[4], c[2] / c[3] if c[3] else 0, c[3])):
        fam, coord, k, n, rounds = cell
        c = patterns[cell]
        row = {"family": fam, "coord": int(coord), "rounds_played": rounds, "k": k, "N": n, "ratio": ratio_label(k, n), "n_honest": c["_n"]}
        for p in pat_names:
            row[p] = c[p]
        for st in sorted((s for s in c if s.startswith("first_defect@")), key=lambda s: _stage_ord(s.split("@")[1])):
            row[st] = c[st]
        pattern_rows.append(row)
    return round_rows, pattern_rows, flips


def _pattern(start_ok: bool, end_ok: bool, n_changes: int) -> str:
    # n_changes counts answer changes across round0..vote; 0 or 1 change means a single move (or none)
    if start_ok and end_ok:
        return "always_correct" if n_changes == 0 else "wavered_held"
    if not start_ok and not end_ok:
        return "never_correct" if n_changes == 0 else "wavered_stayed_wrong"
    if start_ok and not end_ok:
        return "defected" if n_changes <= 1 else "wavered_lost"
    return "converted" if n_changes <= 1 else "wavered_recovered"


def _stage_ord(st: str) -> int:
    return 10_000 if st == "vote" else int(st)


def _flip(g, agent, prev_stage, stage, direction, prev_ans, ans, buckets) -> dict:
    return {
        "trial": g["trial"], "path": g["path"], "rounds_played": g["n_rounds"], "question_id": g["question_id"],
        "bucket": buckets.get(g["question_id"][:8], ""), "category": g["category"],
        "family": g["family"], "coord": int(g["coord"]), "N": g["N"], "k": g["k"], "ratio": ratio_label(g["k"], g["N"]),
        "agent": agent, "from_stage": prev_stage, "to_stage": stage, "direction": direction,
        "prev_answer": prev_ans, "new_answer": ans,
        "new_answer_is_deceiver_answer": int(ans is not None and ans in g["dec_answers"][stage]),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def print_summary(round_rows: list[dict]) -> None:
    """Honest accuracy by stage, one line per family x coord x ratio (N pooled)."""
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in round_rows:
        key = (r["family"], r["coord"], r["rounds_played"], r["ratio"])
        a = agg[key][r["stage"]]
        a[0] += r["n_graded"] * (r["honest_acc"] or 0)
        a[1] += r["n_graded"]
    stages = sorted({r["stage"] for r in round_rows}, key=_stage_ord)
    head = f"{'family':<20} {'coord':>5} {'rnds':>4} {'k/N':>5} " + " ".join(f"{('r'+s if s!='vote' else s):>6}" for s in stages)
    print(head)
    for key in sorted(agg, key=lambda k: (k[0], k[1], k[2], float(k[3]))):
        fam, coord, rounds, ratio = key
        cells = []
        for s in stages:
            num, den = agg[key][s]
            cells.append(f"{num/den:6.2f}" if den else f"{'':>6}")
        print(f"{fam:<20} {coord:>5} {rounds:>4} {ratio:>5} " + " ".join(cells))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="one or more scientifically named result directories")
    ap.add_argument("--out", default="experimental_files/analysis")
    ap.add_argument("--bucket-csv", default="experimental_files/question_selection/gemini-3.8-flash/selection.csv")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    paths = []
    for d in args.runs:
        paths += [p for p in sorted(glob.glob(os.path.join(d, "games", "*.json"))) if not p.endswith(" 2.json")]
    buckets = {}
    if args.bucket_csv and os.path.exists(args.bucket_csv):
        buckets = {r["id"][:8]: f"{r['correct_of_k']}/4" for r in csv.DictReader(open(args.bucket_csv))}

    with ProcessPoolExecutor(args.workers) as ex:
        games = [g for g in ex.map(extract, paths, chunksize=16) if g]
    print(f"{len(paths)} files, {len(games)} graded games")

    round_rows, pattern_rows, flips = build(games, buckets)
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "honest_by_round.csv"), round_rows)
    write_csv(os.path.join(args.out, "honest_trajectory_patterns.csv"), pattern_rows)
    write_csv(os.path.join(args.out, "flip_events.csv"), flips)
    print(f"wrote {len(round_rows)} round rows, {len(pattern_rows)} pattern rows, {len(flips)} flip events to {args.out}/")
    print()
    print_summary(round_rows)


if __name__ == "__main__":
    main()
