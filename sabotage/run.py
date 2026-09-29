"""Grid runner: questions x N x k x R x seeds, resumable, parallel across games.

Example (do not run until ready):
  python -m sabotage.run --out results/pilot \
      --ids-file experimental_files/question_selection/gemini-3.8-flash/selected.json \
      --models gemini-3.8-flash --groups 4+0,4+2,4+3 \
      --rounds 7 --seeds 0 --parallel-games 4

Each game is checkpointed to <out>/games/<qid8>_<model>_N{N}_k{k}_R{R}_s{seed}.json after EVERY
model call (status "running"), then graded (status "complete" + grades). Re-running resumes
partial games from their last completed call and skips finished ones.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import judge, llm, questions
from .deliberation import GameConfig, run_game
from .prompts import load_prompts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def game_path(out: str, qid: str, cfg: GameConfig) -> str:
    return os.path.join(out, "games", f"{qid[:8]}_{cfg.tag()}.json")


def _done(path: str, need_dynamics: bool) -> bool:
    if not os.path.exists(path):
        return False
    try:
        g = json.load(open(path))
    except json.JSONDecodeError:
        return False
    if g.get("status") != "complete" or "grades" not in g or "outcome" not in g["grades"]:
        return False
    if need_dynamics and "dynamics" not in g["grades"]:
        return False
    return True


def _save(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def build_grid(args) -> list[tuple[dict, GameConfig]]:
    qs = questions.select(ids=_split(args.ids), ids_file=args.ids_file,
                          probe_model=args.probe_model, probe_correct=not args.probe_wrong,
                          probe_results=args.probe_results, limit=args.limit)
    if not qs:
        sys.exit("no questions selected (use --ids, --ids-file or --probe-model)")
    if args.groups:   # "4+0,4+2" = (honest, deceivers) pairs
        shapes = []
        for g in args.groups.split(","):
            h, d = g.split("+")
            shapes.append((int(h) + int(d), int(d)))
    else:
        shapes = [(N, k) for N in _ints(args.n_agents) for k in _ints(args.n_deceivers) if k <= N]
    if args.models:   # homogeneous groups, one grid per model
        model_pairs = [(m, m) for m in _split(args.models)]
    else:
        model_pairs = [(args.honest_model, args.deceiver_model or args.honest_model)]
    grid = []
    for q, (N, k), R, s, (hm, dm) in itertools.product(qs, shapes, _ints(args.rounds), _ints(args.seeds), model_pairs):
        grid.append((q, GameConfig(n_agents=N, n_deceivers=k, n_delib_rounds=R,
                                   honest_model=hm, deceiver_model=dm,
                                   effort=None if args.effort == "none" else args.effort,
                                   seed=s, max_tokens=args.max_tokens,
                                   coordinate=args.coordinate, coord_passes=args.coord_passes,
                                   stop_after=args.stop_after)))
    return grid


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def estimate_calls(grid) -> int:
    n = 0
    for _, cfg in grid:
        rounds = cfg.n_rounds if cfg.stop_after is None else min(cfg.stop_after, cfg.n_delib_rounds) + 1
        n += cfg.n_agents * (2 * rounds + 1)
        if cfg.coordinate and cfg.n_deceivers >= 2:   # pre-game chat + plan, then one chat pass per round
            n += cfg.n_deceivers * (cfg.coord_passes + 1) + cfg.n_deceivers * cfg.coord_passes * rounds
    return n


def branch_sibling(branch_dir: str, qid: str, cfg: GameConfig, complete_only: bool = False) -> dict | None:
    """Transcript in branch_dir for the same question and config except stop_after (own tag excluded when it is
    complete_only, since that case is handled by _done). Returns the one with the most rounds."""
    import glob as _glob, re as _re
    base = _re.sub(r"_stop\d+$", "", cfg.tag())
    cands = [p for p in _glob.glob(os.path.join(branch_dir, "games", f"{qid[:8]}_{base}*.json")) if " 2.json" not in p]
    cands = [p for p in cands if _re.fullmatch(rf"{_re.escape(qid[:8])}_{_re.escape(base)}(_stop\d+)?\.json", os.path.basename(p))]
    best = None
    for p in cands:
        try:
            g = json.load(open(p))
        except json.JSONDecodeError:
            continue
        if g.get("question_id") != qid or g.get("tag") == cfg.tag():
            continue
        if complete_only and not (g.get("status") == "complete" and "outcome" in (g.get("grades") or {})):
            continue
        if best is None or len(g["rounds"]) > len(best["rounds"]):
            best = g
    return best


def complete_rounds(g: dict, cfg: GameConfig) -> int:
    """Number of leading rounds whose public and reflection stages are complete for every agent."""
    n = 0
    for rd in g["rounds"]:
        if len(rd["public"]) == cfg.n_agents and len(rd["reflection"]) == cfg.n_agents:
            n += 1
        else:
            break
    return n


def branch_seed(branch_dir: str, qid: str, cfg: GameConfig) -> dict | None:
    """Seed a partial transcript for cfg from an UNFINISHED sibling transcript (other stop point): keep pregame +
    rounds up to cfg's last round, drop votes and grades. Finished siblings are never touched (see _settled)."""
    from dataclasses import asdict
    best = branch_sibling(branch_dir, qid, cfg)
    if best is None or (best.get("status") == "complete" and "outcome" in (best.get("grades") or {})):
        return None
    last = cfg.n_delib_rounds if cfg.stop_after is None else min(cfg.stop_after, cfg.n_delib_rounds)
    seeded = dict(best)
    keep = min(last + 1, max(complete_rounds(best, cfg), 0) + 1)   # complete rounds plus at most one partial round
    seeded["rounds"] = [dict(rd) for rd in best["rounds"][:min(last + 1, len(best["rounds"]))][:keep]]
    seeded["votes"], seeded["votes_text"] = {}, {}
    seeded.pop("grades", None); seeded.pop("voted_after_round", None)
    seeded["status"] = "running"; seeded["seconds"] = None; seeded["total_usage"] = {}
    seeded["config"] = asdict(cfg); seeded["tag"] = cfg.tag()
    seeded["branched_from"] = {"tag": best["tag"], "rounds_copied": len(seeded["rounds"])}
    return seeded


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("question selection")
    src.add_argument("--ids", help="comma-separated HLE ids (prefixes ok)")
    src.add_argument("--ids-file", help="JSON list of HLE ids")
    src.add_argument("--probe-model", help="use questions this model answered correctly in --probe-results")
    src.add_argument("--probe-results", help="probe-results JSON used with --probe-model")
    src.add_argument("--probe-wrong", action="store_true", help="...answered INcorrectly instead")
    src.add_argument("--limit", type=int, help="cap number of questions")
    g = ap.add_argument_group("grid")
    g.add_argument("--n-agents", default="5", help="comma list of TOTAL group sizes, e.g. 3,5,9")
    g.add_argument("--n-deceivers", default="0,1", help="comma list, e.g. 0,1,2")
    g.add_argument("--groups", default=None, help="alternative: honest+deceivers pairs, e.g. 4+0,4+2 (overrides --n-agents/--n-deceivers)")
    g.add_argument("--rounds", default="1,5,10", help="deliberation rounds after round 0; comma list")
    g.add_argument("--seeds", default="0", help="comma list; seed sets which agent ids are deceivers")
    g.add_argument("--coordinate", action="store_true",
                   help="private sequential deceiver chat before the game and after every public board (deceiver prompts *_coordinated)")
    g.add_argument("--coord-passes", type=int, default=1, help="posting passes per deceiver chat (default 1)")
    g.add_argument("--stop-after", type=int, default=None,
                   help="take the final vote right after this round while prompts still announce --rounds; game tag gets _stopN")
    g.add_argument("--branch-from", default=None,
                   help="directory whose games share this grid's questions/config except stop-after; a new game is seeded from the "
                        "matching transcript there (truncated or extended), votes re-taken")
    m = ap.add_argument_group("models")
    m.add_argument("--models", default=None, help="comma list; runs a homogeneous group per model (overrides --honest-model/--deceiver-model)")
    m.add_argument("--honest-model", default=None, help="model for honest agents (required unless --models is set)")
    m.add_argument("--deceiver-model", default=None, help="defaults to --honest-model")
    m.add_argument("--effort", default="medium", choices=["low", "medium", "high", "none"])
    m.add_argument("--max-tokens", type=int, default=32000)
    m.add_argument("--judge-model", default=judge.JUDGE_MODEL)
    m.add_argument("--judge-concurrency", type=int, default=24, help="max concurrent judge calls (process-wide)")
    m.add_argument("--judge-provider", default="openai", choices=["openai", "litellm"], help="route judge calls direct to OpenAI or through a LiteLLM proxy")
    r = ap.add_argument_group("execution")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--parallel-games", type=int, default=4, help="games run concurrently")
    r.add_argument("--gemini-litellm-share", type=float, default=0.0,
                   help="fraction of Gemini calls to route through the LiteLLM proxy (e.g. 0.5); model name stays the same")
    r.add_argument("--concurrency", default=None,
                   help="per-provider in-flight caps, e.g. anthropic=48,xai=48,openai=24,gemini=16")
    r.add_argument("--dynamics", action="store_true", help="also run challenge/contamination audits (more judge calls)")
    r.add_argument("--dry-run", action="store_true", help="print the grid and call estimate, then exit")
    r.add_argument("--force", action="store_true", help="re-run games even if already saved")
    args = ap.parse_args(argv)

    if not args.models and not args.honest_model:
        ap.error("one of --models or --honest-model is required")

    judge.JUDGE_MODEL = args.judge_model
    judge.set_judge_concurrency(args.judge_concurrency)
    judge.set_judge_provider(args.judge_provider)
    if args.gemini_litellm_share:
        llm.set_route_split("gemini", args.gemini_litellm_share)
    if args.concurrency:
        for kv in args.concurrency.split(","):
            prov, n = kv.split("=")
            llm.set_concurrency(prov.strip(), int(n))
    grid = build_grid(args)
    def _settled(q, c):
        """Done at this config, or (with --branch-from) already complete at the same config with another stop point:
        finished transcripts are never truncated or re-voted."""
        if _done(game_path(args.out, q["id"], c), args.dynamics):
            return True
        if args.branch_from:
            sib = branch_sibling(args.branch_from, q["id"], c, complete_only=True)
            return sib is not None
        return False
    todo = [(q, c) for q, c in grid if args.force or not _settled(q, c)]
    n_calls = estimate_calls(todo)
    print(f"{len(grid)} games in grid, {len(todo)} to run, ~{n_calls} agent LLM calls "
          f"(+ ~{sum(c.n_agents * (c.n_rounds + 1) for _, c in todo)} judge calls)", flush=True)
    if args.dry_run:
        for q, c in todo[:50]:
            print(f"  {q['id'][:8]} {c.tag()}  [{q['category']}]")
        if len(todo) > 50:
            print(f"  ... {len(todo) - 50} more")
        return

    os.makedirs(os.path.join(args.out, "games"), exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        eff = dict(vars(args))   # record only what took effect: --groups overrides --n-agents/--n-deceivers, --models overrides --honest-model/--deceiver-model
        if args.groups:
            eff["n_agents"] = eff["n_deceivers"] = None
        if args.models:
            eff["honest_model"] = eff["deceiver_model"] = None
        json.dump(eff, f, indent=1)
    prompts = load_prompts()
    logf = open(os.path.join(args.out, "run.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    def one(q, cfg):
        path = game_path(args.out, q["id"], cfg)
        try:
            partial = None
            if os.path.exists(path) and not args.force:
                try:
                    partial = json.load(open(path))   # complete or partial transcript
                except json.JSONDecodeError:
                    partial = None
            if partial is None and args.branch_from:
                sib = branch_sibling(args.branch_from, q["id"], cfg)
                if sib is not None and cfg.stop_after is not None:
                    # a transcript already past the stop round is polled where it stands, never rolled back
                    last_idx = complete_rounds(sib, cfg) - 1
                    if last_idx > cfg.stop_after:
                        from dataclasses import replace as _replace
                        cfg = _replace(cfg, stop_after=min(last_idx, cfg.n_delib_rounds))
                        path = game_path(args.out, q["id"], cfg)
                        if os.path.exists(path):
                            try:
                                partial = json.load(open(path))
                            except json.JSONDecodeError:
                                partial = None
                if partial is None:
                    partial = branch_seed(args.branch_from, q["id"], cfg)
                    if partial:
                        _save(path, partial)
                        log(f"  [{cfg.tag()}] branched from {partial['branched_from']['tag']} ({partial['branched_from']['rounds_copied']} rounds copied)")
            game = run_game(q, cfg, prompts, log=log, save=lambda g: _save(path, g), resume=partial)
            judge.grade_game(game, dynamics=args.dynamics, save=lambda g: _save(path, g))
            _save(path, game)
            o = game["grades"]["outcome"]
            return path, f"OK  {q['id'][:8]} {cfg.tag()}  honest votes correct {o['votes_correct_honest']}/{o['n_honest']}  majority_all={o['majority_correct_all']}  {game.get('seconds')}s"
        except Exception as e:  # noqa: BLE001
            return path, f"FAIL {q['id'][:8]} {cfg.tag()}  {type(e).__name__}: {str(e)[:300]}\n{traceback.format_exc()[-800:]}"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel_games) as ex:
        futs = [ex.submit(one, q, c) for q, c in todo]
        for i, f in enumerate(as_completed(futs), 1):
            _, msg = f.result()
            log(f"[{i}/{len(todo)}] {msg}")
    log(f"done in {(time.time() - t0) / 60:.1f} min")
    logf.close()
    from .analyze import summarize
    summarize(args.out)


if __name__ == "__main__":
    main()
