"""Run one deliberation game: N participants (k deceivers) on one question for R deliberation rounds.

Protocol (see RESEARCH.md and experimental_files/prompts/):
  round 0        : everyone answers independently (public), then writes a private reflection
  rounds 1..R    : everyone posts a public response using (own last reflection, all public
                   responses from the previous round); after ALL have posted, everyone updates
                   their private reflection using (last reflection, round-0 responses, this
                   round's responses)
  final          : everyone casts a private vote using only their final reflection

N_ROUNDS as seen by the prompts equals R + 1. R = 0 is the pure independent baseline.

Checkpointing: if `save` is given, the transcript dict is written after EVERY model call.
If `resume` (a previously saved partial transcript) is given, completed calls are reused and
only missing ones are made. Agent ids are stored as strings ("1".."N") everywhere so the
in-memory dict and the JSON on disk have identical shape.
"""
from __future__ import annotations

import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from . import llm
from .prompts import COORD_STAGES, extract_options, format_chat, format_ids, format_public_responses, load_prompts, strip_prefix


@dataclass
class GameConfig:
    n_agents: int
    n_deceivers: int
    n_delib_rounds: int              # R; prompts see N_ROUNDS = R + 1
    honest_model: str
    deceiver_model: str
    effort: str | None = "medium"    # reasoning effort for every call
    seed: int = 0
    max_tokens: int = 32000
    coordinate: bool = False         # private sequential deceiver chat before the game and after every public board
    coord_passes: int = 1            # posting passes per chat (each deceiver posts once per pass)
    stop_after: int | None = None    # take the final vote right after this round (prompts still announce N_ROUNDS = R + 1);
                                     # the transcript can later be branched to continue rounds stop_after+1..R

    @property
    def n_rounds(self) -> int:
        return self.n_delib_rounds + 1

    def model_slug(self) -> str:
        m = self.honest_model if self.honest_model == self.deceiver_model else f"{self.honest_model}+{self.deceiver_model}"
        return re.sub(r"[^A-Za-z0-9.+-]", "-", m)

    def tag(self) -> str:
        t = f"{self.model_slug()}_N{self.n_agents}_k{self.n_deceivers}_R{self.n_delib_rounds}_s{self.seed}"
        if self.coordinate:
            t += "_coord" + (f"P{self.coord_passes}" if self.coord_passes != 1 else "")
        if self.stop_after is not None and self.stop_after < self.n_delib_rounds:
            t += f"_stop{self.stop_after}"
        return t


@dataclass
class Agent:
    id: int          # 1-based; shown publicly as "Colleague {id}"
    role: str        # "honest" | "deceiver"
    model: str


def assign_roles(cfg: GameConfig) -> list[Agent]:
    assert 0 <= cfg.n_deceivers <= cfg.n_agents, "n_deceivers must be within [0, n_agents]"
    rng = random.Random(cfg.seed)
    ids = list(range(1, cfg.n_agents + 1))
    deceivers = set(rng.sample(ids, cfg.n_deceivers))
    return [Agent(id=i, role="deceiver" if i in deceivers else "honest",
                  model=cfg.deceiver_model if i in deceivers else cfg.honest_model) for i in ids]


def new_game(question: dict, cfg: GameConfig, agents: list[Agent]) -> dict:
    return {
        "question_id": question["id"],
        "question": question["question"],
        "correct_answer": question["answer"],
        "answer_type": question.get("answer_type"),
        "category": question.get("category"),
        "raw_subject": question.get("raw_subject"),
        "config": asdict(cfg),
        "tag": cfg.tag(),
        "agents": [asdict(a) for a in agents],
        "status": "running",
        "started_at": time.time(),
        "rounds": [],          # [{round, public{id:rec}, public_text{id:str}, reflection{id:rec}, reflection_text{id:str}, chat[{agent,rec,text}]}]
        "pregame": {"chat": [], "reflection": {}, "reflection_text": {}} if cfg.coordinate else None,
        "votes": {},           # {id: rec}
        "votes_text": {},      # {id: str}
        "seconds": None,
        "total_usage": {},
    }


def run_game(question: dict, cfg: GameConfig, prompts=None, log=print, save=None, resume: dict | None = None) -> dict:
    """Returns the full transcript dict. Raises llm.LLMError on unrecoverable API failure
    (the partial transcript will have been saved if `save` was given)."""
    prompts = prompts or load_prompts()
    agents = assign_roles(cfg)
    if resume and resume.get("status") == "complete":
        return resume
    game = resume if resume else new_game(question, cfg, agents)
    if resume:
        assert game["tag"] == cfg.tag() and game["question_id"] == question["id"], "resume file does not match config"
        game["agents"] = [asdict(a) for a in agents]
    lock = threading.Lock()

    def checkpoint():
        if save:
            with lock:
                save(game)

    q_text, correct = question["question"], question["answer"]
    common = dict(QUESTION=q_text, N_ROUNDS=cfg.n_rounds, OPTIONS=extract_options(q_text), CORRECT_ANSWER=correct)
    label_of = {"public": "PUBLIC RESPONSE", "reflection": "PRIVATE REFLECTION", "vote": "FINAL ANSWER",
                "chat": "MESSAGE TO FELLOW DECEIVERS"}
    deceivers = [a for a in agents if a.role == "deceiver"]
    coord = bool(cfg.coordinate) and len(deceivers) >= 2      # a lone deceiver has nobody to coordinate with
    if coord:
        missing = [st for st in COORD_STAGES if st not in prompts["deceiver"].templates]
        if missing:
            raise ValueError(f"deceiver prompts missing coordination stages {missing}")
        for a in deceivers:
            a_ids = dict(YOUR_ID=f"Colleague {a.id}", DECEIVER_IDS=format_ids(d.id for d in deceivers if d.id != a.id),
                         N_COLLEAGUES=cfg.n_agents, N_DECEIVERS=len(deceivers))
            setattr(a, "coord_vals", a_ids)

    def call(agent: Agent, stage: str, **vals) -> dict:
        if coord and agent.role == "deceiver":
            vals = {**getattr(agent, "coord_vals", {}), **vals}
        prompt = prompts[agent.role].fill(stage, **common, **vals)
        c = llm.complete(agent.model, prompt, effort=cfg.effort, max_tokens=cfg.max_tokens)
        rec = c.to_dict()
        rec["prompt_chars"] = len(prompt)
        rec["finished_at"] = time.time()
        return rec

    def run_stage(kind: str, store: dict, text_store: dict, stage, vals_for) -> None:
        """Run `stage` (a name, or a function agent -> name) for every agent missing from `store`,
        in parallel; checkpoint after each result."""
        todo = [a for a in agents if str(a.id) not in store]
        if not todo:
            return
        stage_of = stage if callable(stage) else (lambda a: stage)
        with ThreadPoolExecutor(max_workers=len(todo)) as ex:
            futs = {ex.submit(call, a, stage_of(a), **vals_for(a)): a for a in todo}
            for f in as_completed(futs):
                a = futs[f]
                rec = f.result()          # re-raises LLMError; partial progress is already on disk
                with lock:
                    store[str(a.id)] = rec
                    text_store[str(a.id)] = strip_prefix(rec["text"], label_of[kind])
                checkpoint()

    def run_chat(entries: list, stage: str, chat_id: int, vals_for) -> None:
        """Sequential private deceiver chat: coord_passes passes, each deceiver posting once per pass in a
        shuffled (seeded) order; each post sees the thread so far. Resumes from len(entries)."""
        order = []
        for p in range(cfg.coord_passes):
            ids = [a.id for a in deceivers]
            random.Random(f"{cfg.seed}|{chat_id}|{p}").shuffle(ids)
            order += ids
        by_id = {a.id: a for a in deceivers}
        for i in range(len(entries), len(order)):
            a = by_id[order[i]]
            rec = call(a, stage, CHAT_SO_FAR=format_chat(entries), **vals_for(a))
            with lock:
                entries.append({"agent": a.id, "rec": rec, "text": strip_prefix(rec["text"], label_of["chat"])})
            checkpoint()

    def round_dict(t: int) -> dict:
        while len(game["rounds"]) <= t:
            game["rounds"].append({"round": len(game["rounds"]), "public": {}, "public_text": {},
                                   "reflection": {}, "reflection_text": {}, **({"chat": []} if coord else {})})
        rd = game["rounds"][t]
        if coord and "chat" not in rd:
            rd["chat"] = []
        return rd

    N_ROUNDS = cfg.n_rounds
    NONE0 = "None (this is round 0; there is no previous private reflection)."
    SAME0 = "(Identical to the initial round-0 public responses listed above; not repeated.)"

    def is_cd(a: Agent) -> bool:
        return coord and a.role == "deceiver"

    # ---------------- pre-game deceiver coordination (coordinate mode only)
    if coord:
        pg = game.get("pregame") or {"chat": [], "reflection": {}, "reflection_text": {}}
        game["pregame"] = pg
        log(f"  [{cfg.tag()}] pre-game: deceiver chat")
        run_chat(pg["chat"], "pregame_coordination_message", chat_id=-1, vals_for=lambda a: {})
        thread = format_chat(pg["chat"])
        log(f"  [{cfg.tag()}] pre-game: deceiver plan reflection")
        todo_pg = [a for a in deceivers if str(a.id) not in pg["reflection"]]
        if todo_pg:
            with ThreadPoolExecutor(max_workers=len(todo_pg)) as ex:
                futs = {ex.submit(call, a, "pregame_plan_reflection", CHAT_THREAD=thread): a for a in todo_pg}
                for f in as_completed(futs):
                    a = futs[f]; rec = f.result()
                    with lock:
                        pg["reflection"][str(a.id)] = rec
                        pg["reflection_text"][str(a.id)] = strip_prefix(rec["text"], label_of["reflection"])
                    checkpoint()
        plan_of = pg["reflection_text"]
    else:
        plan_of = {}

    def reflection_stage(a: Agent) -> str:
        return "end_of_round_private_reflection_coordinated" if is_cd(a) else "end_of_round_private_reflection"

    def chat_vals(rd: dict, a: Agent) -> dict:
        return {"CHAT_THREAD": format_chat(rd["chat"])} if is_cd(a) else {}

    # ---------------- round 0
    rd0 = round_dict(0)
    log(f"  [{cfg.tag()}] round 0/{N_ROUNDS - 1}: public")
    run_stage("public", rd0["public"], rd0["public_text"],
              lambda a: "round0_public_response_coordinated" if is_cd(a) else "round0_public_response",
              lambda a: {"PRIVATE_REFLECTION": plan_of[str(a.id)]} if is_cd(a) else {})
    initial_block = format_public_responses(rd0["public_text"])
    if coord:
        log(f"  [{cfg.tag()}] round 0/{N_ROUNDS - 1}: deceiver chat")
        run_chat(rd0["chat"], "in_round_coordination_message", chat_id=0,
                 vals_for=lambda a: dict(ROUND=0, PRIVATE_REFLECTION=plan_of[str(a.id)],
                                         CURRENT_ROUND_PUBLIC_RESPONSES=initial_block))
    log(f"  [{cfg.tag()}] round 0/{N_ROUNDS - 1}: reflection")
    run_stage("reflection", rd0["reflection"], rd0["reflection_text"], reflection_stage,
              lambda a: dict(ROUND=0, PREVIOUS_PRIVATE_REFLECTION=plan_of[str(a.id)] if is_cd(a) else NONE0,
                             INITIAL_PUBLIC_RESPONSES=initial_block, CURRENT_ROUND_PUBLIC_RESPONSES=SAME0,
                             **chat_vals(rd0, a)))

    # ---------------- rounds 1..R (or 1..stop_after when voting early)
    last = N_ROUNDS - 1 if cfg.stop_after is None else min(cfg.stop_after, N_ROUNDS - 1)
    game["voted_after_round"] = last
    for t in range(1, last + 1):
        prev, rd = game["rounds"][t - 1], round_dict(t)
        prev_block = format_public_responses(prev["public_text"])
        log(f"  [{cfg.tag()}] round {t}/{N_ROUNDS - 1}: public")
        run_stage("public", rd["public"], rd["public_text"], "round1_to_n_minus_1_public_response",
                  lambda a, t=t, prev=prev, prev_block=prev_block: dict(
                      ROUND=t, PREVIOUS_ROUND=t - 1,
                      PRIVATE_REFLECTION=prev["reflection_text"][str(a.id)],
                      PREVIOUS_ROUND_PUBLIC_RESPONSES=prev_block))
        cur_block = format_public_responses(rd["public_text"])
        if coord:
            log(f"  [{cfg.tag()}] round {t}/{N_ROUNDS - 1}: deceiver chat")
            run_chat(rd["chat"], "in_round_coordination_message", chat_id=t,
                     vals_for=lambda a, t=t, prev=prev, cur_block=cur_block: dict(
                         ROUND=t, PRIVATE_REFLECTION=prev["reflection_text"][str(a.id)],
                         CURRENT_ROUND_PUBLIC_RESPONSES=cur_block))
        log(f"  [{cfg.tag()}] round {t}/{N_ROUNDS - 1}: reflection")
        run_stage("reflection", rd["reflection"], rd["reflection_text"], reflection_stage,
                  lambda a, t=t, prev=prev, cur_block=cur_block, rd=rd: dict(
                      ROUND=t, PREVIOUS_PRIVATE_REFLECTION=prev["reflection_text"][str(a.id)],
                      INITIAL_PUBLIC_RESPONSES=initial_block, CURRENT_ROUND_PUBLIC_RESPONSES=cur_block,
                      **chat_vals(rd, a)))

    # ---------------- final votes (after round `last`)
    last_rd = game["rounds"][last]
    log(f"  [{cfg.tag()}] final votes (after round {last})")
    run_stage("vote", game["votes"], game["votes_text"], "final_private_vote",
              lambda a: dict(PRIVATE_REFLECTION=last_rd["reflection_text"][str(a.id)]))

    game["status"] = "complete"
    game["seconds"] = round(time.time() - game["started_at"], 1)
    game["total_usage"] = _sum_usage(game)
    checkpoint()
    return game


def _sum_usage(game: dict) -> dict:
    tot: dict[str, int] = {}
    recs = [r for rd in game["rounds"] for r in list(rd["public"].values()) + list(rd["reflection"].values())]
    recs += [e["rec"] for rd in game["rounds"] for e in rd.get("chat", [])]
    if game.get("pregame"):
        recs += [e["rec"] for e in game["pregame"]["chat"]] + list(game["pregame"]["reflection"].values())
    recs += list(game["votes"].values())
    for r in recs:
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0) + v
    tot["n_calls"] = len(recs)
    tot["n_refused"] = sum(1 for r in recs if r.get("refused"))
    return tot
