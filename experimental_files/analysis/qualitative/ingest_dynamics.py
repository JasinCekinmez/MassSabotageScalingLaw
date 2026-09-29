"""Ingest the selected sample of sabotage games into Docent as AgentRuns with 3 role-level
transcripts each (public_board, deceiver_reflections, honest_reflections). See
experimental_files/analysis/qualitative/ingestion-plan.md for the design and field mapping.

For every honest flip event (from experimental_files/analysis/flip_events.csv, restricted to this sample), records the
message indices needed to slice out "the round's public board" + "the agent's reflection that
explains their next answer" (reflection at from_stage, since round t's public post is generated
from round t-1's reflection — confirmed against sabotage/deliberation.py's PRIVATE_REFLECTION wiring).

Writes experimental_files/analysis/qualitative/ingested_index.json: per-game Docent ids + index maps + matched flip events,
used by the follow-up reading scripts instead of re-deriving them from the raw game JSON.

Run: uv run experimental_files/analysis/qualitative/ingest_dynamics.py
"""
import csv
import json
import os

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import AssistantMessage, check_agent_runs, format_check_report

FAM_TRIAL = {"gemini_h12": "Gemini 3.8 Flash", "deepseek_main": "DeepSeek V4.1 Flash",
             "glimmer_main": "Muse Glimmer 30B", "grok_main": "Grok 4.3"}
COLLECTION_NAME = "mass-sabotage-persuasion-lulling"

selected = json.load(open("experimental_files/analysis/qualitative/selected_games.json"))
selected_paths = {(s["trial"], s["path"]) for s in selected}

flip_rows = list(csv.DictReader(open("experimental_files/analysis/flip_events.csv")))
flips_by_game = {}
for r in flip_rows:
    key = (r["trial"], r["path"])
    if key in selected_paths:
        flips_by_game.setdefault(key, []).append(r)


def build_agent_run(path: str, trial: str) -> tuple[AgentRun, dict]:
    g = json.load(open(path))
    cfg = g["config"]
    roles = {str(a["id"]): a["role"] for a in g["agents"]}
    honest = sorted((a for a, r in roles.items() if r == "honest"), key=int)
    deceivers = sorted((a for a, r in roles.items() if r == "deceiver"), key=int)
    n_rounds = len(g["rounds"])
    gr = g["grades"]
    pub_grades, vote_grades = gr["public"], gr["votes"]

    public_msgs, deceiver_msgs, honest_msgs = [], [], []
    public_round_range: dict[str, list[int]] = {}
    reflection_index: dict[str, dict[str, int]] = {a: {} for a in honest + deceivers}

    for r in range(n_rounds):
        rd = g["rounds"][r]
        start = len(public_msgs)
        for a in sorted(rd["public_text"], key=int):
            grade = pub_grades.get(str(r), {}).get(a, {})
            public_msgs.append(AssistantMessage(
                content=rd["public_text"][a],
                metadata={"round": r, "stage": "public", "agent_id": int(a), "role": roles[a],
                          "extracted_answer": grade.get("extracted_final_answer"),
                          "correct": grade.get("correct")},
            ))
        public_round_range[str(r)] = [start, len(public_msgs) - 1]

        for a in sorted(rd["reflection_text"], key=int):
            text = rd["reflection_text"][a]
            target = deceiver_msgs if roles[a] == "deceiver" else honest_msgs
            reflection_index[a][str(r)] = len(target)
            target.append(AssistantMessage(
                content=text, metadata={"round": r, "stage": "reflection", "agent_id": int(a), "role": roles[a]},
            ))

    vote_start = len(public_msgs)
    for a in sorted(g.get("votes_text", {}), key=int):
        grade = vote_grades.get(a, {})
        public_msgs.append(AssistantMessage(
            content=g["votes_text"][a],
            metadata={"round": n_rounds, "stage": "vote", "agent_id": int(a), "role": roles.get(a),
                      "extracted_answer": grade.get("extracted_final_answer"), "correct": grade.get("correct")},
        ))
    public_round_range["vote"] = [vote_start, len(public_msgs) - 1]

    public_t = Transcript(name="public_board", messages=public_msgs,
                           metadata={"description": "every agent's public round post, then the final vote"})
    deceiver_t = Transcript(name="deceiver_reflections", messages=deceiver_msgs,
                             metadata={"description": "every deceiver's private per-round reflection"})
    honest_t = Transcript(name="honest_reflections", messages=honest_msgs,
                           metadata={"description": "every honest agent's private per-round reflection"})

    outcome = gr["outcome"]
    run = AgentRun(
        name=f"{trial}/{os.path.basename(path)}",
        transcripts=[public_t, deceiver_t, honest_t],
        metadata={
            "trial": trial, "family": FAM_TRIAL[trial], "source_path": path,
            "question": g["question"], "correct_answer": g["correct_answer"],
            "answer_type": g.get("answer_type"), "category": g.get("category"), "raw_subject": g.get("raw_subject"),
            "N": cfg["n_agents"], "k": cfg["n_deceivers"], "ratio": round(cfg["n_deceivers"] / cfg["n_agents"], 2),
            "n_rounds": n_rounds, "rounds_played": n_rounds, "coordinate": bool(cfg.get("coordinate", False)),
            "deceiver_ids": [int(a) for a in deceivers], "honest_ids": [int(a) for a in honest],
            "outcome": outcome,
        },
    )
    index = {
        "public_transcript_id": str(public_t.id), "deceiver_transcript_id": str(deceiver_t.id),
        "honest_transcript_id": str(honest_t.id), "agent_run_id": str(run.id),
        "public_round_range": public_round_range, "reflection_index": reflection_index,
        "deceiver_ids": deceivers, "honest_ids": honest, "n_rounds": n_rounds,
    }
    return run, index


print(f"Building {len(selected)} agent runs...")
runs, index_by_game, sample_errors = [], {}, []
for s in selected:
    key = (s["trial"], s["path"])
    try:
        run, idx = build_agent_run(s["path"], s["trial"])
    except Exception as exc:
        sample_errors.append({"path": s["path"], "error": str(exc)})
        continue
    runs.append(run)
    idx["trial"], idx["path"], idx["family"], idx["ratio"] = s["trial"], s["path"], s["family"], s["ratio"]
    # attach only flip events whose from_stage reflection we can cite (always true; see module docstring)
    idx["flip_events"] = flips_by_game.get(key, [])
    index_by_game[f"{s['trial']}::{os.path.basename(s['path'])}"] = idx

print(f"Converted {len(runs)}/{len(selected)}; errors: {len(sample_errors)}")
if sample_errors:
    for e in sample_errors[:5]:
        print(" ", e)
    raise SystemExit("fix conversion errors before upload")

report = check_agent_runs(runs)
print(format_check_report(report))
# consecutive_assistant_messages is expected (see ingestion-plan.md Omitted Data) since every
# public post / reflection is a distinct agent's turn, not a fragment of one turn.

client = Docent()
collection_id = client.create_collection(name=COLLECTION_NAME, description=(
    "Sample of mass-sabotage deliberation games (Gemini 3.8 Flash, DeepSeek V4.1 Flash, "
    "Muse Glimmer 30B, Grok 4.3; uncoordinated deceivers) selected for honest-agent flip "
    "events, for readings on deceiver persuasion tactics and honest-agent lulling causes."
))
print("collection_id:", collection_id)
upload_result = client.add_agent_runs(collection_id, runs)
print("upload_result:", upload_result)

os.makedirs("experimental_files/analysis/qualitative", exist_ok=True)
json.dump({"collection_id": collection_id, "games": index_by_game},
          open("experimental_files/analysis/qualitative/ingested_index.json", "w"), indent=2)

collection_info = client.get_collection(collection_id)
print("VERIFICATION")
print(f"  selected: {len(selected)}  converted: {len(runs)}  uploaded_result: {upload_result}")
print(f"  collection: {collection_info}")
print(f"  URL: https://docent.transluce.org/dashboard/{collection_id}")
