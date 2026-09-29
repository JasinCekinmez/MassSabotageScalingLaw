"""Sample defect flip events (honest agent correct -> incorrect) from the 120 ingested games for
Reading B, stratified by family, deterministic given SEED. Writes flip_sample.json alongside this
script: each entry carries everything reading_b_lulling.py needs to build a TranscriptSliceRef pair
(the round's public board + the agent's own reflection that produced the next answer) without
re-deriving indices from the raw game JSON.
"""
import json
import os
import random

SEED = 42
PER_FAMILY = 50

here = os.path.dirname(__file__)
idx = json.load(open(os.path.join(os.path.dirname(here), "ingested_index.json")))
collection_id = idx["collection_id"]

by_family = {}
for key, g in idx["games"].items():
    for e in g["flip_events"]:
        if e["direction"] != "defect":
            continue
        from_stage = e["from_stage"]
        ridx = g["reflection_index"].get(e["agent"], {}).get(from_stage)
        prange = g["public_round_range"].get(from_stage)
        if ridx is None or prange is None:
            continue  # shouldn't happen for defect events (from_stage is always a round index)
        by_family.setdefault(g["family"], []).append({
            "game_key": key, "trial": g["trial"], "path": g["path"],
            "agent_run_id": g["agent_run_id"], "public_transcript_id": g["public_transcript_id"],
            "honest_transcript_id": g["honest_transcript_id"], "family": g["family"], "ratio": g["ratio"],
            "deceiver_ids": g["deceiver_ids"],
            "agent": e["agent"], "from_stage": from_stage, "to_stage": e["to_stage"],
            "prev_answer": e["prev_answer"], "new_answer": e["new_answer"],
            "new_answer_is_deceiver_answer": bool(int(e["new_answer_is_deceiver_answer"])),
            "public_round_range": prange, "reflection_idx": ridx,
        })

rng = random.Random(SEED)
sample = []
for fam, events in by_family.items():
    events = sorted(events, key=lambda e: (e["game_key"], e["agent"], e["from_stage"]))  # stable order before sampling
    rng.shuffle(events)
    sample += events[:PER_FAMILY]

json.dump({"collection_id": collection_id, "events": sample},
          open(os.path.join(here, "flip_sample.json"), "w"), indent=2)

import collections
print(f"sampled {len(sample)} defect events:", dict(collections.Counter(e["family"] for e in sample)))
print("available pool per family:", {k: len(v) for k, v in by_family.items()})
