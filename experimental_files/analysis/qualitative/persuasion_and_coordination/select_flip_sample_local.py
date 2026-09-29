"""Sample honest 'defect' flip events (correct -> incorrect) directly from experimental_files/analysis/flip_events.csv,
for both the four uncoordinated model families and the Gemini/Grok coordinated conditions -- fully local, no
Docent dependency. Reuses the SAME 30-game selections already made (selected_games.json,
selected_coordinated_games.json) so the flip sample is drawn from a set of games we've already
looked at closely, not a fresh unrelated sample.
"""
import csv
import collections
import json
import os
import random

SEED = 42
PER_GROUP = 50  # per family (uncoordinated) or per coordinated trial

HERE = os.path.dirname(__file__)
sel_paths = set()
for f in ["selected_games.json"]:
    for s in json.load(open(os.path.join(os.path.dirname(HERE), f))):
        sel_paths.add((s["trial"], s["path"]))
for f in ["selected_coordinated_games.json"]:
    for s in json.load(open(os.path.join(HERE, f))):
        sel_paths.add((s["trial"], s["path"]))

FAM = {"gemini_h12": "Gemini 3.8 Flash", "deepseek_main": "DeepSeek V4.1 Flash",
       "glimmer_main": "Muse Glimmer 30B", "grok_main": "Grok 4.3",
       "gemini_coordinated": "Gemini 3.8 Flash (coordinated)", "grok_coordinated": "Grok 4.3 (coordinated)"}

rows = list(csv.DictReader(open("experimental_files/analysis/flip_events.csv")))
rows = [r for r in rows if r["direction"] == "defect" and (r["trial"], r["path"]) in sel_paths
        and r["from_stage"] in ("0", "1")]  # scoped to round 0/1, where ~half of all defections happen

by_group = collections.defaultdict(list)
for r in rows:
    by_group[FAM[r["trial"]]].append(r)

rng = random.Random(SEED)
sample = []
for fam, evs in by_group.items():
    evs = sorted(evs, key=lambda e: (e["path"], e["agent"], e["from_stage"]))
    rng.shuffle(evs)
    sample += [dict(e, family=fam, coordinate=("coordinated" in fam)) for e in evs[:PER_GROUP]]

out_path = os.path.join(HERE, "flip_sample_local.json")
json.dump(sample, open(out_path, "w"), indent=2)
c = collections.Counter(e["family"] for e in sample)
print(f"sampled {len(sample)} defect events:", dict(c))
print("available pool per group:", {k: len(v) for k, v in by_group.items()})
