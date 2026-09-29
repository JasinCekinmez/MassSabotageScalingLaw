"""Same stratified-by-flip-events selection as select_games.py, applied to the two coordinated
trials (8 = Gemini 3.8 Flash, 10 = Grok 4.3), so the coordination comparison uses a comparably
selected sample to the uncoordinated one.
"""
import csv, collections, json, os

PER_TRIAL_CAP = 30
FAM_TRIAL = {"gemini_coordinated": "Gemini 3.8 Flash", "grok_coordinated": "Grok 4.3"}

rows = list(csv.DictReader(open("experimental_files/analysis/flip_events.csv")))
rows = [r for r in rows if r["trial"] in FAM_TRIAL and r["k"] != "0"]

by_path = collections.defaultdict(list)
for r in rows:
    by_path[(r["trial"], r["path"], r["ratio"])].append(r)

per_trial_ratio = collections.defaultdict(list)
for (trial, path, ratio), evs in by_path.items():
    per_trial_ratio[(trial, ratio)].append((len(evs), path))
for k in per_trial_ratio:
    per_trial_ratio[k].sort(reverse=True)

selected = []
for trial in FAM_TRIAL:
    ratios = sorted({k[1] for k in per_trial_ratio if k[0] == trial}, key=float)
    per_ratio_cap = max(1, PER_TRIAL_CAP // len(ratios))
    picked = []
    for ratio in ratios:
        cand = per_trial_ratio[(trial, ratio)][:per_ratio_cap]
        picked += [(trial, path, ratio, n) for n, path in cand]
    if len(picked) < PER_TRIAL_CAP:
        rest = []
        for ratio in ratios:
            rest += [(trial, path, ratio, n) for n, path in per_trial_ratio[(trial, ratio)][per_ratio_cap:]]
        rest.sort(key=lambda x: -x[3])
        picked += rest[: PER_TRIAL_CAP - len(picked)]
    selected += picked[:PER_TRIAL_CAP]

out = [{"trial": t, "path": p, "family": FAM_TRIAL[t] + " (coordinated)", "ratio": r, "n_flip_events": n}
       for t, p, r, n in selected]
here = os.path.dirname(__file__)
json.dump(out, open(os.path.join(here, "selected_coordinated_games.json"), "w"), indent=2)
c = collections.Counter(o["family"] for o in out)
print(f"selected {len(out)} coordinated games:", dict(c))
