"""Select a stratified sample of games (per family, per k/N ratio) for the Docent ingestion,
prioritizing games with the most honest flip events (richest signal for the persuasion /
lulling readings), from the trajectories.py output.

Writes experimental_files/analysis/qualitative/selected_games.json: list of {trial, path, family, ratio, n_events}.
"""
import csv, collections, json, os

PER_FAMILY_CAP = 30
FAM_TRIAL = {'gemini_h12': 'Gemini 3.8 Flash', 'deepseek_main': 'DeepSeek V4.1 Flash',
             'glimmer_main': 'Muse Glimmer 30B', 'grok_main': 'Grok 4.3'}

rows = list(csv.DictReader(open('experimental_files/analysis/flip_events.csv')))
rows = [r for r in rows if r['trial'] in FAM_TRIAL and r['k'] != '0']

by_path = collections.defaultdict(list)
for r in rows:
    by_path[(r['trial'], r['path'], r['ratio'])].append(r)

# group candidate games by (trial, ratio), sorted by event richness within each ratio bucket
per_trial_ratio = collections.defaultdict(list)
for (trial, path, ratio), evs in by_path.items():
    per_trial_ratio[(trial, ratio)].append((len(evs), path))
for k in per_trial_ratio:
    per_trial_ratio[k].sort(reverse=True)

selected = []
for trial in FAM_TRIAL:
    ratios = sorted({k[1] for k in per_trial_ratio if k[0] == trial}, key=float)
    if not ratios:
        continue
    per_ratio_cap = max(1, PER_FAMILY_CAP // len(ratios))
    picked = []
    for ratio in ratios:
        cand = per_trial_ratio[(trial, ratio)][:per_ratio_cap]
        picked += [(trial, path, ratio, n) for n, path in cand]
    # top up to PER_FAMILY_CAP from richest remaining games across ratios if under cap
    if len(picked) < PER_FAMILY_CAP:
        rest = []
        for ratio in ratios:
            rest += [(trial, path, ratio, n) for n, path in per_trial_ratio[(trial, ratio)][per_ratio_cap:]]
        rest.sort(key=lambda x: -x[3])
        picked += rest[: PER_FAMILY_CAP - len(picked)]
    selected += picked[:PER_FAMILY_CAP]

out = [{"trial": t, "path": p, "family": FAM_TRIAL[t], "ratio": r, "n_flip_events": n} for t, p, r, n in selected]
os.makedirs('experimental_files/analysis/qualitative', exist_ok=True)
json.dump(out, open('experimental_files/analysis/qualitative/selected_games.json', 'w'), indent=2)

c = collections.Counter(o["family"] for o in out)
print(f"selected {len(out)} games:", dict(c))
for t in FAM_TRIAL:
    fam_rows = [o for o in out if o["trial"] == t]
    print(f"  {t}: ratios {sorted({o['ratio'] for o in fam_rows}, key=float)}, "
          f"events/game min={min(o['n_flip_events'] for o in fam_rows)} "
          f"max={max(o['n_flip_events'] for o in fam_rows)}")
