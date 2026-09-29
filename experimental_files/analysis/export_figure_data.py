"""Regenerate the frozen trial-level CSVs consumed by the paper figure scripts.

The release manifest is the single source of truth for result locations. Outputs
are written to ``experimental_files/analysis/figures`` and can be checked by
running the plotting scripts in that directory.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experimental_files"
OUT = Path(__file__).resolve().parent / "figures"
MANIFEST = json.loads((EXPERIMENTS / "manifest.json").read_text())

TACTICS = [
    "fabricate_or_misrepresent_evidence",
    "misleading_inference",
    "reframe_or_equivocate",
    "selective_skepticism",
    "concede_and_pivot",
    "exploit_uncertainty",
    "social_or_consensus_pressure",
    "bare_assertion",
]
MODEL_IDS = {
    "Gemini 3.8 Flash": "gemini-3.8-flash",
    "Grok 4.3": "grok-4.3",
    "DeepSeek V4.1 Flash": "together/deepseek-ai/DeepSeek-V4.1-Flash",
    "Muse Glimmer": "together/meta-models/Muse-Glimmer-30B",
}


def paths(items: list[str]) -> list[Path]:
    return [ROOT / item for item in items]


def per_game(result_dirs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path / "per_game.csv").assign(result_dir=str(path.relative_to(ROOT))) for path in result_dirs]
    data = pd.concat(frames, ignore_index=True)
    data["initially_correct"] = (data.honest_r0_acc * data.n_honest).round().astype(int)
    data["defections"] = data.n_switched.astype(int)
    return data


def buckets(model: str) -> dict[str, int]:
    selection = pd.read_csv(ROOT / MANIFEST["question_selection"][model])
    return dict(zip(selection.id, selection.correct_of_k))


def glimmer_snapshot(data: pd.DataFrame, counts: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    """Match the frozen paper snapshot by keeping the earliest completed trials per composition."""
    keep = []
    for row in counts.itertuples():
        pattern = str(result_dir / "games" / f"*_N{row.N}_k{row.k}_R*_s0_stop4.json")
        files = sorted(glob.glob(pattern), key=lambda item: Path(item).stat().st_mtime)
        prefixes = [re.match(r"([0-9a-f]+)_", Path(item).name).group(1) for item in files[: int(row.trials)]]
        subset = data[(data.N == row.N) & (data.k == row.k)]
        subset = subset[subset.question_id.str[:8].isin(prefixes)]
        assert len(subset) == row.trials, (row.N, row.k, len(subset), row.trials)
        keep.append(subset)
    return pd.concat(keep, ignore_index=True)


COLS = ["model", "question_id", "bucket", "N", "k", "initially_correct", "defections"]


def export_main_grid() -> pd.DataFrame:
    fixed = pd.read_csv(OUT / "defection_by_composition.csv")
    rows = []
    for model, entries in MANIFEST["main_grid"].items():
        data = per_game(paths(entries))
        expected = fixed[fixed.model == model]
        if model == "Muse Glimmer":
            data = glimmer_snapshot(data, expected, paths(entries)[0])
        data["bucket"] = data.question_id.map(buckets(model))
        assert data.bucket.notna().all(), model
        assert not data.duplicated(["question_id", "N", "k"]).any(), model
        data["model"] = model
        aggregate = data.groupby(["N", "k"]).agg(
            trials=("question_id", "size"),
            initially_correct=("initially_correct", "sum"),
            defections=("defections", "sum"),
        ).reset_index()
        check = aggregate.merge(expected, on=["N", "k"], suffixes=("", "_expected"))
        for column in ("trials", "initially_correct", "defections"):
            assert (check[column] == check[f"{column}_expected"]).all(), (model, column)
        rows.append(data[COLS])
    output = pd.concat(rows).sort_values(["model", "N", "k", "question_id"])
    output.to_csv(OUT / "defection_by_trial.csv", index=False)
    return output


def export_coordination(main: pd.DataFrame) -> None:
    rows = []
    for model, entry in MANIFEST["coordinated_deceivers"].items():
        independent = main[main.model == model].copy()
        coordinated = per_game([ROOT / entry])
        coordinated["bucket"] = coordinated.question_id.map(buckets(model))
        coordinated["model"] = model
        keys = ["question_id", "N", "k"]
        coordinated = coordinated[coordinated.k >= 2]
        paired = coordinated[keys].merge(independent[independent.k >= 2][keys], on=keys, validate="one_to_one")
        base = independent[independent.k == 0]
        matched_independent = paired.merge(independent, on=keys, validate="one_to_one")
        matched_coordinated = paired.merge(coordinated, on=keys, validate="one_to_one")
        for condition, data in (
            ("independent", pd.concat([base, matched_independent], ignore_index=True)),
            ("coordinated", matched_coordinated),
        ):
            data = data.copy()
            data["condition"] = condition
            rows.append(data[["model", "condition"] + COLS[1:]])
    output = pd.concat(rows).sort_values(["model", "condition", "N", "k", "question_id"])
    output.to_csv(OUT / "coordination_by_trial.csv", index=False)


def export_heterogeneous() -> None:
    labels = {
        "Gemini / DeepSeek": ("Gemini", "DeepSeek"),
        "Gemini / Grok": ("Gemini", "Grok"),
        "Glimmer / DeepSeek": ("Glimmer", "DeepSeek"),
        "Glimmer / Grok": ("Glimmer", "Grok"),
    }
    bucket_map = buckets("Gemini 3.8 Flash")
    rows = []
    for pairing, entry in MANIFEST["heterogeneous"].items():
        honest, deceiver = labels[pairing]
        data = per_game([ROOT / entry])
        data["bucket"] = data.question_id.map(bucket_map)
        data["honest_model"] = honest
        data["deceiver_model"] = deceiver
        rows.append(data[["honest_model", "deceiver_model", "question_id", "bucket", "N", "k", "initially_correct", "defections"]])
    output = pd.concat(rows).sort_values(["honest_model", "deceiver_model", "N", "k", "question_id"])
    assert not output.duplicated(["honest_model", "deceiver_model", "question_id", "N", "k"]).any()
    output.to_csv(OUT / "heterogeneous_by_trial.csv", index=False)


def export_tactics(main: pd.DataFrame) -> None:
    source = EXPERIMENTS / "analysis" / "qualitative" / "persuasion_and_coordination" / "level1_per_deceiver_round.json"
    full_ids = set(main.question_id)

    def full_question_id(path: str) -> str:
        prefix = Path(path).name.split("_")[0]
        matches = {qid for qid in full_ids if qid.startswith(prefix)}
        assert len(matches) == 1, path
        return matches.pop()

    rows = []
    for record in json.loads(source.read_text()):
        match = re.search(r"_N(\d+)_k(\d+)_", record["path"])
        present = {item["tactic"] for item in record["tactics_present"]}
        rows.append({
            "model": record["family"].replace("Muse Glimmer 30B", "Muse Glimmer"),
            "coordinated": int(record["coordinate"]),
            "question_id": full_question_id(record["path"]),
            "N": int(match.group(1)),
            "k": int(match.group(2)),
            "agent": record["agent"],
            "round": int(record["round"]),
            **{tactic: int(tactic in present) for tactic in TACTICS},
        })
    output = pd.DataFrame(rows).sort_values(["model", "coordinated", "question_id", "N", "k", "round", "agent"])
    output.to_csv(OUT / "tactics_by_message.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_grid = export_main_grid()
    export_coordination(main_grid)
    export_heterogeneous()
    export_tactics(main_grid)
    print(f"wrote figure data to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
