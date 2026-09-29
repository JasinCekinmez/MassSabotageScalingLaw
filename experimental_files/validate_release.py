"""Fast structural checks for the public release tree."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experimental_files"
MANIFEST = json.loads((EXPERIMENTS / "manifest.json").read_text())
ALLOWED_AGENTS = set(MANIFEST["reported_agent_models"])
OPAQUE_NAME = re.compile(r"^trial\d+$")
CONFLICT_COPY = re.compile(r" \d+(?=\.[^.]+$)")
GENERATED_SUFFIXES = {".log", ".html", ".tmp", ".pyc"}
GENERATED_DIRS = {"__pycache__", ".ruff_cache", ".pytest_cache"}


def result_directories() -> list[Path]:
    entries: list[str] = []
    entries.extend(path for paths in MANIFEST["main_grid"].values() for path in paths)
    entries.extend(MANIFEST["coordinated_deceivers"].values())
    entries.extend(MANIFEST["heterogeneous"].values())
    return [ROOT / entry for entry in entries]


def check_manifest_paths() -> None:
    paths = result_directories()
    paths.extend(ROOT / path for path in MANIFEST["question_selection"].values())
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    assert not missing, f"manifest paths do not exist: {missing}"


def check_result_metadata() -> None:
    for directory in result_directories():
        for required in ("args.json", "games", "per_game.csv", "summary.csv"):
            assert (directory / required).exists(), f"missing {directory / required}"
        args = json.loads((directory / "args.json").read_text())
        participants = []
        if args.get("models"):
            # --models defines a homogeneous group and overrides the legacy
            # honest/deceiver defaults that older args.json files retained.
            participants.extend(model.strip() for model in args["models"].split(","))
        else:
            participants.extend(args.get(field) for field in ("honest_model", "deceiver_model") if args.get(field))
        unexpected = set(participants) - ALLOWED_AGENTS
        assert not unexpected, f"unreported participant model(s) in {directory}: {sorted(unexpected)}"
        filenames = (path.name.lower() for path in (directory / "games").glob("*.json"))
        excluded = [name for name in filenames if "claude" in name or re.search(r"(^|_)gpt-", name)]
        assert not excluded, f"excluded GPT/Claude deliberation data in {directory}: {excluded[:3]}"


def check_release_hygiene() -> None:
    bad_names = []
    conflict_copies = []
    generated = []
    oversized = []
    for path in ROOT.rglob("*"):
        if any(part in {".git", ".venv", "hle", "local_archive"} for part in path.parts):
            continue
        if OPAQUE_NAME.match(path.name):
            bad_names.append(str(path.relative_to(ROOT)))
        if path.is_file() and CONFLICT_COPY.search(path.name):
            conflict_copies.append(str(path.relative_to(ROOT)))
        if path.is_dir() and (path.name in GENERATED_DIRS or path.name.endswith(".egg-info")):
            generated.append(str(path.relative_to(ROOT)))
        if path.is_file():
            if path.suffix in GENERATED_SUFFIXES:
                generated.append(str(path.relative_to(ROOT)))
            if path.stat().st_size >= 95 * 1024 * 1024:
                oversized.append(str(path.relative_to(ROOT)))
    assert not bad_names, f"opaque trial directory names remain: {bad_names}"
    assert not conflict_copies, f"conflict-copy filenames remain: {conflict_copies}"
    assert not generated, f"generated files remain in release tree: {generated[:10]}"
    assert not oversized, f"files near GitHub's 100 MB limit: {oversized}"


def check_frozen_figure_data() -> None:
    figures = EXPERIMENTS / "analysis" / "figures"
    expected_rows = {
        "defection_by_trial.csv": 4788,
        "heterogeneous_by_trial.csv": 1711,
        "tactics_by_message.csv": 2018,
    }
    for name, expected in expected_rows.items():
        with (figures / name).open(newline="") as handle:
            rows = sum(1 for _ in csv.reader(handle)) - 1
        assert rows == expected, f"{name}: expected {expected} rows, found {rows}"


def check_statistics_analyses() -> None:
    required = [
        "main_statistics.py",
        "coordination_statistics.py",
        "heterogeneous_statistics.py",
        "behavioral_statistics.py",
        "verify_statistics.py",
        "behavioral_counts.json",
    ]
    missing = [name for name in required if not (EXPERIMENTS / "analysis" / name).is_file()]
    assert not missing, f"missing manuscript statistics analyses: {missing}"


def main() -> None:
    check_manifest_paths()
    check_result_metadata()
    check_release_hygiene()
    check_frozen_figure_data()
    check_statistics_analyses()
    print("release structure is valid")


if __name__ == "__main__":
    main()
