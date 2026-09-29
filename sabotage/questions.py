"""Select HLE questions for deliberation games."""
from __future__ import annotations

import json
import os

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "hle", "data", "data", "test-00000-of-00001.parquet")

COLS = ["id", "question", "answer", "answer_type", "category", "raw_subject", "rationale"]


def load_hle(text_only: bool = True) -> dict[str, dict]:
    t = pq.read_table(PARQUET, columns=COLS + ["image"]).to_pylist()
    out = {}
    for r in t:
        if text_only and r["image"]:
            continue
        r.pop("image", None)
        out[r["id"]] = r
    return out


def from_probe(model: str, path: str, correct: bool = True) -> list[str]:
    """Question IDs that ``model`` answered correctly (or incorrectly) in a probe-results JSON file."""
    res = json.load(open(path))
    want = "yes" if correct else "no"
    return [qid for qid, q in res.items()
            if q["responses"].get(model, {}).get("judge", {}).get("correct") == want]


def select(ids: list[str] | None = None, ids_file: str | None = None,
           probe_model: str | None = None, probe_correct: bool = True,
           probe_results: str | None = None,
           limit: int | None = None) -> list[dict]:
    hle = load_hle()
    chosen: list[str] = []
    if ids:
        chosen += ids
    if ids_file:
        data = json.load(open(ids_file))
        chosen += data if isinstance(data, list) else list(data)
    if probe_model:
        if not probe_results:
            raise ValueError("probe_results is required when probe_model is set")
        chosen += from_probe(probe_model, probe_results, probe_correct)
    seen, ordered = set(), []
    for qid in chosen:
        full = next((k for k in hle if k.startswith(qid)), None)
        if full is None:
            raise KeyError(f"question id {qid!r} not found among text-only HLE questions")
        if full not in seen:
            seen.add(full); ordered.append(full)
    if limit:
        ordered = ordered[:limit]
    return [hle[q] for q in ordered]
