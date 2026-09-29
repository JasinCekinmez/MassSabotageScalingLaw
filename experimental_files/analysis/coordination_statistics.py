"""Reproduce the manuscript's coordinated-deceiver comparisons.

Run from any directory:
  python experimental_files/analysis/coordination_statistics.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "figures/coordination_by_trial.csv"


def pooled_rate(frame: pd.DataFrame) -> float:
    return 100 * frame.defections.sum() / frame.initially_correct.sum()


def paired_permutation(frame: pd.DataFrame, seed: int, permutations: int) -> dict:
    paired = frame.pivot(
        index=["question_id", "N", "k"],
        columns="condition",
        values=["defections", "initially_correct"],
    ).dropna()
    by_question = paired.groupby(level="question_id").sum()
    independent = by_question[[
        ("defections", "independent"), ("initially_correct", "independent")
    ]].to_numpy()
    coordinated = by_question[[
        ("defections", "coordinated"), ("initially_correct", "coordinated")
    ]].to_numpy()
    observed = 100 * (
        coordinated[:, 0].sum() / coordinated[:, 1].sum()
        - independent[:, 0].sum() / independent[:, 1].sum()
    )
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, permutations, 1000):
        count = min(1000, permutations - start)
        swaps = rng.integers(0, 2, size=(count, len(by_question)))
        perm_independent = independent.sum(axis=0) + swaps @ (coordinated - independent)
        perm_coordinated = coordinated.sum(axis=0) - swaps @ (coordinated - independent)
        differences = 100 * (
            perm_coordinated[:, 0] / perm_coordinated[:, 1]
            - perm_independent[:, 0] / perm_independent[:, 1]
        )
        extreme += int(np.sum(np.abs(differences) >= abs(observed) - 1e-12))
    return {
        "matched_questions": len(by_question),
        "delta_percentage_points": float(observed),
        "two_sided_p": (extreme + 1) / (permutations + 1),
    }


def coordinated_r_squared(frame: pd.DataFrame) -> float:
    baseline = frame[(frame.condition == "independent") & (frame.k == 0)]
    coordinated = frame[frame.condition == "coordinated"]
    data = pd.concat([baseline, coordinated]).drop_duplicates(["question_id", "N", "k"])
    data = data.assign(proportion=data.k / data.N)
    cells = data.groupby("proportion", as_index=False).agg(
        defections=("defections", "sum"), initially_correct=("initially_correct", "sum")
    )
    x = cells.proportion.to_numpy()
    y = (cells.defections / cells.initially_correct).to_numpy()
    weights = cells.initially_correct.to_numpy()
    design = np.column_stack([np.ones(len(cells)), x])
    beta = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    fitted = design @ beta
    center = np.average(y, weights=weights)
    return float(1 - np.sum(weights * (y - fitted) ** 2) / np.sum(weights * (y - center) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=HERE / "coordination_statistics.json")
    parser.add_argument("--permutations", type=int, default=49999)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    result = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "permutations": args.permutations,
        "seed": args.seed,
        "models": {},
    }
    for model, frame in data.groupby("model", sort=False):
        nonzero = frame[frame.k > 0]
        rates = {
            condition: pooled_rate(part)
            for condition, part in nonzero.groupby("condition", sort=False)
        }
        comparison = paired_permutation(nonzero, args.seed, args.permutations)
        comparison["rates_percent"] = rates
        comparison["coordinated_weighted_r_squared"] = coordinated_r_squared(frame)
        result["models"][model] = comparison
        print(
            f"{model}: {rates['independent']:.1f}% -> {rates['coordinated']:.1f}%, "
            f"delta={comparison['delta_percentage_points']:.2f}, "
            f"p={comparison['two_sided_p']:.5f}, "
            f"coordinated R^2={comparison['coordinated_weighted_r_squared']:.2f}"
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
