"""Reproduce the main-grid statistics reported in the manuscript.

Run from any directory:
  python experimental_files/analysis/main_statistics.py

The input is regenerated from the released per-game result tables by
``export_figure_data.py``. No model or API calls are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM


HERE = Path(__file__).resolve().parent
INPUT = HERE / "figures/defection_by_trial.csv"
MODEL_ORDER = ["Gemini 3.8 Flash", "Grok 4.3", "DeepSeek V4.1 Flash", "Muse Glimmer"]


def weighted_line(frame: pd.DataFrame) -> dict:
    cells = frame.groupby("proportion", as_index=False).agg(
        defections=("defections", "sum"), initially_correct=("initially_correct", "sum")
    )
    cells["rate"] = cells.defections / cells.initially_correct
    x = cells.proportion.to_numpy()
    y = cells.rate.to_numpy()
    weights = cells.initially_correct.to_numpy()
    design = np.column_stack([np.ones(len(cells)), x])
    beta = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    fitted = design @ beta
    center = np.average(y, weights=weights)
    r2 = 1 - np.sum(weights * (y - fitted) ** 2) / np.sum(weights * (y - center) ** 2)
    return {
        "rates_percent": {f"{level:.8g}": float(100 * rate) for level, rate in zip(x, y)},
        "slope_percentage_points_per_0.1": float(10 * beta[1]),
        "weighted_r_squared": float(r2),
    }


def permutation_p(frame: pd.DataFrame, seed: int, permutations: int) -> float:
    # Collapse configurations that share k/N, then permute the four proportion
    # labels within question. The WLS weights are initially-correct agents.
    data = frame.groupby(["question_id", "proportion"], as_index=False).agg(
        defections=("defections", "sum"), initially_correct=("initially_correct", "sum")
    )
    x = data.proportion.to_numpy()
    d = data.defections.to_numpy(dtype=float)
    c = data.initially_correct.to_numpy(dtype=float)
    groups = [indices.to_numpy() for _, indices in data.groupby("question_id").groups.items()]

    def slope(labels: np.ndarray) -> float:
        center = np.sum(c * labels) / np.sum(c)
        return np.sum((labels - center) * d) / np.sum(c * (labels - center) ** 2)

    observed = slope(x)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        shuffled = x.copy()
        for indices in groups:
            shuffled[indices] = rng.permutation(shuffled[indices])
        extreme += abs(slope(shuffled)) >= abs(observed) - 1e-15
    return (extreme + 1) / (permutations + 1)


def design(frame: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    strata = frame.model + ":bucket=" + frame.bucket.astype(str)
    return pd.concat([pd.get_dummies(strata, dtype=float), frame[terms]], axis=1)


def fit(frame: pd.DataFrame, terms: list[str]):
    response = np.column_stack([frame.defections, frame.initially_correct - frame.defections])
    fitted = GLM(response, design(frame, terms), family=Binomial()).fit(maxiter=500, tol=1e-10)
    assert fitted.converged
    return fitted


def model_comparison(data: pd.DataFrame) -> dict:
    count = fit(data, ["k", "log2_N"])
    proportion = fit(data, ["proportion", "log2_N"])
    combined = fit(data, ["k", "proportion", "log2_N"])
    add_proportion = float(count.deviance - combined.deviance)
    add_count = float(proportion.deviance - combined.deviance)
    per_model = {}
    for model, frame in data.groupby("model", sort=False):
        # Use model-specific bucket intercepts through the same design helper.
        count_model = fit(frame, ["k", "log2_N"])
        proportion_model = fit(frame, ["proportion", "log2_N"])
        per_model[model] = {
            "count_minus_proportion_deviance": float(count_model.deviance - proportion_model.deviance)
        }
    return {
        "count_deviance": float(count.deviance),
        "proportion_deviance": float(proportion.deviance),
        "count_minus_proportion_deviance": float(count.deviance - proportion.deviance),
        "add_proportion_to_count": {"likelihood_ratio": add_proportion, "p": float(chi2.sf(add_proportion, 1))},
        "add_count_to_proportion": {"likelihood_ratio": add_count, "p": float(chi2.sf(add_count, 1))},
        "per_model": per_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=HERE / "main_statistics.json")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data["proportion"] = data.k / data.N
    data["log2_N"] = np.log2(data.N)
    trends = {}
    for model in MODEL_ORDER:
        frame = data[data.model == model]
        trend = weighted_line(frame)
        trend["within_question_permutation_p"] = permutation_p(
            frame, seed=args.seed, permutations=args.permutations
        )
        trends[model] = trend
        print(
            f"{model}: slope={trend['slope_percentage_points_per_0.1']:.1f}, "
            f"p={trend['within_question_permutation_p']:.4f}, "
            f"R^2={trend['weighted_r_squared']:.2f}"
        )

    comparison = model_comparison(data[data.initially_correct > 0].copy())
    result = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "permutations": args.permutations,
        "seed": args.seed,
        "trends": trends,
        "proportion_vs_count": comparison,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "count - proportion deviance:",
        f"{comparison['count_minus_proportion_deviance']:.1f}",
    )
    print(
        "add proportion to count:",
        f"LR={comparison['add_proportion_to_count']['likelihood_ratio']:.1f}, "
        f"p={comparison['add_proportion_to_count']['p']:.3g}",
    )
    print(
        "add count to proportion:",
        f"LR={comparison['add_count_to_proportion']['likelihood_ratio']:.2f}, "
        f"p={comparison['add_count_to_proportion']['p']:.2f}",
    )


if __name__ == "__main__":
    main()
