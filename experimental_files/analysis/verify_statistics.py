"""Run every manuscript statistics analysis and verify its headline values."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(name: str) -> None:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, str(HERE / name)], cwd=ROOT, env=env, check=True)


def load(name: str):
    return json.loads((HERE / name).read_text())


def close(actual, expected, tolerance=0.051):
    assert np.allclose(actual, expected, atol=tolerance), (actual, expected)


def compare_tree(actual, expected, path="root"):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, f"{path}: missing {key}"
            compare_tree(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: different lengths"
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            compare_tree(got, wanted, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert np.isclose(actual, expected, rtol=1e-9, atol=1e-11), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def main() -> None:
    run("export_figure_data.py")
    for script in (
        "main_statistics.py",
        "coordination_statistics.py",
        "heterogeneous_statistics.py",
        "behavioral_statistics.py",
    ):
        run(script)

    expected_hashes = {
        "coordination_by_trial.csv": "34318e99ffb55f3a8caaf6ca4ec236a18a06c7de5d00f29f3785af9d465bca6c",
        "defection_by_composition.csv": "424c6b222c337e710744d79166fbcc91315f7924d98d41dd595b4703bdbfc2b1",
        "defection_by_trial.csv": "b489d7ab9d3a8a570b91ae8dc643513114720fde99437211475127958b5c6974",
        "heterogeneous_by_trial.csv": "a5214da644c6283d0a464cb0115982c4ad087a483bf31f11cfa91fc78e6b3307",
        "tactics_by_message.csv": "b9e2b173021bcfff5984d76b7be441cd10bb3ec9997df970499b60db513d0227",
    }
    for name, expected in expected_hashes.items():
        actual = hashlib.sha256((HERE / "figures" / name).read_bytes()).hexdigest()
        assert actual == expected, (name, actual, expected)

    main_stats = load("main_statistics.json")
    trends = main_stats["trends"]
    close([trends[name]["slope_percentage_points_per_0.1"] for name in trends], [5.3, 4.2, 2.1, 5.7])
    close([trends[name]["weighted_r_squared"] for name in trends], [0.97, 0.90, 0.82, 0.97])
    comparison = main_stats["proportion_vs_count"]
    close(comparison["count_minus_proportion_deviance"], 61.5)
    close(comparison["add_proportion_to_count"]["likelihood_ratio"], 62.1)
    close(comparison["add_count_to_proportion"]["likelihood_ratio"], 0.59)
    close(comparison["add_count_to_proportion"]["p"], 0.44)

    coordination = load("coordination_statistics.json")["models"]
    for model, expected in {
        "Gemini 3.8 Flash": [29.2, 21.9, -7.29, 0.96],
        "Grok 4.3": [30.0, 24.6, -5.37, 0.98],
    }.items():
        result = coordination[model]
        close(
            [result["rates_percent"]["independent"], result["rates_percent"]["coordinated"],
             result["delta_percentage_points"], result["coordinated_weighted_r_squared"]],
            expected,
        )

    heterogeneous = load("heterogeneous_statistics.json")
    heterogeneous_expected = load("heterogeneous_statistics.expected.json")
    for section in ("data", "linearity", "analyses"):
        compare_tree(heterogeneous[section], heterogeneous_expected[section], f"heterogeneous.{section}")
    summary = heterogeneous["manuscript_summary"]
    close(list(summary["rates_percent_by_honest_model"].values()), [19.5, 37.7])
    close(list(summary["rates_percent_by_deceiver_model"].values()), [26.8, 21.2])
    pooled_trend = heterogeneous["analyses"]["Pooled"]["trend"]
    close(pooled_trend["rates_percent"], [21.2, 26.8])
    close(pooled_trend["delta_percentage_points"], 5.54)
    assert sum(test["two_sided_p"] < 0.05 for test in summary["deceiver_comparisons_by_composition"].values()) == 3

    behavior_output = load("behavioral_statistics.json")
    behavior_expected = load("behavioral_statistics.expected.json")
    compare_tree(behavior_output["tests"], behavior_expected["tests"], "behavior.tests")
    behavior = {test["name"]: test for test in behavior_output["tests"]}
    close(
        [behavior[name]["t"] for name in (
            "round_change_co0_concede_and_pivot",
            "round_change_co0_selective_skepticism",
            "round_change_co0_reframe_or_equivocate",
        )],
        [16.80, 11.51, 9.59],
    )
    assert behavior["by_round2_DeepSeek V4.1 Flash"]["holm_p"] < 0.01
    print("All manuscript headline statistics reconstructed successfully.")


if __name__ == "__main__":
    main()
