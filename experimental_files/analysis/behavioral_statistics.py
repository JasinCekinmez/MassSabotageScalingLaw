"""Reproduce the comparisons in Behavioral Analysis of Deliberation.

Run: python experimental_files/analysis/behavioral_statistics.py
Requires numpy, pandas, scipy, and statsmodels. No model/API calls.

OLS contrasts estimate differences in the displayed pooled proportions. Binary
outcomes are messages (tactics), trials (planning), or first-defection events
(timing). Reflection contrasts subtract the two category indicators within each
reflection. All standard errors cluster on question, retaining dependence across
agents, rounds, compositions, models, and conditions for the same question.
Tests are two-sided, use CR1 finite-sample covariance correction and G-1 Student
t degrees of freedom, and apply Holm adjustment within each named comparison
family: five round changes, three planning contrasts, two coordination contrasts,
four timing tests, and two reflection contrasts. Timing tests use a 50% null;
all other contrasts use a zero-difference null.

The coded samples were selected for rich flip activity, not randomly sampled:
their contrasts describe the coded sample and are not population causal effects.
Archived counts preserve the samples behind the manuscript; ongoing runs do not
change them. The timing snapshot includes the first defect transition for each
honest agent with any defect event (not just final-vote defections).

Covariance/reference distribution documentation:
https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html
"""

from pathlib import Path
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
COUNTS = HERE / "behavioral_counts.json"
TACTICS = HERE / "figures/tactics_by_message.csv"
RESULTS = []


def test(name, family, y, groups, x=None, null=0):
    y = np.asarray(y, dtype=float)
    design = np.ones((len(y), 1))
    if x is not None:
        design = np.column_stack([design, np.asarray(x, dtype=float)])
    fit = sm.OLS(y, design).fit(
        cov_type="cluster", use_t=True,
        cov_kwds={"groups": np.asarray(groups), "use_correction": True,
                  "df_correction": True},
    )
    contrast = np.zeros(design.shape[1])
    contrast[-1] = 1
    tested = fit.t_test((contrast, null))
    statistic = float(tested.tvalue.item())
    df = int(tested.df_denom)
    p = float(tested.pvalue.item())
    assert df == len(set(groups)) - 1
    assert np.isclose(p, 2 * student_t.sf(abs(statistic), df))
    estimate = float(tested.effect.item())
    if x is not None:
        x = np.asarray(x)
        assert np.isclose(estimate, y[x == 1].mean() - y[x == 0].mean())
    else:
        assert np.isclose(estimate, y.mean())
    result = dict(name=name, family=family, n=len(y), clusters=df + 1,
                  estimate=estimate, null=null, t=statistic, df=df, p=p)
    RESULTS.append(result)


def main():
    snapshot = json.loads(COUNTS.read_text())
    tables = {key: pd.DataFrame(snapshot[key]["rows"], columns=snapshot[key]["columns"])
              for key in ["planning", "reflections", "timing"]}
    tactics = pd.read_csv(TACTICS)
    tactics["model"] = tactics.model.str.replace(" (coordinated)", "", regex=False)
    keys = ["model", "coordinated", "question_id", "N", "k", "agent", "round"]
    assert not tactics.duplicated(keys).any()
    assert tactics.groupby(keys[:-1]).size().eq(2).all()
    assert set(tactics["round"]) == {0, 1}

    for coordinated, labels in [
        (0, ["concede_and_pivot", "selective_skepticism", "reframe_or_equivocate"]),
        (1, ["concede_and_pivot", "selective_skepticism"]),
    ]:
        part = tactics[tactics.coordinated == coordinated]
        for label in labels:
            test(f"round_change_co{coordinated}_{label}", "round_changes",
                 part[label], part.question_id, part["round"])

    for model in ["Gemini 3.8 Flash", "Grok 4.3"]:
        part = tactics[tactics.model == model]
        test(f"coordination_reframing_{model}", "coordination_reframing",
             part.reframe_or_equivocate, part.question_id, part.coordinated)

    planning = tables["planning"]
    assert planning.groupby("model").size().tolist() == [30, 30]
    for label, expected in [("suspicion_management", [27, 15]),
                            ("cross_round_reinforcement", [25, 18]),
                            ("coordinated_pivot", [29, 6])]:
        assert planning.groupby("model")[label].sum().tolist() == expected
        test(label, "planning", planning[label], planning.question_id,
             planning.model.eq("Gemini 3.8 Flash"))

    for model, part in tables["timing"].groupby("family"):
        outcomes, questions = [], []
        for row in part.itertuples():
            assert 0 <= row.early <= row.total
            outcomes.extend([1] * row.early + [0] * (row.total - row.early))
            questions.extend([row.question_id] * row.total)
        test(f"by_round2_{model}", "timing", outcomes, questions, null=0.5)

    reflections = tables["reflections"]
    assert len(reflections) == 25
    for label in ["reframe_or_equivocate", "exploit_uncertainty"]:
        outcome = (reflections.category.eq(label).astype(int)
                   - reflections.category.eq("fabricate_or_misrepresent_evidence").astype(int))
        test(f"reflection_{label}", "reflections", outcome, reflections.question_id)

    for family in sorted({r["family"] for r in RESULTS}):
        subset = [r for r in RESULTS if r["family"] == family]
        adjusted = multipletests([r["p"] for r in subset], method="holm")[1]
        for result, p in zip(subset, adjusted):
            result["holm_p"] = float(p)
    output = {
        "method": "Two-sided OLS contrasts with question-clustered CR1 covariance, G-1 t reference, Holm within comparison family",
        "input_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [COUNTS, TACTICS]},
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "statsmodels"]},
        "tests": RESULTS,
    }
    HERE.joinpath("behavioral_statistics.json").write_text(json.dumps(output, indent=2) + "\n")
    for result in RESULTS:
        print(f"{result['name']}: t({result['df']})={result['t']:.4f}, Holm p={result['holm_p']:.6g}")


if __name__ == "__main__":
    main()
