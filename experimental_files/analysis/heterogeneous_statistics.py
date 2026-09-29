"""Reproduce Section 2.3 trend and proportion-versus-count analyses.

Run from any directory with Python, numpy, pandas, scipy, and statsmodels:
  python experimental_files/analysis/heterogeneous_statistics.py

Only two adversarial proportions occur in this experiment. A curvature/linearity
test is therefore not identifiable; the trend analysis tests their difference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.multitest import multipletests


PAIR_FOLDERS = {
    "Gemini / DeepSeek": "gemini-honest_deepseek-deceiver",
    "Gemini / Grok": "gemini-honest_grok-deceiver",
    "Glimmer / DeepSeek": "glimmer-honest_deepseek-deceiver",
    "Glimmer / Grok": "glimmer-honest_grok-deceiver",
}
TERMS = {"proportion": ["proportion", "log2_N"],
         "count": ["k", "log2_N"],
         "combined": ["proportion", "k", "log2_N"]}


def read_data(root):
    selection_path = root / "experimental_files/question_selection/gemini-3.8-flash/selection.csv"
    selection = pd.read_csv(selection_path).set_index("id").correct_of_k
    files = [selection_path]
    parts = []
    for pair, folder in PAIR_FOLDERS.items():
        path = root / "experimental_files/results/heterogeneous" / folder / "per_game.csv"
        frame = pd.read_csv(path)
        files.append(path)
        frame["pair"] = pair
        frame["bucket"] = frame.question_id.map(selection)
        initially_correct = frame.honest_r0_acc * frame.n_honest
        assert np.allclose(initially_correct, np.rint(initially_correct), atol=1e-8)
        frame["C"] = np.rint(initially_correct).astype(int)
        frame["D"] = frame.n_switched.astype(int)
        frame["proportion"] = frame.k / frame.N
        frame["log2_N"] = np.log2(frame.N)
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    assert not data.bucket.isna().any()
    data["bucket"] = data.bucket.astype(int)
    assert not data.duplicated(["pair", "question_id", "N", "k"]).any()
    assert ((data.D >= 0) & (data.D <= data.C)).all()
    assert set(zip(data.N, data.k)) == {(5, 1), (10, 2), (7, 3), (14, 6)}
    fingerprints = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files}
    return data, fingerprints


def design(frame, model):
    strata = frame.pair + ":bucket=" + frame.bucket.astype(str)
    intercepts = pd.get_dummies(strata, dtype=float)
    matrix = pd.concat([intercepts, frame[TERMS[model]]], axis=1)
    assert np.linalg.matrix_rank(matrix) == matrix.shape[1]
    return matrix


def fit_models(frame, verify=False):
    usable = frame[frame.C > 0].copy().reset_index(drop=True)
    response = np.column_stack([usable.D, usable.C - usable.D])
    models, summaries = {}, {}
    for name in TERMS:
        matrix = design(usable, name)
        fitted = GLM(response, matrix, family=Binomial()).fit(
            maxiter=200, tol=1e-10, cov_type="cluster",
            cov_kwds={"groups": usable.question_id, "use_correction": True},
            use_t=True,
        )
        assert fitted.converged
        models[name] = fitted
        intervals = fitted.conf_int()
        summaries[name] = {
            "deviance": float(fitted.deviance), "aic": float(fitted.aic),
            "n_parameters": matrix.shape[1], "n_usable_trials": len(usable),
            "n_question_clusters": usable.question_id.nunique(),
            "terms": {term: {
                "coefficient": float(fitted.params[term]),
                "question_clustered_se": float(fitted.bse[term]),
                "question_clustered_t_p": float(fitted.pvalues[term]),
                "question_clustered_95_ci": intervals.loc[term].tolist(),
            } for term in TERMS[name]},
        }
        if verify and name != "combined":
            # Independent optimizer checks the likelihood and gradient of the
            # trial-level fit; this is not an additional statistical test.
            x, successes, totals = matrix.to_numpy(), usable.D.to_numpy(), usable.C.to_numpy()
            def objective(beta):
                eta = x @ beta
                return float(np.sum(totals * np.logaddexp(0, eta) - successes * eta))
            def gradient(beta):
                return x.T @ (totals * expit(x @ beta) - successes)
            checked = minimize(objective, fitted.params.to_numpy() + 0.01,
                               jac=gradient, method="BFGS", options={"gtol": 1e-8})
            assert abs(objective(checked.x) - objective(fitted.params)) < 1e-6
            assert np.max(np.abs(gradient(fitted.params))) < 1e-5
    comparisons = {
        "count_minus_proportion_deviance": float(models["count"].deviance - models["proportion"].deviance),
        "count_minus_proportion_aic": float(models["count"].aic - models["proportion"].aic),
    }
    for restricted, added in [("count", "proportion"), ("proportion", "k")]:
        statistic = float(models[restricted].deviance - models["combined"].deviance)
        comparisons["add_" + added + "_to_" + restricted] = {
            "question_clustered_t_p": float(models["combined"].pvalues[added]),
            "binomial_independence_likelihood_ratio": statistic,
            "binomial_independence_likelihood_ratio_p": float(chi2.sf(statistic, 1)),
        }
    return {"models": summaries, "comparison": comparisons}


def question_bootstrap_weights(data, rng, n_bootstrap):
    questions = data[["question_id", "bucket"]].drop_duplicates().sort_values("question_id")
    assert not questions.question_id.duplicated().any()
    questions = questions.reset_index(drop=True)
    weights = np.zeros((n_bootstrap, len(questions)), dtype=int)
    # Preserve the evaluation's question-difficulty stratification. Each draw
    # retains all trials and all model pairings for a sampled question.
    for _, indices in questions.groupby("bucket").groups.items():
        indices = np.asarray(list(indices))
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)),
                                             size=n_bootstrap)
    return questions.question_id.tolist(), weights


def trend_test(frame, questions, weights, rng, n_permutations, paired_across_all=False):
    grouped = frame.groupby(["question_id", "proportion"])[["D", "C"]].sum()
    levels = sorted(frame.proportion.unique())
    arrays = []
    for level in levels:
        arrays.append(grouped.xs(level, level="proportion").reindex(questions, fill_value=0).to_numpy())
    low, high = arrays
    rates = [100 * a[:, 0].sum() / a[:, 1].sum() for a in arrays]
    boot_low, boot_high = weights @ low, weights @ high
    boot_delta = 100 * (boot_high[:, 0] / boot_high[:, 1] - boot_low[:, 0] / boot_low[:, 1])

    # The permutation analysis uses complete four-composition question blocks.
    # In the pooled test the same low/high swap is applied to all four model
    # pairings for a question, preserving their dependence.
    expected = 4 * (4 if paired_across_all else 1)
    complete = frame.groupby("question_id").size()
    complete = complete[complete == expected].index
    mask = np.isin(questions, complete)
    a, b = low[mask], high[mask]
    observed = 100 * (b[:, 0].sum() / b[:, 1].sum() - a[:, 0].sum() / a[:, 1].sum())
    extreme = 0
    for start in range(0, n_permutations, 1000):
        count = min(1000, n_permutations - start)
        swaps = rng.integers(0, 2, size=(count, len(a)))
        perm_low = a.sum(axis=0) + swaps @ (b - a)
        perm_high = b.sum(axis=0) - swaps @ (b - a)
        differences = 100 * (perm_high[:, 0] / perm_high[:, 1] - perm_low[:, 0] / perm_low[:, 1])
        extreme += int(np.sum(np.abs(differences) >= abs(observed) - 1e-12))
    return {
        "proportions": levels, "rates_percent": rates, "delta_percentage_points": rates[1] - rates[0],
        "question_bootstrap_delta_95_ci": np.quantile(boot_delta, [0.025, 0.975]).tolist(),
        "paired_permutation_questions": len(a), "paired_permutation_delta_percentage_points": observed,
        "paired_permutation_two_sided_p": (extreme + 1) / (n_permutations + 1),
    }


def deceiver_comparisons_by_composition(data, seed, n_permutations):
    """Paired DeepSeek-versus-Grok comparisons, pooled across honest models."""
    results = {}
    for (n_agents, n_deceivers), frame in data.groupby(["N", "k"], sort=True):
        grouped = frame.groupby(["question_id", "deceiver_label"])[["D", "C"]].sum().unstack()
        grouped = grouped.dropna()
        deepseek = grouped[[('D', 'DeepSeek'), ('C', 'DeepSeek')]].to_numpy()
        grok = grouped[[('D', 'Grok'), ('C', 'Grok')]].to_numpy()
        observed = 100 * (
            deepseek[:, 0].sum() / deepseek[:, 1].sum()
            - grok[:, 0].sum() / grok[:, 1].sum()
        )
        rng = np.random.default_rng(seed)
        extreme = 0
        for start in range(0, n_permutations, 1000):
            count = min(1000, n_permutations - start)
            swaps = rng.integers(0, 2, size=(count, len(grouped)))
            perm_grok = grok.sum(axis=0) + swaps @ (deepseek - grok)
            perm_deepseek = deepseek.sum(axis=0) - swaps @ (deepseek - grok)
            differences = 100 * (
                perm_deepseek[:, 0] / perm_deepseek[:, 1]
                - perm_grok[:, 0] / perm_grok[:, 1]
            )
            extreme += int(np.sum(np.abs(differences) >= abs(observed) - 1e-12))
        results[f"{n_agents} total / {n_deceivers} deceivers"] = {
            "paired_questions": len(grouped),
            "deepseek_minus_grok_percentage_points": float(observed),
            "two_sided_p": (extreme + 1) / (n_permutations + 1),
        }
    return results


def bootstrap_comparison(data, questions, weights, original_difference):
    cells = data[["pair", "bucket", "N", "k", "proportion", "log2_N"]].drop_duplicates()
    cells = cells.sort_values(["pair", "bucket", "N"]).reset_index(drop=True)
    cells["cell"] = np.arange(len(cells))
    linked = data.merge(cells, on=["pair", "bucket", "N", "k", "proportion", "log2_N"], validate="many_to_one")
    q_index = {q: i for i, q in enumerate(questions)}
    successes = np.zeros((len(questions), len(cells)))
    totals = np.zeros_like(successes)
    for row in linked.itertuples():
        successes[q_index[row.question_id], row.cell] += row.D
        totals[q_index[row.question_id], row.cell] += row.C
    matrices = {name: design(cells, name).to_numpy() for name in ("proportion", "count")}

    def compare(y, n):
        active = n > 0
        values = {}
        for name, matrix in matrices.items():
            x = matrix[active]
            assert np.linalg.matrix_rank(x) == x.shape[1]
            result = GLM(np.column_stack([y[active], n[active] - y[active]]), x,
                         family=Binomial()).fit(maxiter=200, tol=1e-10)
            assert result.converged
            values[name] = result.deviance
        return float(values["count"] - values["proportion"])

    # Aggregating equal-predictor rows changes absolute deviances but not their
    # difference; verify this before accelerating the bootstrap with aggregation.
    assert abs(compare(successes.sum(axis=0), totals.sum(axis=0)) - original_difference) < 1e-6
    boot_y, boot_n = weights @ successes, weights @ totals
    differences = []
    for i, (y, n) in enumerate(zip(boot_y, boot_n), 1):
        differences.append(compare(y, n))
        if i % 250 == 0:
            print(f"Question bootstrap: {i}/{len(weights)}", flush=True)
    return {
        "count_minus_proportion_deviance_95_ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "fraction_resamples_favoring_proportion": float(np.mean(np.asarray(differences) > 0)),
        "successful_resamples": len(differences),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--counts", type=Path, help="Use an archived count snapshot instead of the original run files.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=49999)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.counts:
        snapshot = json.loads(args.counts.read_text())
        data = pd.DataFrame(snapshot["rows"], columns=snapshot["columns"])
        hashes = snapshot["input_sha256"]
        assert not data.duplicated(["pair", "question_id", "N", "k"]).any()
        assert ((data.D >= 0) & (data.D <= data.C)).all()
        assert set(zip(data.N, data.k)) == {(5, 1), (10, 2), (7, 3), (14, 6)}
        data["proportion"] = data.k / data.N
        data["log2_N"] = np.log2(data.N)
    else:
        data, hashes = read_data(args.data_root)
    rng = np.random.default_rng(args.seed)
    data["honest_label"] = data.pair.str.split(" / ").str[0]
    data["deceiver_label"] = data.pair.str.split(" / ").str[1]
    questions, weights = question_bootstrap_weights(data, rng, args.bootstrap)
    result = {
        "seed": args.seed, "bootstrap_resamples": args.bootstrap, "permutations": args.permutations,
        "versions": {m: importlib.metadata.version(m) for m in ["numpy", "pandas", "scipy", "statsmodels"]},
        "input_sha256": hashes,
        "data": {"trials": len(data), "questions": len(questions), "initially_correct": int(data.C.sum()),
                 "defections": int(data.D.sum()), "zero_initially_correct_trials": int((data.C == 0).sum())},
        "linearity": {"identifiable": False, "unique_proportions": sorted(data.proportion.unique()),
                      "reason": "Two predictor values cannot distinguish a straight line from curvature."},
        "analyses": {},
    }
    result["manuscript_summary"] = {
        "rates_percent_by_honest_model": {
            name: float(100 * part.D.sum() / part.C.sum())
            for name, part in data.groupby("honest_label", sort=True)
        },
        "rates_percent_by_deceiver_model": {
            name: float(100 * part.D.sum() / part.C.sum())
            for name, part in data.groupby("deceiver_label", sort=True)
        },
        "rates_percent_by_proportion": {
            str(level): float(100 * part.D.sum() / part.C.sum())
            for level, part in data.groupby("proportion", sort=True)
        },
        "deceiver_comparisons_by_composition": deceiver_comparisons_by_composition(
            data, args.seed, args.permutations
        ),
    }
    assert np.linalg.matrix_rank(np.column_stack([np.ones(len(data)), data.proportion, data.proportion**2])) == 2
    for name, frame in [("Pooled", data)] + list(data.groupby("pair", sort=False)):
        analysis = fit_models(frame, verify=name == "Pooled")
        analysis["trend"] = trend_test(frame, questions, weights, rng, args.permutations,
                                       paired_across_all=name == "Pooled")
        result["analyses"][name] = analysis
        print(name, "delta deviance:", analysis["comparison"]["count_minus_proportion_deviance"],
              "trend p:", analysis["trend"]["paired_permutation_two_sided_p"], flush=True)
    pair_ps = [result["analyses"][name]["trend"]["paired_permutation_two_sided_p"] for name in PAIR_FOLDERS]
    for name, adjusted in zip(PAIR_FOLDERS, multipletests(pair_ps, method="holm")[1]):
        result["analyses"][name]["trend"]["holm_adjusted_p_across_four_pairings"] = float(adjusted)
    pooled = result["analyses"]["Pooled"]
    pooled["comparison"]["question_bootstrap"] = bootstrap_comparison(
        data, questions, weights, pooled["comparison"]["count_minus_proportion_deviance"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    columns = ["question_id", "pair", "bucket", "N", "k", "C", "D"]
    snapshot = {"input_sha256": hashes, "columns": columns,
                "rows": data[columns].values.tolist()}
    args.output.with_name(args.output.stem + "_counts.json").write_text(
        json.dumps(snapshot, allow_nan=False) + "\n")
    print("Saved", args.output, flush=True)
    print(json.dumps(pooled["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
