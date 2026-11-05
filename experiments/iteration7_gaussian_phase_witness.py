"""Deterministic exact-algebra audit of fixed thermal Gaussian phase witnesses.

Only n=1, r=2/5 and phases 0/pi on the three- and four-cycles are evaluated.
There is no parameter search, fitting, random sampling or performance claim.
All probability identities are checked with Fraction arithmetic; the existing
Gaussian and forest code is used only as an independent small numerical check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from gbslab.exact import ThresholdModel
from gbslab.positive_forest import ForestCompletionSampler


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction_record(value):
    return {"exact": str(value), "decimal": float(value)}


def determinant(matrix):
    a = [list(row) for row in matrix]
    result = F(1)
    for col in range(len(a)):
        pivot = next((i for i in range(col, len(a)) if a[i][col]), None)
        if pivot is None:
            return F(0)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            result = -result
        diagonal = a[col][col]
        result *= diagonal
        for row in range(col + 1, len(a)):
            factor = a[row][col] / diagonal
            for j in range(col + 1, len(a)):
                a[row][j] -= factor * a[col][j]
            a[row][col] = F(0)
    return result


def cycle_number_covariance(modes, sign):
    matrix = [[F(int(i == j)) for j in range(modes)] for i in range(modes)]
    for i in range(modes):
        j = (i + 1) % modes
        matrix[i][j] = matrix[j][i] = F(2, 5) * (sign if i == modes - 1 else 1)
    return matrix


def exact_law(number_covariance):
    m = len(number_covariance)
    vacuum = []
    for mask in range(1 << m):
        indices = [j for j in range(m) if (mask >> j) & 1]
        matrix = [[number_covariance[i][j] + int(i == j) for j in indices] for i in indices]
        vacuum.append(1 / determinant(matrix))
    target = []
    full = (1 << m) - 1
    for clicks in range(1 << m):
        zeros = full ^ clicks
        subsets = [s for s in range(1 << m) if s & clicks == s]
        target.append(sum(((-1)**s.bit_count() * vacuum[zeros | s] for s in subsets), F(0)))
    assert sum(target) == 1 and min(target) > 0
    return vacuum, target


def event(target, *, zeros=(), ones=()):
    return sum((v for mask, v in enumerate(target)
                if all(not (mask >> j) & 1 for j in zeros)
                and all((mask >> j) & 1 for j in ones)), F(0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/iteration7_gaussian_phase_witness")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new directory; the fixed algebra audit is not silently overwritten")
    sources = [Path(__file__).resolve()] + [ROOT / "src/gbslab" / (name + ".py") for name in
        ("positive_forest", "positive_completion", "exact", "conditional_memory", "schur_memory", "zero_memory")]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    args.output.mkdir(parents=True)
    rows = []
    for m in (3, 4):
        matrices, vacua, targets, covariances = [], [], [], []
        numerical_differences, physicality = [], []
        for sign in (1, -1):
            N = cycle_number_covariance(m, sign)
            v, p = exact_law(N)
            n_float = np.asarray(N, dtype=float)
            block = np.eye(m) + 2 * n_float
            V = np.block([[block, np.zeros((m, m))], [np.zeros((m, m)), block]])
            model = ThresholdModel(V)
            p_float = np.asarray(p, dtype=float)
            numeric = model.full_distribution()
            discrepancy = float(np.max(np.abs(numeric - p_float)))
            assert discrepancy < 3e-14
            matrices.append(n_float); vacua.append(v); targets.append(p); covariances.append(V)
            numerical_differences.append(discrepancy)
            physicality.append({"minimum_N_eigenvalue": float(np.linalg.eigvalsh(n_float)[0]),
                                "minimum_uncertainty_eigenvalue": model.minimum_uncertainty_eigenvalue,
                                "target_diagnostics": model.diagnostics.copy()})
        assert vacua[0][:-1] == vacua[1][:-1]
        delta_v = vacua[0][-1] - vacua[1][-1]
        assert all(a - b == (-1)**mask.bit_count() * delta_v
                   for mask, (a, b) in enumerate(zip(*targets)))
        tv = sum(abs(a - b) for a, b in zip(*targets)) / 2
        assert tv == 2**(m - 1) * abs(delta_v)
        F_modes = tuple(range(m - 1))
        prefix_mass = event(targets[0], ones=F_modes)
        assert prefix_mass == event(targets[1], ones=F_modes)
        posteriors = [p[-1] / prefix_mass for p in targets]
        row = {"modes": m, "n": "1", "r": "2/5", "phases": ["0", "pi"],
               "analytic_minimum_N_eigenvalue_lower_bound": "1/5",
               "proper_vacuum_marginals_identical_exactly": True,
               "full_vacuum": [fraction_record(v[-1]) for v in vacua],
               "delta_full_vacuum": fraction_record(delta_v),
               "all_click_probability": [fraction_record(p[-1]) for p in targets],
               "full_target_tv": fraction_record(tv),
               "common_Q_information_model_minimax_TV_lower_bound": fraction_record(tv / 2),
               "last_mode": m - 1, "all_one_prefix_mass": fraction_record(prefix_mass),
               "true_last_mode_posteriors": [fraction_record(p) for p in posteriors],
               "posterior_separation": fraction_record(abs(posteriors[0] - posteriors[1])),
               "exact_target_fractions": [[str(v) for v in p] for p in targets],
               "gaussian_numeric_max_abs_errors": numerical_differences,
               "physicality_checks": physicality}
        arrays = {"N": np.asarray(matrices), "covariance": np.asarray(covariances),
                  "target": np.asarray(targets, dtype=float), "vacuum": np.asarray(vacua, dtype=float)}
        if m == 4:
            conditionals = []
            tree_scores = []
            for p in targets:
                branches = []
                scores = []
                for outcome in (0, 1):
                    zeros, ones = ((3,), ()) if outcome == 0 else ((), (3,))
                    w = event(p, zeros=zeros, ones=ones)
                    nodes = {j: event(p, zeros=zeros, ones=ones + (j,)) / w for j in F_modes}
                    pairs = {pair: event(p, zeros=zeros, ones=ones + pair) / w
                             for pair in combinations(F_modes, 2)}
                    branches.append({"outcome": outcome, "nodes": {str(k): str(v) for k, v in nodes.items()},
                                     "all_pairs": {str(k): str(v) for k, v in pairs.items()}})
                    scores.append(w * pairs[(0, 1)] * pairs[(1, 2)] / nodes[1])
                conditionals.append(branches)
                tree_scores.append(scores)
            assert conditionals[0] == conditionals[1]
            assert tree_scores[0] == tree_scores[1]
            approximation = tree_scores[0][1] / sum(tree_scores[0])
            assert approximation == F(88804, 165533)
            q_exact, sampler_checks, exact_errors = [], [], []
            for p, V in zip(targets, covariances):
                q = p.copy()
                q[7], q[15] = prefix_mass * (1 - approximation), prefix_mass * approximation
                sampler = ForestCompletionSampler(ThresholdModel(V), 0,
                                                    forest="chain", selection="click_influence")
                law = sampler.enumerate_law()
                difference = float(np.max(np.abs(law["distribution"] - np.asarray(q, dtype=float))))
                assert law["failure_mass"] == 0 and difference < 3e-14
                q_exact.append(q)
                sampler_checks.append({"failure_mass": law["failure_mass"], "max_abs_error": difference,
                    "next_click_at_all_one_prefix": sampler.conditional_click_probability((1, 1, 1))})
                exact_errors.append(sum(abs(a - b) for a, b in zip(p, q)) / 2)
            conditional_bound = abs(delta_v) / 2
            assert prefix_mass * abs(posteriors[0] - posteriors[1]) / 2 == conditional_bound
            assert max(exact_errors) >= conditional_bound
            assert q_exact[0] != q_exact[1]  # Actual sampler sees full zero events on other paths.
            row.update({"current_conditioned_singleton_and_all_pair_marginals_identical": True,
                "current_conditioned_marginals": conditionals[0],
                "chain_tree_edges": [[0, 1], [1, 2]],
                "common_chain_posterior": fraction_record(approximation),
                "actual_chain_k0_Qs_are_identical": False,
                "actual_chain_k0_TVs": [fraction_record(v) for v in exact_errors],
                "actual_sampler_conditional_minimax_TV_lower_bound": fraction_record(conditional_bound),
                "actual_chain_k0_exact_law_checks": sampler_checks})
            arrays["chain_k0_distribution"] = np.asarray(q_exact, dtype=float)
        path = args.output / f"cycle{m}.npz"
        np.savez_compressed(path, **arrays)
        row["arrays_sha256"] = sha(path)
        rows.append(row)
    stable = all(sha(ROOT / name) == digest for name, digest in source_hashes.items())
    assert stable
    record = {"schema": "gbslab.iteration7_gaussian_phase_witness.v1", "passed": True,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Fixed deterministic exact-algebra examples, not search, random sampling or an algorithmic hardness result",
              "parameter_search_performed": False, "new_random_samples": 0,
              "phase_values_are_part_of_analytic_construction": True,
              "full_covariance_algorithms_are_not_ruled_out": True,
              "actual_sampler_and_restricted_information_model_bounds_are_distinct": True,
              "source_hashes": source_hashes, "sources_stable": stable,
              "python": sys.version, "numpy": np.__version__, "rows": rows}
    (args.output / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    for name in source_hashes:
        destination = args.output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    print(json.dumps({"passed": True, "record_sha256": sha(args.output / "record.json"),
                      "cycle4": {key: rows[1][key] for key in (
                          "full_target_tv", "true_last_mode_posteriors", "common_chain_posterior",
                          "actual_chain_k0_TVs", "actual_sampler_conditional_minimax_TV_lower_bound")}}, indent=2))


if __name__ == "__main__":
    main()
