"""Reproduce one post-hoc local-error counterexample; performs no sampling.

All event probabilities below are independently summed from the complete
7-mode target probability table.  This script does not import the proposed
positive_completion module or choose a retained set from its implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gbslab.exact import ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for the reproduced record")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    seed, modes, current = 0, 7, 6
    rng = np.random.default_rng(seed)
    unitary = haar_unitary(modes, rng)
    squeezing = rng.uniform(0.2, 1.4, modes)
    transmissivity = rng.uniform(0.1, 1.0, modes)
    covariance = covariance_from_interferometer(unitary, squeezing, transmissivity)
    model = ThresholdModel(covariance, max_exact_modes=modes)
    target = model.full_distribution()
    masks = np.arange(1 << modes, dtype=np.uint64)
    zeros, retained, omitted = (5,), (3, 4), (0, 1, 2)

    def event(z, c):
        zm = sum(1 << j for j in z)
        cm = sum(1 << j for j in c)
        return float(np.sum(target[((masks & zm) == 0) & ((masks & cm) == cm)],
                            dtype=np.longdouble))

    def masses(a, extra=()):
        z = zeros + ((current,) if a == 0 else ())
        c = retained + tuple(extra) + ((current,) if a == 1 else ())
        return event(z, c)

    base = [masses(a) for a in (0, 1)]
    exact = [masses(a, omitted) for a in (0, 1)]
    exact_click = exact[1] / sum(exact)
    rows = []
    for name, blocks in (("product", ((0,), (1,), (2,))),
                         ("pair_block", ((0, 1), (2,)))):
        branch_rows, log_scores = [], []
        for a in (0, 1):
            joint_blocks = [masses(a, block) for block in blocks]
            likelihoods = [mass / base[a] for mass in joint_blocks]
            log_score = float(np.log(base[a]) + sum(
                np.log(mass) - np.log(base[a]) for mass in joint_blocks))
            log_scores.append(log_score)
            true_likelihood = exact[a] / base[a]
            product_likelihood = float(np.prod(likelihoods))
            branch_rows.append({
                "outcome": a,
                "base_joint_mass": base[a],
                "block_joint_masses": joint_blocks,
                "block_likelihoods": likelihoods,
                "true_full_joint_mass": exact[a],
                "true_full_likelihood": true_likelihood,
                "product_likelihood": product_likelihood,
                "correlation_ratio_C": true_likelihood / product_likelihood,
                "log_score": log_score,
            })
        scaled = np.exp(np.asarray(log_scores) - max(log_scores))
        predicted = float(scaled[1] / np.sum(scaled))
        rows.append({
            "method": name,
            "blocks": blocks,
            "branches": branch_rows,
            "click_probability": predicted,
            "absolute_conditional_error": abs(predicted - exact_click),
        })

    arrays = args.output / "counterexample.npz"
    np.savez_compressed(arrays, covariance=covariance, target_probabilities=target,
                        unitary=unitary, squeezing=squeezing,
                        transmissivity=transmissivity)
    record = {
        "schema_version": 1,
        "analysis_role": "posthoc_counterexample_to_local_error_monotonicity",
        "pre_registered": False,
        "sampling_runs": 0,
        "formal_iteration6_case_member": False,
        "claim": "A larger disjoint block can have greater one-step probability error.",
        "not_claimed": ["global TV ordering", "statistical confirmation",
                        "speed advantage", "new algorithm or theorem novelty"],
        "discovery": (
            "Exploratory search, separate from all frozen formal cases. Earlier "
            "homogeneous m=6, r=.9, eta=.65, k=0 searches over seeds 0..99 did "
            "not find the requested full-TV or all-click-final-prefix local "
            "inequality. The heterogeneous m=7 search planned seeds 0..199, "
            "prefixes in ascending bit-mask order and k in (0,1,2), and stopped "
            "at this seed=0, prefix=31, k=2 local counterexample."
        ),
        "seed": seed,
        "rng": "numpy.random.default_rng (PCG64)",
        "construction_order": ["haar_unitary(7, rng)",
                               "rng.uniform(.2, 1.4, 7)",
                               "rng.uniform(.1, 1., 7)"],
        "numpy_version": np.__version__,
        "modes": modes,
        "squeezing": squeezing.tolist(),
        "transmissivity": transmissivity.tolist(),
        "covariance_convention": "zero mean, hbar=2, xxpp; loss argument is transmissivity",
        "covariance_array_sha256": hashlib.sha256(
            np.asarray(covariance, dtype="<f8", order="C").tobytes()).hexdigest(),
        "target_array_sha256": hashlib.sha256(
            np.asarray(target, dtype="<f8", order="C").tobytes()).hexdigest(),
        "arrays_file": arrays.name,
        "arrays_sha256": sha256(arrays),
        "target_encoding": "little-endian physical-mode mask; 1 means >=1 photon",
        "measurement_order": list(range(modes)),
        "history": [1, 1, 1, 1, 1, 0],
        "current_mode": current,
        "retention_rule": "recent",
        "k": 2,
        "zeros": zeros,
        "retained": retained,
        "omitted": omitted,
        "block_order": "measurement",
        "conditioning_history_probability": sum(exact),
        "exact_click_probability": exact_click,
        "rows": rows,
        "pair_error_greater_than_product": (
            rows[1]["absolute_conditional_error"] > rows[0]["absolute_conditional_error"]),
        "minimum_target_probability": float(np.min(target)),
        "target_probability_sum": float(np.sum(target, dtype=np.longdouble)),
        "minimum_uncertainty_eigenvalue": model.minimum_uncertainty_eigenvalue,
        "target_numerical_diagnostics": model.diagnostics,
        "strict_floating_point_certificate": False,
        "numerical_scope": (
            "Full 128-cell inclusion-exclusion target in float64 with long-double "
            "accumulation. Complete here means no model/Fock cutoff; no outward "
            "rounded numerical certificate is asserted."
        ),
        "source_sha256": {
            "reproduce.py": sha256(__file__),
            "src/gbslab/exact.py": sha256(ROOT / "src/gbslab/exact.py"),
            "src/gbslab/gaussian.py": sha256(ROOT / "src/gbslab/gaussian.py"),
        },
    }
    (args.output / "case.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "exact_click_probability": exact_click,
                      "rows": [{k: row[k] for k in
                                ("method", "click_probability", "absolute_conditional_error")}
                               for row in rows],
                      "pair_error_greater_than_product": record["pair_error_greater_than_product"]},
                     indent=2))


if __name__ == "__main__":
    main()
