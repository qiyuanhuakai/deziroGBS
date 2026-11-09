#!/usr/bin/env python3
"""Predeclared analytic pure-core resource budget; no GBS samples are drawn.

Use ``freeze`` first, then ``run`` in the same output directory.  The second
command verifies the frozen configuration/source before any budget calculation.
Floating-point tails are not outward-rounded certificates.  Nothing here
provides a distance bound for an existing forest sampling distribution.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.special import gammaln, logsumexp
from scipy.stats import nbinom

from gbslab.gaussian import covariance_from_interferometer, haar_unitary


ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    return {name: sha(ROOT / name) for name in
            ("experiments/iteration9_core_budget.py", "src/gbslab/gaussian.py")}


def freeze(config_path, output):
    if output.exists():
        raise FileExistsError("Freeze requires a new output directory")
    config = json.loads(config_path.read_text())
    rows = len(config["profiles"]) * len(config["modes"]) * len(config["tv_budgets"])
    if rows != config["planned_comparison_rows"] or 2 * rows != config["planned_cutoff_calculations"]:
        raise ValueError("Planned matrix counts are inconsistent")
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.json")
    sources = source_hashes()
    for name in sources:
        path = output / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    plan = {"status": "frozen_before_budget_calculation", "frozen_utc": utc(),
            "config_sha256": sha(config_path), "source_hashes": sources,
            "comparison_rows": rows, "cutoff_calculations": 2 * rows,
            "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "scope": "Analytic scalar budget and one six-mode covariance identity only; no target laws or samples"}
    save(output / "plan.json", plan)
    print(json.dumps(plan, sort_keys=True), flush=True)


def parameters(r, eta):
    if not math.isfinite(r) or r < 0 or not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("This decomposition requires finite r>=0 and uniform eta in [0,1]")
    a = 1.0 - eta + eta * math.exp(-2 * r)
    b = 1.0 - eta + eta * math.exp(2 * r)
    s = -0.5 * math.log(a)
    identity = 1.0 + 4.0 * eta * (1.0 - eta) * math.sinh(r)**2
    return {"a": a, "b": b, "s": s, "a_times_b": a * b,
            "a_times_b_exact_identity_float": identity,
            "identity_absolute_residual": abs(a * b - identity),
            "residual_antisqueezed_variance": b - 1.0 / a}


def minimal_cutoff(modes, squeeze, epsilon, maximum_pairs):
    """Smallest nonnegative even K using the fixed SciPy SF convention."""
    shape = modes / 2.0
    p = 1.0 / math.cosh(squeeze)**2
    tail_budget = epsilon**2
    if not 0 < p <= 1 or not 0 < tail_budget < 1:
        raise ValueError("Invalid negative-binomial parameter or error budget")

    def tail(j):
        value = float(nbinom.sf(j, shape, p))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ArithmeticError("Nonfinite/out-of-range negative-binomial tail")
        return value

    lower, upper = -1, 0
    while tail(upper) > tail_budget:
        lower, upper = upper, max(1, 2 * upper)
        if upper > maximum_pairs:
            upper = maximum_pairs
            if tail(upper) > tail_budget:
                raise ArithmeticError("Predeclared cutoff resource cap exhausted")
            break
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if tail(middle) <= tail_budget:
            upper = middle
        else:
            lower = middle
    value = tail(upper)
    previous = tail(upper - 1) if upper else None
    if value > tail_budget or (previous is not None and previous <= tail_budget):
        raise ArithmeticError("Returned cutoff is not the minimal even cutoff")
    return {"squeezing": squeeze, "shape": shape, "success_probability": p,
            "mean_photons": modes * math.sinh(squeeze)**2,
            "cutoff_pairs": upper, "cutoff_photons": 2 * upper,
            "tail_budget": tail_budget, "tail_at_cutoff": value,
            "previous_even_cutoff": 2 * (upper - 1) if upper else None,
            "tail_at_previous_even_cutoff": previous,
            "budget_minus_tail": tail_budget - value,
            "previous_tail_minus_budget": previous - tail_budget if previous is not None else None,
            "sqrt_tail_float": math.sqrt(value),
            "rigorous_outward_certificate": False}


def recurrence_tail(shape, p, cutoff, goal, maximum_terms):
    """Independent positive PMF recurrence, avoiding 1-CDF cancellation.

    For shape>=1 the successive PMF ratio decreases to 1-p. Once below one,
    current_term*ratio/(1-ratio) bounds the unadded tail in exact arithmetic.
    Its floating evaluation is a termination diagnostic, not a certificate.
    """
    if p == 1.0:
        return {"tail": 0.0, "terms": 1, "unadded_geometric_remainder_float": 0.0,
                "summed_total_mass": 1.0}
    alpha, success = np.longdouble(shape), np.longdouble(p)
    failure = np.longdouble(1) - success
    if shape < 1:
        raise ValueError("This recurrence-tail remainder implementation assumes shape>=1")
    term = np.exp(alpha * np.log(success))
    total, tail = np.longdouble(0), np.longdouble(0)
    for j in range(maximum_terms):
        if not np.isfinite(term) or term < 0 or (j == 0 and term == 0):
            raise ArithmeticError("Recurrence PMF lost representability")
        total += term
        if j > cutoff:
            tail += term
        ratio = failure * (np.longdouble(j) + alpha) / np.longdouble(j + 1)
        if j >= cutoff and ratio < 1:
            remainder = term * ratio / (1 - ratio)
            if remainder <= np.longdouble(goal):
                return {"tail": float(tail), "terms": j + 1,
                        "unadded_geometric_remainder_float": float(remainder),
                        "summed_total_mass": float(total)}
        following = term * ratio
        if term > 0 and ratio > 0 and following == 0:
            raise ArithmeticError("Recurrence underflow before its fixed termination goal")
        term = following
    raise ArithmeticError("Predeclared recurrence resource cap exhausted")


def dimension_log10(modes, cutoff):
    if modes < 1 or cutoff < 0 or cutoff % 2:
        raise ValueError("Dimension requires positive modes and nonnegative even cutoff")
    photons = np.arange(0, cutoff + 1, 2, dtype=float)
    terms = gammaln(modes + photons) - gammaln(modes) - gammaln(photons + 1)
    return float(logsumexp(terms) / math.log(10.0))


def evaluate_budget(modes, squeeze, epsilon, config):
    result = minimal_cutoff(modes, squeeze, epsilon, config["cutoff_maximum_pairs"])
    for name, pairs in (("cutoff", result["cutoff_pairs"]),
                         ("previous", result["cutoff_pairs"] - 1)):
        if pairs < 0:
            result["recurrence_" + name] = None
            continue
        check = recurrence_tail(result["shape"], result["success_probability"], pairs,
            config["recurrence_absolute_remainder_goal"], config["recurrence_maximum_terms"])
        expected = result["tail_at_cutoff"] if name == "cutoff" else result["tail_at_previous_even_cutoff"]
        check["absolute_difference_from_scipy"] = abs(check["tail"] - expected)
        check["passed"] = (check["absolute_difference_from_scipy"]
                            <= config["tail_absolute_agreement_tolerance"]
                            and abs(check["summed_total_mass"] - 1) <= config["tail_absolute_agreement_tolerance"])
        if not check["passed"]:
            raise ArithmeticError("Independent negative-binomial recurrence disagrees")
        result["recurrence_" + name] = check
    result["log10_even_core_dimension"] = dimension_log10(modes, result["cutoff_photons"])
    result["log10_one_complex128_vector_bytes"] = result["log10_even_core_dimension"] + math.log10(16)
    result["dimension_is_full_sampler_cost"] = False
    return result


def numerical_validation(config, output):
    checks, arrays = [], {}
    tolerance = config["covariance_check"]["maximum_absolute_identity_residual"]
    m = config["covariance_check"]["modes"]
    seed = config["covariance_check"]["haar_seed"]
    U = haar_unitary(m, np.random.default_rng(seed))
    O = np.block([[U.real, -U.imag], [U.imag, U.real]])
    omega = np.block([[np.zeros((m, m)), np.eye(m)], [-np.eye(m), np.zeros((m, m))]])
    arrays.update(interferometer=U, passive_symplectic=O)
    for profile in config["profiles"]:
        r, eta = profile["squeezing"], profile["transmissivity"]
        p = parameters(r, eta)
        V = covariance_from_interferometer(U, r, loss=eta)
        W = (O * np.r_[np.full(m, p["a"]), np.full(m, 1 / p["a"])] [None, :]) @ O.T
        residual = (O * np.r_[np.zeros(m), np.full(m, p["b"] - 1 / p["a"])] [None, :]) @ O.T
        identity_error = float(np.max(np.abs(V - W - residual)))
        pure_error = float(np.max(np.abs(W @ omega @ W - omega)))
        minimum = float(np.linalg.eigvalsh((residual + residual.T) / 2)[0])
        mean_error = abs((float(np.trace(W)) - 2 * m) / 4 - m * math.sinh(p["s"])**2)
        record = {"name": "pure_plus_psd_covariance_" + profile["name"],
                  "modes": m, "haar_seed": seed, **p,
                  "maximum_identity_error": identity_error,
                  "maximum_pure_symplectic_error": pure_error,
                  "minimum_residual_eigenvalue": minimum,
                  "mean_core_photon_identity_error": mean_error,
                  "passed": bool(identity_error <= tolerance and pure_error
                      <= config["covariance_check"]["pure_symplectic_identity_tolerance"]
                      and minimum >= config["covariance_check"]["minimum_eigenvalue_tolerance"]
                      and p["a_times_b"] >= 1 - tolerance and p["identity_absolute_residual"] <= tolerance
                      and mean_error <= tolerance)}
        checks.append(record)
        arrays[profile["name"] + "_target_covariance"] = V
        arrays[profile["name"] + "_pure_core_covariance"] = W
        arrays[profile["name"] + "_residual_covariance"] = residual
    np.savez_compressed(output / "covariance_check.npz", **arrays)
    boundary_m = config["boundary_checks"]["modes"]
    for profile in config["profiles"]:
        r = profile["squeezing"]
        for eta in (0.0, 1.0):
            p = parameters(r, eta)
            passed = (abs(p["s"] - (r if eta else 0.0)) <= tolerance
                      and abs(p["residual_antisqueezed_variance"]) <= tolerance
                      and abs(p["a_times_b"] - 1) <= tolerance)
            for epsilon in config["boundary_checks"]["tv_budgets"]:
                actual = minimal_cutoff(boundary_m, p["s"], epsilon, config["cutoff_maximum_pairs"])
                expected = minimal_cutoff(boundary_m, r if eta else 0.0, epsilon, config["cutoff_maximum_pairs"])
                passed = passed and actual["cutoff_photons"] == expected["cutoff_photons"]
            checks.append({"name": f"eta_boundary_{profile['name']}_{eta:g}", **p,
                           "passed": bool(passed)})
    for eta in config["boundary_checks"]["vacuum_r_0_at_eta"]:
        p = parameters(0.0, eta)
        passed = p["a"] == p["b"] == 1 and p["s"] == 0 and p["residual_antisqueezed_variance"] == 0
        for epsilon in config["boundary_checks"]["tv_budgets"]:
            cut = minimal_cutoff(boundary_m, p["s"], epsilon, config["cutoff_maximum_pairs"])
            passed = passed and cut["cutoff_photons"] == 0 and cut["tail_at_cutoff"] == 0
        checks.append({"name": f"vacuum_boundary_eta_{eta:g}", "passed": bool(passed)})
    for profile in config["profiles"]:
        s = parameters(profile["squeezing"], profile["transmissivity"])["s"]
        p = 1 / math.cosh(s)**2
        differences = [abs(float(nbinom.sf(j, 1, p)) - (1 - p)**(j + 1)) for j in (0, 1, 3, 10)]
        checks.append({"name": "m2_geometric_" + profile["name"],
            "maximum_absolute_difference": max(differences),
            "passed": bool(max(differences) <= config["tail_absolute_agreement_tolerance"])})
    dimension_errors = []
    for modes in config["dimension_check_modes"]:
        for cutoff in config["dimension_check_cutoffs"]:
            exact = sum(math.comb(modes + n - 1, n) for n in range(0, cutoff + 1, 2))
            dimension_errors.append(abs(dimension_log10(modes, cutoff) - math.log10(exact)))
    checks.append({"name": "stable_log_dimension_vs_exact_integer_sum", "comparisons": len(dimension_errors),
                   "maximum_absolute_log10_difference": max(dimension_errors),
                   "passed": bool(max(dimension_errors) <= config["dimension_log10_check_tolerance"])})
    return checks


def write_readme(output, config, rows, metadata, validation):
    lines = ["# Pure-core photon-budget feasibility", "",
        "This is a predeclared analytic resource calculation, not a sampling implementation.",
        "No target P/Q, forest samples, or fitted parameters are used. The exact trace-distance",
        "argument is described in `research/iteration9_core_budget_theory.md`; the numerical",
        "tails below are ordinary floating-point evaluations, not outward-rounded certificates.", "",
        "For uniform loss, write a=1-eta+eta exp(-2r), b=1-eta+eta exp(2r), and",
        "s=-log(a)/2. The exact covariance decomposition is V=W+C, where the pure core W",
        "has squeeze s and C is positive semidefinite because ab=1+4 eta(1-eta)sinh(r)^2.",
        "For identical core modes, N_core=2J with J negative-binomial(shape=m/2, p=sech(s)^2).",
        "The minimal even K is chosen with Pr(N_core>K)<=epsilon_TV^2; normalized pure-core",
        "truncation then has exact trace distance sqrt(tail) before numerical evaluation.", "",
        "The same random-displacement channel and measurement contract that exact distance.",
        "In hbar=2 conventions use real quadrature shift d~N(0,C) and alpha=(d_q+i*d_p)/2.",
        "C can be singular; a Gaussian pushforward defines the distribution without a density determinant.",
        "After displacement the output can have odd photon number and unbounded photon support.",
        "The even-sector dimension below belongs only to the truncated pure-core vector.",
        "It is not the cost or support of a completed displaced-threshold sampling algorithm,",
        "and the error budget is not a certificate for any existing forest sampler.", "",
        "The raw comparison uses squeeze r for the unattenuated pure input. It does not describe",
        "the mixed target's measured photon number. No speed or novelty claim is made.", "",
        "## Large-system resource check", "",
        "`log10 D` is the logarithm of sum_even_n<=K binomial(m+n-1,n). A single dense",
        "complex128 core vector would require 16D bytes, before any workspace or measurement cost.", "",
        "| Profile | m | TV budget | Core mean N | Core K | Core log10 D | Raw-input K | Raw-input log10 D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["modes"] not in (256, 1024):
            continue
        c, r = row["core"], row["unattenuated_pure_input_reference"]
        lines.append(f"| {row['profile']} | {row['modes']} | {row['epsilon_tv']:g} | {c['mean_photons']:.6g} | {c['cutoff_photons']} | {c['log10_even_core_dimension']:.3f} | {r['cutoff_photons']} | {r['log10_even_core_dimension']:.3f} |")
    large = [row for row in rows if row["modes"] == 1024]
    minimum = min(row["core"]["log10_even_core_dimension"] for row in large)
    maximum = max(row["core"]["log10_even_core_dimension"] for row in large)
    lines += ["", f"At m=1024 the core dimensions in this fixed panel range from 10^{minimum:.3f}",
        f"to 10^{maximum:.3f}. The corresponding minimum single-vector storage is about",
        f"10^{minimum + math.log10(16):.3f} bytes. The reduced cutoff does not make dense enumeration practical.",
        "For any fixed nonzero core squeezing, mean core photons grows linearly with m;",
        "this computation establishes no polynomial scaling in m for the dense core representation.", "",
        "## Validation and provenance", "",
        f"The {len(rows)} predeclared comparisons contain {2*len(rows)} cutoff calculations.",
        "The accepted cutoff and previous even cutoff tails are both stored. An independent",
        "positive-term long-double PMF recurrence checks the SciPy survival probabilities.",
        "The recurrence geometric remainder is a floating-point termination diagnostic, not an interval certificate.",
        f"Validation passed: {validation['passed']}; checks: {len(validation['checks'])};",
        f"maximum recurrence/SciPy absolute tail difference: {validation['maximum_tail_absolute_difference']:.6g}.",
        "The one six-mode Haar covariance check, eta=0/1 and vacuum limits, m=2 geometric",
        "tails, and stable log-dimension against exact small integer sums are archived separately.",
        f"Plan frozen: {metadata['plan_frozen_utc']}; evaluation started: {metadata['started_utc']}.",
        f"Configuration SHA256: `{metadata['config_sha256']}`.", "",
        "Files: `budget_rows.json` contains every size/profile/epsilon and both full budget records;",
        "`validation.json` contains checks; `covariance_check.npz` contains U, W, C and target V;",
        "`plan.json`, `config.json`, `source_snapshot/`, `metadata.json` and `manifest.json` bind provenance.", "",
        "Reproduce in a new directory with:", "",
        "```sh", "PYTHONPATH=src python experiments/iteration9_core_budget.py freeze --output results/new_core_budget",
        "PYTHONPATH=src python experiments/iteration9_core_budget.py run --output results/new_core_budget", "```", ""]
    (output / "README.md").write_text("\n".join(lines))


def run(config_path, output):
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "frozen_before_budget_calculation":
        raise ValueError("A frozen, unevaluated plan is required")
    if (sha(config_path) != plan["config_sha256"] or sha(output / "config.json") != plan["config_sha256"]
            or source_hashes() != plan["source_hashes"]):
        raise ValueError("Configuration or source changed after plan freeze")
    if (output / "metadata.json").exists() or (output / "budget_rows.json").exists():
        raise FileExistsError("Do not overwrite or rerun a prior budget attempt")
    config = json.loads((output / "config.json").read_text())
    started = perf_counter()
    metadata = {"status": "running", "started_utc": utc(), "plan_frozen_utc": plan["frozen_utc"],
        "plan_sha256": sha(plan_path), "config_sha256": plan["config_sha256"],
        "source_hashes": plan["source_hashes"], "python": sys.version,
        "numpy": np.__version__, "scipy": scipy.__version__, "floating_point_certificate": False,
        "no_target_P_or_Q_loaded": True, "no_sampling": True}
    save(output / "metadata.json", metadata)
    try:
        checks = numerical_validation(config, output)
        rows = []
        for profile in config["profiles"]:
            p = parameters(profile["squeezing"], profile["transmissivity"])
            for modes in config["modes"]:
                for epsilon in config["tv_budgets"]:
                    core = evaluate_budget(modes, p["s"], epsilon, config)
                    raw = evaluate_budget(modes, profile["squeezing"], epsilon, config)
                    rows.append({"profile": profile["name"], "modes": modes,
                        "input_squeezing": profile["squeezing"], "transmissivity": profile["transmissivity"],
                        "epsilon_tv": epsilon, "parameters": p, "core": core,
                        "unattenuated_pure_input_reference": raw,
                        "core_to_raw_mean_ratio": core["mean_photons"] / raw["mean_photons"],
                        "core_minus_raw_log10_dimension": core["log10_even_core_dimension"] - raw["log10_even_core_dimension"]})
        expected = {(p["name"], m, e) for p in config["profiles"] for m in config["modes"] for e in config["tv_budgets"]}
        if (len(rows) != config["planned_comparison_rows"] or len(rows) != len(expected)
                or {(row["profile"], row["modes"], row["epsilon_tv"]) for row in rows} != expected):
            raise ArithmeticError("Incomplete or duplicated resource matrix")
        errors = [record["absolute_difference_from_scipy"] for row in rows
            for budget in (row["core"], row["unattenuated_pure_input_reference"])
            for record in (budget["recurrence_cutoff"], budget["recurrence_previous"]) if record is not None]
        checks.append({"name": "all_cutoff_and_previous_tail_recurrences", "comparisons": len(errors),
                       "maximum_absolute_difference": max(errors),
                       "passed": bool(max(errors) <= config["tail_absolute_agreement_tolerance"])})
        checks.append({"name": "complete_predeclared_matrix", "rows": len(rows), "passed": True})
        validation = {"passed": all(check["passed"] for check in checks), "checks": checks,
                      "maximum_tail_absolute_difference": max(errors),
                      "ordinary_floating_point_checks_not_certified_enclosures": True}
        save(output / "budget_rows.json", rows)
        save(output / "validation.json", validation)
        if not validation["passed"]:
            raise ArithmeticError("A predeclared numerical identity check failed")
        if source_hashes() != plan["source_hashes"] or sha(config_path) != plan["config_sha256"]:
            raise RuntimeError("Source or configuration changed during calculation")
        metadata.update(status="complete", completed_utc=utc(), elapsed_seconds=perf_counter() - started,
                        comparison_rows=len(rows), cutoff_calculations=2 * len(rows),
                        source_and_config_unchanged=True, all_checks_passed=True)
        save(output / "metadata.json", metadata)
        write_readme(output, config, rows, metadata, validation)
        manifest = {str(path.relative_to(output)): sha(path)
                    for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        save(output / "manifest.json", {"sha256": manifest})
        print(json.dumps({key: metadata[key] for key in ("status", "comparison_rows", "cutoff_calculations", "elapsed_seconds")}), flush=True)
    except Exception as error:
        metadata.update(status="failed", failed_utc=utc(), error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=perf_counter() - started)
        save(output / "metadata.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/iteration9_core_budget.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/iteration9_core_budget")
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.phase](args.config, args.output)


if __name__ == "__main__":
    main()
