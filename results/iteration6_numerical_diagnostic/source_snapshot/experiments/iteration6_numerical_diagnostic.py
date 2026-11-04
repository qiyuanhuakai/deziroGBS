"""Post hoc numerical diagnosis of the six fixed m12 pair_k4 primary laws.

This runs only after the primary 18-instance/126-law experiment is complete.
It leaves every primary file and algorithm parameter unchanged and replaces
only the event evaluator with the saved complete-target table oracle.  The
same Gaussian click-influence selector and frozen traversal are reused.

This is NOT a new primary result, a new sample, a numerical certificate, or a
speed result.  The full-P oracle is exponential and may inherit corrections
from the original floating Gaussian target.  Distances between three laws
are reported separately; their differences are not an additive decomposition
of approximation and numerical errors.  Table failures remain in the law.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import scipy

from iteration6_completion import extended_law, make_sampler, plain, save, sha


ROOT = Path(__file__).resolve().parents[1]
METHOD = {"id": "pair_k4", "completion": "pair_block", "k": 4}
PROFILES = (
    {"name": "lossy", "squeezing": .6, "transmissivity": .35},
    {"name": "bright", "squeezing": .9, "transmissivity": .65},
    {"name": "stress", "squeezing": 1.2, "transmissivity": .85},
)
SEEDS = (240611, 240612)
CASE_IDS = tuple(f"m12_{p['name']}_s{s}" for s in SEEDS for p in PROFILES)
SCOPE = {
    "analysis_role": "post_hoc_numerical_diagnostic",
    "selection_reason": "A primary m12 pair_k4 numerical failure was observed; inspect all six fixed m12 pair_k4 cases",
    "case_selection_was_preregistered": False,
    "all_cases_in_fixed_m12_stratum_included": True,
    "new_samples_or_inputs": False,
    "primary_results_replaced": False,
    "algorithm_or_budget_tuned": False,
    "table_oracle_is_exponential_reference": True,
    "table_oracle_allowed_for_speed_claims": False,
    "floating_point_certificate": False,
    "successes_renormalized": False,
    "interpretation": "Separate complete-law distances; not an additive error decomposition or replacement primary result",
    "selector_scope": "Identical Gaussian selector source, covariance, configuration and tie rules; event evaluator alone is replaced",
}


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(value.shape), "dtype": value.dtype.str},
                             sort_keys=True).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def extended_tv(first, first_failure, second, second_failure=0.0):
    return float(.5 * (np.abs(first - second).sum() + abs(first_failure - second_failure)))


def validate_law(distribution, failure, modes, label):
    require(distribution.shape == (1 << modes,), f"{label}: wrong law dimension")
    require(np.all(np.isfinite(distribution)) and np.all(distribution >= 0),
            f"{label}: law is not finite/nonnegative")
    require(np.isfinite(failure) and 0 <= failure <= 1, f"{label}: invalid failure mass")
    require(abs(float(np.sum(distribution, dtype=np.longdouble)) + failure - 1) < 2e-11,
            f"{label}: extended law not normalized")


def failure_summary(failures):
    groups = defaultdict(lambda: {"prefixes": 0, "mass": 0.0})
    for item in failures:
        error = item["error"]
        if error.startswith("relative event indicator"):
            reason = "relative_event_indicator_guard"
        elif error == "completion log-score roundoff guard failed":
            reason = "propagated_log_score_indicator_guard"
        else:
            reason = item["error_type"] + ": " + error
        groups[reason]["prefixes"] += 1
        groups[reason]["mass"] += item["mass"]
    return dict(sorted(groups.items()))


def validate_primary(primary):
    """Read/bind the completed archive before creating any diagnostic output."""
    metadata = read_json(primary / "metadata.json")
    require(metadata.get("status") == "complete", "Primary metadata must be complete")
    require(metadata.get("sources_and_binary_stable") is True
            and metadata.get("crosschecks_passed") is True,
            "Primary source stability and crosschecks must pass")
    require(metadata.get("instances") == 18 and metadata.get("method_evaluations") == 126,
            "Primary must contain all 18 instances and 126 laws")
    config = read_json(primary / "config.json")
    require(sha(primary / "config.json") == metadata["config_sha256"], "Primary config hash differs")
    require(config["modes"] == [8, 10, 12] and config["seeds_per_mode"] == list(SEEDS)
            and config["profiles"] == list(PROFILES), "Primary fixed input family differs")
    expected_options = {"selector": "click_influence", "order": "natural physical-mode order",
                        "block_order": "measurement", "cache_size": 8192,
                        "max_relative_roundoff": 1e-6, "full_law_max_modes": 12,
                        "planned_instances": 18, "planned_method_evaluations": 126}
    require(all(config.get(k) == v for k, v in expected_options.items()),
            "Primary configuration differs from the fixed diagnostic scope")
    require([s for s in config["methods"] if s["id"] == "pair_k4"] == [METHOD],
            "Primary pair_k4 specification differs")
    gate_path = primary / "validation_gate.json"
    gate = read_json(gate_path)
    require(sha(gate_path) == metadata["validation_sha256"] and gate.get("passed") is True,
            "Primary validation gate differs or did not pass")
    sources = metadata["source_hashes"]
    require(all(p in sources for p in ("experiments/iteration6_completion.py",
            "src/gbslab/positive_completion.py", "src/gbslab/conditional_memory.py",
            "src/gbslab/zero_memory.py", "src/gbslab/schur_memory.py", "src/gbslab/exact.py")),
            "Primary source binding is incomplete")
    for name, digest in sources.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Primary source path must be a safe relative path")
        require(sha(ROOT / relative) == digest
                and sha(primary / "source_snapshot" / relative) == digest,
                f"Frozen source changed: {name}")
    require(gate["source_hashes"]["src/gbslab/positive_completion.py"]
            == sources["src/gbslab/positive_completion.py"], "Primary module/gate binding differs")
    inputs = read_json(primary / "inputs.json")
    metrics = read_json(primary / "metrics.json")
    checks = read_json(primary / "crosschecks.json")
    expected_cases = {f"m{m}_{p['name']}_s{s}"
                      for m in (8, 10, 12) for s in SEEDS for p in PROFILES}
    require(len(inputs) == 18 and {x["case_id"] for x in inputs} == expected_cases,
            "Primary input manifest is incomplete or duplicated")
    expected_rows = {(case, spec["id"]) for case in expected_cases for spec in config["methods"]}
    require(len(metrics) == 126 and {(x["case_id"], x["method"]) for x in metrics} == expected_rows,
            "Primary metric matrix is incomplete or duplicated")
    require(len(checks) == 42 and all(x.get("passed") is True for x in checks),
            "Primary crosscheck archive is incomplete")
    inputs = {x["case_id"]: x for x in inputs}
    rows = {(x["case_id"], x["method"]): x for x in metrics}
    bound_paths = [primary / name for name in ("metadata.json", "config.json", "validation_gate.json",
                                              "inputs.json", "metrics.json", "crosschecks.json")]
    bound_paths.extend(primary / "source_snapshot" / name for name in sources)
    cases = []
    for case_id in CASE_IDS:
        directory = primary / case_id
        target_path, law_path, row_path = (directory / name for name in
                                         ("target.npz", "pair_k4.npz", "pair_k4.json"))
        entry = inputs[case_id]
        require(entry["modes"] == 12 and sha(target_path) == entry["input_sha256"],
                f"Target input differs: {case_id}")
        row = read_json(row_path)
        require({k: v for k, v in row.items() if k != "failures"} == rows[(case_id, "pair_k4")],
                f"Per-case primary report differs from metrics: {case_id}")
        require(row["modes"] == 12 and row["completion"] == "pair_block" and row["k"] == 4
                and row["status"] in ("complete", "complete_with_failure_mass"),
                f"Invalid primary method record: {case_id}")
        require(sha(law_path) == row["law_sha256"], f"Primary law hash differs: {case_id}")
        with np.load(target_path, allow_pickle=False) as data:
            require(set(data.files) == {"covariance", "target"}, "Unexpected target archive fields")
            V, P = data["covariance"].copy(), data["target"].copy()
        with np.load(law_path, allow_pickle=False) as data:
            require(set(data.files) == {"distribution", "failure_mass"}, "Unexpected primary law fields")
            q, failure = data["distribution"].copy(), float(data["failure_mass"])
        require(V.shape == (24, 24) and np.all(np.isfinite(V)), "Invalid saved covariance")
        validate_law(P, 0., 12, "target")
        validate_law(q, failure, 12, "primary")
        require(failure == row["failure_mass"], f"Primary failure mass differs: {case_id}")
        require(abs(extended_tv(q, failure, P) - row["extended_tv"]) < 2e-14,
                f"Primary TV differs from stored complete law: {case_id}")
        failures = row["failures"]
        prefixes = [tuple(item["prefix"]) for item in failures]
        require(len(failures) == row["unresolved_prefixes"] and len(set(prefixes)) == len(prefixes),
                f"Invalid failure prefix inventory: {case_id}")
        for prefix, item in zip(prefixes, failures):
            require(len(prefix) < 12 and all(v in (0, 1) for v in prefix)
                    and np.isfinite(item["mass"]) and item["mass"] > 0,
                    f"Invalid primary failure entry: {case_id}")
            require(not any(prefix[:n] in set(prefixes) for n in range(len(prefix))),
                    f"Primary failure prefixes overlap: {case_id}")
        require(abs(sum(x["mass"] for x in failures) - failure) < 2e-14,
                f"Failure prefixes do not account for primary failure mass: {case_id}")
        bound_paths.extend((target_path, law_path, row_path))
        cases.append({"case_id": case_id, "input": entry, "report": row,
                      "covariance": V, "target": P, "distribution": q, "failure_mass": failure})
    return config, metadata, cases, {str(p.relative_to(primary)): sha(p) for p in bound_paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=ROOT / "results/iteration6_completion")
    parser.add_argument("--output", type=Path, default=ROOT / "results/iteration6_numerical_diagnostic")
    parser.add_argument("--check-only", action="store_true",
                        help="Check completed-primary bindings without enumeration or output writes")
    args = parser.parse_args()
    primary, output = args.primary.resolve(), args.output.resolve()
    require(output != primary and not output.is_relative_to(primary) and not primary.is_relative_to(output),
            "Diagnostic output must be separate from the primary archive")
    config, primary_metadata, cases, primary_hashes = validate_primary(primary)
    if args.check_only:
        print(json.dumps({"primary_bindings_passed": True, "cases": list(CASE_IDS),
                          "case_count": len(cases), "enumerations_performed": 0}))
        return
    if output.exists():
        raise FileExistsError("Use a new output directory; no overwrite or silent diagnostic retry")
    source_hashes = {**primary_metadata["source_hashes"],
                     str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    output.mkdir(parents=True)
    for name in source_hashes:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    metadata = {"schema": "gbslab.iteration6_numerical_diagnostic.v1", "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(), **SCOPE,
                "primary_directory": str(primary), "primary_completed_utc": primary_metadata["completed_utc"],
                "method": METHOD, "fixed_case_ids": list(CASE_IDS), "source_hashes": source_hashes,
                "primary_file_hashes": primary_hashes,
                "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}
    save(output / "metadata.json", metadata)
    rows = []
    overall_start = perf_counter()
    for case in cases:
        case_id = case["case_id"]
        directory = output / case_id
        directory.mkdir()
        started = perf_counter()
        sampler, model = make_sampler(case["covariance"], METHOD, config, target=case["target"])
        setup_seconds = perf_counter() - started
        table_q, table_failure, table_failures, elapsed = extended_law(sampler, 12)
        validate_law(table_q, table_failure, 12, "table reference")
        primary_q, primary_failure, P = case["distribution"], case["failure_mass"], case["target"]
        distance = extended_tv(primary_q, primary_failure, table_q, table_failure)
        primary_tv = extended_tv(primary_q, primary_failure, P)
        table_tv = extended_tv(table_q, table_failure, P)
        require(abs(primary_tv - table_tv) <= distance + 2e-14,
                "Complete-law distances violate the reverse triangle inequality")
        np.savez_compressed(directory / "laws.npz", target=P,
                            primary_distribution=primary_q, primary_failure_mass=np.asarray(primary_failure),
                            table_distribution=table_q, table_failure_mass=np.asarray(table_failure))
        row = {"case_id": case_id, "method": "pair_k4", "modes": 12,
               "primary_to_target_extended_tv": primary_tv,
               "table_to_target_extended_tv": table_tv,
               "primary_to_table_extended_tv": distance,
               "primary_minus_table_target_tv": primary_tv - table_tv,
               "primary_failure_mass": primary_failure, "table_failure_mass": table_failure,
               "primary_failure_prefixes": len(case["report"]["failures"]),
               "table_failure_prefixes": len(table_failures),
               "primary_failure_reasons": failure_summary(case["report"]["failures"]),
               "table_failure_reasons": failure_summary(table_failures),
               "input_sha256": case["input"]["input_sha256"],
               "covariance_array_sha256": array_sha(case["covariance"]),
               "target_array_sha256": array_sha(P),
               "primary_law_sha256": case["report"]["law_sha256"],
               "diagnostic_laws_sha256": sha(directory / "laws.npz"),
               "source_hashes": source_hashes,
               "table_setup_seconds_descriptive_only": setup_seconds,
               "table_enumeration_seconds_descriptive_only": elapsed}
        save(directory / "report.json", {**row, **SCOPE,
             "primary_failures": case["report"]["failures"], "table_failures": table_failures,
             "saved_target_diagnostics": case["input"]["target_diagnostics"],
             "table_diagnostics": sampler.cache_info(), "table_model_diagnostics": model.diagnostics})
        rows.append({**row, "report_sha256": sha(directory / "report.json")})
        print(json.dumps({key: row[key] for key in ("case_id", "primary_to_target_extended_tv",
              "table_to_target_extended_tv", "primary_to_table_extended_tv",
              "primary_failure_mass", "table_failure_mass")}), flush=True)
    primary_stable = all(sha(primary / name) == digest for name, digest in primary_hashes.items())
    sources_stable = all(sha(ROOT / name) == digest for name, digest in source_hashes.items())
    metadata.update(status="complete" if primary_stable and sources_stable else "investigation_required",
                    completed_utc=datetime.now(timezone.utc).isoformat(), case_count=len(rows),
                    primary_files_unchanged=primary_stable, sources_unchanged=sources_stable,
                    elapsed_seconds=perf_counter() - overall_start)
    save(output / "summary.json", {**SCOPE, "status": metadata["status"], "case_count": len(rows),
         "primary_files_unchanged": primary_stable, "sources_unchanged": sources_stable,
         "source_hashes": source_hashes, "rows": rows})
    save(output / "metadata.json", metadata)
    (output / "README.md").write_text(
        "# Iteration 6 post hoc numerical diagnostic\n\n"
        "This separate diagnostic was chosen after a numerical failure was observed in the primary run. "
        "It includes all six fixed m=12 inputs for pair_k4, without changing k, the completion rule, "
        "the Gaussian selector, guard threshold, or the primary archive. No new samples are drawn.\n\n"
        "The saved full target P supplies event probabilities through TableEventOracle. This is an "
        "exponential reference and inherits the saved target's floating arithmetic and any recorded "
        "corrections. Its timings cannot support native or scalable speed claims.\n\n"
        "Each case preserves both full laws and their failure symbols. Three distances are reported: "
        "primary to P, table reference to P, and primary to table reference. Their differences are "
        "not an additive decomposition of formula and numerical errors. A table improvement diagnoses "
        "oracle/guard sensitivity; it does not replace the primary result or establish a certified "
        "exact-arithmetic law. Primary failure reasons and masses remain in each report.\n")
    print(json.dumps({"status": metadata["status"], "cases": len(rows),
                      "summary_sha256": sha(output / "summary.json"),
                      "primary_files_unchanged": primary_stable,
                      "sources_unchanged": sources_stable}), flush=True)
    if not primary_stable or not sources_stable:
        raise SystemExit("Archived inputs or frozen sources changed during diagnosis")


if __name__ == "__main__":
    main()
