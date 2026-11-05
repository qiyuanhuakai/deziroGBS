"""Fixed post hoc event-oracle diagnosis of 18 iteration-7 primary laws.

Prepare a manifest before diagnosis with --prepare-manifest, then run only
after the complete 9-input/72-law primary experiment has finished.  This
includes every m10/m12 lossy/bright/stress input for pair_k4, chain_k4 and
tree_k4.  It does not select only failed laws or replace primary results.

The saved full target supplies TableEventOracle events.  The same Gaussian
selector, static MI tree, parameters, guards and protected traversal remain.
The table is an exponential floating reference, not a numerical certificate
or a scalable sampler.  Report three extended-law distances separately;
they are not an additive approximation/numerical-error decomposition.

The archive-validation pattern and small reporting helpers follow the frozen
iteration6_numerical_diagnostic.py.  No iteration-6 source is modified.
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
import traceback

import numpy as np
import scipy

from iteration7_forest import extended_law, make_sampler, plain, save, sha


ROOT = Path(__file__).resolve().parents[1]
SEED = 340701
MODES = (10, 12)
PROFILES = (
    {"name": "lossy", "squeezing": .6, "transmissivity": .35},
    {"name": "bright", "squeezing": .9, "transmissivity": .65},
    {"name": "stress", "squeezing": 1.2, "transmissivity": .85},
)
METHODS = (
    {"id": "pair_k4", "completion": "pair_block", "k": 4},
    {"id": "chain_k4", "completion": "chain", "k": 4},
    {"id": "tree_k4", "completion": "static_mi_tree", "k": 4},
)
PRIMARY_METHODS = (
    {"id": "retained_k4", "completion": "none", "k": 4},
    {"id": "retained_k6", "completion": "none", "k": 6},
    {"id": "pair_k2", "completion": "pair_block", "k": 2},
    METHODS[0],
    {"id": "chain_k2", "completion": "chain", "k": 2},
    METHODS[1],
    {"id": "tree_k2", "completion": "static_mi_tree", "k": 2},
    METHODS[2],
)
CASE_IDS = tuple(f"m{m}_{p['name']}_s{SEED}" for m in MODES for p in PROFILES)
FIXED_ROWS = tuple({"case_id": case, "method": method["id"]}
                   for case in CASE_IDS for method in METHODS)
GATE_SOURCES = {
    "src/gbslab/positive_forest.py", "tests/test_positive_forest.py",
    "src/gbslab/positive_completion.py", "src/gbslab/conditional_memory.py",
    "src/gbslab/zero_memory.py", "src/gbslab/schur_memory.py",
    "src/gbslab/exact.py", "src/gbslab/gaussian.py",
}
PRIMARY_SOURCES = (GATE_SOURCES - {"tests/test_positive_forest.py"}) | {
    "experiments/iteration7_forest.py", "src/gbslab/cpp_click.py"}
SCOPE = {
    "analysis_role": "post_hoc_numerical_diagnostic",
    "selection_reason": "Nonzero primary failure mass was observed for m10 lossy in pair_k4, chain_k4 and tree_k4; inspect all fixed m10 and m12 inputs for these three methods",
    "case_selection_was_preregistered": False,
    "fixed_before_diagnostic_enumeration": True,
    "all_cases_in_fixed_m10_m12_stratum_included": True,
    "new_samples_or_inputs": False,
    "primary_results_replaced": False,
    "algorithm_or_budget_tuned": False,
    "table_oracle_is_exponential_reference": True,
    "table_oracle_allowed_for_speed_claims": False,
    "floating_point_certificate": False,
    "successes_renormalized": False,
    "interpretation": "Separate complete extended-law distances; not an additive error decomposition or replacement primary result",
    "selector_and_tree_scope": "Identical Gaussian source, saved covariance, selector, static unconditional MI tree, tie rules, budgets and guards; event evaluator alone is replaced",
}


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_path(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts,
            "A bound source path must be a safe relative path")
    return path


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
            f"{label}: law must be finite and nonnegative")
    require(np.isfinite(failure) and 0 <= failure <= 1, f"{label}: invalid failure mass")
    require(abs(float(np.sum(distribution, dtype=np.longdouble)) + failure - 1) < 2e-11,
            f"{label}: extended law is not normalized")


def validate_failures(failures, failure, count, modes, label):
    prefixes = [tuple(item["prefix"]) for item in failures]
    prefix_set = set(prefixes)
    require(len(failures) == count and len(prefix_set) == len(prefixes),
            f"{label}: invalid or duplicate failure prefixes")
    for prefix, item in zip(prefixes, failures):
        require(len(prefix) < modes and all(v in (0, 1) for v in prefix)
                and np.isfinite(item["mass"]) and item["mass"] > 0,
                f"{label}: invalid failure entry")
        require(not any(prefix[:n] in prefix_set for n in range(len(prefix))),
                f"{label}: failure prefixes overlap")
    require(abs(sum(x["mass"] for x in failures) - failure) < 2e-14,
            f"{label}: failure prefix mass differs")


def failure_summary(failures):
    groups = defaultdict(lambda: {"prefixes": 0, "mass": 0.0})
    for item in failures:
        error = item["error"]
        if error.startswith("relative event indicator"):
            reason = "relative_event_indicator_guard"
        elif error in ("completion log-score roundoff guard failed",
                       "forest log-score roundoff guard failed"):
            reason = "propagated_log_score_indicator_guard"
        else:
            reason = item["error_type"] + ": " + error
        groups[reason]["prefixes"] += 1
        groups[reason]["mass"] += item["mass"]
    return dict(sorted(groups.items()))


def validate_context(primary):
    """Bind frozen science/configuration; callable while primary is running."""
    metadata = read_json(primary / "metadata.json")
    require(metadata.get("status") in ("running", "complete"),
            "Primary must be running or complete, without an investigation status")
    config = read_json(primary / "config.json")
    require(sha(primary / "config.json") == metadata["config_sha256"], "Primary config hash differs")
    require(config["modes"] == [8, 10, 12] and config["seeds_per_mode"] == [SEED]
            and config["profiles"] == list(PROFILES) and config["methods"] == list(PRIMARY_METHODS),
            "Primary fixed inputs or methods differ")
    expected = {"selector": "click_influence", "order": "natural physical-mode order",
                "block_order": "measurement", "cache_size": 8192,
                "max_relative_roundoff": 1e-6, "full_law_max_modes": 12,
                "planned_instances": 9, "planned_method_evaluations": 72}
    require(all(config.get(k) == v for k, v in expected.items()),
            "Primary parameters differ from the fixed diagnostic scope")
    gate = read_json(primary / "validation_gate.json")
    require(sha(primary / "validation_gate.json") == metadata["validation_sha256"]
            and gate.get("passed") is True and GATE_SOURCES.issubset(gate["source_hashes"]),
            "Primary validation gate is incomplete, changed or failed")
    sources = metadata["source_hashes"]
    require(PRIMARY_SOURCES.issubset(sources), "Primary source binding is incomplete")
    for name, digest in sources.items():
        path = relative_path(name)
        require(sha(ROOT / path) == digest and sha(primary / "source_snapshot" / path) == digest,
                f"Frozen primary source changed: {name}")
    for name, digest in gate["source_hashes"].items():
        require(sha(ROOT / relative_path(name)) == digest, f"Validation dependency changed: {name}")
        if name in sources:
            require(sources[name] == digest, f"Primary/gate source binding differs: {name}")
    binary = ROOT / ".cache/click-prefix-native/click_sampler"
    native_gate = read_json(ROOT / "results/iteration5_prefix_validation/validation.json")
    require(sha(binary) == metadata["native_executable_sha256"]
            and native_gate.get("passed") is True
            and native_gate["executable_sha256"] == metadata["native_executable_sha256"],
            "Primary native baseline binary/gate binding differs")
    source_hashes = {**sources, **gate["source_hashes"],
                     str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    return config, metadata, source_hashes


def prepare_manifest(primary, output):
    require(not output.exists(), "Manifest preparation requires a new output directory")
    _, metadata, sources = validate_context(primary)
    observed = []
    for method in METHODS:
        path = primary / f"m10_lossy_s{SEED}" / (method["id"] + ".json")
        report = read_json(path)
        require(report["failure_mass"] > 0, "The stated post hoc motivating observation is missing")
        observed.append({"case_id": report["case_id"], "method": report["method"],
                         "failure_mass": report["failure_mass"],
                         "primary_report_sha256": sha(path)})
    manifest = {"schema": "gbslab.iteration7_numerical_diagnostic_plan.v1", **SCOPE,
                "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "primary_directory": str(primary), "primary_status_at_plan": metadata["status"],
                "primary_config_sha256": metadata["config_sha256"],
                "primary_validation_sha256": metadata["validation_sha256"],
                "primary_native_executable_sha256": metadata["native_executable_sha256"],
                "source_hashes": sources, "modes": list(MODES), "profiles": list(PROFILES),
                "methods": list(METHODS), "fixed_case_ids": list(CASE_IDS),
                "fixed_rows": list(FIXED_ROWS), "planned_instances": 6,
                "planned_method_evaluations": 18, "observations_available_at_plan": observed}
    output.mkdir(parents=True)
    save(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "manifest.json"),
                      "manifest_sha256": sha(output / "manifest.json"),
                      "primary_status": metadata["status"], "planned_evaluations": 18,
                      "enumerations_performed": 0}), flush=True)


def validate_primary(primary):
    """Validate all 9 targets and 72 laws, not just the diagnostic subset."""
    config, metadata, sources = validate_context(primary)
    require(metadata.get("status") == "complete", "Primary metadata must be complete before diagnosis")
    require(metadata.get("sources_and_binary_stable") is True
            and metadata.get("crosschecks_passed") is True
            and metadata.get("complete_matrix_keys_verified") is True,
            "Primary source stability and complete crosschecks must pass")
    require(metadata.get("instances") == 9 and metadata.get("method_evaluations") == 72,
            "Primary must contain all 9 instances and 72 laws")
    inputs = read_json(primary / "inputs.json")
    metrics = read_json(primary / "metrics.json")
    checks = read_json(primary / "crosschecks.json")
    expected_cases = {f"m{m}_{p['name']}_s{SEED}": (m, p)
                      for m in (8, 10, 12) for p in PROFILES}
    require(len(inputs) == 9 and {x["case_id"] for x in inputs} == set(expected_cases),
            "Primary input manifest is incomplete or duplicated")
    expected_rows = {(case, method["id"]) for case in expected_cases for method in PRIMARY_METHODS}
    require(len(metrics) == 72 and {(x["case_id"], x["method"]) for x in metrics} == expected_rows,
            "Primary law matrix is incomplete or duplicated")
    expected_checks = {(case, "retained_k4", "independent_native_baseline") for case in expected_cases}
    expected_checks.update((case, method["id"], "independent_table_events_same_selector")
        for case, (m, _) in expected_cases.items() if m == 8
        for method in PRIMARY_METHODS if method["completion"] != "none")
    require(len(checks) == 27 and all(x.get("passed") is True for x in checks)
            and {(x["case_id"], x["method"], x["kind"]) for x in checks} == expected_checks
            and metadata.get("crosscheck_count") == metadata.get("expected_crosscheck_count") == 27,
            "Primary crosscheck archive is incomplete or duplicated")
    journal = [json.loads(line) for line in (primary / "records.jsonl").read_text().splitlines() if line.strip()]
    require(journal == metrics, "Primary journal differs from its completed metric matrix")
    input_map = {x["case_id"]: x for x in inputs}
    metric_map = {(x["case_id"], x["method"]): x for x in metrics}
    cases = []
    for case_id, (m, profile) in expected_cases.items():
        directory = primary / case_id
        entry = input_map[case_id]
        require(entry["modes"] == m and entry["profile"] == profile
                and entry["seed"] == SEED + 1000*m
                and sha(directory / "target.npz") == entry["input_sha256"],
                f"Saved target or its fixed input specification differs: {case_id}")
        with np.load(directory / "target.npz", allow_pickle=False) as data:
            require(set(data.files) == {"covariance", "target"}, "Unexpected target archive fields")
            V, P = data["covariance"].copy(), data["target"].copy()
        require(V.shape == (2*m, 2*m) and np.all(np.isfinite(V)), "Invalid saved covariance")
        validate_law(P, 0., m, case_id + " target")
        for method in PRIMARY_METHODS:
            label = case_id + "/" + method["id"]
            report = read_json(directory / (method["id"] + ".json"))
            require({k: v for k, v in report.items() if k != "failures"}
                    == metric_map[(case_id, method["id"])], f"Per-law report differs: {label}")
            require(report["modes"] == m and report["profile"] == profile["name"]
                    and report["seed"] == entry["seed"] and report["k"] == method["k"]
                    and report["completion"] == method["completion"], f"Method specification differs: {label}")
            law_path = directory / (method["id"] + ".npz")
            require(sha(law_path) == report["law_sha256"], f"Primary law hash differs: {label}")
            with np.load(law_path, allow_pickle=False) as data:
                require(set(data.files) == {"distribution", "failure_mass"}, "Unexpected primary law fields")
                q, failure = data["distribution"].copy(), float(data["failure_mass"])
            validate_law(q, failure, m, label)
            require(report["failure_mass"] == failure and report["status"] == (
                    "complete" if failure == 0 else "complete_with_failure_mass"),
                    f"Failure mass/status differs: {label}")
            require(abs(extended_tv(q, failure, P) - report["extended_tv"]) < 2e-14,
                    f"Stored TV differs from the complete law: {label}")
            validate_failures(report["failures"], failure, report["unresolved_prefixes"], m, label)
            if m in MODES and method in METHODS:
                cases.append({"case_id": case_id, "method": method, "modes": m,
                              "input": entry, "report": report, "covariance": V,
                              "target": P, "distribution": q, "failure_mass": failure})
    require([{"case_id": x["case_id"], "method": x["method"]["id"]} for x in cases]
            == list(FIXED_ROWS), "Diagnostic matrix differs from its fixed order")
    # Protect the whole completed primary archive, including native outputs,
    # saved crosschecks and sources, as well as all target and law archives.
    primary_hashes = {str(p.relative_to(primary)): sha(p)
                      for p in sorted(primary.rglob("*")) if p.is_file()}
    return config, metadata, sources, cases, primary_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=ROOT / "results/iteration7_forest")
    parser.add_argument("--output", type=Path, default=ROOT / "results/iteration7_numerical_diagnostic")
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--prepare-manifest", action="store_true")
    options.add_argument("--check-only", action="store_true",
                         help="Validate the completed primary and fixed manifest without enumeration")
    args = parser.parse_args()
    primary, output = args.primary.resolve(), args.output.resolve()
    require(output != primary and not output.is_relative_to(primary) and not primary.is_relative_to(output),
            "Diagnostic output must be separate from the primary archive")
    if args.prepare_manifest:
        prepare_manifest(primary, output)
        return
    manifest_path = output / "manifest.json"
    manifest = read_json(manifest_path)
    manifest_sha = sha(manifest_path)
    config, primary_metadata, source_hashes, cases, primary_hashes = validate_primary(primary)
    require(all(manifest.get(k) == value for k, value in SCOPE.items())
            and manifest["fixed_rows"] == list(FIXED_ROWS)
            and manifest["fixed_case_ids"] == list(CASE_IDS)
            and manifest["methods"] == list(METHODS)
            and manifest["planned_instances"] == 6 and manifest["planned_method_evaluations"] == 18,
            "Prepared diagnostic scope differs")
    require(manifest["primary_directory"] == str(primary)
            and manifest["primary_config_sha256"] == primary_metadata["config_sha256"]
            and manifest["primary_validation_sha256"] == primary_metadata["validation_sha256"]
            and manifest["primary_native_executable_sha256"] == primary_metadata["native_executable_sha256"]
            and manifest["source_hashes"] == source_hashes, "Prepared bindings changed")
    for observed in manifest["observations_available_at_plan"]:
        path = primary / observed["case_id"] / (observed["method"] + ".json")
        require(sha(path) == observed["primary_report_sha256"], "The motivating primary report changed")
    if args.check_only:
        print(json.dumps({"primary_bindings_passed": True, "primary_instances": 9,
                          "primary_laws": 72, "fixed_diagnostic_evaluations": len(cases),
                          "bound_primary_files": len(primary_hashes), "enumerations_performed": 0}))
        return
    require({p.name for p in output.iterdir()} == {"manifest.json"},
            "Output must contain only its prepared manifest; no overwrite or silent retry")
    for name in source_hashes:
        destination = output / "source_snapshot" / relative_path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copy2(primary / "config.json", output / "primary_config.json")
    shutil.copy2(primary / "validation_gate.json", output / "primary_validation_gate.json")
    metadata = {"schema": "gbslab.iteration7_numerical_diagnostic.v1", "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(), **SCOPE,
                "manifest_sha256": manifest_sha, "primary_directory": str(primary),
                "primary_completed_utc": primary_metadata["completed_utc"],
                "primary_instances_validated": 9, "primary_laws_validated": 72,
                "primary_crosschecks_validated": 27, "fixed_rows": list(FIXED_ROWS),
                "source_hashes": source_hashes, "primary_file_hashes": primary_hashes,
                "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}
    save(output / "metadata.json", metadata)
    rows = []
    overall_start = perf_counter()
    with (output / "run.log").open("x") as log, (output / "records.jsonl").open("x") as journal:
        def emit(value):
            line = json.dumps(plain(value), allow_nan=False)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        try:
            emit({"status": "starting_post_hoc_diagnostic", "evaluations": 18,
                  "manifest_sha256": manifest_sha, "new_samples": False})
            for case in cases:
                case_id, method, m = case["case_id"], case["method"], case["modes"]
                directory = output / case_id
                directory.mkdir(exist_ok=True)
                started = perf_counter()
                sampler, model = make_sampler(case["covariance"], method, config, target=case["target"])
                setup_seconds = perf_counter() - started
                table_q, table_failure, table_failures, elapsed = extended_law(sampler, m)
                validate_law(table_q, table_failure, m, case_id + " table " + method["id"])
                validate_failures(table_failures, table_failure, len(table_failures), m, "table reference")
                primary_q, primary_failure, P = case["distribution"], case["failure_mass"], case["target"]
                primary_tv = extended_tv(primary_q, primary_failure, P)
                table_tv = extended_tv(table_q, table_failure, P)
                distance = extended_tv(primary_q, primary_failure, table_q, table_failure)
                require(abs(primary_tv-table_tv) <= distance+2e-14
                        and distance <= primary_tv+table_tv+2e-14,
                        "Complete-law distances violate triangle inequalities")
                law_path = directory / (method["id"] + ".npz")
                np.savez_compressed(law_path, covariance=case["covariance"], target=P,
                    primary_distribution=primary_q, primary_failure_mass=np.asarray(primary_failure),
                    table_distribution=table_q, table_failure_mass=np.asarray(table_failure))
                row = {"case_id": case_id, "method": method["id"], "modes": m,
                    "profile": case["input"]["profile"]["name"], "seed": case["input"]["seed"],
                    "completion": method["completion"], "k": method["k"],
                    "primary_to_target_extended_tv": primary_tv,
                    "table_to_target_extended_tv": table_tv,
                    "primary_to_table_extended_tv": distance,
                    "primary_failure_mass": primary_failure, "table_failure_mass": table_failure,
                    "primary_failure_prefixes": len(case["report"]["failures"]),
                    "table_failure_prefixes": len(table_failures),
                    "primary_failure_reasons": failure_summary(case["report"]["failures"]),
                    "table_failure_reasons": failure_summary(table_failures),
                    "input_sha256": case["input"]["input_sha256"],
                    "covariance_array_sha256": array_sha(case["covariance"]),
                    "target_array_sha256": array_sha(P),
                    "primary_law_sha256": case["report"]["law_sha256"],
                    "diagnostic_laws_sha256": sha(law_path),
                    "table_setup_seconds_descriptive_only": setup_seconds,
                    "table_enumeration_seconds_descriptive_only": elapsed}
                report_path = directory / (method["id"] + ".json")
                save(report_path, {**row, **SCOPE, "source_hashes": source_hashes,
                    "primary_failures": case["report"]["failures"], "table_failures": table_failures,
                    "saved_target_diagnostics": case["input"]["target_diagnostics"],
                    "table_diagnostics": sampler.cache_info(), "table_model_diagnostics": model.diagnostics})
                rows.append({**row, "report_sha256": sha(report_path)})
                journal.write(json.dumps(plain(rows[-1]), allow_nan=False) + "\n")
                journal.flush()
                emit({key: row[key] for key in ("case_id", "method", "primary_to_target_extended_tv",
                    "table_to_target_extended_tv", "primary_to_table_extended_tv",
                    "primary_failure_mass", "table_failure_mass")})
            require([{"case_id": r["case_id"], "method": r["method"]} for r in rows] == list(FIXED_ROWS),
                    "Diagnostic matrix is incomplete or duplicated")
            primary_stable = all(sha(primary / name) == digest for name, digest in primary_hashes.items())
            sources_stable = all(sha(ROOT / name) == digest for name, digest in source_hashes.items())
            snapshots_stable = all(sha(output / "source_snapshot" / name) == digest
                                   for name, digest in source_hashes.items())
            manifest_stable = sha(manifest_path) == manifest_sha
            stable = primary_stable and sources_stable and snapshots_stable and manifest_stable
            metadata.update(status="complete" if stable else "investigation_required",
                completed_utc=datetime.now(timezone.utc).isoformat(), case_count=6,
                method_evaluations=len(rows), primary_files_unchanged=primary_stable,
                sources_unchanged=sources_stable, source_snapshots_match=snapshots_stable,
                manifest_unchanged=manifest_stable, elapsed_seconds=perf_counter()-overall_start)
            save(output / "summary.json", {**SCOPE, "status": metadata["status"], "case_count": 6,
                "method_evaluations": len(rows), "manifest_sha256": manifest_sha,
                "primary_files_unchanged": primary_stable, "sources_unchanged": sources_stable,
                "source_snapshots_match": snapshots_stable, "manifest_unchanged": manifest_stable,
                "source_hashes": source_hashes, "rows": rows})
            emit({"status": metadata["status"], "cases": 6, "method_evaluations": len(rows),
                  "summary_sha256": sha(output / "summary.json"),
                  "primary_files_unchanged": primary_stable, "sources_unchanged": sources_stable})
            if not stable:
                raise RuntimeError("Archived primary files, frozen sources or diagnostic manifest changed")
        except BaseException as error:
            if metadata["status"] == "running":
                metadata.update(status="aborted", completed_utc=datetime.now(timezone.utc).isoformat())
            metadata.update(completed_evaluations_before_abort=len(rows),
                            exception_type=type(error).__name__, exception=str(error))
            traceback.print_exc(file=log)
            log.flush()
            save(output / "metadata.json", metadata)
            raise
    metadata["run_log_sha256"] = sha(output / "run.log")
    metadata["records_sha256"] = sha(output / "records.jsonl")
    save(output / "metadata.json", metadata)
    (output / "README.md").write_text(
        "# Iteration 7 post hoc numerical diagnostic\n\n"
        "The scope was fixed before these diagnostic enumerations, after primary m10 lossy failures "
        "had been observed. All six m10/m12 inputs and all three k4 completion methods are included: "
        "18 laws, not only failed cases. Every primary target/law and frozen scientific source was "
        "validated and bound. The primary archive, algorithms and thresholds remain unchanged.\n\n"
        "The saved full target P supplies TableEventOracle probabilities. Gaussian retained-click "
        "selection, the static unconditional-MI tree and all tie rules are identical to the primary "
        "configuration. The full-P table is an exponential floating reference which inherits any "
        "recorded target corrections. It provides neither a numerical certificate nor a scalable "
        "sampler. Timings are descriptive only. No new samples are drawn.\n\n"
        "Every report preserves both laws and explicit failure masses. Primary-to-P, table-to-P "
        "and primary-to-table extended TV are separate distances, not an additive decomposition "
        "of approximation and numerical error. Table results diagnose oracle/guard sensitivity and "
        "never replace primary results. NumericalProbabilityError retains reachable failure mass; "
        "program errors or traversal arithmetic failures stop diagnosis. No retry, epsilon adjustment "
        "or conditioning on successful outputs is performed.\n")


if __name__ == "__main__":
    main()
