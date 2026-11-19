"""Execute the predeclared native shared-Schur quality/cost/regression panel.

All complete distributions include an explicit failure symbol. Native timing
uses fresh processes and includes setup. No completed failed batch is replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from gbslab.cpp_click import CppClickError
from gbslab.cpp_forest import CppForestError
from gbslab.exact import ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from iteration8_common import (ROOT, TracedTableForest, compare_trace_selector,
    extended_law, finish_run, make_python_sampler, native_run, plain, save,
    sha, start_run, total_variation, validate_law)


def native_attempt(V, spec, config, args, **options):
    started = perf_counter()
    try:
        result = native_run(V, spec, config, args.retained, args.forest, **options)
        return result, None, perf_counter() - started
    except (CppClickError, CppForestError, subprocess.TimeoutExpired) as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc),
                 "returncode": getattr(exc, "returncode", None),
                 "payload": getattr(exc, "payload", None),
                 "stderr": str(getattr(exc, "stderr", ""))}
        return None, error, perf_counter() - started


def finish_matrix(args, metadata, binaries, rows, checks, started, **extra):
    passed = all(c.get("passed") is True for c in checks)
    complete_laws = all(r.get("status") != "native_run_failed" for r in rows)
    save(args.output / "metrics.json", rows)
    save(args.output / "crosschecks.json", checks)
    stable = finish_run(args.output, metadata, binaries,
        status="complete" if passed and complete_laws else "investigation_required",
        method_evaluations=len(rows), crosscheck_count=len(checks), crosschecks_passed=passed,
        native_runs_completed=complete_laws, elapsed_seconds=perf_counter() - started,
        complete_matrix_keys_verified=True, **extra)
    print(json.dumps({k: metadata[k] for k in ("status", "method_evaluations", "crosscheck_count",
                                             "crosschecks_passed", "elapsed_seconds")}), flush=True)
    if not stable or not passed or not complete_laws:
        raise SystemExit("Panel retained for investigation; no failed cases were replaced")


def quality(args, config, metadata, binaries):
    rows, checks, inputs = [], [], []
    started = perf_counter()
    with (args.output / "records.jsonl").open("w") as journal:
        for m in config["modes"]:
            for base in config["seeds_per_mode"]:
                seed = base + 1000 * m
                U = haar_unitary(m, np.random.default_rng(seed))
                for profile in config["profiles"]:
                    case = f"m{m}_{profile['name']}_s{base}"
                    directory = args.output / case
                    directory.mkdir()
                    V = covariance_from_interferometer(U, profile["squeezing"], loss=profile["transmissivity"])
                    model = ThresholdModel(V, max_exact_modes=m)
                    P = model.full_distribution()
                    np.savez_compressed(directory / "target.npz", covariance=V, target=P)
                    inputs.append({"case_id": case, "modes": m, "seed": seed, "seed_base": base,
                        "profile": profile, "input_sha256": sha(directory / "target.npz"),
                        "target_diagnostics": model.diagnostics})
                    for spec in config["methods"]:
                        new = spec["completion"] != "none"
                        native, error, wall = native_attempt(V, spec, config, args,
                            enumerate_distribution=True, trace=(new and m == 8))
                        row = {"case_id": case, "modes": m, "seed": seed, "seed_base": base,
                            "profile": profile["name"], "method": spec["id"], "k": spec["k"],
                            "completion": spec["completion"], "process_wrapper_seconds": wall}
                        save(directory / (spec["id"] + "_native.json"), native if error is None else error)
                        python = make_python_sampler(V, spec, config, args.library)
                        qp, fp, failp, pyseconds = extended_law(python, m)
                        np.savez_compressed(directory / (spec["id"] + "_python.npz"), distribution=qp, failure_mass=fp)
                        save(directory / (spec["id"] + "_python.json"), {"failure_mass": fp,
                            "failures": failp, "enumeration_seconds": pyseconds, "diagnostics": python.cache_info()})
                        if error is None:
                            failure = float(native.get("failure_mass", 0))
                            q = validate_law(native["distribution"], failure, m)
                            np.savez_compressed(directory / (spec["id"] + ".npz"), distribution=q, failure_mass=failure)
                            row.update(status="complete" if failure == 0 else "complete_with_failure_mass",
                                extended_tv=total_variation(q, failure, P), failure_mass=failure,
                                unresolved_prefixes=len(native.get("failures", [])),
                                timings_seconds=native["timings_seconds"],
                                diagnostics=native.get("enumeration_kernel_diagnostics", native.get("enumeration_diagnostics")),
                                law_sha256=sha(directory / (spec["id"] + ".npz")))
                            delta = total_variation(q, failure, qp, fp)
                            check = {"case_id": case, "method": spec["id"], "kind": "python_orchestration",
                                "law_tv_difference": delta, "python_failure_mass": fp,
                                "passed": delta < config["crosscheck_tv_tolerance"]}
                            if new and m == 8:
                                trace = native["enumeration_trace"]
                                decisions = compare_trace_selector(trace, python)
                                reference = TracedTableForest(P, trace)
                                try:
                                    qt, ft, failt, _ = extended_law(reference, m)
                                    table_delta = total_variation(q, failure, qt, ft)
                                    table_check = {"case_id": case, "method": spec["id"], "kind": "independent_table_same_native_R_E",
                                        "law_tv_difference": table_delta, "reference_failure_mass": ft,
                                        "reference_extended_tv": total_variation(qt, ft, P),
                                        "selector_trace_check": decisions,
                                        "passed": table_delta < config["crosscheck_tv_tolerance"] and decisions["passed"]}
                                    np.savez_compressed(directory / (spec["id"] + "_table.npz"), distribution=qt, failure_mass=ft)
                                except RuntimeError as exc:
                                    # Missing traced descendants after a failed native prefix are
                                    # a coverage failure, never a reference mass of zero.
                                    table_check = {"case_id": case, "method": spec["id"], "kind": "independent_table_same_native_R_E",
                                        "passed": False, "error": str(exc), "selector_trace_check": decisions}
                                checks.append(table_check)
                                save(directory / (spec["id"] + "_table_check.json"), table_check)
                        else:
                            row.update(status="native_run_failed", extended_tv=None, failure_mass=None, error=error)
                            check = {"case_id": case, "method": spec["id"], "kind": "python_orchestration",
                                "passed": False, "reason": "native full law unavailable", "python_failure_mass": fp}
                            if new and m == 8:
                                checks.append({"case_id": case, "method": spec["id"], "kind": "independent_table_same_native_R_E",
                                               "passed": False, "reason": "native full law unavailable"})
                        checks.append(check)
                        save(directory / (spec["id"] + ".json"), row)
                        rows.append(row)
                        journal.write(json.dumps(plain(row), allow_nan=False) + "\n")
                        journal.flush()
                        print(json.dumps({k: row[k] for k in ("case_id", "method", "status", "extended_tv", "failure_mass")}), flush=True)
    expected = {(f"m{m}_{p['name']}_s{s}", method["id"]) for m in config["modes"]
                for s in config["seeds_per_mode"] for p in config["profiles"] for method in config["methods"]}
    if len(rows) != config["planned_method_evaluations"] or len(rows) != len(expected) or {(r["case_id"], r["method"]) for r in rows} != expected:
        raise RuntimeError("Missing/duplicate quality matrix keys")
    expected_checks = {(case, method, "python_orchestration") for case, method in expected}
    expected_checks |= {(case, method, "independent_table_same_native_R_E") for case, method in expected
                        if case.startswith("m8_") and not method.startswith("retained")}
    if len(checks) != len(expected_checks) or {(c["case_id"], c["method"], c["kind"]) for c in checks} != expected_checks:
        raise RuntimeError("Missing/duplicate cross-check matrix keys")
    save(args.output / "inputs.json", inputs)
    finish_matrix(args, metadata, binaries, rows, checks, started, instances=len(inputs))


def timing(args, config, metadata, binaries):
    quality_meta = json.loads((args.quality / "metadata.json").read_text())
    if quality_meta.get("status") != "complete":
        raise ValueError("Timing requires the fixed completed quality panel")
    if quality_meta["binary_hashes"] != metadata["binary_hashes"]:
        raise ValueError("Quality and timing must use the same exact binaries")
    if quality_meta["config_sha256"] != metadata["config_sha256"]:
        raise ValueError("Quality and timing configurations differ")
    input_bindings = {r["case_id"]: r["input_sha256"]
                      for r in json.loads((args.quality / "inputs.json").read_text())}
    settings = config["timing"]
    specs = {s["id"]: s for s in config["methods"]}
    jobs = [(p, n, rep, method) for p in settings["profiles"] for n in settings["sample_counts"]
            for rep in range(settings["repetitions"]) for method in specs]
    np.random.default_rng(settings["job_shuffle_seed"]).shuffle(jobs)
    save(args.output / "job_order.json", jobs)
    rows = []
    started = perf_counter()
    with (args.output / "records.jsonl").open("w") as journal:
        for ordinal, (profile, n, rep, method) in enumerate(jobs):
            case = f"m{settings['modes']}_{profile}_s{settings['seed_base']}"
            target_path = args.quality / case / "target.npz"
            if sha(target_path) != input_bindings[case]:
                raise ValueError("Timing input differs from its quality-panel binding")
            with np.load(target_path, allow_pickle=False) as f:
                V = f["covariance"]
            native, error, wall = native_attempt(V, specs[method], config, args,
                samples=n, seed=settings["sample_seed_base"] + rep)
            row = {"ordinal": ordinal, "case_id": case, "method": method, "requested_samples": n,
                   "repetition": rep, "seed": settings["sample_seed_base"] + rep,
                   "input_sha256": sha(target_path), "process_wrapper_seconds": wall,
                   "status": "complete" if error is None else "native_run_failed"}
            if error is None:
                row.update(timings_seconds=native["timings_seconds"],
                    completed_samples=native.get("completed_samples", native.get("sample_count")),
                    kernel_diagnostics=native.get("sampling_kernel_diagnostics"),
                    diagnostics=native.get("selector_sampling_diagnostics", native.get("diagnostics")))
                if row["completed_samples"] != n:
                    raise ValueError("A purportedly successful timing run is incomplete")
            else:
                payload = error.get("payload") or {}
                row.update(error=error, completed_samples=payload.get("completed_samples", 0))
            save(args.output / f"run_{ordinal:03d}.json", native if error is None else error)
            rows.append(row)
            journal.write(json.dumps(plain(row), allow_nan=False) + "\n")
            journal.flush()
            print(json.dumps({k: row[k] for k in ("ordinal", "case_id", "method", "requested_samples", "repetition", "status")}), flush=True)
    if len(rows) != settings["planned_runs"] or len(set(jobs)) != len(rows):
        raise RuntimeError("Missing/duplicate timing jobs")
    finish_matrix(args, metadata, binaries, rows, [], started,
                  successful_runs=sum(r["status"] == "complete" for r in rows),
                  quality_metadata_sha256=sha(args.quality / "metadata.json"))


def regression(args, config, metadata, binaries):
    panel = config["historical_numerical_regression"]
    specs = [s for s in config["methods"] if s["id"] in panel["methods"]]
    rows, checks = [], []
    started = perf_counter()
    from gbslab.batched_forest import install_canonical_selector
    from gbslab.positive_forest import ForestCompletionSampler
    from gbslab.positive_completion import TableEventOracle
    for m in panel["modes"]:
        for profile in panel["profiles"]:
            case = f"m{m}_{profile}_s{panel['seed_base']}"
            source = ROOT / panel["source"] / case
            directory = args.output / case
            directory.mkdir()
            with np.load(source / "target.npz", allow_pickle=False) as f:
                V, P = f["covariance"], f["target"]
            for spec in specs:
                native, error, wall = native_attempt(V, spec, config, args, enumerate_distribution=True)
                old = json.loads((source / (spec["id"] + ".json")).read_text())
                row = {"case_id": case, "modes": m, "profile": profile, "method": spec["id"],
                    "input_sha256": sha(source / "target.npz"), "archived_row_sha256": sha(source / (spec["id"] + ".json")),
                    "old_extended_tv": old["extended_tv"], "old_failure_mass": old["failure_mass"],
                    "process_wrapper_seconds": wall, "status": "complete" if error is None else "native_run_failed"}
                save(directory / (spec["id"] + "_native.json"), native if error is None else error)
                model = ThresholdModel(V, max_exact_modes=m)
                reference = ForestCompletionSampler(model, spec["k"], forest=spec["completion"],
                    selection=config["selector"], event_oracle=TableEventOracle(P),
                    cache_size=config["python_cache_size"], max_relative_roundoff=config["max_relative_roundoff"])
                install_canonical_selector(reference)
                qt, ft, failt, _ = extended_law(reference, m)
                np.savez_compressed(directory / (spec["id"] + "_table.npz"), distribution=qt, failure_mass=ft)
                if error is None:
                    failure = float(native["failure_mass"])
                    q = validate_law(native["distribution"], failure, m)
                    np.savez_compressed(directory / (spec["id"] + ".npz"), distribution=q, failure_mass=failure)
                    delta = total_variation(q, failure, qt, ft)
                    row.update(extended_tv=total_variation(q, failure, P), failure_mass=failure,
                        table_extended_tv=total_variation(qt, ft, P), table_failure_mass=ft,
                        table_law_tv_difference=delta, kernel_diagnostics=native.get("enumeration_kernel_diagnostics"),
                        law_sha256=sha(directory / (spec["id"] + ".npz")))
                    checks.append({"case_id": case, "method": spec["id"], "kind": "historical_table_reference",
                        "law_tv_difference": delta, "passed": delta < config["crosscheck_tv_tolerance"]})
                else:
                    row.update(error=error, extended_tv=None, failure_mass=None)
                    checks.append({"case_id": case, "method": spec["id"], "kind": "historical_table_reference", "passed": False})
                rows.append(row)
                save(directory / (spec["id"] + ".json"), row)
                print(json.dumps({k: row[k] for k in ("case_id", "method", "status", "old_failure_mass", "failure_mass", "extended_tv")}), flush=True)
    expected = {(f"m{m}_{profile}_s{panel['seed_base']}", method)
                for m in panel["modes"] for profile in panel["profiles"] for method in panel["methods"]}
    if (len(rows) != panel["planned_evaluations"] or len(rows) != len(expected)
            or {(r["case_id"], r["method"]) for r in rows} != expected):
        raise RuntimeError("Historical regression panel is incomplete")
    finish_matrix(args, metadata, binaries, rows, checks, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["quality", "timing", "regression"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/iteration8_batched_forest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation", type=Path, default=ROOT / "results/iteration8_validation/validation.json")
    parser.add_argument("--library", type=Path, default=ROOT / ".cache/iteration8-batched-native/libbatched_forest.so")
    parser.add_argument("--retained", type=Path, default=ROOT / ".cache/iteration8-retained-native/click_sampler")
    parser.add_argument("--forest", type=Path, default=ROOT / ".cache/iteration8-forest-native/batched_forest_sampler")
    parser.add_argument("--quality", type=Path, default=ROOT / "results/iteration8_quality")
    args = parser.parse_args()
    args.output = args.output or ROOT / ("results/iteration8_" + args.phase)
    config = json.loads(args.config.read_text())
    if config["order"] != "natural physical-mode order" or config["native_prefix_cache_entries"] != 0:
        raise ValueError("This runner implements the fixed natural-order/no-prefix-cache panel")
    metadata, binaries = start_run(args.output, args.config, args.validation,
                                  args.library, args.retained, args.forest, args.phase)
    {"quality": quality, "timing": timing, "regression": regression}[args.phase](args, config, metadata, binaries)


if __name__ == "__main__":
    main()
