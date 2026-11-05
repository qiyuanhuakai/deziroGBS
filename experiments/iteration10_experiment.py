"""Run the frozen coherent-core construction-quality and native-cost panels.

The core quality calculation integrates the ideal displacement channel and
threshold POVM on an explicitly rounded/normalized exported coefficient vector.
It is not the full finite-arithmetic sampler law or its failure probability.
No reference calculation is used to generate or select the timing samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from time import perf_counter
import traceback

import numpy as np

from gbslab.coherent_core import (CoherentCoreError, coherent_core_resources,
                                 coherent_core_state, run_coherent_core)
from gbslab.cpp_click import CppClickError, run_cpp_click
from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from iteration10_common import (ROOT, finish_run, plain, save, sha, start_run,
                                total_variation, utc, validate_law)
from iteration10_reference import (ReferenceNumericalError, ReferenceResourceError,
                                   core_state_reference, gaussian_averaged_law)


CORE_SCOPE = ("construction_reference: exported coefficients converted to complex128 "
              "and explicitly normalized, under the ideal Gaussian displacement "
              "channel and threshold POVM; not the finite native sampler law")
RETAINED_SCOPE = ("successfully enumerated native conditional-chain law under ideal "
                  "continuous uniforms, extended by failure; an aborted enumeration "
                  "does not provide an extended law")


def case_id(modes, profile, seed_base):
    return f"m{modes}_{profile}_s{seed_base}"


def methods(core_cutoffs, retained_budgets):
    return ([{"method": f"core_k{k}", "family": "coherent_core", "cutoff": k}
             for k in core_cutoffs]
            + [{"method": f"retained_k{k}", "family": "retained", "retained_budget": k}
               for k in retained_budgets])


def quality_plan(config):
    cases = []
    for m in config["primary_modes"]:
        for base in config["seed_bases"]:
            for profile in config["profiles"]:
                cases.append({"case_id": case_id(m, profile["name"], base),
                    "modes": m, "seed_base": base, "seed": base + 1000*m,
                    "profile": profile["name"], "panel": "primary",
                    "methods": methods(config["core_cutoffs"], config["retained_budgets"])})
    s = config["supplement"]
    cases.append({"case_id": case_id(s["modes"], s["profile"], s["seed_base"]),
        "modes": s["modes"], "seed_base": s["seed_base"],
        "seed": s["seed_base"] + 1000*s["modes"], "profile": s["profile"],
        "panel": "supplement", "methods": methods(s["core_cutoffs"], s["retained_budgets"])})
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate fixed quality input")
    primary = [c for c in cases if c["panel"] == "primary"]
    counts = Counter(s["family"] for c in primary for s in c["methods"])
    if (len(primary) != config["planned_primary_inputs"]
            or counts["coherent_core"] != config["planned_primary_core_laws"]
            or counts["retained"] != config["planned_primary_retained_laws"]):
        raise ValueError("Fixed quality counts disagree with the configuration")
    return cases


def timing_plan(config):
    cases, jobs = [], []
    timing = config["timing"]
    # This generation order and the RNG are fixed before any run.  Saving the
    # complete shuffled list precedes input preparation and all native calls.
    for entry in timing["cases"]:
        c = {**entry, "case_id": case_id(entry["modes"], entry["profile"], entry["seed_base"]),
             "seed": entry["seed_base"] + 1000*entry["modes"]}
        cases.append(c)
        for count in entry["sample_counts"]:
            for rep in range(timing["repeats"]):
                for spec in methods(entry["core_cutoffs"], entry["retained_budgets"]):
                    jobs.append({"case_id": c["case_id"], **spec, "samples": count,
                        "requested_samples": count, "replicate": rep,
                        "seed": timing["sample_seed_base"] + rep})
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate fixed timing input")
    keys = [(j["case_id"], j["method"], j["samples"], j["replicate"]) for j in jobs]
    if len(jobs) != timing["planned_runs"] or len(set(keys)) != len(jobs):
        raise ValueError("Fixed timing keys disagree with the configuration")
    np.random.default_rng(timing["shuffle_seed"]).shuffle(jobs)
    for ordinal, job in enumerate(jobs):
        job["ordinal"] = ordinal
    return cases, jobs


def profile_for(config, name):
    found = [p for p in config["profiles"] if p["name"] == name]
    if len(found) != 1:
        raise ValueError("A fixed profile must exist exactly once: " + name)
    return found[0]


def core_options(args, config):
    return {"executable": args.core, "timeout": config["timing"]["per_run_timeout_seconds"],
        "max_basis": config["native"]["maximum_basis"],
        "max_stage_basis": config["native"]["maximum_sum_stage_basis"]}


def retained_options(args, config):
    t = config["timing"]
    return {"executable": args.retained, "timeout": t["per_run_timeout_seconds"],
        "selection": t["retained_selector"], "vacuum_kernel": t["retained_vacuum_kernel"],
        "prefix_cache_entries": t["retained_prefix_cache_entries"]}


def error_record(exc):
    result = {"type": type(exc).__name__, "message": str(exc),
        "returncode": getattr(exc, "returncode", None),
        "numerical_failure": getattr(exc, "numerical_failure", None),
        "stderr": str(getattr(exc, "stderr", "") or "")}
    if isinstance(exc, subprocess.TimeoutExpired):
        result["timeout_seconds"] = exc.timeout
        for key in ("stdout", "stderr"):
            value = getattr(exc, key, None)
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            result[key] = value
    return result


def known_native_failure(exc):
    if isinstance(exc, subprocess.TimeoutExpired):
        return "native_timeout"
    if isinstance(exc, CoherentCoreError) and exc.numerical_failure:
        return "native_numerical_failed"
    if isinstance(exc, CppClickError):
        if isinstance(exc.__cause__, subprocess.TimeoutExpired):
            return "native_timeout"
        if isinstance(exc.payload, dict) and exc.payload.get("error_type") == "numerical_probability":
            return "native_numerical_failed"
    # Invalid arguments, malformed output, changed executable, and unexpected
    # internal errors are programming/provenance errors, never numeric samples.
    return None


def save_native_failure(path, exc):
    payload = getattr(exc, "payload", None)
    if isinstance(payload, dict):
        save(path, payload)
    else:
        save(path, {"status": "unavailable", "error": error_record(exc),
                    "native_payload_available": False})
    return sha(path)


def emit(journal, event, **values):
    record = {"utc": utc(), "event": event, **values}
    journal.write(json.dumps(plain(record), allow_nan=False) + "\n")
    journal.flush()
    print(json.dumps(plain(record), allow_nan=False), flush=True)


def checkpoint(output, rows, checks=None, inputs=None):
    save(output / "metrics.json", rows)
    if checks is not None:
        save(output / "crosschecks.json", checks)
    if inputs is not None:
        save(output / "inputs.json", inputs)


def artifact_hashes(output):
    return {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "metadata.json"
            and "source_snapshot" not in p.relative_to(output).parts}


def native_coefficient_evidence(native, reference, config, path):
    occ = np.asarray(native["occupations"], dtype=np.int64)
    amp = np.asarray(native["amplitudes"], dtype=np.clongdouble)
    ref_occ = np.asarray(reference["occupations"], dtype=np.int64)
    ref_amp = np.asarray(reference["amplitudes"], dtype=np.complex128)
    lookup = {tuple(n): a for n, a in zip(occ, amp)}
    if len(lookup) != len(occ) or set(lookup) != set(map(tuple, ref_occ)):
        raise ReferenceNumericalError("Native and independent occupation bases disagree")
    aligned = np.asarray([lookup[tuple(n)] for n in ref_occ], dtype=np.clongdouble)
    difference = aligned - ref_amp.astype(np.clongdouble)
    native_norm2 = np.sum(np.abs(aligned)**2, dtype=np.longdouble)
    reference_norm2 = np.sum(np.abs(ref_amp.astype(np.clongdouble))**2, dtype=np.longdouble)
    maximum = np.max(np.abs(difference))
    l2 = np.sqrt(np.sum(np.abs(difference)**2, dtype=np.longdouble))
    rounded = aligned.astype(np.complex128)
    if not np.all(np.isfinite(rounded)) or np.any((aligned != 0) & (rounded == 0)):
        raise ReferenceNumericalError("Native-to-complex128 conversion lost a nonzero coefficient")
    before = float(np.linalg.norm(rounded))
    if not np.isfinite(before) or before <= 0:
        raise ReferenceNumericalError("Converted coefficient normalization is unresolved")
    reference_input = rounded / before
    after = np.sqrt(np.sum(np.abs(reference_input.astype(np.clongdouble))**2, dtype=np.longdouble))
    tail_difference = abs(np.longdouble(native["tail_probability"]) - np.longdouble(reference["tail_probability"]))
    retained_difference = abs(np.longdouble(native["retained_weight"]) - np.longdouble(reference["retained_weight"]))
    settings = config["reference"]
    absolute_tolerance = settings["coefficient_comparison_max_absolute_tolerance"]
    norm_tolerance = settings["coefficient_norm_comparison_tolerance"]
    tail_tolerance = settings["native_tail_comparison_absolute_tolerance"]
    passed = bool(maximum <= absolute_tolerance and l2 <= norm_tolerance
        and abs(native_norm2 - 1) <= norm_tolerance and abs(reference_norm2 - 1) <= norm_tolerance
        and abs(after**2 - 1) <= norm_tolerance
        and tail_difference <= tail_tolerance and retained_difference <= norm_tolerance
        and not np.any(reference_input[ref_occ.sum(axis=1) % 2 == 1] != 0))
    comparison = {"passed": passed, "basis_size": len(occ), "occupation_sets_equal": True,
        "native_norm_squared": native_norm2, "reference_norm_squared": reference_norm2,
        "maximum_absolute_error": maximum, "l2_error": l2,
        "tail_absolute_error": tail_difference, "retained_weight_absolute_error": retained_difference,
        "reference_input_norm_before": before, "reference_input_norm_after": after,
        "reference_input_squared_norm_after": after**2,
        "conversion_l2_change": np.sqrt(np.sum(np.abs(aligned-rounded.astype(np.clongdouble))**2, dtype=np.longdouble)),
        "normalization_l2_change": np.sqrt(np.sum(np.abs(rounded.astype(np.clongdouble)-reference_input.astype(np.clongdouble))**2, dtype=np.longdouble)),
        "maximum_absolute_tolerance": absolute_tolerance, "norm_and_l2_tolerance": norm_tolerance,
        "tail_absolute_tolerance": tail_tolerance, "outward_rounded_certificate": False}
    np.savez_compressed(path, native_occupations=occ, native_amplitudes=amp,
        reference_occupations=ref_occ, reference_amplitudes=ref_amp,
        native_aligned_amplitudes=aligned, reference_input_amplitudes=reference_input,
        displacement_covariance=reference["displacement_covariance"],
        pure_covariance=reference["pure_covariance"], target_covariance=reference["target_covariance"],
        native_retained_weight=np.longdouble(native["retained_weight"]),
        native_tail_probability=np.longdouble(native["tail_probability"]),
        reference_tail_probability=np.longdouble(reference["tail_probability"]),
        reference_input_norm_before=before, reference_input_norm_after=after)
    state = dict(reference, occupations=ref_occ, amplitudes=reference_input, even_initial_support=True)
    return comparison, state


def core_quality(args, config, case, spec, U, V, P, row, checks, journal):
    directory, method = args.output / case["case_id"], spec["method"]
    profile, cutoff = profile_for(config, case["profile"]), spec["cutoff"]
    native_path = directory / (method + "_native.json")
    native = coherent_core_state(U, profile["squeezing"], profile["transmissivity"], cutoff,
                                 **core_options(args, config))
    save(native_path, native)
    row.update(native_result_sha256=sha(native_path), native_timings_seconds=native["timings_seconds"],
               resources=native["resources"], native_diagnostics=native["diagnostics"])
    reference = core_state_reference(U, profile["squeezing"], profile["transmissivity"], cutoff,
                                    max_basis=config["native"]["maximum_basis"])
    coefficient_path = directory / (method + "_coefficients.npz")
    comparison, state = native_coefficient_evidence(native, reference, config, coefficient_path)
    save(directory / (method + "_coefficient_check.json"), comparison)
    row.update(coefficient_comparison=comparison, coefficient_sha256=sha(coefficient_path),
        independent_state_diagnostics=reference["diagnostics"],
        target_covariance_max_absolute_difference=float(np.max(np.abs(reference["target_covariance"] - V))),
        nominal_trace_bound=float(np.sqrt(np.longdouble(native["tail_probability"]))),
        tail_probability=native["tail_probability"])
    checks.append({"case_id": case["case_id"], "method": method, "kind": "native_independent_coefficients",
                   **comparison})
    if not comparison["passed"]:
        row.update(status="coefficient_check_failed", error="Frozen coefficient consistency check failed")
        return
    settings = config["reference"]
    emit(journal, "analytic_reference_started", case_id=case["case_id"], method=method,
         modes=case["modes"], cutoff=cutoff, max_pairs=settings["maximum_compatible_unordered_pairs"],
         max_moments=settings["maximum_wick_moments_per_subset"])
    law = gaussian_averaged_law(state, displacement_covariance=state["displacement_covariance"],
        max_modes=settings["maximum_modes"], max_pairs=settings["maximum_compatible_unordered_pairs"],
        max_moments=settings["maximum_wick_moments_per_subset"],
        negative_atom_tolerance=settings["negative_atom_tolerance"],
        sum_error_tolerance=settings["sum_error_tolerance"],
        correction_l1_tolerance=settings["maximum_correction_l1"])
    q = validate_law(law["distribution"], case["modes"])
    path = directory / (method + ".npz")
    np.savez_compressed(path, distribution=q, raw_distribution=law["raw_distribution"],
                        vacuum_marginals=law["vacuum_marginals"])
    tv = total_variation(q, P)
    nominal_pass = bool(tv <= row["nominal_trace_bound"] + settings["nominal_trace_bound_check_absolute_tolerance"])
    row.update(law_sha256=sha(path), reference_diagnostics=law["diagnostics"],
        nominal_bound_check_passed=nominal_pass,
        nominal_bound_check_scope="floating consistency check only, not a rigorous numerical error certificate",
        status="complete" if nominal_pass else "nominal_bound_check_failed")
    if nominal_pass:
        row["construction_reference_tv"] = tv
    else:
        row.update(observed_construction_reference_tv=tv, error="Nominal core budget consistency check failed")
    checks.append({"case_id": case["case_id"], "method": method, "kind": "nominal_trace_budget_consistency",
        "passed": nominal_pass, "construction_reference_tv": tv, "nominal_trace_bound": row["nominal_trace_bound"],
        "absolute_tolerance": settings["nominal_trace_bound_check_absolute_tolerance"],
        "outward_rounded_certificate": False})


def retained_quality(args, config, case, spec, V, P, row, checks):
    directory, method = args.output / case["case_id"], spec["method"]
    native = run_cpp_click(V, max_clicks=spec["retained_budget"], samples=0,
        raw_samples=False, enumerate_distribution=True, **retained_options(args, config))
    path = directory / (method + "_native.json")
    save(path, native)
    row["native_result_sha256"] = sha(path)
    # The frozen retained enumerator either traverses every positive branch or
    # aborts the whole invocation.  Only a successful complete traversal permits
    # reporting zero numerical-failure mass; no incomplete array is substituted.
    failure = float(native.get("failure_mass", 0.0))
    q = validate_law(native["distribution"], case["modes"], failure)
    path = directory / (method + ".npz")
    np.savez_compressed(path, distribution=q, failure_mass=failure)
    row.update(status="complete", extended_tv=total_variation(q, P, failure), failure_mass=failure,
        law_sha256=sha(path), native_timings_seconds=native["timings_seconds"],
        native_diagnostics=native.get("enumeration_diagnostics"),
        native_failure_scope="all reachable prefixes of the completed native enumeration")
    checks.append({"case_id": case["case_id"], "method": method, "kind": "retained_extended_law_mass",
        "passed": True, "sum_with_failure": float(np.sum(q, dtype=np.longdouble)) + failure})


def quality(args, config, metadata, binaries):
    cases = quality_plan(config)
    expected = {(c["case_id"], s["method"]) for c in cases for s in c["methods"]}
    rows, checks, inputs = [], [], []
    started = perf_counter()
    save(args.output / "matrix_plan.json", cases)
    checkpoint(args.output, rows, checks, inputs)
    with (args.output / "records.jsonl").open("w") as journal:
        try:
            for case in cases:
                directory = args.output / case["case_id"]
                directory.mkdir()
                emit(journal, "target_started", case_id=case["case_id"], modes=case["modes"])
                U = haar_unitary(case["modes"], np.random.default_rng(case["seed"]))
                p = profile_for(config, case["profile"])
                V = covariance_from_interferometer(U, p["squeezing"], loss=p["transmissivity"])
                target_started = perf_counter()
                model = ThresholdModel(V, max_exact_modes=case["modes"])
                P = validate_law(model.full_distribution(), case["modes"])
                target_path = directory / "target.npz"
                np.savez_compressed(target_path, unitary=U, covariance=V, target=P)
                input_row = {k: v for k, v in case.items() if k != "methods"}
                input_row.update(input_path=case["case_id"] + "/target.npz", input_sha256=sha(target_path),
                    target_seconds=perf_counter()-target_started, target_diagnostics=model.diagnostics,
                    minimum_uncertainty_eigenvalue=model.minimum_uncertainty_eigenvalue)
                save(directory / "target.json", input_row)
                inputs.append(input_row)
                for spec in case["methods"]:
                    row = {**input_row, **spec, "status": "running", "construction_reference_tv": None,
                           "extended_tv": None, "failure_mass": None,
                           "quality_scope": CORE_SCOPE if spec["family"] == "coherent_core" else RETAINED_SCOPE}
                    row_started = perf_counter()
                    emit(journal, "quality_row_started", case_id=case["case_id"], method=spec["method"])
                    try:
                        if spec["family"] == "coherent_core":
                            core_quality(args, config, case, spec, U, V, P, row, checks, journal)
                        else:
                            retained_quality(args, config, case, spec, V, P, row, checks)
                    except ReferenceResourceError as exc:
                        row.update(status="reference_resource_rejected", error=error_record(exc),
                                   resource_diagnostics=exc.diagnostics)
                    except (ReferenceNumericalError, NumericalProbabilityError) as exc:
                        row.update(status="reference_numerical_failed", error=error_record(exc))
                    except (CoherentCoreError, CppClickError, subprocess.TimeoutExpired) as exc:
                        status = known_native_failure(exc)
                        row["native_result_sha256"] = save_native_failure(directory / (spec["method"] + "_native.json"), exc)
                        row.update(status=status or "aborted_program_error", error=error_record(exc))
                        if status is None:
                            raise
                    except BaseException as exc:
                        row.update(status="aborted_program_error", error=error_record(exc))
                        raise
                    finally:
                        row["elapsed_seconds"] = perf_counter() - row_started
                        rows.append(row)
                        save(directory / (spec["method"] + ".json"), row)
                        checkpoint(args.output, rows, checks, inputs)
                        emit(journal, "quality_row_finished", case_id=case["case_id"], method=spec["method"],
                             status=row["status"], elapsed_seconds=row["elapsed_seconds"],
                             construction_reference_tv=row["construction_reference_tv"],
                             extended_tv=row["extended_tv"], failure_mass=row["failure_mass"])
            observed = [(r["case_id"], r["method"]) for r in rows]
            if len(observed) != len(expected) or set(observed) != expected:
                raise RuntimeError("Quality matrix keys are missing, duplicated, or expanded")
            emit(journal, "quality_matrix_finished", rows=len(rows), inputs=len(inputs))
        except BaseException as exc:
            save(args.output / "aborted.json", {"error": error_record(exc), "traceback": traceback.format_exc()})
            finish_run(args.output, metadata, binaries, status="aborted", complete_matrix_keys_verified=False,
                method_evaluations=len(rows), elapsed_seconds=perf_counter()-started,
                artifact_hashes=artifact_hashes(args.output))
            raise
    complete = all(r["status"] == "complete" for r in rows) and all(c["passed"] for c in checks)
    stable = finish_run(args.output, metadata, binaries,
        status="complete" if complete else "investigation_required", complete_matrix_keys_verified=True,
        method_evaluations=len(rows), quality_inputs=len(inputs),
        completed_laws=sum(r["status"] == "complete" for r in rows),
        status_counts=dict(Counter(r["status"] for r in rows)),
        crosscheck_count=len(checks), crosschecks_passed=all(c["passed"] for c in checks),
        elapsed_seconds=perf_counter()-started, artifact_hashes=artifact_hashes(args.output),
        core_quality_scope=CORE_SCOPE, retained_quality_scope=RETAINED_SCOPE,
        native_core_global_failure_probability=None, no_sampling_quality_certificate=True)
    print(json.dumps({k: metadata[k] for k in ("status", "method_evaluations", "completed_laws",
                                               "crosscheck_count", "elapsed_seconds")}), flush=True)
    if not stable:
        raise RuntimeError("Sources or binaries changed during the quality matrix")


def timing_input_arrays(args, config, cases, metadata):
    quality_directory = args.quality
    qm = json.loads((quality_directory / "metadata.json").read_text())
    if (qm.get("complete_matrix_keys_verified") is not True
            or qm.get("sources_and_binaries_stable") is not True
            or qm["source_hashes"] != metadata["source_hashes"]
            or qm["binary_hashes"] != metadata["binary_hashes"]
            or qm["config_sha256"] != metadata["config_sha256"]):
        raise ValueError("Timing requires the same frozen, completed quality matrix")
    quality_inputs = {r["case_id"]: r for r in json.loads((quality_directory / "inputs.json").read_text())}
    arrays, inputs = {}, []
    for case in cases:
        cid = case["case_id"]
        qrow = quality_inputs.get(cid)
        if qrow is not None:
            path = quality_directory / cid / "target.npz"
            if sha(path) != qrow["input_sha256"]:
                raise ValueError("The shared quality input changed before timing")
            with np.load(path, allow_pickle=False) as stored:
                # Only physical inputs are accessed here.  No P/Q or histogram
                # is used to tune any timing parameter or choose a replacement.
                U, V = stored["unitary"].copy(), stored["covariance"].copy()
        else:
            p = profile_for(config, case["profile"])
            U = haar_unitary(case["modes"], np.random.default_rng(case["seed"]))
            V = covariance_from_interferometer(U, p["squeezing"], loss=p["transmissivity"])
        directory = args.output / cid
        directory.mkdir()
        path = directory / "input.npz"
        np.savez_compressed(path, unitary=U, covariance=V)
        inputs.append({**case, "input_path": cid + "/input.npz", "input_sha256": sha(path),
            "quality_case_id": cid if qrow else None,
            "quality_target_sha256": qrow["input_sha256"] if qrow else None,
            "full_quality_law_available": qrow is not None})
        arrays[cid] = (U, V)
    return arrays, inputs, {"quality_metadata_sha256": sha(quality_directory / "metadata.json"),
                           "quality_inputs_sha256": sha(quality_directory / "inputs.json")}


def timing(args, config, metadata, binaries):
    cases, jobs = timing_plan(config)
    save(args.output / "job_order.json", jobs)
    rows = []
    checkpoint(args.output, rows)
    started = perf_counter()
    with (args.output / "records.jsonl").open("w") as journal:
        try:
            arrays, inputs, quality_bindings = timing_input_arrays(args, config, cases, metadata)
            save(args.output / "inputs.json", inputs)
            input_map = {r["case_id"]: r for r in inputs}
            boundary = config["resource_boundary"]
            resources = coherent_core_resources(boundary["modes"], boundary["cutoff"],
                max_basis=config["native"]["maximum_basis"],
                max_stage_basis=config["native"]["maximum_sum_stage_basis"])
            save(args.output / "resource_boundary.json", {**boundary, "resources": resources,
                "preflight_rejected": not resources["feasible"], "backend_invoked": False,
                "state_allocated": False, "is_numerical_failure": False,
                "is_replacement_timing_input": False})
            for job in jobs:
                source = input_map[job["case_id"]]
                U, V = arrays[job["case_id"]]
                row = {**job, "modes": source["modes"], "profile": source["profile"],
                    "seed_base": source["seed_base"], "input_path": source["input_path"],
                    "input_sha256": source["input_sha256"],
                    "quality_case_id": source["quality_case_id"],
                    "quality_target_sha256": source["quality_target_sha256"],
                    "status": "running", "completed_samples": None}
                raw_path = args.output / f"run_{job['ordinal']:03d}.json"
                result, request_started = None, perf_counter()
                emit(journal, "timing_run_started", **job)
                try:
                    if job["family"] == "coherent_core":
                        p = profile_for(config, source["profile"])
                        result = run_coherent_core(U, p["squeezing"], p["transmissivity"], job["cutoff"],
                            samples=job["samples"], seed=job["seed"], raw_samples=True,
                            **core_options(args, config))
                    else:
                        result = run_cpp_click(V, max_clicks=job["retained_budget"], samples=job["samples"],
                            seed=job["seed"], raw_samples=True, enumerate_distribution=False,
                            **retained_options(args, config))
                    row.update(status="complete", completed_samples=job["samples"])
                    save(raw_path, result)
                except (CoherentCoreError, CppClickError, subprocess.TimeoutExpired) as exc:
                    status = known_native_failure(exc)
                    result = getattr(exc, "payload", None)
                    row.update(status=status or "aborted_program_error", error=error_record(exc))
                    save_native_failure(raw_path, exc)
                    if isinstance(result, dict):
                        row["completed_samples"] = result.get("completed_samples")
                    if status is None:
                        raise
                except BaseException as exc:
                    row.update(status="aborted_program_error", error=error_record(exc))
                    if not raw_path.exists():
                        save(raw_path, {"status": "unavailable", "error": error_record(exc),
                                        "native_payload_available": False})
                    raise
                finally:
                    row["request_wall_seconds"] = perf_counter() - request_started
                    row["wrapper_seconds"] = (result.get("wrapper_wall_seconds", row["request_wall_seconds"])
                                               if isinstance(result, dict) else row["request_wall_seconds"])
                    row["timings_seconds"] = result.get("timings_seconds") if isinstance(result, dict) else None
                    for name in ("wrapper_input_validation_seconds", "wrapper_process_wall_seconds"):
                        row[name] = result.get(name) if isinstance(result, dict) else None
                    timing_values = row["timings_seconds"]
                    row["native_input_setup_sampling_seconds"] = (sum(timing_values[name] for name in
                        ("input", "setup", "sampling")) if timing_values is not None else None)
                    if isinstance(result, dict):
                        row["diagnostics"] = result.get("diagnostics")
                        row["resources"] = result.get("resources")
                        row["tail_probability"] = result.get("tail_probability")
                        row["nominal_trace_bound"] = result.get("trace_distance_bound")
                        row["preserved_raw_sample_count"] = (len(result["raw_samples"])
                            if result.get("raw_samples") is not None else None)
                    row["raw_result_sha256"] = sha(raw_path)
                    rows.append(row)
                    checkpoint(args.output, rows)
                    emit(journal, "timing_run_finished", ordinal=row["ordinal"], case_id=row["case_id"],
                         method=row["method"], samples=row["samples"], replicate=row["replicate"],
                         status=row["status"], completed_samples=row["completed_samples"],
                         timings_seconds=row["timings_seconds"], wrapper_seconds=row["wrapper_seconds"])
            key = lambda r: (r["case_id"], r["method"], r["samples"], r["replicate"])
            if len(rows) != len(jobs) or {key(r) for r in rows} != {key(j) for j in jobs}:
                raise RuntimeError("Timing matrix keys are missing, duplicated, or expanded")
            emit(journal, "timing_matrix_finished", runs=len(rows))
        except BaseException as exc:
            save(args.output / "aborted.json", {"error": error_record(exc), "traceback": traceback.format_exc()})
            finish_run(args.output, metadata, binaries, status="aborted", complete_matrix_keys_verified=False,
                timing_runs=len(rows), elapsed_seconds=perf_counter()-started,
                artifact_hashes=artifact_hashes(args.output))
            raise
    stable = finish_run(args.output, metadata, binaries,
        status="complete" if all(r["status"] == "complete" for r in rows) else "investigation_required",
        complete_matrix_keys_verified=True, timing_runs=len(rows),
        status_counts=dict(Counter(r["status"] for r in rows)), elapsed_seconds=perf_counter()-started,
        artifact_hashes=artifact_hashes(args.output), **quality_bindings,
        primary_cost="native setup plus sampling", additional_native_cost="input plus setup plus sampling",
        timing_scope={"core_setup": "all stage bases/fibers, unitary check, core parameters, B, Fock coefficients, normalization and tail budget",
            "retained_setup": "covariance symmetry/physicality checks, base Gaussian state and selector preparation",
            "input": "native resource checks and CSV parsing, before setup; differs by backend",
            "core_sampling": "sample buffers, RNG, correlated displacement, coherent state copies, all sequential measurement contractions",
            "wrapper": "validation/provenance, CSV I/O, process/JSON/output validation; preserved separately",
            "outside_native_components": "CLI argument parsing, serialization; full process wall includes these",
            "reference": "zero for formal sampling; no required state preparation is moved here"},
        no_full_law_for_m32=True, completed_batch_is_not_global_failure_certificate=True)
    print(json.dumps({k: metadata[k] for k in ("status", "timing_runs", "status_counts", "elapsed_seconds")}), flush=True)
    if not stable:
        raise RuntimeError("Sources or binaries changed during the timing matrix")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("quality", "timing"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/iteration10_coherent_core.json")
    parser.add_argument("--gate", type=Path, default=ROOT / "results/iteration10_validation/validation.json")
    parser.add_argument("--core", type=Path, default=ROOT / ".cache/coherent-core-native/coherent_core_sampler")
    parser.add_argument("--retained", type=Path, default=ROOT / ".cache/iteration8-retained-native/click_sampler")
    parser.add_argument("--quality", type=Path, default=ROOT / "results/iteration10_quality")
    args = parser.parse_args()
    args.output = args.output or ROOT / ("results/iteration10_" + args.phase)
    config = json.loads(args.config.read_text())
    # Validate both matrices without computing a target, state or sample.
    quality_plan(config)
    timing_plan(config)
    metadata, binaries = start_run(args.output, args.config, args.gate, args.core, args.retained,
                                   "iteration10_" + args.phase)
    (quality if args.phase == "quality" else timing)(args, config, metadata, binaries)


if __name__ == "__main__":
    main()
