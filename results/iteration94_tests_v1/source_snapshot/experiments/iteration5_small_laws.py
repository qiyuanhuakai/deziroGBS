#!/usr/bin/env python3
"""Supplementary complete-law audit of the four frozen Frugal timing inputs.

Run only AFTER the formal timing process finishes.  Inputs are the saved
performance NPZ files, not regenerated circuits.  Both native interfaces are
called with zero requested samples and complete-law enumeration enabled.
The resulting costs are audit overhead and are never new performance trials.

This deterministic review was added after the performance configuration was
frozen.  It is neither preregistered statistical confirmation nor a new iid
sample experiment, and does not consume or alter the primary quality alpha.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from gbslab.cpp_click import CppClickError, run_cpp_click
from gbslab.exact import ThresholdModel
from gbslab.frugal_scaling import frugal_native_sample
from gbslab.gbs_identity import covariance_sha256


ROOT = Path(__file__).resolve().parents[1]
LAW_TOLERANCE = 1e-10
UPSTREAM_COMMIT = "fa8b796f2afebd4845a2e335e457735952617ef8"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def checked_law(values, modes):
    """Expose tiny normalization corrections instead of silently hiding them.

    Native Q vectors must already be nonnegative and normalized to tolerance;
    this is not a repair for an invalid sampler.  Both raw and normalized
    vectors are archived.  TV is computed on the normalized numerical laws.
    """
    raw = np.asarray(values, dtype=float)
    if (raw.shape != (1 << modes,) or not np.all(np.isfinite(raw))
            or np.any(raw < 0) or np.any(raw > 1)):
        raise ArithmeticError("Complete law has invalid shape, support, or finite probabilities")
    mass = float(np.sum(raw, dtype=np.longdouble))
    if abs(mass-1) > LAW_TOLERANCE:
        raise ArithmeticError("Complete law is not normalized within the fixed tolerance")
    normalized = raw/mass
    correction = float(np.sum(np.abs(normalized-raw), dtype=np.longdouble))
    return raw.copy(), normalized, {
        "raw_probability_sum": mass, "normalization_l1_change": correction,
        "normalization_tolerance": LAW_TOLERANCE,
        "minimum_probability": float(raw.min()), "maximum_probability": float(raw.max()),
        "zero_probability_patterns": int(np.count_nonzero(raw == 0)),
        "strict_floating_point_certificate": False,
    }


def tv(p, q):
    return .5*float(np.sum(np.abs(p-q), dtype=np.longdouble))


def binary_binding(performance, role, executable, gate_path):
    archive_path = performance/f"{role}_build_and_validation.json"
    archived = read(archive_path)
    current_build_path = executable.parent/"build.json"
    current_build, current_gate = read(current_build_path), read(gate_path)
    digest = sha(executable)
    if (archived.get("executable_sha256") != digest
            or current_build.get("executable_sha256") != digest
            or current_gate.get("executable_sha256") != digest
            or archived.get("validation_sha256") != sha(gate_path)
            or current_gate.get("passed") is not True
            or current_build != archived.get("build")
            or current_gate != archived.get("validation")):
        raise ValueError(f"{role} binary/build/gate no longer matches the formal performance run")
    if role == "candidate":
        source = ROOT/"experiments/click_sampler.cpp"
        if (current_build.get("source_sha256") != sha(source)
                or current_gate.get("source_sha256") != sha(source)):
            raise ValueError("Candidate enumeration source is not bound to the prefix-cache gate")
        checks = current_gate.get("checks")
        if (not isinstance(checks, list) or not checks
                or any(c.get("passed") is not True for c in checks)
                or current_gate.get("tests_run", 0) <= 0
                or current_gate.get("tests_passed") != current_gate.get("tests_run")
                or current_gate.get("skipped") != 0
                or current_gate.get("source_hashes_stable_during_tests") is not True
                or current_gate.get("build_metadata_matches_sources_and_binary") is not True):
            raise ValueError("Candidate equivalence gate contains a failed check")
    elif (current_build.get("upstream_commit") != UPSTREAM_COMMIT
            or current_build.get("precision") != "float64"
            or current_gate.get("build_cache_key") != current_build.get("cache_key")):
        raise ValueError("Frugal enumeration is not the pinned validated double-precision build")
    return {
        "executable": relative(executable), "executable_sha256": digest,
        "performance_archive": relative(archive_path), "performance_archive_sha256": sha(archive_path),
        "current_build_path": relative(current_build_path), "current_build_sha256": sha(current_build_path),
        "validation_path": relative(gate_path), "validation_sha256": sha(gate_path),
        "build_cache_key": current_build.get("cache_key"), "gate_passed": True,
        "same_binary_as_timing": True,
    }


def freeze_inputs(args):
    metadata_path = args.performance/"metadata.json"
    metadata = read(metadata_path)
    if metadata.get("status") not in ("complete", "complete_with_failures"):
        raise RuntimeError("Formal performance run is not complete; do not compete with timed runs")
    config_path = args.performance/"config.json"
    config, metrics, summary = (read(config_path), read(args.performance/"metrics.json"),
                                read(args.performance/"summary.json"))
    if metadata.get("config_sha256") != sha(config_path):
        raise ValueError("Performance configuration hash is inconsistent")
    if summary.get("rows") != len(metrics) or summary.get("all_same_seed_summary_checks_passed") is not True:
        raise ValueError("Performance completion/paired-equivalence metadata is inconsistent")
    if (config.get("official_frugal_modes") != [8, 12]
            or [p["name"] for p in config["profiles"]] != ["lossy", "bright"]
            or config.get("candidate") != {"selector": "click_influence", "k": 4, "vacuum_kernel": "product"}
            or config.get("official_frugal_cutoff") != 5
            or config.get("cache_capacities") != [0, 65536]):
        raise ValueError("This supplementary audit is restricted to the four originally declared timing inputs/settings")
    bindings = {
        "candidate": binary_binding(args.performance, "candidate", args.candidate_executable, args.candidate_validation),
        "frugal": binary_binding(args.performance, "frugal", args.frugal_executable, args.frugal_validation),
    }
    protected = {p: sha(p) for p in (metadata_path, config_path, args.performance/"metrics.json",
        args.performance/"summary.json", args.performance/"candidate_build_and_validation.json",
        args.performance/"frugal_build_and_validation.json", args.candidate_executable,
        args.frugal_executable, args.candidate_validation, args.frugal_validation,
        args.candidate_executable.parent/"build.json", args.frugal_executable.parent/"build.json")}
    for name in ("experiments/click_sampler.cpp", "src/gbslab/cpp_click.py", "src/gbslab/frugal_scaling.py"):
        path = ROOT/name
        if metadata["source_hashes"].get(path.name) != sha(path):
            raise ValueError(f"Performance wrapper/source changed before supplementary enumeration: {name}")
        protected[path] = sha(path)
    cases = []
    for modes in config["official_frugal_modes"]:
        for profile in config["profiles"]:
            case_id = f"m{modes}_{profile['name']}"
            path = args.performance/f"input_{case_id}.npz"
            input_sha = sha(path)
            selected = [r for r in metrics if r["case_id"] == case_id]
            expected_groups = {(n, method, repetition)
                for n in config["sample_counts_by_modes"][str(modes)]
                for method in ("uncached", "prefix_cached", "official_frugal")
                for repetition in range(config["repetitions"])}
            observed_groups = {(r["samples"], r["method"], r["repetition"]) for r in selected}
            if (len(selected) != len(expected_groups) or observed_groups != expected_groups
                    or any(r.get("input_sha256") != input_sha or r["modes"] != modes
                           or r["profile"] != profile["name"] for r in selected)):
                raise ValueError(f"Saved input is not identical to every declared timing attempt: {case_id}")
            with np.load(path, allow_pickle=False) as archive:
                V = np.array(archive["covariance"], dtype=float, copy=True)
            if V.shape != (2*modes, 2*modes):
                raise ValueError("Saved performance covariance has the wrong dimension")
            protected[path] = input_sha
            cases.append({"case_id": case_id, "modes": modes, "profile": profile["name"],
                "input_path": relative(path), "input_sha256": input_sha,
                "covariance_sha256": covariance_sha256(V),
                "performance_attempts": len(selected),
                "performance_failed_attempts": sum(r["status"] != "ok" for r in selected)})
    return config, cases, bindings, protected


def error_record(exc):
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "payload": getattr(exc, "payload", None), "returncode": getattr(exc, "returncode", None),
            "stderr": stderr}


def run(args):
    # This check precedes directory creation, all target evaluation and all
    # native calls.  There is deliberately no bypass for running timings.
    config, cases, bindings, protected = freeze_inputs(args)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh supplementary-audit directory; do not replace previous results")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    sources = [Path(__file__), ROOT/"src/gbslab/exact.py", ROOT/"src/gbslab/cpp_click.py",
               ROOT/"src/gbslab/frugal_scaling.py", ROOT/"experiments/click_sampler.cpp",
               ROOT/"experiments/frugal_sampling_adapter.cpp"]
    for source in sources:
        shutil.copy2(source, snapshot/source.name)
        protected[source] = sha(source)
    generated_trace = args.frugal_executable.parent/"generated/frugal_trace.hpp"
    if generated_trace.exists():
        shutil.copy2(generated_trace, snapshot/generated_trace.name)
        protected[generated_trace] = sha(generated_trace)
    save(args.output/"performance_config.json", config)
    save(args.output/"binary_bindings.json", bindings)
    save(args.output/"case_plan.json", cases)
    for role in bindings:
        shutil.copy2(args.performance/f"{role}_build_and_validation.json",
                     args.output/f"{role}_build_and_validation.json")
    metadata = {
        "schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "performance_directory": relative(args.performance), "planned_inputs": len(cases),
        "performance_bindings": {relative(p): value for p, value in protected.items()},
        "source_hashes": {p.name: sha(p) for p in snapshot.iterdir()},
        "sample_count_per_native_call": 0, "new_sample_draws": 0,
        "new_performance_trials": 0, "native_timer_fields_used_for_performance": False,
        "statistical_alpha_used": None,
        "audit_design": "Supplementary deterministic complete-law quality review added after the performance configuration was frozen; all four predeclared small Frugal timing inputs are included, without new input generation or favorable-case selection.",
        "preregistered_independent_statistical_confirmation": False,
        "scope": "Full numerical P and induced Q on 8/12-mode synthetic threshold targets; not an extension of the primary quality confidence family or a claim about large Jiuzhang instances.",
        "law_semantics": "Complete conditional-chain laws under ideal continuous uniforms, including actual computed threshold saturation where applicable; not finite-PRNG-state enumeration.",
        "frugal_enumeration_scope": "Generated forced-branch upstream SIMD trace, independently validated against the official sampler. This trace is an adapter audit feature, not an official Frugal full-distribution API.",
        "raw_array_semantics": "Raw means before this audit's normalization step. ThresholdModel's internal guarded clipping/normalization, if any, is reported separately in each target model diagnostic.",
        "strict_floating_point_certificate": False,
    }
    save(args.output/"metadata.json", metadata)
    rows = []
    caught = (CppClickError, OSError, RuntimeError, ValueError, ArithmeticError,
              KeyError, TypeError, subprocess.TimeoutExpired)
    for case in cases:
        case_id, modes = case["case_id"], case["modes"]
        with np.load(args.performance/f"input_{case_id}.npz", allow_pickle=False) as archive:
            V = np.array(archive["covariance"], dtype=float, copy=True)
        row = {**case, "status": "ok", "patterns": 1 << modes, "new_samples": 0,
               "candidate_tv": None, "frugal_tv": None,
               "candidate_over_frugal_tv": None, "candidate_lower_tv": None,
               "strict_floating_point_certificate": False}
        arrays, diagnostics = {"covariance": V}, {}
        print(json.dumps({"starting": case_id, "phase": "complete_law_audit", "samples": 0}), flush=True)
        try:
            target = ThresholdModel(V, max_exact_modes=12)
            raw, normalized, details = checked_law(target.full_distribution(), modes)
            arrays.update(target_p_raw=raw, target_p=normalized)
            diagnostics["target"] = {"status": "ok", "law": details, "model": target.cache_info(),
                                     "minimum_uncertainty_eigenvalue": target.minimum_uncertainty_eigenvalue}
        except caught as exc:
            diagnostics["target"] = error_record(exc)
        try:
            native = run_cpp_click(V, samples=0, raw_samples=False, enumerate_distribution=True,
                selection=config["candidate"]["selector"], max_clicks=config["candidate"]["k"],
                vacuum_kernel=config["candidate"]["vacuum_kernel"], prefix_cache_entries=65536,
                order=tuple(range(modes)), seed=0, executable=args.candidate_executable,
                timeout=args.timeout)
            save(args.output/f"native_candidate_{case_id}.json", native)
            if (native.get("executable_sha256") != bindings["candidate"]["executable_sha256"]
                    or native.get("covariance_sha256") != case["covariance_sha256"]
                    or native.get("sample_count") != 0
                    or native.get("prefix_cache", {}).get("entries_used") != 0):
                raise ValueError("Candidate enumeration target/binary/zero-sample binding failed")
            raw, normalized, details = checked_law(native["distribution"], modes)
            arrays.update(candidate_q_raw=raw, candidate_q=normalized)
            diagnostics["candidate"] = {"status": "ok", "law": details,
                "enumeration": native.get("enumeration_diagnostics"),
                "prefix_cache": native.get("prefix_cache"),
                "native_record": f"native_candidate_{case_id}.json",
                "native_record_sha256": sha(args.output/f"native_candidate_{case_id}.json")}
        except caught as exc:
            diagnostics["candidate"] = error_record(exc)
            save(args.output/f"candidate_failure_{case_id}.json", diagnostics["candidate"])
        try:
            native = frugal_native_sample(V, config["official_frugal_cutoff"], sample_count=0, seed=0,
                enumerate_distribution=True, executable=args.frugal_executable, timeout=args.timeout)
            save(args.output/f"native_frugal_{case_id}.json", native)
            trace = native.get("trace") or {}
            if (native.get("sample_count") != 0 or native.get("modes") != modes
                    or native.get("cutoff") != 5 or native.get("upstream_commit") != UPSTREAM_COMMIT
                    or native.get("real_precision") != 2 or native.get("omp_threads") != 1
                    or trace.get("nonfinite_threshold_path_steps") != 0
                    or trace.get("official_api") is not False):
                raise ValueError("Frugal enumeration target/trace/zero-sample binding failed")
            raw, normalized, details = checked_law(native["distribution"], modes)
            arrays.update(frugal_q_raw=raw, frugal_q=normalized)
            diagnostics["frugal"] = {"status": "ok", "law": details, "trace": trace,
                "native_record": f"native_frugal_{case_id}.json",
                "native_record_sha256": sha(args.output/f"native_frugal_{case_id}.json")}
        except caught as exc:
            diagnostics["frugal"] = error_record(exc)
            save(args.output/f"frugal_failure_{case_id}.json", diagnostics["frugal"])

        for role in ("candidate", "frugal"):
            if "target_p" in arrays and f"{role}_q" in arrays:
                row[f"{role}_tv"] = tv(arrays["target_p"], arrays[f"{role}_q"])
                row[f"{role}_tv_raw_arrays"] = tv(arrays["target_p_raw"], arrays[f"{role}_q_raw"])
                row[f"{role}_tv_normalization_change_bound"] = .5*(
                    diagnostics["target"]["law"]["normalization_l1_change"]
                    + diagnostics[role]["law"]["normalization_l1_change"])
        if row["candidate_tv"] is not None and row["frugal_tv"] is not None:
            c, f = row["candidate_tv"], row["frugal_tv"]
            row["candidate_minus_frugal_tv"] = c-f
            row["candidate_over_frugal_tv"] = c/f if f else None
            row["candidate_lower_tv"] = c < f
            row["candidate_to_frugal_tv"] = tv(arrays["candidate_q"], arrays["frugal_q"])
        else:
            row["status"] = "incomplete_evaluation"
        row["diagnostics"] = diagnostics
        law_path = args.output/f"laws_{case_id}.npz"
        np.savez_compressed(law_path, **arrays)
        row.update(laws_path=law_path.name, laws_sha256=sha(law_path))
        save(args.output/f"case_{case_id}.json", row)
        rows.append(row)
        save(args.output/"partial_summary.json", {"complete": False, "rows": rows,
                                                 "planned_inputs": len(cases), "new_samples": 0})
        print(json.dumps({"completed": case_id, "status": row["status"],
                          "candidate_tv": row["candidate_tv"], "frugal_tv": row["frugal_tv"]}), flush=True)

    changed = [relative(path) for path, before in protected.items() if sha(path) != before]
    if changed:
        metadata.update(status="binding_changed", changed_bindings=changed,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        save(args.output/"metadata.json", metadata)
        raise RuntimeError("A protected input, executable, source, or gate changed during the audit")
    complete = [r for r in rows if r["status"] == "ok"]
    summary = {"schema_version": 1, "rows": rows, "planned_inputs": len(cases),
        "complete_inputs": len(complete), "incomplete_inputs": len(rows)-len(complete),
        "candidate_lower_tv_inputs": sum(r["candidate_lower_tv"] for r in complete),
        "new_samples": 0, "new_performance_trials": 0, "statistical_alpha_used": None,
        "performance_binding_unchanged": True, "strict_floating_point_certificate": False,
        "design": metadata["audit_design"], "preregistered_independent_statistical_confirmation": False,
        "law_semantics": metadata["law_semantics"],
        "scope": metadata["scope"], "binary_bindings": bindings,
        "inputs_sha256": {relative(p): value for p, value in protected.items()}}
    save(args.output/"summary.json", summary)
    metadata.update(status="complete" if len(complete) == len(cases) else "complete_with_failures",
        complete_inputs=len(complete), completed_utc=datetime.now(timezone.utc).isoformat(),
        summary_sha256=sha(args.output/"summary.json"), performance_binding_unchanged=True)
    save(args.output/"metadata.json", metadata)
    print(json.dumps({"output": relative(args.output), "complete_inputs": len(complete),
        "candidate_lower_tv_inputs": summary["candidate_lower_tv_inputs"],
        "new_samples": 0, "new_performance_trials": 0}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", type=Path, default=ROOT/"results/iteration5_performance")
    parser.add_argument("--output", type=Path, default=ROOT/"results/iteration5_small_laws")
    parser.add_argument("--candidate-executable", type=Path, default=ROOT/".cache/click-prefix-native/click_sampler")
    parser.add_argument("--candidate-validation", type=Path, default=ROOT/"results/iteration5_prefix_validation/validation.json")
    parser.add_argument("--frugal-executable", type=Path, default=ROOT/".cache/frugal-native/frugal_sampling_adapter")
    parser.add_argument("--frugal-validation", type=Path, default=ROOT/".cache/frugal-native/validation.json")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    run(args)
