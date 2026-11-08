"""Frozen iteration-5 samples, including generation-only official JZ1 control.

Sampling and evaluation are separate stages. Evaluation never redraws samples.
The official adapter saves a complete batch before any later analysis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
from time import perf_counter

import numpy as np

from gbslab.cpp_click import run_cpp_click, CppClickError
from gbslab.exact import ThresholdModel
from gbslab.frugal_evidence import (frugal_evidence_sample, FrugalEvidenceError,
    assess_frugal_samples, UPSTREAM_COMMIT, SAMPLER_BODY_SHA256)
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.gbs_identity import (ConfidencePlan, assess_gbs_identity,
    assess_native_samples, covariance_sha256, fixed_batch_protocol,
    default_witness_subsets)
from gbslab.jiuzhang_gap import assess_jiuzhang_gap
from iteration4_quality import normalize_native_run

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/iteration5_quality.json"
DEFAULT_OUTPUT = ROOT / "results/iteration5_quality"
DEFAULT_VALIDATION = {
    "candidate": ROOT / "results/iteration5_prefix_validation/validation.json",
    "frugal": ROOT / "results/iteration4_frugal_checkpoint_validation/validation.json",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def cases(config):
    result = []
    for m in config["synthetic_modes"]:
        for profile in config["profiles"]:
            result.append({"id": f"m{m}_{profile['name']}", "modes": m,
                "profile": profile, "synthetic": True, "method": "candidate",
                "count": config["sample_count_by_modes"][str(m)],
                "pilot": config["pilot_size_by_modes"][str(m)],
                "seed": config["sample_seed_base"] + len(result)})
    jz = config["jiuzhang1"]
    for method in ("candidate", "frugal"):
        result.append({"id": f"jiuzhang1_{method}", "modes": 100,
            "synthetic": False, "method": method, "count": jz["sample_count"],
            "pilot": jz["pilot_size"], "seed": jz["sample_seed"]})
    if len(result) != config["planned_comparisons"]:
        raise ValueError("Frozen comparison count mismatch")
    return result


def covariance(config, case):
    if not case["synthetic"]:
        spec = config["jiuzhang1"]
        path = ROOT / spec["input"]
        if digest(path) != spec["input_file_sha256"]:
            raise ValueError("Author covariance file hash changed")
        V = np.load(path, allow_pickle=False)
        if covariance_sha256(V) != spec["canonical_covariance_sha256"]:
            raise ValueError("Author covariance canonical hash changed")
        return V
    m, profile = case["modes"], case["profile"]
    U = haar_unitary(m, np.random.default_rng(config["interferometer_seed"] + m))
    return covariance_from_interferometer(U, profile["squeezing"],
                                         loss=profile["transmissivity"])


def confidence_allocation(config):
    """Check the declared union budget without reallocating unused tests."""
    family = cases(config)
    keys = ("familywise_alpha", "old_identity_suite_alpha", "marginal_suite_alpha")
    for key in keys:
        value = config[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value < 1):
            raise ValueError(f"{key} must be a finite probability in (0,1)")
    if (config["old_identity_suite_alpha"] + config["marginal_suite_alpha"]
            > config["familywise_alpha"] + 1e-15):
        raise ValueError("Evidence channels exceed the frozen familywise alpha")
    return (ConfidencePlan(tuple(c["id"] for c in family), config["old_identity_suite_alpha"]),
            config["marginal_suite_alpha"] / len(family))


def materialize_evidence_plan(config, output):
    """Archive deterministic indices; never infer a schedule from sample values.

    The first Frugal attempt was launched from the frozen textual rule before
    this materialization helper existed. Record the actual archive time and
    any already-started attempts, rather than backdating a prelaunch check.
    """
    from gbslab.marginal_witness import default_marginal_windows, fixed_split_protocol
    plan, marginal_alpha = confidence_allocation(config)
    schedules, protocols = {}, {}
    family = cases(config)
    for case in family:
        windows = default_marginal_windows(case["modes"], width=config["marginal_window_width"])
        schedules[case["id"]] = {
            "marginal_windows": [list(w) for w in windows],
            "original_witness_subsets": [list(w) for w in default_witness_subsets(case["modes"])],
            "marginal_comparison_alpha": marginal_alpha,
        }
        protocols[case["id"]] = {
            "original": fixed_batch_protocol(case["count"], source="iteration5_frozen_first_batch"),
            "marginal": fixed_split_protocol(case["count"], case["pilot"], windows=windows,
                source="iteration5_frozen_first_batch_and_positional_split"),
        }
    records = {"witness_schedules.json": schedules, "split_protocols.json": protocols,
               "confidence_plan.json": plan.as_dict()}
    preexisting = []
    for name, expected in records.items():
        path = output / name
        if path.exists():
            preexisting.append(name)
            if json.loads(path.read_text()) != expected:
                raise ValueError(f"Frozen evidence plan differs from current deterministic rule: {name}")
        else:
            save(path, expected)
    manifest = output / "witness_schedule_manifest.json"
    hashes = {name: digest(output / name) for name in records}
    if manifest.exists():
        archived = json.loads(manifest.read_text())
        if archived.get("config_sha256") != digest(output / "config.json") or archived.get("artifact_sha256") != hashes:
            raise ValueError("Frozen evidence-plan manifest is inconsistent")
    else:
        attempts = [c["id"] for c in family if (output / c["id"] / "attempt.json").exists()]
        completed = [c["id"] for c in family if (output / c["id"] / "completion.json").exists()]
        save(manifest, {
            "materialized_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": digest(output / "config.json"), "artifact_sha256": hashes,
            "preexisting_artifacts": preexisting,
            "sampling_attempts_already_started": attempts,
            "completed_batch_artifacts_present": completed,
            "sample_values_read_to_choose_schedule": False,
            "rule_and_split_fixed_in_config_before_sampling": True,
            "explicit_indices_materialized_before_all_sampling": not attempts,
            "basis": "The frozen configuration fixed the window rule and positional split. These dimension-only indices and hashes were materialized at the timestamp above, independently of every pilot and held-out sample value.",
            "source_hashes": {
                "experiments/iteration5_quality.py": digest(Path(__file__)),
                "src/gbslab/marginal_witness.py": digest(ROOT / "src/gbslab/marginal_witness.py")},
        })
    return schedules, protocols


def initialize(config_path, output):
    config = json.loads(config_path.read_text())
    frozen = output / "config.json"
    if frozen.exists():
        if digest(frozen) != digest(config_path):
            raise ValueError("Output is bound to a different frozen protocol")
        if json.loads((output / "comparison_plan.json").read_text()) != cases(config):
            raise ValueError("Frozen comparison plan differs from the configured family")
        materialize_evidence_plan(config, output)
        return config
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh quality output directory")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, frozen)
    save(output / "comparison_plan.json", cases(config))
    save(output / "metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(config_path), "planned_comparisons": len(cases(config)),
        "familywise_alpha": config["familywise_alpha"],
        "alpha_allocation": {"original_D1": config["old_identity_suite_alpha"],
                             "marginal_D1": config["marginal_suite_alpha"]},
        "platform": platform.platform(), "threads": 1,
        "sampling_and_evaluation_separate": True,
        "scope": config["quality_scope"]})
    materialize_evidence_plan(config, output)
    return config


def validate_binary_gate(executable, method, *, validation_path=None, modes=None):
    """Bind a completed validation gate and adjacent build to this exact binary."""
    executable = Path(executable)
    gate_path = Path(validation_path or DEFAULT_VALIDATION[method])
    gate = json.loads(gate_path.read_text())
    build_path = executable.parent / "build.json"
    build = json.loads(build_path.read_text())
    binary = digest(executable)
    if gate.get("passed") is not True or gate.get("executable_sha256") != binary:
        raise ValueError("Validation gate is absent, failed, or bound to a different executable")
    if build.get("executable_sha256") != binary:
        raise ValueError("Build metadata does not identify the validated executable")
    if gate.get("build_cache_key") is not None and gate["build_cache_key"] != build.get("cache_key"):
        raise ValueError("Validation/build cache keys disagree")
    if method == "candidate":
        source_hash = digest(ROOT / "experiments/click_sampler.cpp")
        if source_hash != gate.get("source_sha256") or source_hash != build.get("source_sha256"):
            raise ValueError("Prefix-cache gate/build do not match the current C++ source")
        checks = gate.get("checks")
        if not isinstance(checks, list) or not checks or any(c.get("passed") is not True for c in checks):
            raise ValueError("Prefix-cache equivalence gate has missing or unsuccessful checks")
        if modes is not None and modes > build.get("maximum_modes", 0):
            raise ValueError("Candidate build does not support the requested mode count")
    elif method == "frugal":
        if (gate.get("generated_density_trace_identical_to_v1") is not True
                or build.get("sample_checkpoint_before_density") is not True
                or gate.get("upstream_commit") != UPSTREAM_COMMIT
                or build.get("upstream_commit") != UPSTREAM_COMMIT
                or gate.get("sampler_body_sha256") != SAMPLER_BODY_SHA256
                or build.get("sampler_body_sha256") != SAMPLER_BODY_SHA256
                or build.get("adapter_source_sha256") != digest(ROOT / "experiments/frugal_evidence_adapter.cpp")):
            raise ValueError("Official Frugal checkpoint build lacks the pinned validation identity")
        if modes is not None and modes not in build.get("modes", ()):
            raise ValueError("Official Frugal build does not support the requested mode count")
    else:
        raise ValueError("Unknown quality comparison method")
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "method": method,
        "validation_path": str(gate_path.resolve()), "validation_sha256": digest(gate_path),
        "build_metadata_sha256": digest(build_path), "build_cache_key": build.get("cache_key"),
        "executable_sha256": binary, "passed": True,
        "scope": "Matching validated native binary and build identity; Python evaluator revisions are recorded separately and do not backdate validation.",
    }


def sample_case(config, case, output, executable, validation_path=None):
    binding = validate_binary_gate(executable, case["method"],
        validation_path=validation_path, modes=case["modes"])
    directory = output / case["id"]
    directory.mkdir()  # Existing/partial attempts must not be replaced.
    V = covariance(config, case)
    if case["synthetic"]:
        np.savez_compressed(directory / "input.npz", covariance=V)
    provenance = {"case": case, "covariance_sha256": covariance_sha256(V),
        "config_sha256": digest(output / "config.json"),
        "executable_sha256": binding["executable_sha256"],
        "validation_binding": {**binding, "checked_before_sampling": True},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT / "src/gbslab/cpp_click.py",
            ROOT / "src/gbslab/frugal_evidence.py")}}
    save(directory / "attempt.json", provenance)
    shutil.copy2(executable.parent / "build.json", directory / "build.json")
    shutil.copy2(binding["validation_path"], directory / "validation_gate.json")
    print(json.dumps({"starting": case["id"], "samples": case["count"]}), flush=True)
    started = perf_counter()
    try:
        if case["method"] == "frugal":
            native = frugal_evidence_sample(V,
                cutoff=config["jiuzhang1"]["frugal_cutoff"], sample_count=case["count"],
                seed=case["seed"], executable=executable, raw_samples=True,
                record_log_probabilities=config["jiuzhang1"]["frugal_record_log_probabilities"],
                timeout=config["jiuzhang1"]["frugal_timeout_seconds"],
                threads=config["threads"], sample_checkpoint_path=directory / "sample_checkpoint.json")
        else:
            sampler = config["candidate"]
            native = run_cpp_click(V, selection=sampler["selector"],
                max_clicks=sampler["k"], samples=case["count"], seed=case["seed"],
                executable=executable, raw_samples=True, record_log_probabilities=True,
                vacuum_kernel=sampler["vacuum_kernel"],
                prefix_cache_entries=sampler["prefix_cache_entries"],
                enumerate_distribution=case["modes"] <= config["full_law_max_modes"],
                timeout=config["candidate_timeout_seconds"])
    except (CppClickError, FrugalEvidenceError, OSError, ValueError, RuntimeError) as exc:
        native = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "payload": getattr(exc, "payload", None), "stderr": getattr(exc, "stderr", None),
            "returncode": getattr(exc, "returncode", None)}
    native["harness_call_seconds"] = perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    native["resource_measurement"] = {
        "native_child_peak_rss_bytes": int(usage.ru_maxrss) * 1024,
        "scope": "RUSAGE_CHILDREN high-water over native children in this worker; --case uses one child; --all does not isolate per-case RSS"}
    if digest(executable) != provenance["executable_sha256"]:
        raise RuntimeError("Binary changed while sampling; saved attempt cannot be claimed valid")
    # Keep complete native evidence; zero truncation or sample filtering.
    (directory / "native.json").write_text(json.dumps(native, allow_nan=False) + "\n")
    save(directory / "completion.json", {"status": native.get("status"),
        "native_sha256": digest(directory / "native.json"),
        "harness_call_seconds": native["harness_call_seconds"],
        "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"completed": case["id"], "status": native.get("status"),
                      "seconds": native["harness_call_seconds"]}), flush=True)


def evaluate(config, output):
    from gbslab.marginal_witness import assess_marginal_witness
    family = cases(config)
    plan, per_marginal_alpha = confidence_allocation(config)
    schedules, protocols = materialize_evidence_plan(config, output)
    save(output / "evaluation_manifest.json", {
        "evaluation_started_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(output / "config.json"),
        "witness_schedules_sha256": digest(output / "witness_schedules.json"),
        "split_protocols_sha256": digest(output / "split_protocols.json"),
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT / "src/gbslab/gbs_identity.py", ROOT / "src/gbslab/marginal_witness.py",
            ROOT / "src/gbslab/jiuzhang_gap.py", ROOT / "src/gbslab/frugal_evidence.py",
            ROOT / "src/gbslab/exact.py", ROOT / "src/gbslab/tv_validation.py")},
        "sampling_was_not_repeated_for_evaluation": True,
    })
    rows = []
    for case in family:
        directory = output / case["id"]
        path = directory / "native.json"
        if not path.exists():
            rows.append({"case_id": case["id"], "status": "not_run"})
            continue
        V = covariance(config, case)
        native = load_completed_attempt(config, case, output, V)
        protocol = protocols[case["id"]]["original"]
        options = dict(plan=plan, comparison_id=case["id"], epsilon=config["epsilon"],
            max_exact_clicks=config["point_probability_max_clicks"], target_id=case["id"],
            full_law_max_modes=config["full_law_max_modes"],
            witness_subsets=schedules[case["id"]]["original_witness_subsets"])
        if case["method"] == "candidate":
            identity = assess_native_samples(V, native, protocol=protocol, **options)
        else:
            identity = assess_frugal_samples(V, native,
                require_log_probabilities=config["jiuzhang1"]["frugal_record_log_probabilities"],
                protocol=protocol, **options)
        windows = schedules[case["id"]]["marginal_windows"]
        split = protocols[case["id"]]["marginal"]
        marginal = assess_marginal_witness(V, native.get("raw_samples"),
            windows=windows, pilot_size=case["pilot"], alpha=per_marginal_alpha,
            epsilon=config["epsilon"], protocol=split,
            sampling_failure=None if native.get("status") == "ok" else native,
            target_id=case["id"])
        combined = combine_identity(identity, marginal, config)
        save(directory / "identity_original.json", identity)
        save(directory / "marginal_witness.json", marginal)
        save(directory / "identity.json", combined)
        manifest = normalize_native_run(case["id"], {"modes": case["modes"],
            "source_count": case["modes"] if case["synthetic"] else 50}, native,
            case["count"], synthetic=case["synthetic"],
            instance_id=None if case["synthetic"] else config["jiuzhang1"]["instance_id"],
            noise_features=["uniform_loss"] if case["synthetic"] else ["mode_dependent_loss"],
            maximum_modes=256 if case["method"] == "candidate" else 144)
        if case["method"] == "frugal":
            t = native.get("timings_seconds", {})
            manifest["timings"].update(setup_seconds=t.get("layout", 0) + t.get("target", 0),
                density_seconds=t.get("density"), density_within_sampling=False,
                scope="generation-only official Frugal including checkpoint; no log-Q replay or full-law enumeration")
        else:
            manifest["timings"]["scope"] = "candidate with path log-Q in sampling; small full-Q enumeration is outside sampling"
        if native.get("status") == "ok":
            click_counts = np.asarray(native["raw_samples"], dtype=np.uint8).sum(axis=1)
            manifest["click_statistics"] = {"observed_mean": float(click_counts.mean()),
                "observed_maximum": int(click_counts.max()), "sample_size": len(click_counts),
                "source": "complete frozen batch, not an inferred target-hardness parameter"}
        save(directory / "run_manifest.json", manifest)
        gaps = {}
        for ref in ("jiuzhang1", "jiuzhang4_l1024"):
            gap = assess_jiuzhang_gap(manifest, combined, reference_id=ref,
                # The target is a mathematical Gaussian law in both cases.
                # manifest.synthetic separately fails the physical-instance gate.
                target_kind="published_gaussian_model",
                epsilon=config["epsilon"])
            gaps[ref] = gap["verdict"]
            save(directory / f"gap_{ref}.json", gap)
        old_w = identity.get("witnesses") or {}
        rows.append({"case_id": case["id"], "modes": case["modes"], "method": case["method"],
            "synthetic": case["synthetic"], "sampling_status": native.get("status"),
            "decision": combined["decision"], "original_decision": identity["decision"],
            "full_tv": (identity.get("full_law") or {}).get("tv"),
            "global_tv_upper": (identity.get("global_tv") or {}).get("upper"),
            "original_witness_lower": old_w.get("tv_lower"),
            "marginal_tv_lower": marginal.get("tv_lower"),
            "target_mean_clicks": old_w.get("target_mean_clicks"),
            "sample_mean_clicks": old_w.get("observed_mean_clicks"),
            "timings_seconds": native.get("timings_seconds"), "jiuzhang_gaps": gaps})
    save(output / "summary.json", {"rows": rows, "familywise_alpha": config["familywise_alpha"],
        "epsilon": config["epsilon"], "quality_scope": config["quality_scope"]})
    print(json.dumps(rows, indent=2))


def combine_identity(identity, marginal, config):
    """Compose disjoint confidence budgets, with conflict detection.

    The new channel may reject or remain inconclusive, never accept alone.
    """
    result = json.loads(json.dumps(identity))
    confidence_allocation(config)
    result["marginal_witness"] = marginal
    result["combined_confidence_family"] = {
        "familywise_alpha": config["familywise_alpha"],
        "original_suite_alpha": config["old_identity_suite_alpha"],
        "marginal_suite_alpha": config["marginal_suite_alpha"],
        "planned_comparisons": config["planned_comparisons"],
        "scope": "union bound across both prespecified suites and all comparisons"}
    if result["decision"] in ("invalid_model", "invalid_evidence", "evaluation_failed", "out_of_scope"):
        return result
    if marginal.get("decision") in ("invalid_model", "invalid_evidence", "evaluation_failed"):
        result["reasons"].append("The additional marginal channel was unavailable; original evidence retained")
        return result
    lower = marginal.get("tv_lower")
    if lower is None:
        return result
    target = result.get("target") or {}
    marginal_target = marginal.get("target") or {}
    target_fields = ("target_id", "covariance_sha256", "modes", "detector_model", "state_model", "quadrature_order", "hbar")
    marginal_alpha = marginal.get("alpha")
    if (any(target.get(key) is None or target.get(key) != marginal_target.get(key) for key in target_fields)
            or not result.get("samples_sha256")
            or result["samples_sha256"] != marginal.get("samples_sha256")
            or marginal.get("statistical_protocol_valid") is not True
            or marginal.get("epsilon") != result["epsilon"]
            or isinstance(lower, bool) or not isinstance(lower, (int, float))
            or not math.isfinite(lower) or not 0 <= lower <= 1
            or isinstance(marginal_alpha, bool) or not isinstance(marginal_alpha, (int, float))
            or not math.isfinite(marginal_alpha)
            or not math.isclose(marginal_alpha,
                config["marginal_suite_alpha"]/config["planned_comparisons"], rel_tol=1e-12)):
        result.update(decision="invalid_evidence")
        result["reasons"].append("Marginal evidence is not bound to the same target, complete samples, valid split, tolerance, and confidence allocation")
        return result
    upper = (result.get("global_tv") or {}).get("upper")
    exact_tv = (result.get("full_law") or {}).get("tv")
    if (upper is not None and lower > upper) or (exact_tv is not None and lower > exact_tv + 1e-10):
        result.update(decision="invalid_evidence", statistical_evidence_conflict=True)
        result["reasons"].append("Held-out marginal lower bound conflicts with full-law/global upper evidence")
    elif lower > result["epsilon"]:
        result.update(decision="rejected", decision_basis="held_out_marginal_tv_lower_bound")
        result["reasons"].append("A pilot-selected event on an independently held-out fixed batch rejects the full-law TV tolerance")
    return result


def load_completed_attempt(config, case, output, V):
    """Check archived sample identity before either evidence channel reads it."""
    directory = output / case["id"]
    path = directory / "native.json"
    attempt = json.loads((directory / "attempt.json").read_text())
    completion = json.loads((directory / "completion.json").read_text())
    native = json.loads(path.read_text())
    build = json.loads((directory / "build.json").read_text())
    if (attempt.get("case") != case or attempt.get("config_sha256") != digest(output / "config.json")
            or attempt.get("covariance_sha256") != covariance_sha256(V)
            or completion.get("native_sha256") != digest(path)
            or completion.get("status") != native.get("status")
            or build.get("executable_sha256") != attempt.get("executable_sha256")):
        raise ValueError("Archived native batch does not match its attempt, completion, covariance, or build binding")
    if native.get("status") == "ok":
        expected = {"executable_sha256": attempt["executable_sha256"],
            "covariance_sha256": attempt["covariance_sha256"], "modes": case["modes"],
            "sample_count": case["count"], "seed": case["seed"]}
        if any(native.get(key) != value for key, value in expected.items()):
            raise ValueError("Completed native batch differs from the fixed executable, model, count, or seed")
    return native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case")
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--candidate-executable", type=Path, default=ROOT / ".cache/click-prefix-native/click_sampler")
    parser.add_argument("--frugal-executable", type=Path, default=ROOT / ".cache/frugal-evidence-checkpoint-native/frugal_evidence_adapter")
    parser.add_argument("--candidate-validation", type=Path, default=DEFAULT_VALIDATION["candidate"])
    parser.add_argument("--frugal-validation", type=Path, default=DEFAULT_VALIDATION["frugal"])
    args = parser.parse_args()
    config = initialize(args.config, args.output)
    if args.evaluate:
        evaluate(config, args.output)
        return
    selected = [c for c in cases(config) if c["id"] == args.case
                or (args.all_candidates and c["method"] == "candidate")]
    if not selected:
        parser.error("Choose a fixed --case, --all-candidates, or --evaluate")
    for case in selected:
        executable = args.frugal_executable if case["method"] == "frugal" else args.candidate_executable
        validation = args.frugal_validation if case["method"] == "frugal" else args.candidate_validation
        sample_case(config, case, args.output, executable, validation_path=validation)


if __name__ == "__main__":
    main()
