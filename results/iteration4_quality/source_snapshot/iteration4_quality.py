"""Run the fixed 32-case cross-mode quality family through both validators.

Timing here is descriptive: the formal kernel performance experiment is
separate and used fresh processes without density logging or validator work.
No failed batch is retried, replaced, or removed from the archived family.
"""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
from collections import Counter
import argparse
import hashlib
import json
import platform
import shutil

import numpy as np

from gbslab.cpp_click import run_cpp_click, CppClickError
from gbslab.experiment import json_safe
from gbslab.gbs_identity import (ConfidencePlan, assess_native_samples,
    fixed_batch_protocol, default_witness_subsets)
from gbslab.jiuzhang_gap import assess_jiuzhang_gap, load_reference_catalog

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False)+"\n")


def normalize_native_run(case_id, spec, native, count, *, synthetic=True,
                         instance_id=None, noise_features=None, maximum_modes=256):
    """Bind the D2 manifest to actual results, never a desired reference size."""
    success = native.get("status") == "ok"
    m = spec["modes"]
    clock = native.get("timings_seconds", {})
    observed = len(native.get("raw_samples") or []) if success else 0
    provenance = {key: native.get(key) for key in (
        "executable_sha256", "build_cache_key", "selector", "k", "order",
        "vacuum_kernel", "covariance_sha256", "seed", "sample_count")}
    return {
        "run_id": case_id, "status": native.get("status", "failed"),
        "synthetic": synthetic, "instance_id": instance_id,
        "covariance_sha256": native.get("covariance_sha256"),
        "modes": m, "output_bins": m, "source_count": spec.get("source_count", m),
        "detector_model": "threshold", "output_encoding": "binary_threshold_bins",
        "noise_features": noise_features if noise_features is not None else ["uniform_loss"],
        "sampling": {"count_requested": count, "count_observed": observed,
                     "iid": True, "no_retries": True, "postselection": "none"},
        "timings": {"wall_seconds": native.get("wrapper_wall_seconds"),
                    "setup_seconds": clock.get("setup"), "sampling_seconds": clock.get("sampling"),
                    "scope": "wrapper including density logging; small cases also include full-Q enumeration; validator excluded",
                    "formal_performance_claim": False},
        "resources": {"cpu": "AMD EPYC 9V74; shared host", "threads": native.get("threads", 1),
                      "gpus": 0, "peak_memory_bytes": native.get("peak_memory_bytes")},
        "implementation": {"maximum_modes": maximum_modes}, "native_provenance": provenance,
    }


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory; no hidden reruns")
    config = json.loads(args.config.read_text())
    inputs = json.loads(args.inputs.read_text())
    gate = json.loads(args.kernel_validation.read_text())
    performance = json.loads(args.performance_summary.read_text())
    binary = digest(args.executable)
    if (inputs["config_sha256"] != digest(args.config) or not gate.get("passed")
            or not gate.get("product_equivalence_passed") or gate["executable_sha256"] != binary):
        raise RuntimeError("Frozen input/build/equivalence gate mismatch")
    if performance.get("paired_cases") != 32 or performance.get("product_cold_wins", 0) <= 16:
        raise RuntimeError("Pre-validation performance choice is not supported")
    cases = inputs["cases"]
    plan = ConfidencePlan(tuple(c["case_id"] for c in cases), config["validation"]["familywise_alpha"])
    if len(cases) != config["validation"]["planned_comparisons"]:
        raise ValueError("Confidence family mismatch")
    args.output.mkdir(parents=True)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    sources = [Path(__file__), ROOT/"src/gbslab/gbs_identity.py", ROOT/"src/gbslab/jiuzhang_gap.py",
               ROOT/"src/gbslab/tv_validation.py", ROOT/"src/gbslab/cpp_click.py",
               ROOT/"src/gbslab/exact.py"]
    for src in sources:
        shutil.copy2(src, snapshot/src.name)
    for src, name in [(args.config, "config.json"), (args.inputs, "inputs_manifest.json"),
                      (args.performance_summary, "kernel_choice_performance.json"),
                      (ROOT/"configs/jiuzhang_reference.json", "reference_catalog.json")]:
        shutil.copy2(src, args.output/name)
    # Both the confidence family and every witness schedule are saved before
    # the first sample. Unused alpha is never moved to another comparison.
    save_json(args.output/"confidence_plan.json", plan.as_dict())
    save_json(args.output/"witness_schedules.json", {
        c["case_id"]: default_witness_subsets(c["spec"]["modes"]) for c in cases})
    metadata = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": digest(args.config), "executable_sha256": binary,
                "source_hashes": {p.name: digest(p) for p in sources},
                "selected_kernel": "product", "selection_basis": "equivalence gates plus independent performance family",
                "plan_sha256": plan.as_dict()["plan_sha256"], "planned_cases": len(cases),
                "platform": platform.platform(), "scope": "synthetic targets only; no Jiuzhang equivalence claim"}
    save_json(args.output/"metadata.json", metadata)
    catalog = load_reference_catalog(args.output/"reference_catalog.json")
    rows = []
    started = perf_counter()
    validation = config["validation"]
    with (args.output/"progress.jsonl").open("x") as progress:
        for index, case in enumerate(cases):
            case_id, spec = case["case_id"], case["spec"]
            m = spec["modes"]
            path = ROOT/case["input"]
            if digest(path) != case["sha256"]:
                raise RuntimeError(f"Frozen input changed: {case_id}")
            loaded = np.load(path)
            covariance = loaded["covariance"]
            count = validation["samples_at_most_64"] if m <= 64 else (
                validation["samples_256"] if m == 256 else validation["samples_100_and_144"])
            seed = validation["sample_seed_offset"]+index
            directory = args.output/case_id
            directory.mkdir()
            protocol = fixed_batch_protocol(count, source="root_fixed_iteration4_cross_mode_protocol")
            save_json(directory/"protocol.json", protocol)
            try:
                native = run_cpp_click(covariance, selection=config["selector"],
                    max_clicks=config["retained_clicks"], samples=count, seed=seed,
                    executable=args.executable, raw_samples=True, record_log_probabilities=True,
                    vacuum_kernel="product", enumerate_distribution=m <= validation["full_target_max_modes"],
                    profile=False, timeout=validation["timeout_seconds"])
            except (CppClickError, ValueError, OSError, TimeoutError) as exc:
                native = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                          "payload": getattr(exc, "payload", None), "stderr": getattr(exc, "stderr", None),
                          "returncode": getattr(exc, "returncode", None), "executable_sha256": binary}
            # The frozen target probabilities, if present, are used only by
            # validation after sampling. The sampler receives the covariance.
            options = {}
            if "target" in loaded:
                target = loaded["target"]
                def target_log_probability(bits, p=target):
                    mask = sum(int(bit) << j for j, bit in enumerate(bits))
                    return float(np.log(p[mask])) if p[mask] > 0 else -float("inf")
                options = {"target_log_probability": target_log_probability,
                           "target_probability_mode": "frozen_complete_numerical_target"}
            evidence = assess_native_samples(covariance, native, plan=plan, comparison_id=case_id,
                epsilon=validation["epsilon"], max_exact_clicks=validation["max_exact_clicks"],
                protocol=protocol, witness_subsets=default_witness_subsets(m), target_id=case_id,
                full_law_max_modes=validation["full_target_max_modes"], keep_scores=True, **options)
            normalized = normalize_native_run(case_id, spec, native, count)
            gap4 = assess_jiuzhang_gap(normalized, evidence, reference_id="jiuzhang4_l1024", catalog=catalog,
                                       epsilon=validation["epsilon"])
            gap1 = assess_jiuzhang_gap(normalized, evidence, reference_id="jiuzhang1", catalog=catalog,
                                       epsilon=validation["epsilon"])
            arrays = {}
            for key in ("raw_samples", "sample_log_probabilities", "distribution"):
                if native.get(key) is not None:
                    arrays[key] = np.asarray(native[key], dtype=np.uint8 if key == "raw_samples" else float)
            if arrays:
                np.savez_compressed(directory/"evidence_arrays.npz", **arrays)
            compact = {k: v for k, v in native.items() if k not in arrays}
            compact["array_archive"] = "evidence_arrays.npz" if arrays else None
            if arrays:
                compact["array_archive_sha256"] = digest(directory/"evidence_arrays.npz")
            save_json(directory/"native.json", compact)
            save_json(directory/"identity.json", evidence)
            save_json(directory/"run_manifest.json", normalized)
            save_json(directory/"gap_jiuzhang1.json", gap1)
            save_json(directory/"gap_jiuzhang4.json", gap4)
            bound, witnesses = evidence.get("global_tv") or {}, evidence.get("witnesses") or {}
            row = {"case_id": case_id, "modes": m, "profile": spec["name"], "samples_requested": count,
                   "sampling_status": native["status"], "decision": evidence["decision"],
                   "evidence_scope": evidence["evidence_scope"], "decision_basis": evidence.get("decision_basis"),
                   "full_tv": (evidence.get("full_law") or {}).get("tv"),
                   "global_tv_upper": bound.get("upper"), "global_score_mean": bound.get("mean_score"),
                   "cutoff_count": bound.get("cutoff_count"),
                   "target_probability_failure_count": bound.get("target_probability_failure_count"),
                   "witness_tv_lower": witnesses.get("tv_lower"),
                   "target_mean_clicks": case["physical_target_mean_clicks"],
                   "observed_mean_clicks": witnesses.get("observed_mean_clicks"),
                   "sampling_seconds_with_logq": native.get("timings_seconds", {}).get("sampling"),
                   "jiuzhang4_verdict": gap4["verdict"],
                   "jiuzhang4_speed_ratio": gap4["timing"]["classical_wall_over_reference_wall"]}
            rows.append(row)
            progress.write(json.dumps(json_safe(row), allow_nan=False)+"\n")
            progress.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
    save_json(args.output/"summary.json", {"rows": rows, "decisions": dict(Counter(r["decision"] for r in rows)),
        "familywise_alpha": plan.familywise_alpha, "epsilon": validation["epsilon"],
        "sampling_failures": sum(r["sampling_status"] != "ok" for r in rows),
        "scope": "Each result concerns only its registered synthetic covariance and the threshold alphabet"})
    metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    comparisons=len(rows), elapsed_seconds=perf_counter()-started)
    save_json(args.output/"metadata.json", metadata)
    print(json.dumps({"status": "complete", "comparisons": len(rows), "elapsed_seconds": metadata["elapsed_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/iteration4_cross_mode.json")
    parser.add_argument("--inputs", type=Path, default=ROOT/"results/iteration4_inputs/manifest.json")
    parser.add_argument("--kernel-validation", type=Path, default=ROOT/"results/iteration4_kernel_validation/first_candidate/metadata.json")
    parser.add_argument("--performance-summary", type=Path, default=ROOT/"results/iteration4_performance/performance_summary.json")
    parser.add_argument("--executable", type=Path, default=ROOT/".cache/click-iteration4-native/click_sampler")
    parser.add_argument("--output", type=Path, default=ROOT/"results/iteration4_quality")
    run(parser.parse_args())
