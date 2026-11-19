"""One fixed actual Jiuzhang1-model batch per native method, with two judges.

This is not a reproduction of the Frugal paper's Jiuzhang2/3 GPU experiments.
Each sampler runs in its own Python worker so native child peak RSS is measured
without inheriting the other method's high-water mark. Failed runs are kept.
"""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys

import numpy as np

from gbslab.cpp_click import run_cpp_click, CppClickError
from gbslab.frugal_evidence import (frugal_evidence_sample, FrugalEvidenceError,
                                    assess_frugal_samples, frugal_storage_lower_bound)
from gbslab.gbs_identity import (ConfidencePlan, assess_native_samples,
    fixed_batch_protocol, default_witness_subsets, covariance_sha256)
from gbslab.jiuzhang_gap import assess_jiuzhang_gap, load_reference_catalog
from iteration4_quality import normalize_native_run, save_json, digest

ROOT = Path(__file__).resolve().parents[1]


def check_input(config):
    path = ROOT/config["input"]
    if digest(path) != config["input_file_sha256"]:
        raise RuntimeError("Published input file hash changed")
    V = np.load(path)
    if hashlib.sha256(np.ascontiguousarray(V).tobytes()).hexdigest() != config["raw_array_sha256"]:
        raise RuntimeError("Published raw covariance hash changed")
    if covariance_sha256(V) != config["canonical_covariance_sha256"]:
        raise RuntimeError("Canonical target covariance hash changed")
    return V


def sample_worker(args):
    config = json.loads(args.config.read_text())
    V = check_input(config)
    start = perf_counter()
    try:
        if args.worker == "candidate":
            native = run_cpp_click(V, selection=config["candidate"]["selector"],
                max_clicks=config["candidate"]["k"], samples=config["samples"],
                seed=config["sample_seed"], executable=args.candidate_executable,
                raw_samples=True, record_log_probabilities=True,
                vacuum_kernel=config["candidate"]["kernel"],
                timeout=config["candidate"]["timeout_seconds"])
        else:
            native = frugal_evidence_sample(V, cutoff=config["baseline"]["K"],
                sample_count=config["samples"], seed=config["sample_seed"],
                executable=args.frugal_executable, record_log_probabilities=True,
                raw_samples=True, timeout=config["baseline"]["timeout_seconds"],
                threads=config["threads"],
                sample_checkpoint_path=args.worker_output.parent/"sample_checkpoint.json")
    except (CppClickError, FrugalEvidenceError, OSError, ValueError, TimeoutError) as exc:
        native = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                  "payload": getattr(exc, "payload", None), "stderr": getattr(exc, "stderr", None),
                  "returncode": getattr(exc, "returncode", None),
                  "executable_sha256": getattr(exc, "executable_sha256", None)}
    # On Linux ru_maxrss is in KiB. This fresh worker has launched only the
    # single native sampler, so RUSAGE_CHILDREN measures that child's maximum.
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    native["resource_measurement"] = {
        "native_child_peak_rss_bytes": int(usage.ru_maxrss)*1024,
        "native_child_user_seconds": usage.ru_utime,
        "native_child_system_seconds": usage.ru_stime,
        "worker_call_wall_seconds": perf_counter()-start,
        "method": "Linux getrusage(RUSAGE_CHILDREN) in a fresh one-native-child worker",
        "excludes": "parent process simultaneous RSS, validation and compiler; child high-water may include fork/exec inherited pages",
    }
    native["peak_memory_bytes"] = int(usage.ru_maxrss)*1024
    save_json(args.worker_output, native)


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory; this protocol never retries a batch")
    config = json.loads(args.config.read_text())
    V = check_input(config)
    plan = ConfidencePlan(tuple(config["comparison_ids"]), config["familywise_alpha"])
    frugal_gate = json.loads(args.frugal_validation.read_text())
    if frugal_gate.get("passed") is not True or frugal_gate.get("executable_sha256") != digest(args.frugal_executable):
        raise RuntimeError("Frugal sampler/density validation does not bind this binary")
    candidate_gate = json.loads(args.candidate_validation.read_text())
    if (not candidate_gate.get("passed") or not candidate_gate.get("product_equivalence_passed")
            or candidate_gate.get("executable_sha256") != digest(args.candidate_executable)):
        raise RuntimeError("Candidate implementation validation does not bind this binary")
    args.output.mkdir(parents=True)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    sources = [Path(__file__), ROOT/"experiments/iteration4_quality.py",
               ROOT/"src/gbslab/frugal_evidence.py", ROOT/"src/gbslab/cpp_click.py",
               ROOT/"src/gbslab/gbs_identity.py", ROOT/"src/gbslab/jiuzhang_gap.py",
               ROOT/"src/gbslab/tv_validation.py", ROOT/"src/gbslab/exact.py"]
    for src in sources:
        shutil.copy2(src, snapshot/src.name)
    for src, name in [(args.config, "config.json"), (args.frugal_validation, "frugal_validation.json"),
                      (args.candidate_validation, "candidate_validation.json"),
                      (ROOT/"configs/jiuzhang_reference.json", "reference_catalog.json")]:
        shutil.copy2(src, args.output/name)
    save_json(args.output/"confidence_plan.json", plan.as_dict())
    save_json(args.output/"witness_schedule.json", default_witness_subsets(len(V)//2))
    save_json(args.output/"frugal_memory_preflight.json", frugal_storage_lower_bound(len(V)//2, config["baseline"]["K"]))
    metadata = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(args.config), "source_hashes": {p.name: digest(p) for p in sources},
        "binary_hashes": {"candidate": digest(args.candidate_executable), "frugal": digest(args.frugal_executable)},
        "scope": config["scope"], "planned_batches": 2, "threads": 1,
        "timing_qualification": "One batch per method; no variance estimate. Candidate density logging is within sampling; Frugal density replay is separate. Neither draw rate establishes equal distributional quality."}
    save_json(args.output/"metadata.json", metadata)
    catalog = load_reference_catalog(args.output/"reference_catalog.json")
    rows = []
    start = perf_counter()
    for method, case_id in zip(("candidate", "frugal"), config["comparison_ids"]):
        directory = args.output/method
        directory.mkdir()
        protocol = fixed_batch_protocol(config["samples"], source="root_fixed_actual_jiuzhang1_model_control")
        save_json(directory/"protocol.json", protocol)
        print(json.dumps({"event": "starting_fixed_batch", "method": method, "count": config["samples"],
                          "modes": len(V)//2, "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
        worker_output = directory/"worker_native.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", method,
            "--worker-output", str(worker_output), "--config", str(args.output/"config.json"),
            "--candidate-executable", str(args.candidate_executable),
            "--frugal-executable", str(args.frugal_executable)]
        worker = subprocess.run(command, text=True, capture_output=True,
                                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        (directory/"worker_stderr.txt").write_text(worker.stderr)
        (directory/"worker_stdout.txt").write_text(worker.stdout)
        if worker.returncode or not worker_output.is_file():
            native = {"status": "failed", "error": "worker did not return a complete result",
                      "returncode": worker.returncode, "stderr": worker.stderr}
        else:
            native = json.loads(worker_output.read_text())
        evaluator = assess_native_samples if method == "candidate" else assess_frugal_samples
        evidence = evaluator(V, native, protocol=protocol, plan=plan, comparison_id=case_id,
            epsilon=config["epsilon"], max_exact_clicks=config["max_exact_clicks"],
            witness_subsets=default_witness_subsets(len(V)//2), target_id=config["instance_id"], keep_scores=True)
        normalized = normalize_native_run(case_id, {"modes": len(V)//2, "source_count": 50},
            native, config["samples"], synthetic=False, instance_id=config["instance_id"],
            noise_features=["mode_dependent_loss"], maximum_modes=256 if method == "candidate" else 144)
        clock = native.get("timings_seconds", {})
        if method == "frugal":
            normalized["timings"]["setup_seconds"] = clock.get("layout", 0)+clock.get("target", 0)
            normalized["resources"]["threads"] = native.get("omp_threads", 1)
        normalized["timings"].update(
            scope="complete wrapper with per-sample density; validator excluded; no compiler or full-law enumeration",
            density_seconds=clock.get("density") if method == "frugal" else None,
            density_within_sampling=method == "candidate", formal_performance_claim=False)
        if native.get("status") == "ok":
            observed = np.asarray(native["raw_samples"], dtype=np.uint8).sum(axis=1)
            normalized["click_statistics"] = {"observed_mean": float(observed.mean()),
                "observed_maximum": int(observed.max()), "sample_size": len(observed),
                "source": "this complete fixed observed batch; maximum is not target typical hardness"}
        gaps = {reference: assess_jiuzhang_gap(normalized, evidence, reference_id=reference,
                  target_kind=config["target_kind"], epsilon=config["epsilon"], catalog=catalog)
                for reference in ("jiuzhang1", "jiuzhang2", "jiuzhang3", "jiuzhang4_l1024")}
        save_json(directory/"identity.json", evidence)
        save_json(directory/"run_manifest.json", normalized)
        for reference, report in gaps.items():
            save_json(directory/f"gap_{reference}.json", report)
        arrays = {key: np.asarray(native[key], dtype=np.uint8 if key == "raw_samples" else float)
                  for key in ("raw_samples", "sample_log_probabilities") if native.get(key) is not None}
        if arrays:
            np.savez_compressed(directory/"evidence_arrays.npz", **arrays)
        compact = {key: value for key, value in native.items() if key not in arrays}
        compact["array_archive"] = "evidence_arrays.npz" if arrays else None
        compact["array_archive_sha256"] = digest(directory/"evidence_arrays.npz") if arrays else None
        save_json(directory/"native.json", compact)
        # Retain the original worker result as well: the archive is small for
        # two batches and preserves failure/phase information verbatim.
        bound, witnesses = evidence.get("global_tv") or {}, evidence.get("witnesses") or {}
        row = {"case_id": case_id, "method": method, "sampling_status": native.get("status"),
               "decision": evidence["decision"], "evidence_scope": evidence["evidence_scope"],
               "global_tv_upper": bound.get("upper"), "global_score_mean": bound.get("mean_score"),
               "cutoff_count": bound.get("cutoff_count"),
               "target_probability_failure_count": bound.get("target_probability_failure_count"),
               "witness_tv_lower": witnesses.get("tv_lower"),
               "target_mean_clicks": witnesses.get("target_mean_clicks"),
               "observed_mean_clicks": witnesses.get("observed_mean_clicks"),
               "timings": normalized["timings"], "resources": normalized["resources"],
               "jiuzhang1_verdict": gaps["jiuzhang1"]["verdict"],
               "jiuzhang4_verdict": gaps["jiuzhang4_l1024"]["verdict"],
               "quantum_speed_ratio": gaps["jiuzhang1"]["timing"]["classical_wall_over_reference_wall"]}
        rows.append(row)
        save_json(args.output/"partial_results.json", rows)
        print(json.dumps(row), flush=True)
    metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=perf_counter()-start, batches=len(rows))
    save_json(args.output/"summary.json", {"rows": rows, "scope": config["scope"],
        "familywise_alpha": plan.familywise_alpha, "epsilon": config["epsilon"]})
    save_json(args.output/"metadata.json", metadata)
    print(json.dumps({"status": "complete", "elapsed_seconds": metadata["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/iteration4_jiuzhang1_control.json")
    parser.add_argument("--candidate-executable", type=Path, default=ROOT/".cache/click-iteration4-native/click_sampler")
    parser.add_argument("--frugal-executable", type=Path, default=ROOT/".cache/frugal-evidence-checkpoint-native/frugal_evidence_adapter")
    parser.add_argument("--candidate-validation", type=Path, default=ROOT/"results/iteration4_kernel_validation/first_candidate/metadata.json")
    parser.add_argument("--frugal-validation", type=Path, required=False)
    parser.add_argument("--output", type=Path, default=ROOT/"results/iteration4_jiuzhang1_control")
    parser.add_argument("--worker", choices=("candidate", "frugal"))
    parser.add_argument("--worker-output", type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        sample_worker(arguments)
    elif arguments.frugal_validation is None:
        parser.error("--frugal-validation is required before any actual-model batch")
    else:
        run(arguments)
