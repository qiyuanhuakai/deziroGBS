"""Run fixed-schedule global-TV point validation, with shared cutoff batches.

The familywise schedule is fixed by configuration before drawing any samples.
Default: 6 Gaussian cases x 2 click cutoffs, alpha=.05/12 per bound.  Coverage
is conditional on correct numerical point probabilities and Q densities.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbslab.exact import ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.schur_memory import SchurMemorySampler
from gbslab.surrogates import distribution_metrics
from gbslab.tv_validation import validate_tv


def snapshot_sources(output):
    destination = output / "source_snapshot" / "gbslab"
    destination.mkdir(parents=True, exist_ok=True)
    combined = hashlib.sha256()
    sources = []
    for path in sorted((ROOT / "src" / "gbslab").glob("*.py")):
        data = path.read_bytes()
        (destination / path.name).write_bytes(data)
        combined.update(path.name.encode())
        combined.update(data)
        sources.append({"file": path.name, "sha256": hashlib.sha256(data).hexdigest()})
    shutil.copy2(__file__, output / "runner_snapshot.py")
    return combined.hexdigest(), sources


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    planned = len(config["modes"]) * len(config["seeds"]) * len(config["exact_click_cutoffs"])
    if planned != config["planned_comparisons"]:
        raise ValueError("fixed familywise comparison count does not match configuration")
    if config["topology"] != "haar" or config["sampler"] != "schur_pairwise":
        raise ValueError("this protocol is fixed to the stated Haar/Schur-pairwise experiment")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "records.jsonl").exists() or (output / "metrics.csv").exists():
        raise FileExistsError("refusing to overwrite an earlier validation run")
    data_dir = output / "instances"
    data_dir.mkdir(exist_ok=True)
    source_hash, source_files = snapshot_sources(output)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    alpha = config["familywise_alpha"] / planned
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "source_sha256": source_hash, "source_files": source_files,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "confidence_schedule": {
            "planned_bounds": planned, "familywise_alpha": config["familywise_alpha"],
            "per_bound_alpha": alpha, "per_bound_nominal_confidence": 1-alpha,
            "familywise_nominal_confidence": 1-config["familywise_alpha"],
            "method": "union bound over the fixed 12 comparisons; shared batches need not be independent",
            "numerical_qualification": "conditional on correct floating point target and Q probabilities; not outward-rounded arithmetic",
        },
        "batch_policy": "512 iid Q draws per case, identical batch reused for cutoffs 4 and 6; no retries",
        "cost_notes": "descriptive shared-CPU timing only; Q caches may be warm across validation cutoffs; each cutoff has a new target oracle",
        "dense_reference_policy": "only m=10, after draws and both validations; never used in fitting or density scoring",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows, cases, failures = [], [], []
    start_all = perf_counter()
    with (output / "records.jsonl").open("w") as journal:
        for m in config["modes"]:
            for seed in config["seeds"]:
                spec = {"modes": m, "seed": seed, "topology": "haar",
                        "squeezing": config["squeezing"], "transmissivity": config["transmissivity"]}
                identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
                U = haar_unitary(m, np.random.default_rng(seed))
                V = covariance_from_interferometer(U, spec["squeezing"], spec["transmissivity"])
                start = perf_counter()
                model = ThresholdModel(V, max_exact_modes=3, vacuum_cache_size=8192)
                sampler = SchurMemorySampler(model, config["retained_clicks"], selection="pairwise")
                setup_seconds = perf_counter() - start
                sample_seed = config["sample_seed_offset"] + seed + 101*m
                rng = np.random.default_rng(sample_seed)
                start = perf_counter()
                try:
                    samples = sampler.sample(config["sample_count"], rng)
                except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                    failure = {"case_id": identity, "spec": spec, "stage": "sampling",
                               "error_type": type(exc).__name__, "error": str(exc), "retries": 0}
                    failures.append(failure)
                    journal.write(json.dumps(failure) + "\n"); journal.flush()
                    print(f"m={m} seed={seed} sampling_failed", flush=True)
                    continue
                draw_seconds = perf_counter() - start
                mean_clicks = float(samples.sum(axis=1).mean())
                sample_hash = hashlib.sha256(samples.tobytes()).hexdigest()
                case_record = {"case_id": identity, "spec": spec, "sample_seed": sample_seed,
                               "samples_sha256": sample_hash, "sample_count": len(samples),
                               "sample_mean_clicks": mean_clicks, "setup_seconds": setup_seconds,
                               "draw_seconds": draw_seconds, "sampler_after_draws": sampler.cache_info(),
                               "model_after_draws": model.cache_info()}
                validations = []
                for cutoff in config["exact_click_cutoffs"]:
                    result = validate_tv(
                        model, sampler, config["sample_count"], cutoff, alpha,
                        samples=samples, keep_scores=True,
                        max_relative_roundoff=config["max_relative_roundoff"])
                    validation = {"case_id": identity, "spec": spec, "cutoff": cutoff,
                                  "samples_sha256": sample_hash, "validation": result}
                    validations.append(validation)
                    print(f"m={m} seed={seed} L={cutoff} mean_y={result['mean_y']:.6g} "
                          f"upper={result['upper']:.6g} cutoff={result['cutoff_count']} "
                          f"target_fail={result['target_probability_failure_count']} "
                          f"q_fail={result['q_probability_failure_count']}", flush=True)
                dense = None
                target = surrogate = None
                if m in config["dense_calibration_modes"]:
                    start = perf_counter()
                    target = ThresholdModel(V).full_distribution()
                    surrogate = sampler.full_distribution()
                    dense = {"metrics": distribution_metrics(target, surrogate),
                             "evaluation_seconds": perf_counter() - start,
                             "timing_order": "after both point-validation cutoffs"}
                    case_record["dense_calibration"] = dense
                input_path = data_dir / f"{identity}.npz"
                arrays = {"U": U, "covariance": V, "samples": samples,
                          "squeezing": np.full(m, spec["squeezing"]),
                          "transmissivity": np.array(spec["transmissivity"])}
                if dense is not None:
                    arrays.update(target=target, surrogate=surrogate)
                np.savez_compressed(input_path, **arrays)
                case_record.update(path=f"instances/{identity}.npz",
                                   sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
                cases.append(case_record)
                for validation in validations:
                    result = validation["validation"]
                    true_tv = dense["metrics"]["tv"] if dense is not None else None
                    validation["dense_true_tv"] = true_tv
                    validation["upper_covers_dense_tv"] = (
                        result["upper"] >= true_tv if true_tv is not None else None)
                    journal.write(json.dumps(validation) + "\n"); journal.flush()
                    rows.append({"case_id": identity, **spec, "sample_seed": sample_seed,
                        "sample_count": len(samples), "sample_mean_clicks": mean_clicks,
                        "max_exact_clicks": validation["cutoff"], "per_bound_alpha": alpha,
                        "familywise_alpha": config["familywise_alpha"], "planned_bounds": planned,
                        "mean_y": result["mean_y"], "statistical_margin": result["statistical_margin"],
                        "upper": result["upper"], "dense_true_tv": true_tv,
                        "upper_covers_dense_tv": validation["upper_covers_dense_tv"],
                        "cutoff_count": result["cutoff_count"],
                        "target_probability_failure_count": result["target_probability_failure_count"],
                        "q_probability_failure_count": result["q_probability_failure_count"],
                        "unique_target_probability_evaluations": result["unique_target_probability_evaluations"],
                        "setup_seconds": setup_seconds, "draw_seconds": draw_seconds,
                        "q_probability_seconds": result["q_probability_seconds"],
                        "target_probability_seconds": result["target_probability_seconds"],
                        "validation_seconds": result["elapsed_seconds"],
                        "strict_floating_point_certificate": result["strict_floating_point_certificate"],
                        "samples_sha256": sample_hash})
    if rows:
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    metadata.update(status="complete" if not failures else "complete_with_failures",
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=perf_counter()-start_all, completed_bounds=len(rows),
                    completed_sample_batches=len(cases), unique_iid_draws=sum(c["sample_count"] for c in cases))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status":metadata["status"],"bounds":len(rows),
                      "batches":len(cases),"elapsed_seconds":metadata["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase2_tv_validation.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
