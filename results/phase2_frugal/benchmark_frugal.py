"""Compare pinned official FrugalGBS on archived, unchanged phase-two inputs.

The instrumented law is the continuous-uniform threshold law of the upstream
floating-point sampler. Target probabilities are used only after generation.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
from time import perf_counter

import numpy as np

from gbslab.frugal import frugal_sample
from gbslab.surrogates import distribution_metrics
from gbslab.experiment import json_safe, click_count_distribution

ROOT = Path(__file__).resolve().parents[1]


def run(suites, output, cutoffs, sample_count):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "records.jsonl").exists():
        raise FileExistsError("Use a fresh output directory")
    build = ROOT / ".cache/frugal-build"
    build_record = json.loads((build / "build.json").read_text())
    validation = json.loads((build / "validation.json").read_text())
    binary = build / "frugal_adapter"
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    if (not validation.get("passed") or
            build_record["executable_sha256"] != binary_sha256 or
            validation["executable_sha256"] != binary_sha256 or
            validation["build_cache_key"] != build_record["cache_key"]):
        raise RuntimeError("Validation, build manifest, and current executable do not match")
    for filename in ("build.json", "validation.json"):
        shutil.copy2(build / filename, output / filename)
    for source in (Path(__file__), ROOT / "src/gbslab/frugal.py",
                   ROOT / "experiments/frugal_adapter.cpp", ROOT / "experiments/build_frugal.py"):
        shutil.copy2(source, output / source.name)
    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "suites": suites, "cutoffs": cutoffs, "sample_count": sample_count,
        "upstream_commit": "fa8b796f2afebd4845a2e335e457735952617ef8",
        "binary_sha256": binary_sha256,
        "precision": "double", "threads": 1,
        "distribution": "instrumented continuous-uniform threshold law of pinned SIMD code",
        "sampling": "unmodified official generate_samples",
        "notes": ["128-draw timings are latency probes, not steady-state throughput.",
                  "Python wrapper/process I/O is separately reported.",
                  "Target P is evaluator-only; no target vector is passed to sampler.",
                  "Cumulant cutoff K is not the same resource as retained-click k.",
                  "No quantum hardware, MPS, or Gaussian-reset baseline in this run."],
        "status": "running",
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    rows, failures = [], []
    started = perf_counter()
    with (output / "records.jsonl").open("w") as journal:
        for suite in suites:
            directory = ROOT / "results" / suite
            manifests = json.loads((directory / "instances_manifest.json").read_text())
            for index, item in enumerate(manifests):
                path = directory / item["path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise RuntimeError("Archived input hash mismatch")
                with np.load(path) as data:
                    V, p = data["covariance"], data["target"]
                laws = {}
                for cutoff in cutoffs:
                    spec = item["spec"]
                    row = {"suite": suite, "case_id": item["case_id"], **spec,
                           "strategy": "frugal", "K": cutoff, "sample_count": sample_count}
                    try:
                        result = frugal_sample(V, cutoff, sample_count,
                            seed=spec["seed"] + 200000, threads=1)
                        q = np.asarray(result.pop("distribution"), dtype=float)
                        if result["trace"]["nonfinite_threshold_path_steps"]:
                            raise FloatingPointError("Nonfinite traced thresholds")
                        row.update(distribution_metrics(p, q))
                        count_p = click_count_distribution(p, spec["modes"])
                        row["click_count_tv"] = float(np.abs(count_p -
                            click_count_distribution(q, spec["modes"])).sum() / 2)
                        timing = result["timings_seconds"]
                        row.update({name + "_seconds": value for name, value in timing.items()})
                        row["cold_batch_seconds"] = timing["layout"] + timing["target"] + timing["sampling"]
                        row["wrapper_wall_seconds_including_evaluation"] = result["wrapper_wall_seconds"]
                        row["saturated_threshold_path_steps"] = result["trace"]["saturated_threshold_path_steps"]
                        row["distribution_normalization_error"] = abs(float(q.sum()) - 1)
                        laws[f"frugal_K{cutoff}"] = q
                        rows.append(row)
                        record = {"status": "ok", "metrics": row,
                                  "input_sha256": item["sha256"], "official_result": result}
                    except (RuntimeError, ArithmeticError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
                        record = {"status": "failed", "suite": suite, "case_id": item["case_id"],
                                  "spec": spec, "K": cutoff, "error_type": type(exc).__name__, "error": str(exc)}
                        failures.append(record)
                    journal.write(json.dumps(json_safe(record)) + "\n")
                    journal.flush()
                np.savez_compressed(output / f"surrogates_{item['case_id']}.npz", **laws)
                print(f"{suite}: {index+1}/{len(manifests)} rows={len(rows)} failures={len(failures)}", flush=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    meta.update(status="complete" if not failures else "complete_with_failures",
                completed_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=perf_counter()-started, rows=len(rows), failures=len(failures))
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({key:meta[key] for key in ("status", "rows", "failures", "elapsed_seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=["phase2_discovery", "phase2_holdout", "phase2_bright"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase2_frugal")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    run(args.suites, args.output, args.cutoffs, args.samples)
