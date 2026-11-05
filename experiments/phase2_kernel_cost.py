"""Paired cold-model cost probe of algebraically equivalent event kernels.

Prepared for a quiet compute window.  This script never enumerates a target or
surrogate distribution.  Each pair uses identical U,V, method-independent RNG
seed and draw count; generated bit arrays must match for the equality check.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
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
from gbslab.zero_memory import ZeroMemorySampler


CONFIG = {
    "modes": [16, 32, 64], "seeds": [7201, 7202], "topology": "haar",
    "squeezing": .6, "transmissivity": .35, "retained_clicks": 2,
    "repeats": 3, "samples_per_method": 128,
    "methods": ["old_pairwise", "schur_pairwise"],
    "sample_seed_offset": 300000,
}


def snapshot_sources(output):
    directory = output / "source_snapshot" / "gbslab"
    directory.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for path in sorted((ROOT / "src" / "gbslab").glob("*.py")):
        content = path.read_bytes()
        (directory / path.name).write_bytes(content)
        h.update(path.name.encode()); h.update(content)
    shutil.copy2(__file__, output / "runner_snapshot.py")
    return h.hexdigest()


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "records.jsonl").exists() or (output / "metrics.csv").exists():
        raise FileExistsError("refusing to overwrite a kernel-cost run")
    inputs = output / "instances"
    inputs.mkdir(exist_ok=True)
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": CONFIG, "source_sha256": snapshot_sources(output),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "protocol": "3 repeats per input; fresh model and sampler caches; alternating method order; same RNG within each pair; no dense enumeration",
        "timing_scope": "model construction, sampler setup, and draws timed separately; U,V construction shared/excluded; CPU cache/OS load not controlled by script",
        "quality_scope": "algebraically equivalent retained-click models; equality of paired samples checked, no new global-quality claim",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows, pairs, failures = [], [], []
    all_start = perf_counter()
    with (output / "records.jsonl").open("w") as journal:
        for m in CONFIG["modes"]:
            for seed in CONFIG["seeds"]:
                U = haar_unitary(m, np.random.default_rng(seed))
                V = covariance_from_interferometer(U, CONFIG["squeezing"], CONFIG["transmissivity"])
                input_hash = hashlib.sha256(U.tobytes() + V.tobytes()).hexdigest()
                case_name = f"m{m}_seed{seed}"
                np.savez_compressed(inputs / f"{case_name}_input.npz", U=U, covariance=V)
                for repeat in range(CONFIG["repeats"]):
                    sample_seed = CONFIG["sample_seed_offset"] + seed + m*101 + repeat*10007
                    order = CONFIG["methods"] if repeat % 2 == 0 else CONFIG["methods"][::-1]
                    records = []
                    draws = {}
                    for method in order:
                        gc.collect()
                        if hashlib.sha256(U.tobytes() + V.tobytes()).hexdigest() != input_hash:
                            raise ArithmeticError("shared optical input mutated before paired run")
                        record = {"modes":m,"seed":seed,"repeat":repeat,"method":method,
                                  "input_sha256":input_hash,"sample_seed":sample_seed,
                                  "sample_count":CONFIG["samples_per_method"]}
                        try:
                            start = perf_counter()
                            model = ThresholdModel(V, max_exact_modes=3, vacuum_cache_size=8192)
                            record["model_setup_seconds"] = perf_counter() - start
                            start = perf_counter()
                            cls = ZeroMemorySampler if method == "old_pairwise" else SchurMemorySampler
                            sampler = cls(model, CONFIG["retained_clicks"], selection="pairwise", cache_size=8192)
                            record["sampler_setup_seconds"] = perf_counter() - start
                            rng = np.random.default_rng(sample_seed)
                            start = perf_counter()
                            samples = sampler.sample(CONFIG["samples_per_method"], rng)
                            record["draw_seconds"] = perf_counter() - start
                            record["setup_seconds"] = record["model_setup_seconds"] + record["sampler_setup_seconds"]
                            record["total_seconds"] = record["setup_seconds"] + record["draw_seconds"]
                            record["sample_mean_clicks"] = float(samples.sum(axis=1).mean())
                            record["samples_sha256"] = hashlib.sha256(samples.tobytes()).hexdigest()
                            record["model_diagnostics"] = model.cache_info()
                            record["sampler_diagnostics"] = sampler.cache_info()
                            record["status"] = "ok"
                            draws[method] = samples
                            del sampler, model
                        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                            record.update(status="failed",error_type=type(exc).__name__,error=str(exc))
                            failures.append(record)
                        records.append(record)
                    if set(draws) == set(CONFIG["methods"]):
                        mismatch = int(np.count_nonzero(draws["old_pairwise"] != draws["schur_pairwise"]))
                        equal = mismatch == 0
                        np.savez_compressed(inputs / f"{case_name}_repeat{repeat}_samples.npz", **draws)
                    else:
                        mismatch, equal = None, None
                    input_unchanged = hashlib.sha256(U.tobytes() + V.tobytes()).hexdigest() == input_hash
                    pair = {"modes":m,"seed":seed,"repeat":repeat,"sample_seed":sample_seed,
                            "method_order":order,"input_sha256":input_hash,"input_unchanged":input_unchanged,
                            "paired_samples_equal":equal,"bit_mismatches":mismatch}
                    pairs.append(pair)
                    for record in records:
                        record.update(paired_samples_equal=equal,bit_mismatches=mismatch,input_unchanged=input_unchanged)
                        journal.write(json.dumps(record) + "\n"); journal.flush()
                        row = {key:value for key,value in record.items()
                               if key not in ("model_diagnostics","sampler_diagnostics")}
                        rows.append(row)
                    print(f"m={m} seed={seed} repeat={repeat} paired_samples_equal={equal} "
                          + " ".join(f"{r['method']}={r.get('draw_seconds',float('nan')):.4f}s" for r in records), flush=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    (output / "pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    metadata.update(status="complete" if not failures else "complete_with_failures",
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=perf_counter()-all_start, method_runs=len(rows),
                    sample_pairs=len(pairs), identical_pairs=sum(pair["paired_samples_equal"] is True for pair in pairs),
                    failed_runs=len(failures))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key:metadata[key] for key in
                      ("status","method_runs","sample_pairs","identical_pairs","elapsed_seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
