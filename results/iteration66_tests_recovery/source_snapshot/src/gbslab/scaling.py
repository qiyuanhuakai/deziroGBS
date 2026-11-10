"""Larger-mode cost probe, deliberately without full-target enumeration.

This is a runtime feasibility measurement, not a distributional accuracy
certificate. A small sample batch is not enough to establish global quality.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .exact import ThresholdModel
from .experiment import make_instance, build_candidate, source_digest, json_safe
from .photon_tail import photon_number_tail


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.csv").exists():
        raise FileExistsError("Refusing to overwrite previous scaling probe")
    snapshot = output / "source_snapshot" / "gbslab"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    metadata = {"config": config, "source_sha256": source_digest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "quality_status": "No full TV/KL evaluated at these sizes. Cost probe only.",
        "timing_notes": "Fresh covariance oracle per method, 8192-entry vacuum cache. No exact reference or dense surrogate enumeration. One cold batch; not a throughput benchmark or speedup claim."}
    rows, failures = [], []
    start_all = perf_counter()
    with (output / "records.jsonl").open("w") as log:
        for m, topology, eta in itertools.product(config["modes"], config["topologies"], config["transmissivity"]):
            spec = {"modes": m, "topology": topology,
                "squeezing": config["squeezing"], "transmissivity": eta, "seed": config["seed"]}
            U, r, eta, V = make_instance(spec)
            for method_index, method in enumerate(config["methods"]):
                row = {**spec, **method, "sample_count": config["samples_per_method"]}
                try:
                    start = perf_counter()
                    oracle = ThresholdModel(V, max_exact_modes=3, vacuum_cache_size=8192)
                    fitted = build_candidate(oracle, method)
                    row["fit_seconds"] = perf_counter() - start
                    row["fit_determinants"] = oracle.cache_info()["determinant_evaluations"]
                    start = perf_counter()
                    samples = fitted.sample(config["samples_per_method"],
                        np.random.default_rng(config["seed"] + 1000 + method_index))
                    row["sample_seconds"] = perf_counter() - start
                    row["total_seconds"] = row["fit_seconds"] + row["sample_seconds"]
                    row["sample_mean_clicks"] = float(samples.sum(axis=1).mean())
                    row["expected_mean_photons"] = float(eta * np.sum(np.sinh(r)**2))
                    row["photon_tail_bound"] = photon_number_tail(r, eta, method["k"]) if method["strategy"].startswith("zero_") else None
                    row["total_determinants"] = oracle.cache_info()["determinant_evaluations"]
                    row["vacuum_cache_peak_entries"] = oracle.cache_info()["vacuum_cache_peak_entries"]
                    row["vacuum_cache_evictions"] = oracle.cache_info()["vacuum_cache_evictions"]
                    row["marginal_cache_bytes"] = oracle.cache_info()["marginal_cache_bytes"]
                    rows.append(row)
                    record = {"status": "ok", "metrics": row, "oracle_diagnostics": oracle.cache_info(),
                        "candidate_diagnostics": getattr(fitted, "diagnostics", {}),
                        "candidate_cache_info": fitted.cache_info() if hasattr(fitted,"cache_info") else None}
                except (ArithmeticError, ValueError, FloatingPointError) as exc:
                    record = {"status": "failed", "spec": spec, "method": method,
                        "error_type": type(exc).__name__, "error": str(exc)}
                    failures.append(record)
                log.write(json.dumps(json_safe(record)) + "\n")
                log.flush()
                print(f"m={m} {topology} eta={eta} {method['strategy']} {record['status']}", flush=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    metadata.update(elapsed_seconds=perf_counter()-start_all, successful_rows=len(rows),
        failed_rows=len(failures), completed_utc=datetime.now(timezone.utc).isoformat())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(json.dumps({"rows":len(rows), "failures":len(failures), "seconds":metadata["elapsed_seconds"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
