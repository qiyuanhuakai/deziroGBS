"""Reproducible experiments with cold fitting and explicit evaluation costs.

The target is used only by the evaluator. Candidate fits receive a separate
fresh Gaussian covariance oracle. Sampling is timed BEFORE dense Q evaluation
so that the latter cannot prewarm an adaptive sampler's cache.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np
import scipy

from .exact import ThresholdModel
from .gaussian import covariance_from_interferometer, haar_unitary, local_unitary
from .surrogates import distribution_metrics, fit_surrogate, kl_chain_decomposition
from .cache import cached_reference


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def source_digest():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def make_instance(spec):
    rng = np.random.default_rng(spec["seed"])
    m = spec["modes"]
    topology = spec["topology"]
    if topology == "haar":
        U = haar_unitary(m, rng)
    elif topology.startswith("local"):
        U = local_unitary(m, int(topology.removeprefix("local")), rng)
    else:
        raise ValueError(f"unknown topology: {topology}")
    r = np.full(m, spec["squeezing"])
    eta = spec["transmissivity"]
    V = covariance_from_interferometer(U, r, loss=eta)
    return U, r, eta, V


def build_candidate(model, method):
    strategy, k = method["strategy"], method["k"]
    if strategy in ("zero_memory", "zero_pairwise"):
        from .zero_memory import ZeroMemorySampler
        return ZeroMemorySampler(model, max_clicks=k,
            selection="pairwise" if strategy == "zero_pairwise" else "recent")
    if strategy.startswith("calibrated_"):
        from .calibration import CalibratedSurrogate
        base = fit_surrogate(model, k, strategy.removeprefix("calibrated_"))
        return CalibratedSurrogate(model, base)
    return fit_surrogate(model, k, strategy)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def click_count_distribution(p, m):
    counts = np.fromiter((i.bit_count() for i in range(1 << m)), dtype=int)
    return np.bincount(counts, weights=p, minlength=m + 1)


def run(config_path: Path, output: Path):
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.csv").exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    inputs_dir = output / "instances"
    inputs_dir.mkdir(exist_ok=True)
    snapshot = output / "source_snapshot" / "gbslab"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(source, snapshot / source.name)
    code_hash = source_digest()
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": digest(config),
        "source_sha256": code_hash,
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "thread_environment": {key: os.environ.get(key) for key in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "timing_notes": "Fresh oracle for every method. Sampling before dense validation; no GPU. Wall times are descriptive, not hardware speedup claims.",
        "statistics_notes": "TV/KL computed from full small distributions, not sampling estimates. Seeds share an interferometer across loss/squeezing; rows are not independent repetitions.",
        "status": "running",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    failure_rows = []
    manifest = []
    specs = [dict(zip(("modes", "topology", "squeezing", "transmissivity", "seed"), values))
        for values in itertools.product(config["modes"], config["topologies"],
            config["squeezing"], config["transmissivity"], config["seeds"])]
    total_start = perf_counter()
    with (output / "records.jsonl").open("w") as journal:
        for index, spec in enumerate(specs):
            case_id = digest(spec)[:16]
            U, r, eta, V = make_instance(spec)
            start = perf_counter()
            p, reference_seconds, reference_diagnostics, cache_hit, cache_key = cached_reference(
                spec, V, config_path.parent.parent / ".cache" / "targets")
            reference_access_seconds = perf_counter() - start
            count_p = click_count_distribution(p, spec["modes"])
            input_path = inputs_dir / f"{case_id}.npz"
            np.savez_compressed(input_path, U=U, squeezing=r,
                transmissivity=np.array(eta), covariance=V, target=p)
            manifest.append({"case_id": case_id, "spec": spec,
                "path": f"instances/{case_id}.npz",
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "reference_seconds": reference_seconds,
                "reference_diagnostics": reference_diagnostics,
                "reference_cache_hit": cache_hit, "reference_cache_key": cache_key})
            for method_index, method in enumerate(config["methods"]):
                row = {"case_id": case_id, **spec, **method,
                    "reference_seconds": reference_seconds,
                    "reference_access_seconds": reference_access_seconds,
                    "reference_cache_hit": cache_hit,
                    "target_vacuum_probability": float(p[0]),
                    "target_mean_clicks": float(np.dot(np.arange(len(count_p)), count_p)),
                    "target_mean_photons": float(eta * np.sum(np.sinh(r) ** 2))}
                try:
                    start = perf_counter()
                    oracle = ThresholdModel(V, vacuum_cache_size=8192)
                    candidate = build_candidate(oracle, method)
                    row["fit_seconds"] = perf_counter() - start
                    after_fit = oracle.cache_info()
                    start = perf_counter()
                    samples = candidate.sample(config["samples_per_method"],
                        np.random.default_rng(spec["seed"] + 100000 + method_index))
                    row["sample_seconds"] = perf_counter() - start
                    row["sample_count"] = config["samples_per_method"]
                    after_sampling = oracle.cache_info()
                    start = perf_counter()
                    q = candidate.full_distribution()
                    row["evaluation_seconds"] = perf_counter() - start
                    row.update(distribution_metrics(p, q))
                    count_q = click_count_distribution(q, spec["modes"])
                    row["click_count_tv"] = float(np.sum(np.abs(count_p-count_q))/2)
                    row["vacuum_abs_error"] = float(abs(p[0]-q[0]))
                    row["sampler_empirical_mean_clicks"] = float(samples.sum(axis=1).mean())
                    row["surrogate_mean_clicks"] = float(np.dot(np.arange(len(count_q)), count_q))
                    row["determinants_fit"] = after_fit["determinant_evaluations"]
                    row["determinants_through_sampling"] = after_sampling["determinant_evaluations"]
                    row["determinants_through_evaluation"] = oracle.cache_info()["determinant_evaluations"]
                    row["reference_l1_numerical_correction"] = reference_diagnostics["largest_l1_numerical_correction"]
                    if method["strategy"] in ("zero_memory", "zero_pairwise"):
                        from .photon_tail import photon_number_tail
                        row["click_tail_coupling_bound"] = float(count_p[method["k"]+1:].sum())
                        row["photon_tail_scalable_bound"] = photon_number_tail(r, eta, method["k"])
                        low_masks = np.array([i for i in range(len(p)) if i.bit_count() <= method["k"]])
                        row["low_click_max_abs_error"] = float(np.max(np.abs(p[low_masks]-q[low_masks])))
                        if row["tv"] > row["click_tail_coupling_bound"] + 1e-8:
                            raise ArithmeticError("observed TV exceeds conservative coupling bound")
                    else:
                        row["click_tail_coupling_bound"] = None
                    diagnostic = {"case_id": case_id, "method": method,
                        "status": "ok", "metrics": row,
                        "oracle_after_fit": after_fit,
                        "oracle_after_sampling": after_sampling,
                        "oracle_after_evaluation": oracle.cache_info(),
                        "candidate_diagnostics": getattr(candidate, "diagnostics", {}),
                        "candidate_cache_info": candidate.cache_info() if hasattr(candidate,"cache_info") else None}
                    if method["strategy"] in ("independent", "previous", "pairwise", "greedy_cmi"):
                        diagnostic["kl_chain"] = kl_chain_decomposition(p, candidate)
                    rows.append(row)
                    if config.get("save_distributions", False):
                        np.savez_compressed(output / f"q_{case_id}_{method_index}.npz", q=q)
                except (ArithmeticError, ValueError, FloatingPointError) as exc:
                    diagnostic = {"case_id": case_id, "spec": spec, "method": method,
                        "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
                    failure_rows.append(diagnostic)
                journal.write(json.dumps(json_safe(diagnostic), ensure_ascii=False) + "\n")
                journal.flush()
            print(f"{index+1}/{len(specs)} {case_id} m={spec['modes']} {spec['topology']} r={spec['squeezing']} eta={eta} rows={len(rows)} failures={len(failure_rows)}", flush=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "instances_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "failures.json").write_text(json.dumps(failure_rows, indent=2) + "\n")
    metadata.update(completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=perf_counter()-total_start, cases=len(specs),
        successful_method_rows=len(rows), failed_method_rows=len(failure_rows),
        status="complete" if not failure_rows else "complete_with_failures")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in
        ("status", "elapsed_seconds", "cases", "successful_method_rows", "failed_method_rows")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
