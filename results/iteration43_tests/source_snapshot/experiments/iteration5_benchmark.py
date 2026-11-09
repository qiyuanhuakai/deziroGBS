"""Cold, fixed-budget prefix-cache performance comparison at unchanged Q.

Each repetition creates a new process/cache; setup and every cache miss are
included. No full-law evaluator or validation warms a timed process. This
script records failures without retries and leaves every raw native record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter

import numpy as np

from gbslab.cpp_click import run_cpp_click, CppClickError
from gbslab.frugal_scaling import frugal_native_sample
from gbslab.gaussian import covariance_from_interferometer, haar_unitary

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked_build(executable, validation):
    sha = digest(executable)
    build = json.loads((executable.parent / "build.json").read_text())
    gate = json.loads(validation.read_text())
    if build.get("executable_sha256") != sha or gate.get("executable_sha256") != sha or gate.get("passed") is not True:
        raise ValueError("Binary/build/validation binding failed")
    flags = build.get("command", []) + build.get("flags", [])
    if not all(f in flags for f in ("-O3", "-march=native", "-ffp-contract=off")):
        raise ValueError("Native benchmark compiler flags do not match the protocol")
    if any(f in flags for f in ("-ffast-math", "-Ofast")):
        raise ValueError("Unsafe floating-point compiler flag")
    return {"executable_sha256": sha, "build": build, "validation": gate,
            "validation_sha256": digest(validation)}


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output; never replace timed attempts")
    config = json.loads(args.config.read_text())
    candidate = checked_build(args.executable, args.validation)
    frugal = checked_build(args.frugal_executable, args.frugal_validation)
    if candidate["build"]["compiler_sha256"] != frugal["build"]["compiler_sha256"]:
        raise ValueError("Candidate and Frugal compiler binary mismatch")
    args.output.mkdir(parents=True)
    shutil.copy2(args.config, args.output / "config.json")
    save(args.output / "candidate_build_and_validation.json", candidate)
    save(args.output / "frugal_build_and_validation.json", frugal)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for src in (Path(__file__), ROOT / "experiments/click_sampler.cpp",
                ROOT / "src/gbslab/cpp_click.py", ROOT / "src/gbslab/frugal_scaling.py"):
        shutil.copy2(src, snapshot / src.name)
    metadata = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(args.config), "threads": config["threads"],
        "source_hashes": {p.name: digest(p) for p in snapshot.iterdir()},
        "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "timing_scope": config["primary_metric"], "claim_scope": config["claim_scope"],
        "dedicated_host": False, "prewarming": False}
    save(args.output / "metadata.json", metadata)
    rows, failures, checks = [], [], []
    started = perf_counter()
    case_index = 0
    with (args.output / "records.jsonl").open("x") as journal:
        for m in config["modes"]:
            U = haar_unitary(m, np.random.default_rng(config["interferometer_seed"] + m))
            for profile in config["profiles"]:
                case_id = f"m{m}_{profile['name']}"
                V = covariance_from_interferometer(U, profile["squeezing"], loss=profile["transmissivity"])
                input_path = args.output / f"input_{case_id}.npz"
                np.savez_compressed(input_path, covariance=V)
                input_sha = digest(input_path)
                methods = ["uncached", "prefix_cached"]
                if m in config["official_frugal_modes"]:
                    methods.append("official_frugal")
                for n in config["sample_counts_by_modes"][str(m)]:
                    for repetition in range(config["repetitions"]):
                        seed = config["sample_seed_base"] + case_index * 101 + repetition
                        shift = (case_index + repetition) % len(methods)
                        outputs = {}
                        for method in methods[shift:] + methods[:shift]:
                            row = {"case_id": case_id, "modes": m, "profile": profile["name"],
                                "method": method, "samples": n, "repetition": repetition,
                                "seed": seed, "input_sha256": input_sha}
                            try:
                                outer = perf_counter()
                                if method == "official_frugal":
                                    native = frugal_native_sample(V, config["official_frugal_cutoff"],
                                        n, seed, executable=args.frugal_executable,
                                        enumerate_distribution=False, timeout=config["timeout_seconds"])
                                    t = native["timings_seconds"]
                                    setup = t["layout"] + t["target"]
                                else:
                                    spec = config["candidate"]
                                    native = run_cpp_click(V, selection=spec["selector"],
                                        max_clicks=spec["k"], samples=n, seed=seed,
                                        vacuum_kernel=spec["vacuum_kernel"],
                                        prefix_cache_entries=config["cache_capacities"][method == "prefix_cached"],
                                        executable=args.executable, raw_samples=False,
                                        profile=False, record_log_probabilities=False,
                                        enumerate_distribution=False, timeout=config["timeout_seconds"])
                                    t = native["timings_seconds"]
                                    setup = t["setup"]
                                row.update(status="ok", setup_seconds=setup,
                                    sampling_seconds=t["sampling"], cold_seconds=setup+t["sampling"],
                                    microseconds_per_draw=t["sampling"] / n * 1e6,
                                    harness_seconds=perf_counter()-outer,
                                    prefix_cache=native.get("prefix_cache"))
                                native.pop("build_metadata", None)
                                record = {"metrics": row, "native": native}
                                outputs[method] = native
                            except (CppClickError, OSError, RuntimeError, ValueError,
                                    ArithmeticError, subprocess.TimeoutExpired) as exc:
                                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                                record = {"metrics": row, "payload": getattr(exc, "payload", None)}
                                failures.append(row)
                            rows.append(row)
                            journal.write(json.dumps(record, allow_nan=False) + "\n")
                            journal.flush()
                        if all(k in outputs for k in ("uncached", "prefix_cached")):
                            a, b = outputs["uncached"], outputs["prefix_cached"]
                            keys = ["histogram_counts", "click_count_histogram", "mode_click_counts", "per_mode_click_counts"]
                            keys = [k for k in keys if k in a or k in b]
                            checks.append({"case_id": case_id, "samples": n,
                                "repetition": repetition, "fields": keys,
                                "same_seed_summary_equal": all(a.get(k) == b.get(k) for k in keys),
                                "scope": "Summary equality only; complete path/conditional equivalence is a separate validation gate"})
                    print(json.dumps({"completed": case_id, "samples": n,
                        "rows": len(rows), "failures": len(failures)}), flush=True)
                case_index += 1
    if digest(args.executable) != candidate["executable_sha256"] or digest(args.frugal_executable) != frugal["executable_sha256"]:
        raise RuntimeError("A benchmark binary changed during execution")
    save(args.output / "metrics.json", rows)
    save(args.output / "failures.json", failures)
    save(args.output / "same_seed_checks.json", checks)
    paired = []
    for case_id in dict.fromkeys(r["case_id"] for r in rows):
        selected = [r for r in rows if r["case_id"] == case_id]
        for n in dict.fromkeys(r["samples"] for r in selected):
            means = {}
            for method in dict.fromkeys(r["method"] for r in selected):
                values = [r for r in selected if r["samples"] == n and r["method"] == method and r["status"] == "ok"]
                if len(values) != config["repetitions"]:
                    continue
                means[method] = {k: float(np.median([r[k] for r in values]))
                    for k in ("setup_seconds", "sampling_seconds", "cold_seconds", "harness_seconds")}
            entry = {"case_id": case_id, "modes": selected[0]["modes"], "samples": n, "medians": means}
            if "uncached" in means and "prefix_cached" in means:
                entry["cold_speedup"] = means["uncached"]["cold_seconds"] / means["prefix_cached"]["cold_seconds"]
                entry["sampling_speedup"] = means["uncached"]["sampling_seconds"] / means["prefix_cached"]["sampling_seconds"]
            if "official_frugal" in means and "prefix_cached" in means:
                entry["frugal_over_cached_sampling"] = means["official_frugal"]["sampling_seconds"] / means["prefix_cached"]["sampling_seconds"]
                entry["frugal_over_cached_cold"] = means["official_frugal"]["cold_seconds"] / means["prefix_cached"]["cold_seconds"]
            paired.append(entry)
    metadata.update(status="complete" if not failures and all(c["same_seed_summary_equal"] for c in checks)
                    else "complete_with_failures", rows=len(rows), failures=len(failures),
        elapsed_seconds=perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat())
    save(args.output / "metadata.json", metadata)
    save(args.output / "summary.json", {"paired": paired, "rows": len(rows), "failures": len(failures),
        "all_same_seed_summary_checks_passed": all(c["same_seed_summary_equal"] for c in checks),
        "claim_scope": config["claim_scope"]})
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/iteration5_prefix_benchmark.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/iteration5_performance")
    p.add_argument("--executable", type=Path, default=ROOT / ".cache/click-prefix-native/click_sampler")
    p.add_argument("--validation", type=Path, default=ROOT / "results/iteration5_prefix_validation/validation.json")
    p.add_argument("--frugal-executable", type=Path, default=ROOT / ".cache/frugal-native/frugal_sampling_adapter")
    p.add_argument("--frugal-validation", type=Path, default=ROOT / ".cache/frugal-native/validation.json")
    run(p.parse_args())
