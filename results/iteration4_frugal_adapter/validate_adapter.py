"""Archive bounded Frugal evidence integration gates; no large K=5 benchmark.

Run with PYTHONPATH=src and an explicitly built evidence executable. The
separate boundary probe is compiled only when --compile-boundary-probe is
given. All timing values here are descriptive validation timings.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np

from gbslab.frugal_evidence import (
    DEFAULT_EXECUTABLE, frugal_evidence_sample, frugal_storage_lower_bound,
)
from gbslab.frugal_scaling import frugal_native_sample

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-boundary-probe", action="store_true")
    args = parser.parse_args()
    executable = DEFAULT_EXECUTABLE
    old = ROOT / ".cache/frugal-native/frugal_sampling_adapter"
    older = ROOT / ".cache/frugal-build/frugal_adapter"
    build_path = executable.parent / "build.json"
    build = json.loads(build_path.read_text())
    old_build = json.loads((old.parent / "build.json").read_text())
    frozen = {str(p): sha(p) for p in [executable, build_path, old, old.parent / "build.json", older,
        ROOT / ".cache/click-native/click_sampler"]}
    assert build["executable_sha256"] == frozen[str(executable)]
    assert old_build["executable_sha256"] == frozen[str(old)]
    boundary = None
    if args.compile_boundary_probe:
        source = OUT / "boundary_probe.cpp"
        probe = executable.parent / "boundary_probe"
        command = build["command"]
        # Keep the compiler, flags, and include directories from this exact
        # build, replacing only its source-file list and output executable.
        start_sources = next(i for i, value in enumerate(command) if value.endswith("frugal_evidence_adapter.cpp"))
        probe_command = command[:start_sources] + [str(source), "-o", str(probe)]
        subprocess.run(probe_command, check=True, capture_output=True, text=True)
        result = subprocess.run([str(probe)], check=True, capture_output=True, text=True)
        boundary = json.loads(result.stdout)
        boundary.update(source_sha256=sha(source), executable_sha256=sha(probe), command=probe_command)
        (OUT / "boundary_validation.json").write_text(json.dumps(boundary, indent=2) + "\n")
    else:
        boundary = json.loads((OUT / "boundary_validation.json").read_text())
    fixtures = []
    archived = ROOT / "results/cpp_frugal_validation"
    for name, k in [("haar8_K4", 4), ("haar8_K5", 5), ("clipping8", 5),
                    ("heterogeneous10", 5), ("bright12", 5)]:
        with np.load(archived / (name + ".npz")) as data:
            fixtures.append((name, np.asarray(data["covariance"]), k, 17, True, True))
    for m in [8, 16, 32, 64, 100, 144]:
        # Analytically soluble last-three-mode thermal fixture exercises high
        # physical bits with nonzero clicks, avoiding a merely all-zero test.
        nbar = np.zeros(m); nbar[-3:] = [.5, 1., 2.]
        fixtures.append((f"thermal_tail_m{m}", np.diag(np.tile(1 + 2 * nbar, 2)), 2, 17, m <= 14, m <= 64))
    rows = []
    for index, (name, V, k, n, enumerate_q, compare_old) in enumerate(fixtures):
        seed = 99100 + index
        out = frugal_evidence_sample(V, k, n, seed, enumerate_distribution=enumerate_q)
        m = len(V) // 2
        samples = np.asarray(out["raw_samples"], dtype=np.uint8)
        logs = np.asarray(out["sample_log_probabilities"])
        row = {"name": name, "modes": m, "cutoff": k, "sample_count": n, "seed": seed,
               "covariance_sha256": out["covariance_sha256"], "density_trace": out["density_trace"],
               "trace": out["trace"], "timings_seconds_descriptive": out["timings_seconds"],
               "peak_rss_bytes_linux": out["peak_rss_bytes_linux"], "trace_workspace_bytes": out["trace_workspace_bytes"]}
        arrays = {"covariance": V, "raw_samples": samples, "sample_log_probabilities": logs}
        if enumerate_q:
            q = np.asarray(out["distribution"])
            masks = samples.astype(np.int64) @ (1 << np.arange(m))
            row["log_q_max_error_to_full_q"] = float(np.max(np.abs(logs - np.log(q[masks]))))
            row["full_q_normalization_error"] = float(abs(q.sum(dtype=np.longdouble) - 1))
            row["off_support_samples"] = int(np.asarray(out["histogram_counts"])[q == 0].sum())
            assert row["log_q_max_error_to_full_q"] < 2e-10 and row["off_support_samples"] == 0
            arrays["q"] = q
        if compare_old:
            reference = frugal_native_sample(V, k, n, seed, enumerate_distribution=enumerate_q, executable=old)
            row["frozen_official_histograms_identical"] = all(out[field] == reference[field] for field in (
                "histogram_counts", "per_mode_click_counts", "click_count_histogram"))
            assert row["frozen_official_histograms_identical"]
            if enumerate_q:
                old_q = np.asarray(reference["distribution"])
                row["tv_to_previous_trace"] = float(.5 * np.abs(q - old_q).sum())
                assert row["tv_to_previous_trace"] < 1e-10
                arrays["q_previous"] = old_q
        if name == "clipping8":
            assert out["trace"]["saturated_threshold_path_steps"] > 0
            untraced = frugal_evidence_sample(V, k, n, seed, record_log_probabilities=False)
            row["ordered_samples_unchanged_by_density_tracing"] = out["raw_samples"] == untraced["raw_samples"]
            assert row["ordered_samples_unchanged_by_density_tracing"]
        if name.startswith("thermal_tail"):
            p = np.array([.5, 1., 2.]) / np.array([1.5, 2., 3.])
            expected = np.where(samples[:, -3:], np.log(p), np.log1p(-p)).sum(axis=1)
            row["analytic_log_q_max_error"] = float(np.max(np.abs(logs - expected)))
            row["clicks_in_last_three_modes"] = int(samples[:, -3:].sum())
            assert samples[:, :-3].sum() == 0 and row["clicks_in_last_three_modes"] > 0
            assert row["analytic_log_q_max_error"] < 2e-9
            arrays["analytic_log_q"] = expected
        np.savez_compressed(OUT / (name + ".npz"), **arrays)
        rows.append(row)
        (OUT / "validation_progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(name, "passed", flush=True)
    for path, digest in frozen.items():
        assert sha(path) == digest, "Binary or build metadata changed during validation: " + path
    source_paths = [ROOT / "experiments/frugal_evidence_adapter.cpp", ROOT / "experiments/build_frugal_evidence.py",
        ROOT / "src/gbslab/frugal_evidence.py", ROOT / "tests/test_frugal_evidence.py",
        ROOT / "experiments/build_frugal.py", Path(__file__),
        executable.parent / "generated/frugal_log_trace.hpp"]
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in source_paths:
        shutil.copy2(path, snapshot / path.name)
    shutil.copy2(build_path, OUT / "build.json")
    shutil.copy2(old.parent / "build.json", OUT / "frozen_reference_build.json")
    record = {"passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "executable_sha256": frozen[str(executable)], "build_cache_key": build["cache_key"],
              "upstream_commit": build["upstream_commit"], "sampler_body_sha256": build["sampler_body_sha256"],
              "precision": build["precision"], "real_precision": build["real_precision"],
              "modes_compiled": build["modes"], "flags": build["flags"],
              "preserved_file_hashes": frozen, "source_hashes": {str(p): sha(p) for p in source_paths},
              "cases": rows, "boundary_probe": boundary,
              "storage_payload_lower_bounds": [frugal_storage_lower_bound(m, 5) for m in [64, 100, 144]],
              "scope": ["Official generate_samples unchanged; separate density trace only.",
                        "All timings descriptive, not a performance comparison.",
                        "Modes 16..144 use soluble product-state fixtures, not correlated global GBS-quality evidence.",
                        "No 100- or 144-mode K5 sampling performed by this validation.",
                        "No retry, postselection, automatic K reduction, or finite-PRNG-density claim."]}
    (OUT / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(rows), "executable_sha256": record["executable_sha256"]}))


if __name__ == "__main__":
    main()
