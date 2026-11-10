"""Archive corrected Gaussian pairwise controls without replacing old results.

The fixed 27-case cold configuration supplies immutable optical inputs and
historical Python controls.  This builder preserves legacy MI tables and
laws, creates a separate canonical retained-click k=4 reference, and checks
the frozen native implementation against it at the unchanged 2e-8 TV gate.
Compilation and performance benchmarking are deliberately separate tasks.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbslab.cpp_click import CppClickError, find_cpp_executable, run_cpp_click
from gbslab.exact import ThresholdModel
from gbslab.schur_memory import SchurMemorySampler
from gbslab.zero_memory import GAUSSIAN_PAIRWISE_MI_POLICY


FULL_Q_TV_GATE = 2e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_class(source):
    """Load our hash-bound archived class without writing into its snapshot."""
    namespace = {"__name__": "gbslab._legacy_pairwise_reference", "__package__": "gbslab"}
    exec(compile(source.read_bytes(), str(source), "exec"), namespace)
    return namespace["ZeroMemorySampler"]


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use an empty output directory; original references are never overwritten")
    executable = find_cpp_executable(args.cpp_executable)
    if executable is None:
        raise FileNotFoundError("build and freeze the canonical native sampler before reference generation")
    binary_sha = sha(executable)
    build_path = executable.parent / "build.json"
    build = json.loads(build_path.read_text())
    if build.get("executable_sha256") != binary_sha:
        raise RuntimeError("native build metadata does not match its binary")
    configuration = json.loads(args.config.read_text())
    cases = configuration["cold"]["cases"]
    identifiers = [case["case_id"] for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("cold reference configuration repeats an optical case")
    old_validation = json.loads(args.legacy_validation.read_text())
    legacy_source = (args.legacy_validation.parent / (args.legacy_validation.stem+"_cases")
                     / "source_snapshot" / "gbslab" / "zero_memory.py")
    expected_legacy_sha = old_validation["python_reference_sources_sha256"]["zero_memory.py"]
    if sha(legacy_source) != expected_legacy_sha:
        raise RuntimeError("archived historical pairwise source hash changed")
    LegacySampler = legacy_class(legacy_source)
    output.mkdir(parents=True, exist_ok=True)
    source_directory = output / "source_snapshot" / "gbslab"
    source_directory.mkdir(parents=True)
    source_hashes = {}
    for source in sorted((ROOT / "src" / "gbslab").glob("*.py")):
        source_hashes[source.name] = sha(source)
        shutil.copy2(source, source_directory / source.name)
    shutil.copy2(__file__, output / "runner_snapshot.py")
    shutil.copy2(legacy_source, output / "legacy_zero_memory.py")
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "policy": GAUSSIAN_PAIRWISE_MI_POLICY,
        "scope": "retained-click Gaussian pairwise k=4 only; historical BN and click-influence controls unchanged",
        "planned_cases": len(cases), "full_q_tv_gate": FULL_Q_TV_GATE,
        "config_path": str(args.config.resolve()), "config_sha256": sha(args.config),
        "legacy_validation_path": str(args.legacy_validation.resolve()),
        "legacy_validation_sha256": sha(args.legacy_validation),
        "legacy_zero_memory_sha256": expected_legacy_sha,
        "python_reference_sources_sha256": source_hashes,
        "runner_sha256": sha(Path(__file__)), "executable_sha256": binary_sha,
        "build_metadata": build, "python": platform.python_version(), "numpy": np.__version__,
        "numerical_qualification": "floating-point full-law comparisons, not interval-certified arithmetic or a generic exact-GBS claim",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    rows = []
    start_all = perf_counter()
    with (output / "records.jsonl").open("w") as journal:
        for case in cases:
            identifier = case["case_id"]
            input_path = ROOT / case["input"]
            if sha(input_path) != case["input_sha256"]:
                raise RuntimeError(f"optical input archive changed: {identifier}")
            legacy_law_path = (ROOT / "results" / (case["suite"]+"_k4_extension")
                               / f"surrogates_{identifier}.npz")
            with np.load(input_path) as data:
                V, target = data["covariance"], data["target"]
            with np.load(legacy_law_path) as data:
                legacy_q = data["schur_pairwise_k4"]
            row = {"case_id": identifier, "suite": case["suite"], "spec": case["spec"],
                   "input": case["input"], "input_sha256": case["input_sha256"],
                   "legacy_law_path": str(legacy_law_path.relative_to(ROOT)),
                   "legacy_law_sha256": sha(legacy_law_path), "policy": GAUSSIAN_PAIRWISE_MI_POLICY,
                   "k": 4, "errors": []}
            started = perf_counter()
            historical = LegacySampler(ThresholdModel(V, max_exact_modes=3), 4, selection="pairwise")
            old_scores = historical.pairwise_mi.copy()
            canonical = SchurMemorySampler(ThresholdModel(V, max_exact_modes=3), 4, selection="pairwise")
            if canonical.cache_info()["pairwise_mi_policy"] != GAUSSIAN_PAIRWISE_MI_POLICY:
                raise RuntimeError("canonical sampler did not select the new Gaussian score policy")
            canonical_q = canonical.full_distribution()
            new_scores = canonical.pairwise_mi.copy()
            # Isolate the score-policy change from the unchanged probability
            # kernel.  The historical score table must recover its old Q.
            legacy_replay = SchurMemorySampler(ThresholdModel(V, max_exact_modes=3), 4, selection="pairwise")
            frozen_scores = old_scores.copy(); frozen_scores.setflags(write=False)
            legacy_replay.pairwise_mi = frozen_scores
            replay_q = legacy_replay.full_distribution()
            replay_tv = float(.5*np.abs(replay_q-legacy_q).sum())
            if replay_tv > FULL_Q_TV_GATE:
                row["errors"].append("historical score table did not reproduce archived historical Q")
            modes = len(V)//2
            zero_pairs = []
            structural_only = old_scores.copy()
            for first in range(modes):
                for second in range(first+1, modes):
                    if np.all(V[np.ix_([first, first+modes], [second, second+modes])] == 0):
                        zero_pairs.append([first, second])
                        structural_only[first, second] = structural_only[second, first] = 0
            row.update(
                legacy_replay_tv=replay_tv,
                legacy_target_tv=float(.5*np.abs(legacy_q-target).sum()),
                canonical_target_tv=float(.5*np.abs(canonical_q-target).sum()),
                canonical_vs_legacy_tv=float(.5*np.abs(canonical_q-legacy_q).sum()),
                structural_zero_pairs=zero_pairs,
                changed_score_entries=int(np.count_nonzero(new_scores != old_scores)),
                canonical_diagnostics=canonical.cache_info(),
                python_reference_seconds=perf_counter()-started,
            )
            arrays = {"distribution": canonical_q, "legacy_distribution": legacy_q,
                      "legacy_replay_distribution": replay_q, "target": target,
                      "pairwise_mi": new_scores, "legacy_pairwise_mi": old_scores,
                      "structural_zero_only_pairwise_mi": structural_only}
            try:
                native = run_cpp_click(V, selection="pairwise", max_clicks=4, samples=0,
                    enumerate_distribution=True, executable=executable, raw_samples=False)
                if native["executable_sha256"] != binary_sha:
                    raise RuntimeError("binary changed during canonical reference validation")
                native_q = np.asarray(native["distribution"])
                arrays["cpp_distribution"] = native_q
                difference = np.abs(native_q-canonical_q)
                row.update(native_status="ok", native_vs_canonical_tv=float(.5*difference.sum()),
                           native_vs_canonical_max_abs=float(difference.max()),
                           native_target_tv=float(.5*np.abs(native_q-target).sum()),
                           native_setup_diagnostics=native.get("setup_diagnostics"),
                           native_enumeration_diagnostics=native.get("enumeration_diagnostics"))
                if row["native_vs_canonical_tv"] > FULL_Q_TV_GATE:
                    row["errors"].append("native full Q failed the unchanged 2e-8 canonical equality gate")
            except CppClickError as exc:
                row.update(native_status="failed", native_error=str(exc), native_error_payload=exc.payload)
                row["errors"].append("native reference evaluation failed")
            destination = output / f"{identifier}.npz"
            np.savez_compressed(destination, **arrays)
            row.update(path=destination.name, sha256=sha(destination),
                       status="passed" if not row["errors"] else "failed")
            rows.append(row)
            journal.write(json.dumps(row, allow_nan=False)+"\n"); journal.flush()
            print(f"{identifier} {row['status']} new-old-TV={row['canonical_vs_legacy_tv']:.7g} "
                  f"native-TV={row.get('native_vs_canonical_tv',float('nan')):.3g}", flush=True)
    if sha(executable) != binary_sha:
        raise RuntimeError("native binary changed during reference generation")
    for name, expected in source_hashes.items():
        if sha(ROOT / "src" / "gbslab" / name) != expected:
            raise RuntimeError(f"Python source changed during reference generation: {name}")
    failures = sum(row["status"] != "passed" for row in rows)
    metadata.update(status="complete" if not failures else "complete_with_failures",
                    passed=failures == 0, completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=perf_counter()-start_all,
                    completed_cases=len(rows), failed_cases=failures,
                    maximum_native_vs_canonical_tv=max(row.get("native_vs_canonical_tv", 0) for row in rows))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    manifest = {**metadata, "cases": rows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    columns = ["case_id", "status", "legacy_target_tv", "canonical_target_tv", "native_target_tv",
               "canonical_vs_legacy_tv", "legacy_replay_tv", "native_vs_canonical_tv",
               "native_vs_canonical_max_abs", "changed_score_entries", "python_reference_seconds", "path", "sha256"]
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({key: metadata[key] for key in
        ("status", "passed", "completed_cases", "failed_cases", "maximum_native_vs_canonical_tv",
         "executable_sha256", "elapsed_seconds")}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cpp_matched_benchmark.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/cpp_pairwise_reference")
    parser.add_argument("--cpp-executable", type=Path, default=ROOT / ".cache/click-native/click_sampler")
    parser.add_argument("--legacy-validation", type=Path, default=ROOT / "results/cpp_validation/native_cached.json")
    run(parser.parse_args())
