"""Bind the full correctness suite to explicit already-built native programs.

No compiler is called. All earlier integration dependencies must be available;
skipped, expected-failure, or unexpected-success tests fail this gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import unittest

from gbslab import cpp_click
from iteration10_common import ROOT, save, sha, source_hashes, utc, verify_frozen_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core", type=Path, default=ROOT / ".cache/iteration10-core-native/coherent_core_sampler")
    parser.add_argument("--retained", type=Path, default=ROOT / ".cache/iteration8-retained-native/click_sampler")
    parser.add_argument("--library", type=Path, default=ROOT / ".cache/iteration8-batched-native/libbatched_forest.so")
    parser.add_argument("--forest", type=Path, default=ROOT / ".cache/iteration8-forest-native/batched_forest_sampler")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new verification directory")
    verify_frozen_config(ROOT / "configs/iteration10_coherent_core.json")
    paths = {name: getattr(args, name).resolve() for name in ("core", "retained", "library", "forest")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    builds = {}
    for name, path in paths.items():
        record = json.loads((path.parent / "build.json").read_text())
        if record.get("executable_sha256", record.get("library_sha256")) != sha(path):
            raise ValueError("Binary does not match its build record: " + name)
        for rel, expected in record.get("source_hashes", {}).items():
            if sha(ROOT / rel) != expected:
                raise ValueError("Build source differs: " + rel)
        for rel, expected in record.get("builder_hashes", {}).items():
            if sha(ROOT / rel) != expected:
                raise ValueError("Build script differs: " + rel)
        if name in ("core", "retained"):
            source = "experiments/coherent_core_sampler.cpp" if name == "core" else "experiments/click_sampler.cpp"
            if "source_sha256" in record and record["source_sha256"] != sha(ROOT / source):
                raise ValueError("Build source mismatch: " + name)
        for dependency in ("quadmath_library", "quadmath_header"):
            if dependency in record and sha(record[dependency]) != record[dependency + "_sha256"]:
                raise ValueError("Build dependency changed: " + dependency)
        builds[name] = record
    cpp_click.DEFAULT_EXECUTABLE = paths["retained"]
    os.environ["GBSLAB_BATCHED_FOREST_LIBRARY"] = str(paths["library"])
    os.environ["GBSLAB_FOREST_EXECUTABLE"] = str(paths["forest"])
    os.environ["GBSLAB_COHERENT_CORE_EXECUTABLE"] = str(paths["core"])
    sources_before = source_hashes()
    binaries_before = {name: sha(path) for name, path in paths.items()}
    legacy_paths = [ROOT / ".cache" / rel for rel in (
        "click-prefix-native/click_sampler", "click-iteration4-native/click_sampler",
        "frugal-native/frugal_sampling_adapter", "frugal-evidence-native/frugal_evidence_adapter",
        "frugal-evidence-checkpoint-native/frugal_evidence_adapter")]
    for path in legacy_paths:
        if not path.is_file():
            raise FileNotFoundError("Required historical integration dependency: " + str(path))
    legacy_before = {str(path.relative_to(ROOT)): sha(path) for path in legacy_paths}
    args.output.mkdir(parents=True)
    started = perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with (args.output / "unit_tests.log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    stable = sources_before == source_hashes() and binaries_before == {n: sha(p) for n, p in paths.items()}
    stable = stable and legacy_before == {str(p.relative_to(ROOT)): sha(p) for p in legacy_paths}
    passed = result.wasSuccessful() and stable and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses
    checks = {}
    for module_name in ("test_coherent_core", "test_coherent_core_independent"):
        module = sys.modules.get(module_name)
        checks[module_name] = getattr(module, "VALIDATION_CHECKS", [])
    record = {"created_utc": utc(), "passed": passed, "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses),
        "source_hashes": sources_before, "binary_hashes": binaries_before,
        "historical_binary_hashes": legacy_before,
        "sources_and_binaries_stable_during_tests": stable,
        "actual_candidate_paths": {n: str(p) for n, p in paths.items()},
        "explicit_native_environment": True, "tests_never_compile": True,
        "coherent_validation_checks": checks, "elapsed_seconds": perf_counter() - started,
        "scope": "Full project regression plus coherent finite-Fock states, local POVM refinement, correlated displacement references, and explicit numerical/resource guards. Not a rigorous floating-point or large-system TV certificate."}
    save(args.output / "validation.json", record)
    for name, path in paths.items():
        shutil.copy2(path.parent / "build.json", args.output / (name + "_build.json"))
    print(json.dumps({k: record[k] for k in ("passed", "tests_run", "failures", "errors", "skipped", "elapsed_seconds")}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
