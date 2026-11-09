"""Bind the complete correctness suite to all three already-built candidates.

No compiler is invoked by this gate. A missing binary, skipped integration
test, source change, or failed numerical fixture makes the gate fail.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import unittest

from gbslab import cpp_click

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/iteration8_validation")
    parser.add_argument("--library", type=Path, default=ROOT / ".cache/iteration8-batched-native/libbatched_forest.so")
    parser.add_argument("--retained", type=Path, default=ROOT / ".cache/iteration8-retained-native/click_sampler")
    parser.add_argument("--forest", type=Path, default=ROOT / ".cache/iteration8-forest-native/batched_forest_sampler")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new verification directory, including after a failed gate")
    binary_paths = {name: getattr(args, name).resolve() for name in ("library", "retained", "forest")}
    for path in binary_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    cpp_click.DEFAULT_EXECUTABLE = binary_paths["retained"]
    os.environ["GBSLAB_BATCHED_FOREST_LIBRARY"] = str(binary_paths["library"])
    os.environ["GBSLAB_FOREST_EXECUTABLE"] = str(binary_paths["forest"])
    source_paths = sorted((ROOT / "src/gbslab").glob("*.py")) + sorted((ROOT / "tests").glob("test_*.py"))
    source_paths += sorted((ROOT / "experiments").glob("*batched_forest*"))
    source_paths += sorted((ROOT / "experiments").glob("iteration8*.py"))
    source_paths += [Path(__file__), ROOT / "experiments/click_sampler.cpp", ROOT / "experiments/build_click_sampler.py",
                     ROOT / "experiments/iteration7_forest.py", ROOT / "experiments/iteration6_completion.py"]
    source_paths = sorted(set(p for p in source_paths if p.is_file()))
    before = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    binaries_before = {name: digest(path) for name, path in binary_paths.items()}
    builds = {}
    for name, path in binary_paths.items():
        metadata = json.loads((path.parent / "build.json").read_text())
        if metadata.get("executable_sha256", metadata.get("library_sha256")) != binaries_before[name]:
            raise ValueError("Candidate does not match its build record: " + name)
        for source, expected in metadata.get("source_hashes", {}).items():
            if digest(ROOT / source) != expected:
                raise ValueError("Build source mismatch: " + source)
        for source, expected in metadata.get("builder_hashes", {}).items():
            if digest(ROOT / source) != expected:
                raise ValueError("Build script mismatch: " + source)
        for dependency in ("quadmath_library", "quadmath_header"):
            if dependency in metadata and digest(metadata[dependency]) != metadata[dependency + "_sha256"]:
                raise ValueError("Build dependency mismatch: " + dependency)
        if name == "retained" and metadata["source_sha256"] != digest(ROOT / "experiments/click_sampler.cpp"):
            raise ValueError("Retained baseline source mismatch")
        builds[name] = metadata
    args.output.mkdir(parents=True)
    started = perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with (args.output / "unit_tests.log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    binaries_after = {name: digest(path) for name, path in binary_paths.items()}
    numerical_module = sys.modules.get("test_batched_forest_numerics")
    numerical_checks = getattr(numerical_module, "VALIDATION_CHECKS", [])
    passed = (result.wasSuccessful() and before == after and binaries_before == binaries_after
              and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses
              and bool(numerical_checks))
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "source_hashes": before, "binary_hashes": binaries_before,
        "sources_and_binaries_stable_during_tests": before == after and binaries_before == binaries_after,
        "actual_candidate_paths": {name: str(path) for name, path in binary_paths.items()},
        "explicit_native_environment": True, "tests_never_compile": True,
        "numerical_validation_checks": numerical_checks,
        "elapsed_seconds": perf_counter() - started,
        "scope": "Analytic event and thermal fixtures, real refinement/failure, state/order/cache invariance, independent table events, native injected uniforms/logQ/partial checkpoint, and all prior project regressions. This is not a large-mode TV or rigorous rounding certificate."}
    (args.output / "validation.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    for name, path in binary_paths.items():
        shutil.copy2(path.parent / "build.json", args.output / (name + "_build.json"))
    print(json.dumps({key: record[key] for key in ("passed", "tests_run", "failures", "errors", "skipped", "elapsed_seconds")}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
