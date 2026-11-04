"""Create a fresh binary-bound prefix-equivalence gate; never compile here.

The old reference must be built independently from research commit 37518d5.
Do not compile the new source into the old-reference path to pass this gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
import unittest

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / ".cache/click-prefix-native/click_sampler")
    parser.add_argument("--archived-executable", type=Path, default=ROOT / ".cache/click-iteration4-native/click_sampler")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh gate directory")
    binary, archived = args.executable.resolve(), args.archived_executable.resolve()
    for path in (binary, archived):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(path)
    if digest(binary) == digest(archived):
        raise ValueError("The old reference and candidate must be different independent builds")
    build = json.loads((binary.parent / "build.json").read_text())
    binary_sha = digest(binary)
    sources = [ROOT / p for p in ("experiments/click_sampler.cpp", "src/gbslab/cpp_click.py",
        "experiments/build_click_sampler.py", "tests/test_prefix_cache.py")]
    before = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    if build.get("executable_sha256") != binary_sha or build.get("source_sha256") != before["experiments/click_sampler.cpp"]:
        raise ValueError("Candidate build/source binding mismatch")
    # Decorators inspect this environment during module import.
    os.environ["GBSLAB_PREFIX_EXECUTABLE"] = str(binary)
    spec = importlib.util.spec_from_file_location("gbslab_prefix_gate_tests", ROOT / "tests/test_prefix_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARCHIVED = archived
    args.output.mkdir(parents=True)
    started = perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    with (args.output / "unit_tests.log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    passed = (result.wasSuccessful() and not result.skipped and before == after
              and digest(binary) == binary_sha and bool(module.VALIDATION_CHECKS))
    record = {"schema_version": 1, "kind": "native_prefix_cache_equivalence", "passed": passed,
        "created_utc": datetime.now(timezone.utc).isoformat(), "executable_sha256": binary_sha,
        "archived_executable_sha256": digest(archived), "source_sha256": before["experiments/click_sampler.cpp"],
        "source_hashes": before, "source_hashes_stable_during_tests": before == after,
        "build_metadata_matches_sources_and_binary": True, "tests_run": result.testsRun,
        "tests_passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "checks": module.VALIDATION_CHECKS, "elapsed_seconds": perf_counter()-started,
        "qualification": "Numerical equivalence on analytic, exhaustive small-prefix and large-bit fixtures; not a global GBS quality certificate"}
    (args.output / "validation.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    shutil.copy2(binary.parent / "build.json", args.output / "build.json")
    for source in sources:
        target = args.output / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(json.dumps({key: record[key] for key in ("passed", "tests_run", "failures", "errors", "skipped", "executable_sha256")}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
