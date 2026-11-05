"""Run the project correctness suite after the conditional-forest reference was added.

No compiler, sampler retry, benchmark, or statistical tuning is performed.
Redirect the ordinary adapter default before importing any integration tests,
so passing legacy tests cannot accidentally refer to an older default build.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import unittest

from gbslab import cpp_click

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / ".cache/click-prefix-native/click_sampler")
    parser.add_argument("--output", type=Path, default=ROOT / "results/iteration7_verification")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh verification directory")
    binary = args.executable.resolve()
    if cpp_click.find_cpp_executable(binary) is None:
        raise FileNotFoundError(binary)
    cpp_click.DEFAULT_EXECUTABLE = binary
    sources = sorted((ROOT / "src/gbslab").glob("*.py")) + sorted((ROOT / "tests").glob("test_*.py"))
    sources += [ROOT / "experiments/click_sampler.cpp", ROOT / "experiments/iteration5_quality.py",
                ROOT / "experiments/iteration5_benchmark.py", ROOT / "experiments/iteration6_completion.py",
                ROOT / "experiments/analyze_iteration6.py", ROOT / "experiments/iteration6_numerical_diagnostic.py",
                ROOT / "experiments/iteration7_forest.py", ROOT / "experiments/analyze_iteration7.py",
                ROOT / "experiments/iteration7_numerical_diagnostic.py", Path(__file__)]
    before = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    binary_sha = digest(binary)
    args.output.mkdir(parents=True)
    started = perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with (args.output / "unit_tests.log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": result.wasSuccessful() and before == after and digest(binary) == binary_sha,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "skipped_tests": [(str(t), why) for t, why in result.skipped],
        "executable_sha256": binary_sha, "adapter_default_executable": str(binary),
        "source_hashes": before, "sources_stable_during_tests": before == after,
        "elapsed_seconds": perf_counter()-started,
        "tests_never_compile": True, "integration_tests_use_current_candidate_binary": True,
        "scope": "Meaningful existing analytic, numerical, support, evidence and adapter regressions plus completion/forest/support/failure-mass and protected-enumeration checks; this is not a full-distribution certificate at large m"}
    (args.output / "validation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(binary.parent / "build.json", args.output / "candidate_build.json")
    print(json.dumps({k: metadata[k] for k in ("passed", "tests_run", "failures", "errors", "skipped", "elapsed_seconds")}), flush=True)
    if not metadata["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
