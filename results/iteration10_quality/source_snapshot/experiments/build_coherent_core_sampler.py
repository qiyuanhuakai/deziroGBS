#!/usr/bin/env python3
"""Explicitly build the small coherent-core Fock prototype; tests never build it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / ".cache/coherent-core-native"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, build_dir=DEFAULT_BUILD_DIR, compiler="g++", native=False, force=False):
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise FileNotFoundError(f"C++ compiler not found: {compiler}")
    source = ROOT / "experiments/coherent_core_sampler.cpp"
    bound = {str(p.relative_to(ROOT)): sha(p) for p in (source, Path(__file__))}
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "coherent_core_sampler"
    staged = build_dir / "coherent_core_sampler.building"
    flags = ["-O3", "-std=c++20", "-ffp-contract=off", "-fno-fast-math",
             "-Wall", "-Wextra", "-Wpedantic"]
    if native:
        flags.append("-march=native")
    command = [compiler_path, *flags, str(source), "-o", str(staged)]
    inputs = {"command": command, "source_hashes": bound,
              "compiler_version": subprocess.check_output([compiler_path, "--version"], text=True),
              "compiler_sha256": sha(compiler_path), "flags": flags}
    cache_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    metadata_path = build_dir / "build.json"
    if not force and binary.is_file() and metadata_path.is_file():
        old = json.loads(metadata_path.read_text())
        if old.get("cache_key") == cache_key and old.get("executable_sha256") == sha(binary):
            return binary
    started = perf_counter()
    process = subprocess.run(command, text=True, capture_output=True)
    (build_dir / "compiler.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"Coherent-core build failed; see {build_dir / 'compiler.log'}\n{process.stderr}")
    if any(sha(ROOT / name) != digest for name, digest in bound.items()):
        raise RuntimeError("Coherent-core sources changed while compiling")
    staged.replace(binary)
    record = {**inputs, "cache_key": cache_key, "executable_sha256": sha(binary),
              "native": native, "threads": 1,
              "built_utc": datetime.now(timezone.utc).isoformat(),
              "build_seconds": perf_counter() - started,
              "compiler_log_sha256": sha(build_dir / "compiler.log"),
              "schema_version": 1, "precision": "complex<long double>; compensated real sums",
              "maximum_modes": 64, "maximum_cutoff": 8,
              "maximum_basis": 100000, "maximum_stage_basis": 1000000,
              "representation": "dense total-photon Fock vector; not MPS",
              "measurement": "complete rank-one refinement of displaced threshold POVM",
              "state": "coherent normalized total-N projection before correlated Gaussian displacement",
              "failure_policy": "stop, preserve completed samples, no refill or clipping",
              "no_implicit_compilation": True}
    temporary = build_dir / "build.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(metadata_path)
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build(build_dir=args.build_dir, compiler=args.compiler, native=args.native, force=args.force))


if __name__ == "__main__":
    main()
