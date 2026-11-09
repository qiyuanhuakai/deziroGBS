#!/usr/bin/env python3
"""Build the shared Schur forest event library with explicit provenance.

No import or test compiles a library implicitly.  Portable CPU code, one
thread, no fast-math and no floating-point contraction are the defaults.
"""
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
DEFAULT_BUILD_DIR = ROOT / ".cache/batched-forest-native"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, build_dir=DEFAULT_BUILD_DIR, compiler="g++", native=False, force=False):
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise FileNotFoundError(f"C++ compiler not found: {compiler}")
    compiler_version = subprocess.check_output([compiler_path, "--version"], text=True)
    source = ROOT / "experiments/batched_forest_kernel.cpp"
    header = ROOT / "experiments/batched_forest_kernel.h"
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (source, header, Path(__file__))}
    quad_library = Path(subprocess.check_output(
        [compiler_path, "-print-file-name=libquadmath.so"], text=True).strip()).resolve()
    include_directory = Path(subprocess.check_output(
        [compiler_path, "-print-file-name=include"], text=True).strip())
    quad_header = include_directory / "quadmath.h"
    if not quad_library.is_file() or not quad_header.is_file():
        raise FileNotFoundError("The selected compiler must supply libquadmath and quadmath.h")
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / "libbatched_forest.so"
    staged = build_dir / "libbatched_forest.building.so"
    flags = ["-O3", "-std=c++20", "-shared", "-fPIC", "-ffp-contract=off",
             "-fno-fast-math", "-Wall", "-Wextra", "-Wpedantic"]
    if native:
        flags.append("-march=native")
    command = [compiler_path, *flags, str(source), "-o", str(staged), "-lquadmath"]
    inputs = {"command": command, "compiler_version": compiler_version,
              "compiler_sha256": sha(compiler_path), "source_hashes": source_hashes,
              "quadmath_library": str(quad_library), "quadmath_library_sha256": sha(quad_library),
              "quadmath_header": str(quad_header), "quadmath_header_sha256": sha(quad_header)}
    cache_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    metadata_path = build_dir / "build.json"
    if not force and library.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("cache_key") == cache_key and metadata.get("library_sha256") == sha(library):
            return library
    started = perf_counter()
    process = subprocess.run(command, text=True, capture_output=True)
    (build_dir / "compiler.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"Native forest build failed; see {build_dir / 'compiler.log'}\n{process.stderr}")
    if not all(sha(ROOT / name) == digest for name, digest in source_hashes.items()):
        raise RuntimeError("Build sources changed during compilation")
    staged.replace(library)
    metadata = {**inputs, "cache_key": cache_key, "library_sha256": sha(library),
                "flags": flags, "native": native, "threads": 1,
                "built_utc": datetime.now(timezone.utc).isoformat(),
                "build_seconds": perf_counter() - started,
                "compiler_log_sha256": sha(build_dir / "compiler.log"),
                "abi_version": 1, "precision_formats": ["C long double", "GNU __float128"],
                "input_precision": "original binary64 covariance, exactly widened before Q construction",
                "indicator_policy": "per-scalar absolute forward indicator, mu=8*format epsilon; not certified intervals",
                "auto_policy": "one long-double batch; on numerical failure at most one independent quad batch, fixed R/E and guard",
                "state_policy": "one canonical zero-sequence state per precision; append canonical prefixes, otherwise replay original input",
                "maximum_modes": 512, "maximum_retained": 16,
                "maximum_local_determinant_dimension": 6,
                "no_implicit_compilation": True}
    temporary = build_dir / "build.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build(build_dir=args.build_dir, compiler=args.compiler,
                native=args.native, force=args.force))


if __name__ == "__main__":
    main()
