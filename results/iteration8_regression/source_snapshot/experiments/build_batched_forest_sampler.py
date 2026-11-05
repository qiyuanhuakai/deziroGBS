#!/usr/bin/env python3
"""Build the complete native forest sampler and archive its source closure.

The unchanged retained-click source supplies the selector and validation.
The shared forest event kernel is compiled as a separate translation unit.
Portable code is the default; --native matches the retained baseline build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from build_click_sampler import DEFAULT_EIGEN, tree_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def build(*, build_dir: Path, eigen: Path, compiler: str = "c++",
          native: bool = False, force: bool = False) -> Path:
    eigen = eigen.resolve()
    if not (eigen / "Eigen/Core").is_file():
        raise FileNotFoundError(f"Eigen headers not found at {eigen}")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise FileNotFoundError(f"C++ compiler not found: {compiler}")
    compiler_version = subprocess.check_output([compiler_path, "--version"], text=True)
    quad_library = Path(subprocess.check_output(
        [compiler_path, "-print-file-name=libquadmath.so"], text=True).strip()).resolve()
    include_directory = Path(subprocess.check_output(
        [compiler_path, "-print-file-name=include"], text=True).strip())
    quad_header = include_directory / "quadmath.h"
    if not quad_library.is_file() or not quad_header.is_file():
        raise FileNotFoundError("Selected compiler must provide libquadmath and quadmath.h")
    source_paths = ["experiments/batched_forest_sampler.cpp",
                    "experiments/batched_forest_kernel.cpp",
                    "experiments/batched_forest_kernel.h",
                    "experiments/click_sampler.cpp"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in source_paths}
    builder_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in ("experiments/build_batched_forest_sampler.py",
                                   "experiments/build_click_sampler.py")}
    build_dir = build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "batched_forest_sampler"
    staged = build_dir / "batched_forest_sampler.building"
    flags = ["-O3", "-std=c++20", "-ffp-contract=off", "-DEIGEN_DONT_PARALLELIZE"]
    if native:
        flags.append("-march=native")
    command = [compiler_path, *flags, "-I", str(eigen),
               str(ROOT / source_paths[0]), str(ROOT / source_paths[1]),
               "-lquadmath", "-o", str(staged)]
    inputs = {"command": command, "compiler_version": compiler_version,
              "compiler_sha256": hashlib.sha256(Path(compiler_path).read_bytes()).hexdigest(),
              "source_hashes": source_hashes,
              "source_sha256": source_hashes[source_paths[0]],
              "builder_hashes": builder_hashes,
              "builder_sha256": builder_hashes["experiments/build_batched_forest_sampler.py"],
              "dependencies": {"eigen": tree_fingerprint(eigen)},
              "quadmath_library": str(quad_library),
              "quadmath_library_sha256": hashlib.sha256(quad_library.read_bytes()).hexdigest(),
              "quadmath_header": str(quad_header),
              "quadmath_header_sha256": hashlib.sha256(quad_header.read_bytes()).hexdigest(),
              "link_libraries": ["quadmath"]}
    cache_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    metadata_path = build_dir / "build.json"
    if not force and executable.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("cache_key") == cache_key
                and metadata.get("executable_sha256")
                == hashlib.sha256(executable.read_bytes()).hexdigest()):
            return executable
    started = time.perf_counter()
    subprocess.run(command, check=True)
    # A concurrently edited source must not be blessed by a stale hash record.
    for name, expected in {**source_hashes, **builder_hashes}.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            staged.unlink(missing_ok=True)
            raise RuntimeError(f"build source changed during compilation: {name}")
    for path, expected in ((quad_library, inputs["quadmath_library_sha256"]),
                           (quad_header, inputs["quadmath_header_sha256"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            staged.unlink(missing_ok=True)
            raise RuntimeError(f"quadmath dependency changed during compilation: {path}")
    staged.replace(executable)
    metadata = {
        **inputs, "cache_key": cache_key, "flags": flags, "native": native,
        "compiler": compiler_version.splitlines()[0],
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "build_seconds": time.perf_counter() - started,
        "implementation": "complete native batched forest sampler ABI 1",
        "threads": 1, "maximum_modes": 256, "maximum_click_budget": 8,
        "maximum_enumeration_modes": 14,
        "measurement_order": "natural only",
        "forest_options": ["chain", "static_mi_tree"],
        "precision_options": ["longdouble", "quad", "auto"],
        "precision_policy": "shared bf_probability; one whole-prefix LD-to-quad refinement only on numerical failure",
        "quad_input": "original validated/symmetrized binary64 covariance, never an LD Schur state",
        "prefix_cache_entries": 0,
        "selector_step_cache": "legacy within-prefix workspace; cleared at each selected prefix",
        "sample_log_probabilities": "actual returned binary64 conditional chain, long-double log/log1p accumulation",
        "failure_policy": "stop sample batch and retain completed raw/logQ; full-law enumeration retains failure mass without renormalization",
        "timing_scope": "setup plus sampling includes state, selector, tree, refinement, RNG, packed storage, requested logQ",
        "source_unchanged_during_build": True,
    }
    temp = build_dir / "build.json.tmp"
    temp.write_text(json.dumps(metadata, indent=2) + "\n")
    temp.replace(metadata_path)
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".cache/batched-forest-build")
    parser.add_argument("--eigen", type=Path,
                        default=Path(os.environ.get("EIGEN3_INCLUDE_DIR", DEFAULT_EIGEN)))
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build(build_dir=args.build_dir, eigen=args.eigen, compiler=args.compiler,
                native=args.native, force=args.force))


if __name__ == "__main__":
    main()
