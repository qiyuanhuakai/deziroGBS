#!/usr/bin/env python3
"""Build the native retained-click sampler, with reproducible provenance.

Portable CPU code is the default. --native adds -march=native; use separate
--build-dir paths to preserve portable/native binaries for comparisons.
No fast-math or fused floating-point contraction is enabled.
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EIGEN = Path(
    "/opt/codex/runtimes/codex-primary-runtime/dependencies/python/"
    "lib/python3.12/site-packages/casadi/include/eigen3"
)


def tree_fingerprint(root: Path) -> dict:
    digest = hashlib.sha256()
    count = total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        content = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
        count += 1
        total += len(content)
    return {"sha256": digest.hexdigest(), "files": count, "bytes": total}


def build(*, build_dir: Path, eigen: Path, compiler: str = "c++",
          native: bool = False, force: bool = False) -> Path:
    eigen = eigen.resolve()
    if not (eigen / "Eigen/Core").is_file():
        raise FileNotFoundError(f"Eigen headers not found at {eigen}")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise FileNotFoundError(f"C++ compiler not found: {compiler}")
    compiler_version = subprocess.check_output([compiler_path, "--version"], text=True)
    compiler_sha256 = hashlib.sha256(Path(compiler_path).read_bytes()).hexdigest()
    source = ROOT / "experiments/click_sampler.cpp"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    builder_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dependencies = {"eigen": tree_fingerprint(eigen)}
    build_dir = build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "click_sampler"
    staged = build_dir / "click_sampler.building"
    flags = ["-O3", "-std=c++20", "-ffp-contract=off", "-DEIGEN_DONT_PARALLELIZE"]
    if native:
        flags.append("-march=native")
    command = [compiler_path, *flags, "-I", str(eigen), str(source), "-o", str(staged)]
    inputs = {"command": command, "compiler_version": compiler_version,
              "compiler_sha256": compiler_sha256, "source_sha256": source_sha256,
              "builder_sha256": builder_sha256, "dependencies": dependencies}
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
    staged.replace(executable)
    metadata = {
        **inputs, "cache_key": cache_key, "flags": flags, "native": native,
        "compiler": compiler_version.splitlines()[0],
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "build_seconds": time.perf_counter() - started,
        "implementation": "native C++ retained-click sampler with Gaussian zero updates",
        "threads": 1, "precision": "double; long-double inclusion-exclusion sums",
        "maximum_modes": 256, "maximum_click_budget": 8,
        "maximum_enumeration_modes": 14,
        "vacuum_kernel_options": {
            "default": "log", "available": ["log", "product"],
            "log": "exp(-sum(log(Cholesky diagonals)))",
            "product": "reciprocal diagonal product; logarithmic fallback on a nonnormal intermediate product",
            "sampling_guards": "unchanged; no fast-math and no score-floor policy change",
        },
        "sample_log_probabilities": {
            "flag": "--record-log-probabilities", "default": False,
            "array_key": "sample_log_probabilities",
            "meaning": "natural log of actual computed conditional chain under ideal continuous uniforms",
            "accumulation": "long double log(p_one) or log1p(-p_one), double output",
            "timing": "allocation and evaluation included in sampling",
        },
        "pairwise_score_policy": {
            "version": 3,
            "structural_zero": "exact zero original 2x2 covariance crossblock gives MI=0",
            "precision": "explicit long-double scalar 2x2 factors and whitening",
            "delta": "qa*qb*expm1(-0.5*log1p(-(sum(T^2)-det(T)^2)))",
            "mutual_information": "sum product_cell*h(signed_delta/product_cell)",
            "h": "(1+x)*log1p(x)-x; h(-1)=1",
            "h_series_abs_x_max": 0.001, "h_series_powers": [2, 3, 4, 5, 6, 7, 8],
            "probability_tolerance": 2e-10, "maximum_l1_correction": 2e-9,
            "score_floor": None, "tie_break": "physical mode index",
        },
    }
    temp = build_dir / "build.json.tmp"
    temp.write_text(json.dumps(metadata, indent=2) + "\n")
    temp.replace(metadata_path)
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".cache/click-build")
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
