#!/usr/bin/env python3
"""Build the pinned official FrugalGBS CPU sampler without editing upstream.

The optional full-distribution trace is generated from the pinned sampler
body. Only its random branch selection is replaced; the official sampler is
also compiled unchanged and is the implementation used for timing/samples.
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
UPSTREAM_COMMIT = "fa8b796f2afebd4845a2e335e457735952617ef8"
SAMPLER_BODY_SHA256 = "9c14a870f2875e808025d9e3535a71ba86144e5639ab14f0b7be5b594dabb3fb"
DEFAULT_EIGEN = Path(
    "/opt/codex/runtimes/codex-primary-runtime/dependencies/python/"
    "lib/python3.12/site-packages/casadi/include/eigen3"
)


def tree_fingerprint(root: Path) -> dict:
    """Hash names and bytes, including Eigen's extensionless public headers."""
    digest = hashlib.sha256()
    count = size = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        content = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(content)
        count += 1
        size += len(content)
    return {"sha256": digest.hexdigest(), "files": count, "bytes": size}


def make_trace_header(upstream: Path) -> str:
    """Keep the upstream SIMD probability arithmetic byte-for-byte intact.

    A hash gate makes an upstream change require an explicit instrumentation
    review. The generated file includes the upstream MIT notice in full.
    """
    source = (upstream / "include/fgbs/cpu/cpu_sampler.hpp").read_text()
    start = source.index(
        "template<usize N, typename T>\nSampleResult<N, T> generate_sample("
    )
    end = source.index(
        "\ntemplate<usize N>\nResult<std::vector<Bitstring<N>>> generate_samples(",
        start,
    )
    function = source[start:end]
    digest = hashlib.sha256(function.encode()).hexdigest()
    if digest != SAMPLER_BODY_SHA256:
        raise RuntimeError(
            "Pinned generate_sample body changed; review the trace before "
            f"building (expected {SAMPLER_BODY_SHA256}, got {digest})"
        )
    body = function[function.index(" {\n") + 3 :]
    initial = "    SampleResult<N, T> result{};"
    branch = (
        "        update_result<N>(rand, result, prev_prob, parity_arr, i, "
        "zero_next_prob);"
    )
    if body.count(initial) != 1 or body.count(branch) != 1:
        raise RuntimeError("Unexpected sampler instrumentation anchors")
    body = body.replace(initial, "    using T = ProbSimd;\n    TracePathResult trace{};")
    body = body.replace(branch, r"""        // Instrumentation: RNG > zero_next/prev is the upstream rule.
        // Every SIMD lane follows the same forced prefix, so lane zero gives
        // the same arithmetic as an ordinary official CPU sample.
        const auto previous = prev_prob.get(0);
        const auto proposed_zero = zero_next_prob.get(0);
        if (!(previous > 0.0) || !std::isfinite(previous)) {
            throw std::runtime_error("Nonpositive/nonfinite reachable prefix mass");
        }
        const auto threshold = proposed_zero / previous;
        if (!std::isfinite(threshold)) {
            ++trace.nonfinite_threshold_path_steps;
        }
        if (threshold < 0.0 || threshold > 1.0) {
            ++trace.saturated_threshold_path_steps;
        }
        // A comparison with NaN is false, matching the official zero branch.
        // Nonfinite values are nevertheless reported rather than hidden.
        const auto p_zero = std::isnan(threshold) ? 1.0
            : std::max(0.0, std::min(1.0, threshold));
        const bool one = ((forced_bits >> i) & std::uint64_t{1}) != 0;
        trace.probability *= one ? (1.0L - p_zero) : p_zero;
        if (trace.probability == 0.0L) {
            return trace;  // An unreachable branch needs no further arithmetic.
        }
        parity_arr[i] = one ? -get_half<T>() : get_half<T>();
        prev_prob = one ? prev_prob - zero_next_prob : zero_next_prob;""")
    if body.count("    return result;") != 1:
        raise RuntimeError("Unexpected sampler return anchor")
    body = body.replace("    return result;", "    return trace;")
    notice = (upstream / "LICENSE.md").read_text()
    return (
        "// Generated instrumentation, NOT an official FrugalGBS API.\n"
        f"// Upstream commit: {UPSTREAM_COMMIT}\n"
        f"// Source function SHA256: {digest}\n"
        "/*\n" + notice + "\n*/\n"
        "#ifndef GBSLAB_FRUGAL_TRACE_HPP\n#define GBSLAB_FRUGAL_TRACE_HPP\n"
        '#include "fgbs/cpu/cpu_sampler.hpp"\n'
        "#include <cstdint>\n#include <stdexcept>\n"
        "#if defined(_MSC_VER)\n#define GBSLAB_RESTRICT __restrict\n"
        "#elif defined(__clang__) || defined(__GNUC__)\n#define GBSLAB_RESTRICT __restrict__\n"
        "#else\n#define GBSLAB_RESTRICT\n#endif\n"
        "namespace fgbs::cpu {\n"
        "struct TracePathResult {\n"
        "    long double probability = 1.0L;\n"
        "    std::uint64_t saturated_threshold_path_steps = 0;\n"
        "    std::uint64_t nonfinite_threshold_path_steps = 0;\n"
        "};\n"
        "template<usize N>\n"
        "TracePathResult trace_sample_probability(\n"
        "    BitstringLayout<N> const& bitstring_layout,\n"
        "    SamplingTarget<N> const& sampling_target,\n"
        "    std::uint64_t forced_bits,\n"
        "    ProbSimd* GBSLAB_RESTRICT partial_marginal_arr,\n"
        "    ProbSimd* GBSLAB_RESTRICT elided_marginal_arr,\n"
        "    ProbSimd* GBSLAB_RESTRICT partial_elided_marginal_arr,\n"
        "    ProbSimd* GBSLAB_RESTRICT double_elided_marginal_arr) {\n"
        + body + "\n} // namespace fgbs::cpu\n#undef GBSLAB_RESTRICT\n#endif\n"
    )


def build(*, build_dir: Path, eigen: Path, xsimd: Path, compiler: str,
          modes: tuple[int, ...], force: bool = False) -> Path:
    upstream = ROOT / "third_party/FrugalGBS"
    if not (eigen / "Eigen/Core").is_file():
        raise FileNotFoundError(f"Eigen headers not found at {eigen}")
    if not (xsimd / "xsimd/xsimd.hpp").is_file():
        raise FileNotFoundError(f"xsimd headers not found at {xsimd}")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise FileNotFoundError(f"C++ compiler not found: {compiler}")
    if not modes or any(n < 2 or n > 16 for n in modes):
        raise ValueError("Compile modes must be integers from 2 through 16")
    build_dir = build_dir.resolve()
    generated = build_dir / "generated"
    (generated / "fgbs").mkdir(parents=True, exist_ok=True)
    global_header = (upstream / "include/fgbs/global.hpp.in").read_text()
    global_header = global_header.replace("@FGBS_REAL_PRECISION@", "2")
    global_header = global_header.replace("@FGBS_MAX_CUTOFF_ORDER@", "5")
    (generated / "fgbs/global.hpp").write_text(global_header)
    trace = make_trace_header(upstream)
    (generated / "frugal_trace.hpp").write_text(trace)
    (generated / "frugal_modes.hpp").write_text(
        "#define GBSLAB_FRUGAL_MODES(M) " + " ".join(f"M({n})" for n in modes) + "\n"
    )
    source = ROOT / "experiments/frugal_adapter.cpp"
    sources = [source, *sorted((upstream / "src/fgbs/core").glob("*.cpp")),
               upstream / "src/fgbs/layout.cpp",
               upstream / "src/fgbs/cpu/cpu_sampler.cpp"]
    executable = build_dir / "frugal_adapter"
    staged_executable = build_dir / "frugal_adapter.building"
    command = [compiler_path, "-O3", "-std=c++20", "-fopenmp",
               "-DFGBS_GPU_ENABLED=0", "-I", str(generated),
               "-I", str(upstream / "include"), "-I", str(eigen),
               "-I", str(xsimd), *map(str, sources), "-o", str(staged_executable)]
    compiler_version = subprocess.check_output([compiler_path, "--version"], text=True)
    compiler_sha256 = hashlib.sha256(Path(compiler_path).read_bytes()).hexdigest()
    dependencies = {"eigen": tree_fingerprint(eigen), "xsimd": tree_fingerprint(xsimd)}
    fingerprint = hashlib.sha256()
    fingerprint.update(json.dumps(command).encode())
    fingerprint.update(compiler_version.encode())
    fingerprint.update(compiler_sha256.encode())
    fingerprint.update(json.dumps(dependencies, sort_keys=True).encode())
    fingerprint.update(trace.encode())
    fingerprint.update(global_header.encode())
    fingerprint.update(repr(modes).encode())
    for path in [*sources, *sorted((upstream / "include").rglob("*hpp"))]:
        fingerprint.update(path.read_bytes())
    cache_key = fingerprint.hexdigest()
    metadata_path = build_dir / "build.json"
    if not force and executable.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
        if (metadata.get("cache_key") == cache_key
                and metadata.get("executable_sha256") == executable_sha256):
            return executable
    started = time.perf_counter()
    subprocess.run(command, check=True)
    staged_executable.replace(executable)
    metadata = {
        "upstream_commit": UPSTREAM_COMMIT, "sampler_body_sha256": SAMPLER_BODY_SHA256,
        "cache_key": cache_key, "command": command, "modes": modes,
        "precision": "float64", "cutoff_max": 5, "gpu": False,
        "build_seconds": time.perf_counter() - started,
        "compiler": compiler_version.splitlines()[0],
        "compiler_sha256": compiler_sha256, "dependencies": dependencies,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "trace": "generated SIMD forced-branch instrumentation; not an official API",
    }
    staged_metadata = build_dir / "build.json.tmp"
    staged_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    staged_metadata.replace(metadata_path)
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".cache/frugal-build")
    parser.add_argument("--eigen", type=Path, default=Path(os.environ.get("EIGEN3_INCLUDE_DIR", DEFAULT_EIGEN)))
    parser.add_argument("--xsimd", type=Path, default=ROOT / "third_party/xsimd/include")
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--modes", default="2,4,8,10,12")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    modes = tuple(sorted(set(int(n) for n in args.modes.split(","))))
    print(build(build_dir=args.build_dir, eigen=args.eigen, xsimd=args.xsimd,
                compiler=args.compiler, modes=modes, force=args.force))


if __name__ == "__main__":
    main()
