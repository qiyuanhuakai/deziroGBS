#!/usr/bin/env python3
"""Build an isolated evidence adapter around pinned official FrugalGBS.

The public generate_samples API remains unchanged. A separately generated
forced-branch trace scores its outputs after sampling, including modes above
64. Neither an old executable nor upstream sources are overwritten.
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

from build_frugal import (
    DEFAULT_EIGEN, SAMPLER_BODY_SHA256, UPSTREAM_COMMIT,
    make_trace_header, tree_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = (8, 10, 12, 16, 32, 64, 100, 144)


def make_log_trace_header(upstream: Path) -> str:
    """Extend the existing hash-gated trace, retaining its arithmetic body.

    The old generator checks the pinned source body and carries its MIT notice.
    Only its branch instrumentation, output struct, and bitstring argument
    change here; the sampling arithmetic stays in upstream SIMD precision.
    """
    trace = make_trace_header(upstream)
    old_branch = """        // Instrumentation: RNG > zero_next/prev is the upstream rule.
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
        prev_prob = one ? prev_prob - zero_next_prob : zero_next_prob;"""
    new_branch = """        // All SIMD lanes follow the same forced physical-bit prefix.
        // This changes branch choice only; upstream marginal arithmetic above
        // and the state updates below retain their original SIMD precision.
        const auto previous = prev_prob.get(0);
        const auto proposed_zero = zero_next_prob.get(0);
        if (!(previous > 0.0) || !std::isfinite(previous)) {
            throw std::runtime_error("Nonpositive/nonfinite reachable prefix mass");
        }
        const auto threshold = proposed_zero / previous;
        const bool one = get_bit<N>(forced_bits, i);
        if (!record_forced_threshold(trace, threshold, one)) return trace;
        parity_arr[i] = one ? -get_half<T>() : get_half<T>();
        prev_prob = one ? prev_prob - zero_next_prob : zero_next_prob;"""
    if trace.count(old_branch) != 1:
        raise RuntimeError("Existing pinned branch trace changed; review instrumentation")
    trace = trace.replace(old_branch, new_branch)
    trace = trace.replace("GBSLAB_FRUGAL_TRACE_HPP", "GBSLAB_FRUGAL_LOG_TRACE_HPP")
    trace = trace.replace("TracePathResult", "TraceLogPathResult")
    trace = trace.replace("trace_sample_probability(", "trace_sample_log_probability(")
    trace = trace.replace("    std::uint64_t forced_bits,", "    Bitstring<N> const& forced_bits,")
    trace = trace.replace("#include <cstdint>", "#include <cstdint>\n#include <limits>")
    trace = trace.replace("    long double probability = 1.0L;", """    long double probability = 1.0L;
    long double log_probability = 0.0L;""")
    helper = r"""
// The official decision is ONE iff U > threshold. Its continuous-uniform
// zero-branch mass is the threshold clipped to [0,1], not proposed_zero itself.
// NaN compares false and therefore selects ZERO; it is still an invalid trace
// for evidence and the adapter reports it explicitly rather than hiding it.
inline bool record_forced_threshold(TraceLogPathResult& trace, double threshold, bool one) {
    if (!std::isfinite(threshold)) ++trace.nonfinite_threshold_path_steps;
    if (threshold < 0.0 || threshold > 1.0) ++trace.saturated_threshold_path_steps;
    const double p_zero = std::isnan(threshold) ? 1.0
        : std::max(0.0, std::min(1.0, threshold));
    const long double chosen = one ? 1.0L - p_zero : p_zero;
    trace.probability *= chosen;
    if (chosen == 0.0L) {
        trace.log_probability = -std::numeric_limits<long double>::infinity();
        return false;
    }
    trace.log_probability += one ? std::log1p(-static_cast<long double>(p_zero))
        : std::log(static_cast<long double>(p_zero));
    return true; // A product underflow does not terminate the log-density trace.
}
"""
    anchor = "};\ntemplate<usize N>\nTraceLogPathResult trace_sample_log_probability("
    if trace.count(anchor) != 1:
        raise RuntimeError("Unexpected log-trace output anchor")
    return trace.replace(anchor, "};\n" + helper + "\ntemplate<usize N>\nTraceLogPathResult trace_sample_log_probability(")


def build(build_dir, eigen, compiler, modes=DEFAULT_MODES, native=False, force=False):
    upstream = ROOT / "third_party/FrugalGBS"
    xsimd = ROOT / "third_party/xsimd/include"
    eigen = Path(eigen)
    modes = tuple(sorted(set(modes)))
    if not modes or any(isinstance(m, bool) or not isinstance(m, int) or m < 2 or m > 144 for m in modes):
        raise ValueError("compile modes must be integers in [2,144]")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise FileNotFoundError(compiler)
    if not (eigen / "Eigen/Core").is_file():
        raise FileNotFoundError(eigen)
    build_dir = Path(build_dir).resolve()
    protected = {ROOT / ".cache/frugal-native", ROOT / ".cache/frugal-build"}
    if build_dir in {p.resolve() for p in protected}:
        raise ValueError("Evidence build requires an isolated build directory")
    generated = build_dir / "generated"
    (generated / "fgbs").mkdir(parents=True, exist_ok=True)
    global_header = (upstream / "include/fgbs/global.hpp.in").read_text().replace(
        "@FGBS_REAL_PRECISION@", "2").replace("@FGBS_MAX_CUTOFF_ORDER@", "5")
    trace = make_log_trace_header(upstream)
    modes_header = "#define GBSLAB_FRUGAL_MODES(M) " + " ".join(f"M({n})" for n in modes) + "\n"
    for name, content in (("fgbs/global.hpp", global_header),
                          ("frugal_log_trace.hpp", trace), ("frugal_modes.hpp", modes_header)):
        (generated / name).write_text(content)
    sources = [ROOT / "experiments/frugal_evidence_adapter.cpp",
               *sorted((upstream / "src/fgbs/core").glob("*.cpp")),
               upstream / "src/fgbs/layout.cpp", upstream / "src/fgbs/cpu/cpu_sampler.cpp"]
    executable = build_dir / "frugal_evidence_adapter"
    staged = build_dir / "frugal_evidence_adapter.building"
    flags = ["-O3", "-std=c++20", "-ffp-contract=off", "-fopenmp", "-DFGBS_GPU_ENABLED=0"]
    if native:
        flags.append("-march=native")
    command = [compiler_path, *flags, "-I", str(generated), "-I", str(upstream / "include"),
               "-I", str(eigen), "-I", str(xsimd), *map(str, sources), "-o", str(staged)]
    compiler_version = subprocess.check_output([compiler_path, "--version"], text=True)
    compiler_sha256 = hashlib.sha256(Path(compiler_path).read_bytes()).hexdigest()
    dependencies = {"eigen": tree_fingerprint(eigen), "xsimd": tree_fingerprint(xsimd)}
    digest = hashlib.sha256(json.dumps({"command": command, "compiler": compiler_version,
        "compiler_sha256": compiler_sha256, "dependencies": dependencies}, sort_keys=True).encode())
    for content in (global_header, trace, modes_header):
        digest.update(content.encode())
    for path in [Path(__file__), ROOT / "experiments/build_frugal.py", *sources,
                 *sorted((upstream / "include").rglob("*.hpp"))]:
        digest.update(path.read_bytes())
    cache_key = digest.hexdigest()
    metadata_path = build_dir / "build.json"
    if not force and executable.is_file() and metadata_path.is_file():
        old = json.loads(metadata_path.read_text())
        if old.get("cache_key") == cache_key and old.get("executable_sha256") == hashlib.sha256(executable.read_bytes()).hexdigest():
            return executable
    start = time.perf_counter()
    with (build_dir / "compile.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Compile failed; see {build_dir / 'compile.log'}")
    staged.replace(executable)
    metadata = {
        "upstream_commit": UPSTREAM_COMMIT, "sampler_body_sha256": SAMPLER_BODY_SHA256,
        "native": native, "modes": modes, "flags": flags, "command": command,
        "compiler": compiler_version.splitlines()[0], "compiler_sha256": compiler_sha256,
        "dependencies": dependencies, "cache_key": cache_key,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "adapter_source_sha256": hashlib.sha256(sources[0].read_bytes()).hexdigest(),
        "generated_trace_sha256": hashlib.sha256(trace.encode()).hexdigest(),
        "build_seconds": time.perf_counter() - start,
        "precision": "float64", "real_precision": 2, "cutoff_max": 5,
        "full_law_max_modes": 14, "sampling_api": "unmodified official generate_samples",
        "density_api": "separate generated forced-branch SIMD trace after sampling",
        "fast_math": False,
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
    return executable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".cache/frugal-evidence-native")
    parser.add_argument("--eigen", type=Path, default=Path(os.environ.get("EIGEN3_INCLUDE_DIR", DEFAULT_EIGEN)))
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--modes", default=",".join(map(str, DEFAULT_MODES)))
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build(args.build_dir, args.eigen, args.compiler,
                tuple(map(int, args.modes.split(","))), args.native, args.force))
