"""Auditable evidence from unchanged official FrugalGBS CPU sampling.

Build explicitly with experiments/build_frugal_evidence.py. This module never
compiles, retries, reduces K, substitutes a different sampler, or postselects.
The recorded log density describes Frugal's computed conditional law, not the
target GBS distribution; use assess_frugal_samples to compare those laws.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTABLE = ROOT / ".cache/frugal-evidence-native/frugal_evidence_adapter"
UPSTREAM_COMMIT = "fa8b796f2afebd4845a2e335e457735952617ef8"
SAMPLER_BODY_SHA256 = "9c14a870f2875e808025d9e3535a71ba86144e5639ab14f0b7be5b594dabb3fb"


class FrugalEvidenceError(RuntimeError):
    """A single run failed; its structured payload and stage log are retained."""
    def __init__(self, message, *, payload=None, stderr="", returncode=None,
                 executable_sha256=None):
        super().__init__(message)
        self.payload = payload
        self.stderr = stderr
        self.returncode = returncode
        self.executable_sha256 = executable_sha256


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside supported range")
    return value


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _covariance(covariance):
    supplied = getattr(covariance, "V", covariance)
    if np.iscomplexobj(supplied):
        raise ValueError("Real xxpp covariance required")
    V = np.asarray(supplied, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] % 2 or not np.isfinite(V).all():
        raise ValueError("Finite even square covariance required")
    m = V.shape[0] // 2
    if not 2 <= m <= 144:
        raise ValueError("Evidence adapter supports 2..144 modes, subject to compiled mode list")
    scale = max(1.0, float(np.max(np.abs(V))))
    if not np.allclose(V, V.T, rtol=0, atol=1e-10 * scale):
        raise ValueError("Symmetric covariance required")
    V = np.ascontiguousarray(V * .5 + V.T * .5)
    eye, zero = np.eye(m), np.zeros((m, m))
    omega = np.block([[zero, eye], [-eye, zero]])
    minimum = float(np.linalg.eigvalsh(V + 1j * omega)[0])
    if minimum < -1e-9 * scale:
        raise ValueError(f"Covariance violates Gaussian uncertainty relation: {minimum:g}")
    return V, minimum


def _counts(values, size, total, name):
    array = np.asarray(values)
    if (array.shape != (size,) or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0) or sum(map(int, array)) != total):
        raise FrugalEvidenceError(f"Invalid {name} counts or total")
    return array


def _build_metadata(executable, sha):
    path = executable.parent / "build.json"
    try:
        metadata = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    except (OSError, ValueError):
        return None
    return metadata if isinstance(metadata, dict) and metadata.get("executable_sha256") == sha else None


def _stage_diagnostics(stderr):
    result = []
    for line in stderr.splitlines():
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except ValueError:
            continue
        if isinstance(value, dict) and "phase" in value:
            result.append(value)
    return result


def frugal_storage_lower_bound(modes, cutoff=5, *, simd_lanes=8):
    """Payload lower bound from upstream containers, not a peak-RSS forecast.

    Includes persistent bitstring layout and float64 target, plus one tracing
    workspace. Excludes allocation overhead and temporary target construction.
    """
    m = _integer(modes, "modes", 2, 144)
    k = _integer(cutoff, "cutoff", 1, min(5, m))
    lanes = _integer(simd_lanes, "simd_lanes", 1)
    entries = sum(math.comb(m, i) for i in range(k + 1))
    bitstring_bytes = 4 * ((m + 31) // 32)
    # upto<N> is uint8 for every supported N <= 144.
    layout = entries * (bitstring_bytes + 1 + k)
    target = 8 * (entries + m)
    workspace = 8 * lanes * (2 * math.comb(m, 2) + 2 * max(1, math.comb(m, 3)))
    return {"modes": m, "cutoff": k, "layout_entries": entries,
            "bitstring_layout_bytes": layout, "target_payload_bytes": target,
            "trace_workspace_bytes": workspace,
            "persistent_payload_plus_one_trace_workspace_bytes": layout + target + workspace,
            "is_peak_rss_estimate": False,
            "excludes": "target-construction temporaries, allocator overhead, cumulant layout, samples, serialization"}


def frugal_evidence_sample(covariance, cutoff=5, sample_count=1024, seed=0, *,
                          executable=None, record_log_probabilities=True,
                          raw_samples=True, enumerate_distribution=False,
                          timeout=1800.0, threads=1):
    """Make exactly one official sampling call and optionally score every output.

    ``timings_seconds['sampling']`` contains unchanged official generation;
    ``density`` contains workspace allocation and subsequent serial log-Q
    tracing. No filtering or reordering occurs. Both runtime K and precision
    are returned explicitly, including on structured native failure.
    """
    start = perf_counter()
    V, uncertainty_minimum = _covariance(covariance)
    m = len(V) // 2
    k = _integer(cutoff, "cutoff", 1, min(5, m))
    n = _integer(sample_count, "sample_count", 0)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    threads = _integer(threads, "threads", 1, 1024)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    for name, flag in (("record_log_probabilities", record_log_probabilities),
                       ("raw_samples", raw_samples), ("enumerate_distribution", enumerate_distribution)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    if enumerate_distribution and m > 14:
        raise ValueError("Full-law cap is 14 modes")
    executable = Path(DEFAULT_EXECUTABLE if executable is None else executable).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Evidence executable not built: {executable}")
    sha = _sha256(executable.read_bytes())
    metadata = _build_metadata(executable, sha)
    if metadata is not None and m not in metadata.get("modes", ()):
        raise ValueError(f"Mode count {m} is not compiled into this evidence executable")
    provenance = {
        "executable_sha256": sha,
        "build_cache_key": metadata.get("cache_key") if metadata else None,
        "build_metadata_matches_binary": metadata is not None,
        "covariance_sha256": _sha256(V.tobytes()),
        "selector": "official_frugal_double_elision", "k": k, "vacuum_kernel": "upstream",
        "seed": seed, "sample_count": n, "modes": m, "cutoff": k,
        "order": list(range(m)), "uniform_source": "upstream per-thread seeded SIMD PRNG",
        "uniforms_sha256": None, "executable": str(executable),
    }
    with tempfile.TemporaryDirectory(prefix="gbs-frugal-evidence-") as directory:
        path = Path(directory) / "covariance.csv"
        np.savetxt(path, V, delimiter=",", fmt="%.17g")
        cmd = [str(executable), "--input", str(path), "--modes", str(m), "--cutoff", str(k),
               "--samples", str(n), "--seed", str(seed), "--threads", str(threads)]
        if raw_samples:
            cmd.append("--raw-samples")
        if record_log_probabilities:
            cmd.append("--record-log-probabilities")
        if enumerate_distribution:
            cmd.append("--enumerate")
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            payload = {**provenance, "status": "error", "error": "timeout", "timeout_seconds": timeout,
                       "stage_diagnostics": _stage_diagnostics(stderr), "no_retry": True,
                       "samples_postselected": False, "wrapper_wall_seconds": perf_counter() - start}
            raise FrugalEvidenceError("Official Frugal evidence run timed out; no retry or K reduction",
                payload=payload, stderr=stderr, executable_sha256=sha) from exc
    try:
        out = json.loads(process.stdout, parse_constant=_reject_json_constant)
        if not isinstance(out, dict):
            raise ValueError("Native output must be a JSON object")
    except ValueError as exc:
        payload = {**provenance, "status": "error", "error": "invalid native JSON",
                   "stage_diagnostics": _stage_diagnostics(process.stderr), "no_retry": True}
        raise FrugalEvidenceError("Invalid native evidence output", payload=payload,
            returncode=process.returncode, stderr=process.stderr, executable_sha256=sha) from exc
    if process.returncode or out.get("status") != "ok":
        out.update({key: value for key, value in provenance.items() if key not in out})
        out["stage_diagnostics"] = _stage_diagnostics(process.stderr)
        out["wrapper_wall_seconds"] = perf_counter() - start
        raise FrugalEvidenceError(out.get("error", "Official Frugal evidence run failed"),
            payload=out, stderr=process.stderr, returncode=process.returncode, executable_sha256=sha)
    if _sha256(executable.read_bytes()) != sha:
        raise FrugalEvidenceError("Executable changed during run", payload=out, executable_sha256=sha)
    expected = {"schema_version": 1, "modes": m, "cutoff": k, "sample_count": n,
                "requested_sample_count": n, "seed": seed, "omp_threads": threads,
                "real_precision": 2, "precision": "float64", "upstream_commit": UPSTREAM_COMMIT,
                "sampler_body_sha256": SAMPLER_BODY_SHA256,
                "sampling_api": "unmodified official generate_samples",
                "log_probability_enabled": bool(record_log_probabilities)}
    if any(out.get(key) != value for key, value in expected.items()):
        raise FrugalEvidenceError("Native evidence metadata mismatch", payload=out, executable_sha256=sha)
    clicks = _counts(out.get("click_count_histogram"), m + 1, n, "click histogram")
    total_clicks = sum(i * int(v) for i, v in enumerate(clicks))
    per_mode = _counts(out.get("per_mode_click_counts"), m, total_clicks, "per-mode")
    if np.any(per_mode > n):
        raise FrugalEvidenceError("Invalid per-mode count above sample count")
    histogram = None
    if m <= 14:
        histogram = _counts(out.get("histogram_counts"), 2**m, n, "physical-mask histogram")
    elif out.get("histogram_counts") != []:
        raise FrugalEvidenceError("Unexpected exponential histogram above cap")
    observed = None
    if raw_samples:
        observed = np.asarray(out.get("raw_samples"))
        if n == 0 and out.get("raw_samples") == []:
            observed = np.empty((0, m), dtype=np.uint8)
        if observed.shape != (n, m) or np.any((observed != 0) & (observed != 1)):
            raise FrugalEvidenceError("Raw samples are not a complete physical-bit batch")
        observed = observed.astype(np.uint8)
        if not np.array_equal(observed.sum(axis=0), per_mode):
            raise FrugalEvidenceError("Raw samples disagree with per-mode counts")
        if not np.array_equal(np.bincount(observed.sum(axis=1).astype(int), minlength=m + 1), clicks):
            raise FrugalEvidenceError("Raw samples disagree with click-count histogram")
        if histogram is not None:
            masks = observed.astype(np.int64) @ (1 << np.arange(m))
            if not np.array_equal(np.bincount(masks, minlength=2**m), histogram):
                raise FrugalEvidenceError("Raw samples disagree with physical-mask histogram")
    elif out.get("raw_samples") is not None:
        raise FrugalEvidenceError("Unexpected unrequested raw samples")
    q = None
    if enumerate_distribution:
        q = np.asarray(out.get("distribution"), dtype=float)
        trace = out.get("trace") or {}
        if (q.shape != (2**m,) or not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1)
                or abs(float(q.sum(dtype=np.longdouble)) - 1) > 1e-11
                or trace.get("nonfinite_threshold_path_steps") != 0):
            raise FrugalEvidenceError("Invalid full Frugal conditional law")
    elif out.get("distribution") is not None:
        raise FrugalEvidenceError("Unexpected unrequested full distribution")
    if record_log_probabilities:
        logs = np.asarray(out.get("sample_log_probabilities"), dtype=float)
        trace = out.get("density_trace") or {}
        if (logs.shape != (n,) or not np.isfinite(logs).all() or np.any(logs > 0)
                or trace.get("nonfinite_threshold_path_steps") != 0
                or trace.get("scored_samples") != n or trace.get("after_timed_sampling") is not True
                or not isinstance(out.get("sample_log_probability_semantics"), str)):
            raise FrugalEvidenceError("Invalid sampled path density or trace metadata")
        if observed is not None:
            unique = {}
            for sample, logq in zip(observed, logs):
                key = bytes(np.packbits(sample))
                if key in unique and abs(unique[key] - logq) > 2e-10:
                    raise FrugalEvidenceError("Repeated sample has inconsistent log Q")
                unique[key] = logq
            if sum(math.exp(value) for value in unique.values()) > 1 + 1e-10:
                raise FrugalEvidenceError("Distinct observed sample masses exceed one")
            if q is not None and n:
                mask = observed.astype(np.int64) @ (1 << np.arange(m))
                if np.any(q[mask] <= 0) or not np.allclose(logs, np.log(q[mask]), rtol=0, atol=2e-10):
                    raise FrugalEvidenceError("Sample log Q disagrees with full conditional law")
    elif out.get("sample_log_probabilities") is not None:
        raise FrugalEvidenceError("Unexpected unrequested sampled density")
    timings = out.get("timings_seconds")
    if not isinstance(timings, dict) or any(name not in timings for name in ("layout", "target", "sampling", "density")):
        raise FrugalEvidenceError("Missing separated generation/density timings")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in timings.values()):
        raise FrugalEvidenceError("Invalid timings")
    out.update(provenance)
    out.update(wrapper_wall_seconds=perf_counter() - start,
               stage_diagnostics=_stage_diagnostics(process.stderr),
               model_uncertainty_minimum=uncertainty_minimum,
               density_in_sampling_timer=False, no_retry=True, samples_postselected=False,
               finite_prng_density_claimed=False)
    return out


def assess_frugal_samples(covariance, evidence, *, protocol=None, **options):
    """Pass complete Frugal evidence to the generic GBS identity evaluator.

    The caller supplies a protocol fixed before sampling and a confidence plan.
    A successful run or a good fit to a few moments is not that attestation.
    """
    from .gbs_identity import assess_gbs_identity, covariance_sha256
    if not isinstance(evidence, dict) or evidence.get("status") != "ok":
        failure = evidence if isinstance(evidence, dict) else {"error": str(evidence)}
        return assess_gbs_identity(covariance, protocol=protocol, sampling_failure=failure, **options)
    if evidence.get("raw_samples") is None or evidence.get("sample_log_probabilities") is None or evidence.get("log_probability_enabled") is not True:
        raise ValueError("Frugal assessment requires raw samples and recorded log probabilities")
    V = np.asarray(getattr(covariance, "V", covariance))
    if evidence.get("modes") != V.shape[0] // 2 or evidence.get("covariance_sha256") != covariance_sha256(V):
        raise ValueError("Frugal evidence is not bound to the supplied covariance")
    if len(evidence["raw_samples"]) != evidence.get("sample_count"):
        raise ValueError("Frugal evidence does not contain the complete batch")
    if isinstance(protocol, dict) and protocol.get("requested_draws") != evidence.get("sample_count"):
        return assess_gbs_identity(V, protocol=protocol,
            sampling_failure={"error": "Frugal batch does not match prespecified requested_draws"}, **options)
    result = assess_gbs_identity(V, candidate_distribution=evidence.get("distribution"),
        samples=evidence["raw_samples"], sample_log_probabilities=evidence["sample_log_probabilities"],
        protocol=protocol, **options)
    result["native_provenance"] = {key: evidence.get(key) for key in (
        "executable_sha256", "build_cache_key", "build_metadata_matches_binary", "covariance_sha256",
        "selector", "k", "vacuum_kernel", "seed", "sample_count", "order", "uniform_source",
        "sample_log_probability_semantics", "uniforms_sha256", "upstream_commit", "sampler_body_sha256",
        "cutoff", "real_precision", "precision", "sampling_api", "density_in_sampling_timer")}
    result["native_provenance"]["binding_scope"] = (
        "Traceability to the declared covariance, pinned sampler, precision, and executable; "
        "not cryptographic verification of implementation correctness")
    return result
