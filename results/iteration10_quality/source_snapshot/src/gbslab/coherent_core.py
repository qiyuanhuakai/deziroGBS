"""Checked native prototype for a coherent photon-truncated Gaussian core.

This is a capped dense Fock representation, not MPS.  The coherent normalized
projection precedes a correlated classical displacement.  Hidden rank-one
measurement channels are marginalized by the physical sampler; their trajectory
probabilities must never be reported as marginal binary logQ.

Imports/tests do not compile anything.  Numerical failures stop the batch and
retain complete outputs in CoherentCoreError.payload.  Positive long-double
probabilities and amplitudes travel as decimal strings, not binary64 JSON
numbers.  sqrt(tail_probability) is only the exact-arithmetic projection budget:
floating state construction, measurement and the finite Gaussian RNG need their
own error analysis; this module does not claim a certified numerical TV bound.
"""
from __future__ import annotations

import hashlib
import json
from math import comb
import os
from pathlib import Path
import re
import subprocess
import tempfile
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTABLE = ROOT / ".cache/coherent-core-native/coherent_core_sampler"
EXECUTABLE_ENV = "GBSLAB_COHERENT_CORE_EXECUTABLE"
MAX_MODES = 64
MAX_CUTOFF = 8
MAX_BASIS = 100_000
MAX_STAGE_BASIS = 1_000_000
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_SAMPLES = 1_000_000
NUMERIC_GUARD = 1e-10


class CoherentCoreError(RuntimeError):
    """Native failure with unmodified completed-sample and diagnostic evidence."""

    def __init__(self, message, *, payload=None, returncode=None, stderr=None):
        super().__init__(message)
        self.payload = payload
        self.returncode = returncode
        self.stderr = stderr
        self.numerical_failure = bool(payload and payload.get("numerical_failure"))


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return value


def _scalar(value, name, low=0, high=np.inf):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and lie in [{low}, {high}]")
    return value


def coherent_core_resources(modes, cutoff, *, samples=0, max_basis=MAX_BASIS,
                            max_stage_basis=MAX_STAGE_BASIS,
                            max_payload_bytes=MAX_PAYLOAD_BYTES,
                            uniforms=False, normals=False, trace=False):
    """Exact combinatorial resource preflight; rejected sizes are not truncated.

    The initial state has only even total sectors.  The live representation and
    every postmeasurement state include ALL occupations of total N<=cutoff.
    Bytes describe major numerical/index/sample containers, not process RSS.
    """
    m = _integer(modes, "modes", 1, MAX_MODES)
    k = _integer(cutoff, "cutoff", 0, MAX_CUTOFF)
    n = _integer(samples, "samples", 0, MAX_SAMPLES)
    cap = _integer(max_basis, "max_basis", 1, MAX_BASIS)
    stage_cap = _integer(max_stage_basis, "max_stage_basis", 1, MAX_STAGE_BASIS)
    byte_cap = _integer(max_payload_bytes, "max_payload_bytes", 1, MAX_PAYLOAD_BYTES)
    dims = [comb(r + k, k) for r in range(m + 1)]
    full = dims[-1]
    even = sum(comb(m + total - 1, total) for total in range(0, k + 1, 2))
    stages = sum(dims)
    fibers = sum(dims[:-1]) * (k + 1)
    real_bytes = np.dtype(np.longdouble).itemsize
    complex_bytes = np.dtype(np.clongdouble).itemsize
    payload = (stages * 9 + fibers * 4 + 2 * m * m * complex_bytes
               + (3 * full + (k + 1) * dims[-2]) * complex_bytes
               + full * real_bytes + n * 8
               + (n * 2 * m * real_bytes if uniforms else 0)
               + (n * m * real_bytes if normals else 0)
               + (n * m * 512 if trace else 0))
    reasons = []
    if full > cap:
        reasons.append("full_basis")
    if stages > stage_cap:
        reasons.append("all_stage_basis")
    if fibers > np.iinfo(np.int32).max:
        reasons.append("fiber_index_capacity")
    if payload > byte_cap:
        reasons.append("planned_payload")
    if trace and n * m > 100_000:
        reasons.append("trace_entries")
    return {"modes": m, "cutoff": k, "full_basis": full, "even_initial_basis": even,
            "all_stage_basis": stages, "fiber_index_entries": fibers,
            "planned_major_payload_bytes": payload, "max_basis": cap,
            "max_stage_basis": stage_cap, "max_payload_bytes": byte_cap,
            "feasible": not reasons, "rejection_reasons": reasons,
            "memory_scope": "major numerical/index/sample containers; not allocator peak, RSS or JSON transport"}


def find_coherent_core_executable(executable=None):
    value = executable if executable is not None else os.environ.get(EXECUTABLE_ENV, DEFAULT_EXECUTABLE)
    path = Path(value)
    if path.is_file() and os.access(path, os.X_OK):
        return path.resolve()
    return None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unitary(value):
    U = np.ascontiguousarray(value, dtype=np.complex128)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or not 1 <= U.shape[0] <= MAX_MODES:
        raise ValueError("U must be a square matrix with 1..64 modes")
    if not np.all(np.isfinite(U)):
        raise ValueError("U must be finite")
    residual = float(np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0]))))
    if residual > 1e-10:
        raise ValueError("U must be unitary within absolute residual 1e-10")
    return U


def _complex_vector(value, length, name):
    result = np.ascontiguousarray(value, dtype=np.complex128)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite complex vector of length {length}")
    return result


def _invalid_json_constant(value):
    raise ValueError("invalid JSON constant: " + value)


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _decode_numbers(value):
    if isinstance(value, str) and _NUMBER.fullmatch(value):
        result = np.longdouble(value)
        if not np.isfinite(result):
            raise ValueError("nonfinite long-double field in native payload")
        return result
    if isinstance(value, list):
        return [_decode_numbers(x) for x in value]
    if isinstance(value, dict):
        return {k: _decode_numbers(v) for k, v in value.items()}
    return value


def _complex_fields(value):
    return (np.asarray(value["amplitudes_real"], dtype=np.longdouble)
            + np.clongdouble(1j) * np.asarray(value["amplitudes_imag"], dtype=np.longdouble))


def _sum_longdouble(values):
    s = np.longdouble(0)
    c = np.longdouble(0)
    for value in values:
        x = np.longdouble(value)
        next_sum = s + x
        c += (s - next_sum) + x if abs(s) >= abs(x) else (x - next_sum) + s
        s = next_sum
    return s + c


def _validate_trace(result, modes, cutoff, packed, injected_uniforms):
    """Check the actual fixed two-uniform latent-channel protocol."""
    prefix = 0
    for at, step in enumerate(result["trace"]):
        sample, mode = divmod(at, modes)
        if mode == 0:
            prefix = 0
        if (step.get("sample") != sample or step.get("mode") != mode
                or step.get("prefix_mask") != prefix):
            raise CoherentCoreError("measurement trace order/prefix mismatch", payload=result)
        p = step["projective_probabilities"]
        u, v, t, q = (step[name] for name in ("uniform_channel", "uniform_split", "t", "q"))
        if (p.shape != (cutoff + 1,) or not np.all(np.isfinite(p)) or np.any(p < 0)
                or not 0 <= u < 1 or not 0 <= v < 1 or not t > 0 or q < 0
                or abs(t + q - 1) > result["numeric_guard"]
                or not np.isfinite(step["alpha"])):
            raise CoherentCoreError("invalid latent-channel trace probabilities", payload=result)
        if injected_uniforms is not None and (u != injected_uniforms[sample, 2 * mode]
                                              or v != injected_uniforms[sample, 2 * mode + 1]):
            raise CoherentCoreError("native trace changed injected uniforms", payload=result)
        order = sorted(range(cutoff + 1), key=lambda j: (p[j], j))
        threshold = u * _sum_longdouble(p)
        cumulative = []
        chosen = None
        for j in order:
            if p[j] == 0:
                continue
            chosen = j
            cumulative.append(p[j])
            if threshold < _sum_longdouble(cumulative):
                break
        if chosen is None or step.get("channel") != chosen:
            raise CoherentCoreError("native sorted-categorical channel disagrees with trace", payload=result)
        if chosen:
            bit = 1
        elif q == 0:
            if step["alpha"] != 0:
                raise CoherentCoreError("uncertified zero Poisson tail in trace", payload=result)
            bit = 0
        elif q <= t:
            bit = int(v < q / (t + q))
        else:
            bit = int(not v < t / (t + q))
        if type(step.get("bit")) is not int or step["bit"] != bit:
            raise CoherentCoreError("native small-tail binary split disagrees with trace", payload=result)
        prefix |= bit << mode
        if mode == modes - 1 and prefix != packed[sample]:
            raise CoherentCoreError("trace binary output differs from packed sample", payload=result)
    partial = result["partial_prefix_length"]
    if partial and prefix != result["partial_prefix_mask"]:
        raise CoherentCoreError("partial latent trace disagrees with checkpoint prefix", payload=result)


def _decode_payload(raw, modes, cutoff, requested, raw_samples, traced, injected_uniforms=None):
    result = _decode_numbers(raw)
    if result.get("schema_version") != 1:
        raise CoherentCoreError("unsupported native coherent-core schema", payload=result)
    if "completed_samples" not in result:
        return result
    count = result["completed_samples"]
    if type(count) is not int or not 0 <= count <= requested:
        raise CoherentCoreError("invalid completed sample count", payload=result)
    if result.get("modes") != modes or result.get("cutoff") != cutoff or result.get("requested_samples") != requested:
        raise CoherentCoreError("native request binding mismatch", payload=result)
    if result["long_double_bytes"] != np.dtype(np.longdouble).itemsize or result["long_double_digits"] < 64:
        raise CoherentCoreError("native/Python long-double format mismatch", payload=result)
    packed = result.get("completed_packed_words")
    if (not isinstance(packed, list) or len(packed) != count
            or any(type(x) is not int or not 0 <= x < 1 << modes for x in packed)):
        raise CoherentCoreError("invalid completed packed checkpoint", payload=result)
    bits = np.zeros((count, modes), dtype=np.uint8)
    for j in range(modes):
        bits[:, j] = [(word >> j) & 1 for word in packed]
    if raw_samples:
        original = np.asarray(result.get("raw_samples"))
        if count == 0 and original.shape == (0,):
            original = original.reshape(0, modes)
        if (original.shape != bits.shape or not np.all((original == 0) | (original == 1))
                or not np.array_equal(original, bits)):
            raise CoherentCoreError("raw and packed samples disagree", payload=result)
        result["raw_samples"] = bits
    elif result.get("raw_samples") is not None:
        raise CoherentCoreError("unrequested raw sample output", payload=result)
    partial = result.get("partial_prefix_length")
    if type(partial) is not int or not 0 <= partial <= modes:
        raise CoherentCoreError("invalid partial prefix length", payload=result)
    if result["diagnostics"]["quantum_uniforms"] != 2 * (count * modes + partial):
        raise CoherentCoreError("quantum uniform count disagrees with completed/partial trace", payload=result)
    if result.get("status") == "ok" and count != requested:
        raise CoherentCoreError("successful batch is incomplete", payload=result)
    if result.get("state") is not None:
        state = result["state"]
        occupations = np.asarray(state["occupations"], dtype=np.int16)
        amplitudes = _complex_fields(state)
        if (occupations.shape != (len(amplitudes), modes) or not np.all(np.isfinite(amplitudes))
                or np.any(occupations < 0) or np.any(occupations.sum(axis=1) > cutoff)):
            raise CoherentCoreError("invalid returned Fock state", payload=result)
        norm = np.sum(np.abs(amplitudes) ** 2, dtype=np.longdouble)
        if abs(norm - 1) > result["numeric_guard"]:
            raise CoherentCoreError("returned coherent core is not normalized", payload=result)
        state.update(occupations=occupations, amplitudes=amplitudes)
        result.update(occupations=occupations, amplitudes=amplitudes)
    if result.get("channels") is not None:
        channels = result["channels"]
        for key in ("projective_probabilities", "projective_raw_weights", "flattened_probabilities"):
            channels[key] = np.asarray(channels[key], dtype=np.longdouble)
        channels["w"] = (np.asarray(channels["w_real"], dtype=np.longdouble)
                         + np.clongdouble(1j) * np.asarray(channels["w_imag"], dtype=np.longdouble))
        channels["remaining_occupations"] = np.asarray(channels["remaining_occupations"], dtype=np.int16)
        states = [None if row is None else _complex_fields(row) for row in channels["remaining_states"]]
        channels["remaining_amplitudes"] = states
        if (channels["projective_probabilities"].shape != (cutoff + 1,)
                or channels["flattened_probabilities"].shape != (cutoff + 2,)
                or np.any(channels["projective_probabilities"] < 0)
                or np.any(channels["flattened_probabilities"] < 0)):
            raise CoherentCoreError("invalid projective/refined probabilities", payload=result)
        if abs(channels["flattened_probabilities"].sum(dtype=np.longdouble) - 1) > result["numeric_guard"]:
            raise CoherentCoreError("refined POVM probabilities are not normalized", payload=result)
        result.update(channels)
    for step in result.get("trace", []):
        step["projective_probabilities"] = np.asarray(step["projective_probabilities"], dtype=np.longdouble)
        step["alpha"] = np.clongdouble(step["alpha_real"]) + np.clongdouble(1j) * step["alpha_imag"]
    if traced and len(result.get("trace", [])) != count * modes + partial:
        raise CoherentCoreError("trace coverage disagrees with complete/partial samples", payload=result)
    if not traced and result.get("trace"):
        raise CoherentCoreError("unrequested measurement trace", payload=result)
    if traced:
        _validate_trace(result, modes, cutoff, packed, injected_uniforms)
    if result.get("distribution") is not None:
        q = np.asarray(result["distribution"], dtype=np.longdouble)
        f = result["failure_mass"]
        if (q.shape != (1 << modes,) or not np.all(np.isfinite(q)) or np.any(q < 0)
                or not 0 <= f <= 1 or abs(q.sum(dtype=np.longdouble) + f - 1) > result["numeric_guard"]):
            raise CoherentCoreError("invalid complete extended law", payload=result)
        result["distribution"] = q
    if result.get("vacuum_marginals") is not None:
        result["vacuum_marginals"] = np.asarray(result["vacuum_marginals"], dtype=np.longdouble)
    return result


def _write_complex(path, values):
    values = np.asarray(values, dtype=np.complex128).reshape(-1)
    columns = np.column_stack((values.real, values.imag))
    np.savetxt(path, columns, fmt="%.17g", delimiter=",")


def _ld_text(value):
    return np.format_float_scientific(np.longdouble(value), precision=21, unique=False, trim="k")


def _invoke(*, action, U=None, squeezing=0, loss=1, cutoff, samples=0, seed=0,
            fixed_alpha=None, uniforms=None, normals=None, raw_samples=False,
            return_state=False, trace=False, occupations=None, amplitudes=None,
            executable=None, timeout=300, numeric_guard=NUMERIC_GUARD,
            max_basis=MAX_BASIS, max_stage_basis=MAX_STAGE_BASIS,
            max_payload_bytes=MAX_PAYLOAD_BYTES, max_enumeration_nodes=1_000_000,
            allow_failure=False):
    begun = perf_counter()
    samples = _integer(samples, "samples", 0, MAX_SAMPLES)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    cutoff = _integer(cutoff, "cutoff", 0, MAX_CUTOFF)
    squeezing = _scalar(squeezing, "squeezing")
    loss = _scalar(loss, "loss", 0, 1)
    guard = _scalar(numeric_guard, "numeric_guard", np.nextafter(0., 1.), 1e-6)
    timeout = _scalar(timeout, "timeout", np.nextafter(0., 1.))
    max_nodes = _integer(max_enumeration_nodes, "max_enumeration_nodes", 1, 1_000_000)
    if action == "channels":
        occ = np.asarray(occupations)
        amp = np.asarray(amplitudes, dtype=np.clongdouble)
        if (occ.ndim != 2 or not 1 <= occ.shape[1] <= MAX_MODES
                or occ.shape[0] != amp.size or amp.ndim != 1 or amp.size == 0
                or not np.all(np.isfinite(amp)) or not np.all(np.isfinite(occ))
                or np.any(occ < 0) or np.any(occ != np.floor(occ)) or np.any(occ.sum(axis=1) > cutoff)):
            raise ValueError("invalid normalized sparse Fock state")
        occ = occ.astype(np.int16)
        if len({tuple(row) for row in occ.tolist()}) != len(occ):
            raise ValueError("Fock occupations must be unique")
        if abs(np.sum(np.abs(amp) ** 2, dtype=np.longdouble) - 1) > guard:
            raise ValueError("input pure state must be normalized within numeric_guard")
        modes = occ.shape[1]
        alpha = _complex_vector([fixed_alpha], 1, "first-mode alpha")
    else:
        U = _unitary(U)
        modes = U.shape[0]
        alpha = None if fixed_alpha is None else _complex_vector(fixed_alpha, modes, "fixed_alpha")
    if action == "vacuum-law" and modes > 12:
        raise ValueError("vacuum full law supports at most 12 modes")
    if action == "refined-law" and modes > 4:
        raise ValueError("refined full law supports at most 4 modes")
    if normals is not None and alpha is not None:
        raise ValueError("fixed_alpha and normals are mutually exclusive")
    if uniforms is not None:
        if np.iscomplexobj(uniforms):
            raise ValueError("uniforms must be real")
        uniforms = np.ascontiguousarray(uniforms, dtype=float)
        if uniforms.shape != (samples, 2 * modes) or not np.all(np.isfinite(uniforms)) or np.any((uniforms < 0) | (uniforms >= 1)):
            raise ValueError("uniforms must have shape (samples, 2*modes), in [0,1)")
    if normals is not None:
        if np.iscomplexobj(normals):
            raise ValueError("normals must be real")
        normals = np.ascontiguousarray(normals, dtype=float)
        if normals.shape != (samples, modes) or not np.all(np.isfinite(normals)):
            raise ValueError("normals must have shape (samples, modes), finite")
    resource = coherent_core_resources(modes, cutoff, samples=samples,
        max_basis=max_basis, max_stage_basis=max_stage_basis, max_payload_bytes=max_payload_bytes,
        uniforms=uniforms is not None, normals=normals is not None, trace=trace)
    if not resource["feasible"]:
        raise ValueError("dense coherent-core resource cap exceeded: " + ", ".join(resource["rejection_reasons"]))
    path = find_coherent_core_executable(executable)
    if path is None:
        raise FileNotFoundError("Compile experiments/build_coherent_core_sampler.py explicitly; no implicit compilation")
    digest = _sha(path)
    build_path = path.parent / "build.json"
    if not build_path.is_file():
        raise ValueError("coherent-core binary requires build.json provenance")
    build = json.loads(build_path.read_text())
    if build.get("executable_sha256") != digest:
        raise ValueError("coherent-core executable differs from build.json")
    required = {"experiments/coherent_core_sampler.cpp", "experiments/build_coherent_core_sampler.py"}
    if not required.issubset(build.get("source_hashes", {})):
        raise ValueError("coherent-core build must bind source and builder")
    for name, expected in build["source_hashes"].items():
        if _sha(ROOT / name) != expected:
            raise ValueError("coherent-core source changed since explicit compilation: " + name)
    validation_seconds = perf_counter() - begun
    with tempfile.TemporaryDirectory(prefix="gbs-coherent-core-") as temporary:
        directory = Path(temporary)
        command = [str(path), "--action", action, "--modes", str(modes), "--cutoff", str(cutoff),
                   "--squeezing", format(squeezing, ".17g"), "--loss", format(loss, ".17g"),
                   "--samples", str(samples), "--seed", str(seed), "--numeric-guard", format(guard, ".17g"),
                   "--max-basis", str(max_basis), "--max-stage-basis", str(max_stage_basis),
                   "--max-payload-bytes", str(max_payload_bytes), "--max-enumeration-nodes", str(max_nodes)]
        if U is not None:
            filename = directory / "unitary.csv"
            _write_complex(filename, U)
            command += ["--unitary", str(filename)]
        if alpha is not None:
            filename = directory / "alpha.csv"
            _write_complex(filename, alpha)
            command += ["--alpha", str(filename)]
        if action == "channels":
            filename = directory / "state.csv"
            with filename.open("w") as stream:
                for row, value in zip(occ, amp):
                    stream.write(",".join([*(str(int(x)) for x in row), _ld_text(value.real), _ld_text(value.imag)]) + "\n")
            command += ["--state", str(filename)]
        for name, values in (("uniforms", uniforms), ("normals", normals)):
            if values is not None:
                filename = directory / (name + ".csv")
                np.savetxt(filename, values, fmt="%.17g", delimiter=",")
                command += ["--" + name, str(filename)]
        if raw_samples:
            command.append("--raw-samples")
        if return_state:
            command.append("--return-state")
        if trace:
            command.append("--trace")
        started = perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        process_seconds = perf_counter() - started
    if _sha(path) != digest:
        raise CoherentCoreError("executable changed during invocation", returncode=process.returncode, stderr=process.stderr)
    try:
        raw = json.loads(process.stdout, parse_constant=_invalid_json_constant)
        result = _decode_payload(raw, modes, cutoff, samples, raw_samples, trace, uniforms)
    except (ValueError, TypeError, KeyError) as exc:
        raise CoherentCoreError("invalid native coherent-core payload: " + str(exc),
                               returncode=process.returncode, stderr=process.stderr) from exc
    result.update(executable_sha256=digest, build_cache_key=build["cache_key"],
                  build_metadata_matches_binary=True, wrapper_input_validation_seconds=validation_seconds,
                  wrapper_process_wall_seconds=process_seconds, wrapper_wall_seconds=perf_counter() - begun,
                  wrapper_stderr=process.stderr)
    if U is not None:
        result["unitary_sha256"] = hashlib.sha256(U.tobytes()).hexdigest()
    if alpha is not None:
        result["fixed_alpha_sha256"] = hashlib.sha256(alpha.tobytes()).hexdigest()
    if uniforms is not None:
        result["uniforms_sha256"] = hashlib.sha256(uniforms.tobytes()).hexdigest()
    if normals is not None:
        result["normals_sha256"] = hashlib.sha256(normals.tobytes()).hexdigest()
    if process.returncode or result.get("status") != "ok":
        if allow_failure:
            return result
        raise CoherentCoreError(result.get("error") or "native coherent-core invocation failed",
                               payload=result, returncode=process.returncode, stderr=process.stderr)
    return result


def coherent_core_state(U, squeezing, loss, cutoff, **options):
    """Construct and export the normalized coherent core, with all basis rows."""
    return _invoke(action="state", U=U, squeezing=squeezing, loss=loss, cutoff=cutoff,
                   return_state=True, **options)


def run_coherent_core(U, squeezing, loss, cutoff, samples=0, seed=0, *,
                      fixed_alpha=None, uniforms=None, normals=None,
                      raw_samples=True, return_state=False, trace=False, **options):
    """Sample the displaced coherent core without resampling failed attempts.

    normals has shape (samples,m); uniforms has shape (samples,2*m).  Two
    quantum uniforms are consumed after each successful measurement contraction.
    The second is used only for projective channel zero.  Fixed alpha suppresses
    classical-displacement draws and is a validation route, not the mixed model.
    """
    return _invoke(action="sample", U=U, squeezing=squeezing, loss=loss, cutoff=cutoff,
                   samples=samples, seed=seed, fixed_alpha=fixed_alpha, uniforms=uniforms,
                   normals=normals, raw_samples=raw_samples, return_state=return_state,
                   trace=trace, **options)


def coherent_core_fixed_law(U, squeezing, loss, cutoff, alpha, *, method="vacuum", **options):
    """Fixed-alpha complete binary law; reference work is never sampling time.

    method='refined' (m<=4) traverses hidden rank-one outcomes of the sampler,
    retaining numerical failure mass.  method='vacuum' (m<=12) obtains vacuum
    marginals and applies Mobius inversion, failing on any negative probability
    or an excessive heuristic absolute indicator.  Neither integrates alpha.
    """
    if method not in ("vacuum", "refined"):
        raise ValueError("method must be 'vacuum' or 'refined'")
    return _invoke(action=method + "-law", U=U, squeezing=squeezing, loss=loss,
                   cutoff=cutoff, fixed_alpha=alpha, **options)


def coherent_measure_channels(occupations, amplitudes, alpha, *, cutoff, **options):
    """Inspect every rank-one channel for a normalized (possibly sparse) state.

    Returns K+1 projective probabilities, K+2 flattened binary/channel masses,
    all remaining occupations of total N<=K, normalized remaining_amplitudes
    (None for a zero channel), and independently computed Poisson t/q.
    """
    return _invoke(action="channels", occupations=occupations, amplitudes=amplitudes,
                   fixed_alpha=alpha, cutoff=cutoff, **options)
