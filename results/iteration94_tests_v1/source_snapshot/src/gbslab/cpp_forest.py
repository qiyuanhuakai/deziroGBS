"""Checked subprocess interface to the complete native batch-event sampler.

This wrapper never compiles, retries, refills failed batches, or normalizes
successful leaves.  A complete enumeration can legitimately have failure mass.
A stopped sample batch raises CppForestError with its completed raw/logQ
checkpoint and full executable/input provenance in ``payload``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from .cpp_click import (_integer, _invalid_json_constant, _read_build_metadata,
                        _sha256, _validate_covariance)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTABLE = ROOT / ".cache/batched-forest-build/batched_forest_sampler"


class CppForestError(RuntimeError):
    """Native failure, with completed samples preserved when the CLI returned them."""

    def __init__(self, message, *, returncode=None, payload=None, stderr=None,
                 executable_sha256=None):
        super().__init__(message)
        self.returncode = returncode
        self.payload = payload
        self.stderr = stderr
        self.executable_sha256 = executable_sha256


def find_cpp_forest_executable(executable=None):
    path = Path(executable) if executable is not None else DEFAULT_EXECUTABLE
    if path.is_file() and os.access(path, os.X_OK):
        return path.resolve()
    return None


def _bad(message, result):
    raise CppForestError(message, payload=result,
                         executable_sha256=result.get("executable_sha256"))


def _number(value, *, nonnegative=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value) and (not nonnegative or value >= 0))


def _bits(values, count, modes, result, name):
    array = np.asarray(values)
    if count == 0 and array.shape == (0,):
        array = array.reshape(0, modes)
    if array.shape != (count, modes) or not np.all((array == 0) | (array == 1)):
        _bad(f"{name} must contain {count} complete binary samples", result)
    return array.astype(np.uint8)


def _validate_checkpoint(result, modes, requested, raw_samples, record_logs):
    count = result.get("completed_samples")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= requested:
        _bad("invalid completed sample count", result)
    if result.get("sample_count") != count:
        _bad("sample_count disagrees with completed_samples", result)
    raw = result.get("raw_samples")
    bits = _bits(raw, count, modes, result, "raw_samples") if raw is not None else None
    if raw_samples and raw is None:
        _bad("requested complete raw samples are missing", result)
    packed = result.get("completed_packed_words")
    if packed is not None:
        words = (modes + 63) // 64
        if (result.get("words_per_sample") != words or not isinstance(packed, list)
                or len(packed) != count * words
                or any(isinstance(x, bool) or not isinstance(x, int)
                       or not 0 <= x < 2**64 for x in packed)):
            _bad("invalid completed packed checkpoint", result)
        decoded = np.zeros((count, modes), dtype=np.uint8)
        for j in range(modes):
            decoded[:, j] = [(packed[s * words + j // 64] >> (j % 64)) & 1
                             for s in range(count)]
        if bits is not None and not np.array_equal(bits, decoded):
            _bad("packed checkpoint disagrees with completed raw samples", result)
        if bits is None:
            bits = decoded
    if result.get("status") != "ok" and packed is None:
        _bad("failed batch has no completed packed checkpoint", result)
    logs = result.get("sample_log_probabilities")
    if record_logs:
        values = np.asarray(logs, dtype=float)
        if (result.get("log_probability_enabled") is not True
                or values.shape != (count,) or not np.all(np.isfinite(values))
                or np.any(values > 0)):
            _bad("invalid or missing actual sample log probabilities", result)
    elif logs is not None or result.get("log_probability_enabled") is not False:
        _bad("unrequested sample log probabilities were returned", result)
    else:
        values = None
    return count, bits, values


def _validate_step(step, modes, k, result, *, allow_failure=False):
    if not isinstance(step, dict):
        _bad("trace row is not an object", result)
    prefix = step.get("prefix")
    if (not isinstance(prefix, list) or len(prefix) >= modes
            or any(type(bit) is not int or bit not in (0, 1) for bit in prefix)):
        _bad("trace has invalid measured prefix", result)
    position = len(prefix)
    if step.get("position") != position or step.get("mode") != position:
        _bad("trace does not follow natural order", result)
    clicks = {j for j, bit in enumerate(prefix) if bit}
    retained, omitted = step.get("retained_clicks"), step.get("omitted_clicks")
    accepted = step.get("accepted") is True
    if accepted or not allow_failure:
        if (not isinstance(retained, list) or len(retained) > k
                or any(type(j) is not int for j in retained)
                or retained != sorted(set(retained)) or not set(retained) <= clicks):
            _bad("trace has invalid retained clicks", result)
        if omitted != sorted(clicks - set(retained)):
            _bad("trace omitted set does not complement retained clicks", result)
        selected = step.get("selection_order")
        if (not isinstance(selected, list) or len(selected) != len(retained)
                or set(selected) != set(retained)):
            _bad("selection-order trace disagrees with retained set", result)
        edges = step.get("forest_edges")
        if not isinstance(edges, list) or len(edges) != max(0, len(omitted) - 1):
            _bad("trace tree does not have a spanning edge count", result)
        parent = {j: j for j in omitted}

        def root(j):
            while parent[j] != j:
                j = parent[j]
            return j

        for edge in edges:
            if (not isinstance(edge, list) or len(edge) != 2
                    or any(type(j) is not int or j not in parent for j in edge)
                    or edge[0] >= edge[1]):
                _bad("trace has invalid physical tree edge", result)
            first, second = root(edge[0]), root(edge[1])
            if first == second:
                _bad("trace tree contains a cycle", result)
            parent[first] = second
        if result["forest"] == "chain" and edges != [list(edge) for edge in zip(omitted[:-1], omitted[1:])]:
            _bad("chain trace changed its measurement-order edge rule", result)
    probability = step.get("p_one")
    if accepted:
        if not _number(probability) or not 0 <= probability <= 1 or step.get("probability") != probability:
            _bad("trace has invalid accepted probability", result)
        if step.get("precision_used") not in ("longdouble", "quad"):
            _bad("trace has invalid precision", result)
        if result["precision_policy"] != "auto" and step["precision_used"] != result["precision_policy"]:
            _bad("trace used a precision outside the explicit policy", result)
        if step.get("refinement_attempted") and (result["precision_policy"] != "auto" or step["precision_used"] != "quad"):
            _bad("trace has invalid refinement state", result)
    elif not allow_failure or probability is not None or not step.get("error"):
        _bad("trace contains an unrecorded or unexpected failed conditional", result)
    return tuple(prefix)


def _validate_sample_evidence(result, modes, k, count, bits, logs, traced, uniforms):
    """Validate completed and partial evidence before a failed batch is raised."""
    partial = result.get("partial_sample_prefix")
    if (not isinstance(partial, list) or len(partial) > modes
            or any(type(bit) is not int or bit not in (0, 1) for bit in partial)
            or result["uniforms_consumed"] != count * modes + len(partial)):
        _bad("partial sample prefix disagrees with exact uniform consumption", result)
    for key, length in (("histogram_counts", 1 << modes if modes <= 14 else 0),
                        ("click_count_histogram", modes + 1)):
        values = result.get(key)
        if not length:
            if values is not None:
                _bad("unexpected exponentially sized histogram", result)
            continue
        if (not isinstance(values, list) or len(values) != length
                or any(type(value) is not int or value < 0 for value in values)
                or sum(values) != count):
            _bad(f"invalid completed-sample {key}", result)
    mode_counts = result.get("mode_click_counts")
    if (not isinstance(mode_counts, list) or len(mode_counts) != modes
            or any(type(value) is not int or not 0 <= value <= count for value in mode_counts)
            or sum(mode_counts) != sum(j * value for j, value in enumerate(result["click_count_histogram"]))):
        _bad("invalid completed-sample per-mode counts", result)
    if bits is not None:
        if mode_counts != bits.sum(axis=0).tolist():
            _bad("per-mode counts disagree with completed raw samples", result)
        if result["click_count_histogram"] != np.bincount(bits.sum(axis=1).astype(int), minlength=modes + 1).tolist():
            _bad("click histogram disagrees with completed raw samples", result)
        if modes <= 14:
            masks = bits.astype(np.int64) @ (1 << np.arange(modes))
            if result["histogram_counts"] != np.bincount(masks, minlength=1 << modes).tolist():
                _bad("full histogram disagrees with completed raw samples", result)
    trace = result.get("conditional_trace")
    if not traced:
        if trace is not None:
            _bad("unrequested sampling trace returned", result)
        return
    if not isinstance(trace, list) or len(trace) != result["uniforms_consumed"]:
        _bad("sampling trace does not match consumed uniforms", result)
    traced_logs = np.zeros(count, dtype=np.longdouble)
    for linear, step in enumerate(trace):
        sample, position = divmod(linear, modes)
        prefix = _validate_step(step, modes, k, result)
        if step.get("sample_index") != sample or len(prefix) != position:
            _bad("sampling trace is out of sample/order sequence", result)
        observed = bits[sample] if sample < count and bits is not None else partial if sample == count else None
        if observed is None:
            continue
        if prefix != tuple(observed[:position]):
            _bad("sampling trace prefix disagrees with completed or partial bits", result)
        probability = np.longdouble(step["p_one"])
        one = observed[position]
        if uniforms is not None and one != int(uniforms[sample, position] < probability):
            _bad("sample bit disagrees with injected uniform and actual transition", result)
        if (one and probability <= 0) or (not one and probability >= 1):
            _bad("sample follows a zero-probability traced branch", result)
        if sample < count:
            traced_logs[sample] += np.log(probability) if one else np.log1p(-probability)
    if logs is not None and bits is not None and not np.allclose(logs, traced_logs, atol=2e-12 * max(1, modes), rtol=0):
        _bad("completed sample logQ disagrees with actual transition trace", result)


def run_cpp_forest(model_or_covariance, *, max_clicks=2, forest="chain",
                   precision="auto", selection="click_influence", samples=0,
                   seed=0, order=None, uniforms=None, enumerate_distribution=False,
                   executable=None, raw_samples=True, trace_conditionals=False,
                   record_log_probabilities=False, vacuum_kernel="product",
                   prefix_cache_entries=0, profile=False, timeout=600.0):
    """Run one fresh native process and validate its complete returned evidence.

    ``distribution`` contains unnormalized successful-leaf masses and
    ``failure_mass`` is the additional symbol. Enumeration never samples first
    unless a positive ``samples`` count was explicitly requested. With tracing,
    ``conditional_trace`` describes sampled steps and ``enumeration_trace``
    describes every reached enumeration prefix, including numeric failures.
    Wrapper wall includes validation, temporary CSVs, binary hashing, process
    startup/output and response checks; native setup/sampling timers are separate.
    """
    started = perf_counter()
    validation_started = perf_counter()
    covariance, minimum_uncertainty = _validate_covariance(model_or_covariance)
    modes = covariance.shape[0] // 2
    k = _integer(max_clicks, "max_clicks", 0, 8)
    count = _integer(samples, "samples", 0)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    if forest not in ("chain", "static_mi_tree"):
        raise ValueError("forest must be chain or static_mi_tree")
    if precision not in ("longdouble", "quad", "auto"):
        raise ValueError("precision must be longdouble, quad, or auto")
    if selection not in ("click_influence", "pairwise"):
        raise ValueError("selection must be click_influence or pairwise")
    if vacuum_kernel not in ("product", "log"):
        raise ValueError("vacuum_kernel must be product or log")
    if _integer(prefix_cache_entries, "prefix_cache_entries", 0) != 0:
        raise ValueError("forest sampler requires prefix_cache_entries=0")
    for name, value in (("enumerate_distribution", enumerate_distribution),
                        ("raw_samples", raw_samples), ("trace_conditionals", trace_conditionals),
                        ("record_log_probabilities", record_log_probabilities), ("profile", profile)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    if order is not None:
        order = tuple(order)
        if (any(isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) for j in order)
                or order != tuple(range(modes))):
            raise ValueError("native forest currently requires natural measurement order")
    if enumerate_distribution and modes > 14:
        raise ValueError("full forest enumeration supports at most 14 modes")
    if not np.isscalar(timeout) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if uniforms is not None:
        if np.iscomplexobj(uniforms):
            raise ValueError("uniforms must be real")
        uniforms = np.ascontiguousarray(uniforms, dtype=float)
        if (uniforms.shape != (count, modes) or not np.all(np.isfinite(uniforms))
                or np.any(uniforms < 0) or np.any(uniforms >= 1)):
            raise ValueError("uniforms must have shape (samples,modes), in [0,1)")
    validation_seconds = perf_counter() - validation_started
    path = find_cpp_forest_executable(executable)
    if path is None:
        raise FileNotFoundError("compiled forest sampler missing; explicitly run experiments/build_batched_forest_sampler.py first")
    binary_sha = _sha256(path.read_bytes())
    metadata = _read_build_metadata(path, binary_sha)
    command = [str(path), "--modes", str(modes), "--k", str(k), "--forest", forest,
               "--precision", precision, "--selector", selection, "--samples", str(count),
               "--seed", str(seed), "--vacuum-kernel", vacuum_kernel]
    with tempfile.TemporaryDirectory(prefix="gbslab-cpp-forest-") as directory:
        input_path = Path(directory) / "covariance.csv"
        np.savetxt(input_path, covariance, delimiter=",", fmt="%.17g")
        command += ["--input", str(input_path)]
        if uniforms is not None:
            uniform_path = Path(directory) / "uniforms.csv"
            np.savetxt(uniform_path, uniforms, delimiter=",", fmt="%.17g")
            command += ["--uniforms", str(uniform_path)]
        for flag, enabled in (("--enumerate", enumerate_distribution), ("--raw-samples", raw_samples),
                              ("--trace-conditionals", trace_conditionals),
                              ("--record-log-probabilities", record_log_probabilities), ("--profile", profile)):
            if enabled:
                command.append(flag)
        process_started = perf_counter()
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=float(timeout), check=False)
        except subprocess.TimeoutExpired as error:
            raise CppForestError("native forest timed out; no retry or refill performed",
                                 executable_sha256=binary_sha) from error
        process_seconds = perf_counter() - process_started
    if _sha256(path.read_bytes()) != binary_sha:
        raise CppForestError("forest binary changed during execution", executable_sha256=binary_sha)
    try:
        result = json.loads(process.stdout, parse_constant=_invalid_json_constant)
    except (TypeError, ValueError) as error:
        raise CppForestError("native forest returned invalid finite JSON", returncode=process.returncode,
                             stderr=process.stderr, executable_sha256=binary_sha) from error
    if not isinstance(result, dict):
        raise CppForestError("native forest returned a non-object result", executable_sha256=binary_sha)
    result.update(executable_sha256=binary_sha,
                  build_cache_key=metadata.get("cache_key") if metadata else None,
                  build_metadata_matches_binary=metadata is not None,
                  covariance_sha256=_sha256(covariance.tobytes()),
                  uniforms_sha256=_sha256(uniforms.tobytes()) if uniforms is not None else None,
                  wrapper_input_minimum_uncertainty_eigenvalue=minimum_uncertainty,
                  wrapper_input_validation_seconds=validation_seconds,
                  wrapper_process_wall_seconds=process_seconds,
                  wrapper_stderr=process.stderr)
    # Argument/setup errors may precede the normal schema. Preserve them as-is.
    if "requested_samples" not in result and (process.returncode or result.get("status") != "ok"):
        result["wrapper_wall_seconds"] = perf_counter() - started
        raise CppForestError(str(result.get("error", "native forest failed before sampling")),
                             returncode=process.returncode, payload=result,
                             stderr=process.stderr, executable_sha256=binary_sha)
    expected = {"modes": modes, "k": k, "forest": forest, "precision_policy": precision,
                "selector": selection, "requested_samples": count, "seed": seed,
                "order": list(range(modes)), "vacuum_kernel": vacuum_kernel,
                "prefix_cache_entries": 0, "abi_version": 1}
    for key, value in expected.items():
        if result.get(key) != value:
            _bad(f"native forest metadata mismatch for {key}", result)
    completed, bits, logs = _validate_checkpoint(result, modes, count, raw_samples, record_log_probabilities)
    consumed = result.get("uniforms_consumed")
    if (isinstance(consumed, bool) or not isinstance(consumed, int)
            or not completed * modes <= consumed <= min(count, completed + 1) * modes):
        _bad("uniform consumption disagrees with complete/partial samples", result)
    timings = result.get("timings")
    if not isinstance(timings, dict) or not all(_number(value, nonnegative=True) for value in timings.values()):
        _bad("native timing fields are invalid", result)
    for key in ("input_seconds", "setup_seconds", "sampling_seconds", "cold_batch_seconds",
                "aggregation_seconds", "enumeration_seconds", "cold_enumeration_seconds"):
        if key not in timings:
            _bad(f"missing native timing field {key}", result)
    if abs(timings["cold_batch_seconds"] - timings["setup_seconds"] - timings["sampling_seconds"]) > 1e-9:
        _bad("cold batch timing omits setup or sampling", result)
    # Validated naming adapter to the retained sampler's transport convention.
    # Original native fields remain available, including phase-specific peaks.
    result["timings_seconds"] = {key.removesuffix("_seconds"): value
                                 for key, value in timings.items()}
    result["kernel_diagnostics"] = {
        "setup": result.get("setup_kernel_diagnostics"),
        "sampling": result.get("sampling_kernel_diagnostics"),
        "enumeration": result.get("enumeration_kernel_diagnostics"),
        "scope": "phase increments; maximum and payload fields are cumulative process values",
    }
    result["wrapper_wall_seconds"] = perf_counter() - started
    _validate_sample_evidence(result, modes, k, completed, bits, logs, trace_conditionals, uniforms)
    if process.returncode or result.get("status") != "ok":
        result["wrapper_wall_seconds"] = perf_counter() - started
        raise CppForestError(str(result.get("error", "native forest failed")),
                             returncode=process.returncode, payload=result,
                             stderr=process.stderr, executable_sha256=binary_sha)
    if completed != count or consumed != count * modes or result.get("partial_sample_prefix") != []:
        _bad("successful batch is incomplete", result)
    if enumerate_distribution:
        law = np.asarray(result.get("distribution"), dtype=float)
        failure_mass = result.get("failure_mass")
        if (result.get("enumeration_complete") is not True or law.shape != (1 << modes,)
                or not np.all(np.isfinite(law)) or np.any(law < 0)
                or not _number(failure_mass, nonnegative=True) or failure_mass > 1
                or abs(np.sum(law, dtype=np.longdouble) + failure_mass - 1) > 2e-11):
            _bad("enumerated successful law plus failure mass is invalid", result)
        failures = result.get("failures")
        if not isinstance(failures, list):
            _bad("enumerated failure records are missing", result)
        prefixes = []
        for failure in failures:
            prefix = failure.get("prefix") if isinstance(failure, dict) else None
            if (not isinstance(prefix, list) or len(prefix) > modes
                    or any(type(bit) is not int or bit not in (0, 1) for bit in prefix)
                    or not _number(failure.get("mass"), nonnegative=True) or failure["mass"] <= 0
                    or failure.get("phase") not in ("conditional", "zero_state_update")
                    or not failure.get("error")):
                _bad("invalid enumerated failure record", result)
            prefixes.append(tuple(prefix))
        ordered = sorted(prefixes)
        if any(later[:len(earlier)] == earlier for earlier, later in zip(ordered, ordered[1:])):
            _bad("failure prefixes overlap; mass would be counted twice", result)
        if abs(sum(failure["mass"] for failure in failures) - failure_mass) > 2e-11:
            _bad("failure records do not sum to failure_mass", result)
        if logs is not None and bits is not None and count:
            masks = bits.astype(np.int64) @ (1 << np.arange(modes))
            probabilities = law[masks]
            if np.any(probabilities <= 0) or not np.allclose(logs, np.log(probabilities), atol=2e-10, rtol=0):
                _bad("actual sample logQ disagrees with enumerated successful law", result)
    elif result.get("distribution") is not None or result.get("failure_mass") is not None or result.get("enumeration_complete"):
        _bad("unrequested full law was returned", result)
    if trace_conditionals:
        if enumerate_distribution:
            rows = result.get("enumeration_trace")
            if not isinstance(rows, list):
                _bad("enumeration trace is missing", result)
            seen = set()
            for row in rows:
                prefix = _validate_step(row, modes, k, result, allow_failure=True)
                if prefix in seen:
                    _bad("enumeration queried a prefix twice", result)
                seen.add(prefix)
            if () not in seen:
                _bad("enumeration trace omitted its root", result)
    elif result.get("conditional_trace") is not None or result.get("enumeration_trace") is not None:
        _bad("unrequested conditional traces were returned", result)
    result["wrapper_wall_seconds"] = perf_counter() - started
    return result
