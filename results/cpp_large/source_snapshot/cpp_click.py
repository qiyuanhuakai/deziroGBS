"""Validated process adapter for the compiled retained-click sampler.

The C++ program implements the same pairwise or click-influence retained-event
model as the Python samplers.  Compiling it does not make that approximate
model an exact generic GBS sampler.  Build explicitly with
``python experiments/build_click_sampler.py``; neither this module nor its
ordinary unit tests invoke a compiler.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTABLE = ROOT / ".cache" / "click-build" / "click_sampler"


class CppClickError(RuntimeError):
    """A compiled run failed, or its reported probabilities were invalid."""

    def __init__(self, message, *, returncode=None, payload=None, stderr="",
                 executable_sha256=None):
        super().__init__(message)
        self.returncode = returncode
        self.payload = payload
        self.stderr = stderr
        self.executable_sha256 = executable_sha256


def find_cpp_executable(executable: str | Path | None = None) -> Path | None:
    """Find an already-built executable, returning None if it is absent.

    An explicit executable path is authoritative.  No search, installation,
    download, compilation, or fallback implementation is performed.
    """
    path = Path(executable) if executable is not None else DEFAULT_EXECUTABLE
    path = path.expanduser().resolve()
    return path if path.is_file() and os.access(path, os.X_OK) else None


def _integer(value, name, minimum, maximum=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return value


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _invalid_json_constant(value):
    raise ValueError(f"nonfinite JSON constant {value}")


def _read_build_metadata(executable, binary_sha):
    path = executable.parent / "build.json"
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text(), parse_constant=_invalid_json_constant)
    except (OSError, ValueError):
        return None
    # A nearby build record is not evidence about a different binary.
    return metadata if isinstance(metadata, dict) and metadata.get("executable_sha256") == binary_sha else None


def _validate_covariance(model_or_covariance):
    supplied = getattr(model_or_covariance, "V", model_or_covariance)
    if np.iscomplexobj(supplied):
        raise ValueError("covariance must be real in xxpp quadrature order")
    covariance = np.asarray(supplied, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or covariance.shape[0] == 0 or covariance.shape[0] % 2):
        raise ValueError("covariance must be a nonempty even-dimensional square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite")
    modes = covariance.shape[0] // 2
    if modes > 256:
        raise ValueError("compiled sampler supports at most 256 modes")
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(covariance, covariance.T, atol=1e-10 * scale, rtol=0):
        raise ValueError("covariance must be symmetric")
    covariance = np.ascontiguousarray(covariance * 0.5 + covariance.T * 0.5)
    # This is input validation, never complete target-distribution fitting.
    eye, zero = np.eye(modes), np.zeros((modes, modes))
    omega = np.block([[zero, eye], [-eye, zero]])
    minimum = float(np.linalg.eigvalsh(covariance + 1j * omega)[0])
    if minimum < -1e-9 * scale:
        raise ValueError(f"covariance violates the Gaussian uncertainty relation: {minimum:g}")
    return covariance, minimum


def _check_probability_array(values, length, name):
    try:
        probability = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CppClickError(f"{name} must be a numeric probability array") from exc
    if (probability.shape != (length,) or not np.all(np.isfinite(probability))
            or np.any(probability < 0) or np.any(probability > 1)
            or abs(float(np.sum(probability, dtype=np.longdouble)) - 1.0) > 1e-10):
        raise CppClickError(f"{name} failed finite, range, shape, or normalization checks")
    return probability


def _check_count_array(values, length, count, name):
    try:
        numbers = np.asarray(values)
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CppClickError(f"{name} must contain integer counts") from exc
    if (numbers.shape != (length,) or not np.all(np.isfinite(numeric))
            or np.any(numeric < 0) or np.any(numeric != np.floor(numeric))
            or sum(int(value) for value in numbers) != count):
        raise CppClickError(f"{name} has invalid counts or total")
    return numbers


def run_cpp_click(
    model_or_covariance,
    *,
    selection: str = "pairwise",
    max_clicks: int = 2,
    samples: int = 0,
    seed: int = 0,
    order=None,
    uniforms=None,
    enumerate_distribution: bool = False,
    executable: str | Path | None = None,
    raw_samples: bool = True,
    trace_conditionals: bool = False,
    profile: bool = False,
    timeout: float = 600.0,
) -> dict:
    """Run the compiled sampler and validate its JSON result.

    Covariances use hbar=2, vacuum I, and xxpp order.  Uniforms must have shape
    (samples, modes), with columns in *measurement* order, and entries in
    [0,1).  The sampled bit is exactly ``uniform < p_one``.  Returned sample
    columns and distribution mask bits use *physical* mode indices; mode 0
    is the least significant distribution bit.  Enumerating Q is an explicit
    optional operation limited to 14 modes and never used to generate draws.

    Seed reproducibility is internal to the C++ generator; the same numeric
    seed does not promise the same stream as NumPy.  Use explicit uniforms
    for paired trajectory comparisons.

    Returned values are JSON-compatible.  ``wrapper_wall_seconds`` includes
    input validation, temporary CSV I/O, executable hashing and process work;
    ``timings_seconds`` contains the program's own component measurements.
    No compiler is invoked.  A nonzero exit, numerical failure, malformed
    result or invalid probability raises CppClickError, without retrying.
    """
    started = perf_counter()
    validation_start = perf_counter()
    covariance, minimum_uncertainty = _validate_covariance(model_or_covariance)
    modes = covariance.shape[0] // 2
    if selection not in ("pairwise", "click_influence"):
        raise ValueError("selection must be 'pairwise' or 'click_influence'")
    k = _integer(max_clicks, "max_clicks", 0, 8)
    count = _integer(samples, "samples", 0)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    for name, flag in (("enumerate_distribution", enumerate_distribution),
                       ("raw_samples", raw_samples),
                       ("trace_conditionals", trace_conditionals), ("profile", profile)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    if enumerate_distribution and modes > 14:
        raise ValueError("full Q enumeration supports at most 14 modes")
    if not np.isscalar(timeout) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    order = tuple(range(modes)) if order is None else tuple(order)
    if (len(order) != modes
            or any(isinstance(j, (bool, np.bool_))
                   or not isinstance(j, (int, np.integer)) for j in order)
            or set(order) != set(range(modes))):
        raise ValueError("order must be a permutation of physical mode indices")
    order = tuple(int(j) for j in order)
    if uniforms is not None:
        if np.iscomplexobj(uniforms):
            raise ValueError("uniforms must be real")
        uniforms = np.ascontiguousarray(uniforms, dtype=float)
        if (uniforms.shape != (count, modes) or not np.all(np.isfinite(uniforms))
                or np.any(uniforms < 0) or np.any(uniforms >= 1)):
            raise ValueError("uniforms must have shape (samples,modes) with values in [0,1)")
    validation_seconds = perf_counter() - validation_start
    path = find_cpp_executable(executable)
    if path is None:
        requested = Path(executable) if executable is not None else DEFAULT_EXECUTABLE
        raise FileNotFoundError(
            f"compiled click sampler missing or not executable at {requested}; "
            "run python experiments/build_click_sampler.py explicitly first")
    binary_sha = _sha256(path.read_bytes())
    build_metadata = _read_build_metadata(path, binary_sha)
    command = [str(path), "--modes", str(modes), "--selector", selection,
               "--k", str(k), "--samples", str(count), "--seed", str(seed),
               "--order", ",".join(map(str, order))]
    with tempfile.TemporaryDirectory(prefix="gbslab-cpp-click-") as directory:
        covariance_path = Path(directory) / "covariance.csv"
        np.savetxt(covariance_path, covariance, delimiter=",", fmt="%.17g")
        command += ["--input", str(covariance_path)]
        if uniforms is not None:
            uniform_path = Path(directory) / "uniforms.csv"
            np.savetxt(uniform_path, uniforms, delimiter=",", fmt="%.17g")
            command += ["--uniforms", str(uniform_path)]
        for flag, enabled in (("--enumerate", enumerate_distribution),
                              ("--raw-samples", raw_samples),
                              ("--trace-conditionals", trace_conditionals),
                              ("--profile", profile)):
            if enabled:
                command.append(flag)
        process_start = perf_counter()
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=float(timeout), check=False)
        except subprocess.TimeoutExpired as exc:
            raise CppClickError("compiled click sampler timed out; no retry performed",
                                executable_sha256=binary_sha) from exc
        process_seconds = perf_counter() - process_start
    if _sha256(path.read_bytes()) != binary_sha:
        raise CppClickError("compiled binary changed during the run",
                            executable_sha256=binary_sha)
    try:
        result = json.loads(completed.stdout, parse_constant=_invalid_json_constant)
    except (TypeError, ValueError) as exc:
        raise CppClickError("compiled click sampler did not return finite JSON",
                            returncode=completed.returncode, stderr=completed.stderr,
                            executable_sha256=binary_sha) from exc
    if (completed.returncode or not isinstance(result, dict)
            or result.get("status") != "ok"):
        error = result.get("error", "compiled click sampler failed") if isinstance(result, dict) else "compiled click sampler failed"
        raise CppClickError(str(error), returncode=completed.returncode,
                            payload=result, stderr=completed.stderr,
                            executable_sha256=binary_sha)
    expected_metadata = {"modes": modes, "selector": selection, "k": k,
                         "sample_count": count, "seed": seed, "order": list(order)}
    for key, expected in expected_metadata.items():
        if result.get(key) != expected:
            raise CppClickError(f"compiled result metadata mismatch for {key}",
                                payload=result, executable_sha256=binary_sha)
    if enumerate_distribution:
        _check_probability_array(result.get("distribution"), 1 << modes, "distribution")
    elif result.get("distribution") is not None:
        raise CppClickError("unrequested full distribution was returned")
    returned_samples = result.get("raw_samples")
    if raw_samples and returned_samples is None:
        raise CppClickError("requested raw_samples are missing")
    if returned_samples is not None:
        bit_array = np.asarray(returned_samples)
        if count == 0 and bit_array.shape == (0,):
            bit_array = bit_array.reshape(0, modes)
        if (bit_array.shape != (count, modes)
                or not np.all((bit_array == 0) | (bit_array == 1))):
            raise CppClickError("raw_samples are not binary or have the wrong shape")
    histogram = result.get("histogram_counts")
    if histogram is not None:
        if modes > 14:
            raise CppClickError("unexpected exponentially sized histogram above 14 modes")
        _check_count_array(histogram, 1 << modes, count, "histogram_counts")
        if returned_samples is not None:
            masks = np.asarray(bit_array, dtype=np.int64) @ (1 << np.arange(modes))
            expected_histogram = np.bincount(masks, minlength=1 << modes)
            if not np.array_equal(expected_histogram, histogram):
                raise CppClickError("sample histogram disagrees with raw physical-mode bits")
    click_histogram = result.get("click_count_histogram")
    if click_histogram is not None:
        _check_count_array(click_histogram, modes + 1, count, "click_count_histogram")
        if returned_samples is not None:
            expected_histogram = np.bincount(np.asarray(bit_array, dtype=np.int64).sum(axis=1),
                                            minlength=modes + 1)
            if not np.array_equal(expected_histogram, click_histogram):
                raise CppClickError("click-count histogram disagrees with raw_samples")
    mode_counts = result.get("mode_click_counts")
    if mode_counts is not None:
        values = np.asarray(mode_counts, dtype=float)
        if (values.shape != (modes,) or not np.all(np.isfinite(values))
                or np.any(values < 0) or np.any(values > count)
                or np.any(values != np.floor(values))):
            raise CppClickError("mode_click_counts has invalid counts")
        if click_histogram is not None and int(values.sum()) != sum(
                j * int(n) for j, n in enumerate(click_histogram)):
            raise CppClickError("mode_click_counts disagrees with click-count histogram")
        if returned_samples is not None and not np.array_equal(values, bit_array.sum(axis=0)):
            raise CppClickError("mode_click_counts disagrees with raw_samples")
    for section in ("diagnostics", "setup_diagnostics", "enumeration_diagnostics"):
        diagnostics = result.get(section)
        if diagnostics is None:
            continue
        if not isinstance(diagnostics, dict):
            raise CppClickError(f"{section} must be a JSON object")
        for key in ("minimum_p_one", "maximum_p_one", "min_p_one", "max_p_one",
                    "minimum_click_probability", "maximum_click_probability"):
            value = diagnostics.get(key)
            if value is not None and (not isinstance(value, (int, float))
                    or not np.isfinite(value) or not 0 <= value <= 1):
                raise CppClickError(f"reported {key} is not a valid conditional probability")
    trace = result.get("conditional_trace")
    if trace_conditionals and trace is None:
        raise CppClickError("requested conditional trace is missing")
    if trace is not None:
        if not isinstance(trace, list) or len(trace) != count*modes:
            raise CppClickError("conditional trace has the wrong length")
        for linear, step in enumerate(trace):
            sample_index, position = divmod(linear, modes)
            mode = order[position]
            if (not isinstance(step, dict) or step.get("sample_index") != sample_index
                    or step.get("position") != position or step.get("mode") != mode):
                raise CppClickError("conditional trace does not follow the requested sample/order")
            probability = step.get("p_one")
            if (not isinstance(probability, (int, float)) or not np.isfinite(probability)
                    or not 0 <= probability <= 1):
                raise CppClickError("conditional trace contains an invalid probability")
            retained = step.get("retained")
            if (not isinstance(retained, list) or len(retained) > k
                    or any(not isinstance(j, int) for j in retained)
                    or len(set(retained)) != len(retained)
                    or not set(retained).issubset(order[:position])):
                raise CppClickError("conditional trace has an invalid retained-click set")
            if returned_samples is not None:
                if any(bit_array[sample_index, j] != 1 for j in retained):
                    raise CppClickError("conditional trace retained a previously unclicked mode")
                if uniforms is not None and bit_array[sample_index, mode] != int(
                        uniforms[sample_index, position] < probability):
                    raise CppClickError("sample bits disagree with the traced uniform threshold rule")
    result.update(
        executable_sha256=binary_sha,
        build_cache_key=build_metadata.get("cache_key") if build_metadata else None,
        build_metadata_matches_binary=build_metadata is not None,
        covariance_sha256=_sha256(covariance.tobytes()),
        uniforms_sha256=_sha256(uniforms.tobytes()) if uniforms is not None else None,
        wrapper_input_minimum_uncertainty_eigenvalue=minimum_uncertainty,
        wrapper_input_validation_seconds=validation_seconds,
        wrapper_process_wall_seconds=process_seconds,
        wrapper_wall_seconds=perf_counter() - started,
        wrapper_stderr=completed.stderr,
    )
    return result


class _UniformSequence:
    """A deliberately minimal scalar RNG interface for paired validation."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float).ravel()
        self.position = 0

    def random(self):
        if self.position >= len(self._values):
            raise AssertionError("sampler consumed more than one uniform per mode")
        value = self._values[self.position]
        self.position += 1
        return float(value)


def _python_sampler(covariance, selection, k, order):
    from .conditional_memory import SchurConditionalMemorySampler
    from .exact import ThresholdModel
    from .schur_memory import SchurMemorySampler

    model = ThresholdModel(covariance, max_exact_modes=3, vacuum_cache_size=8192)
    cls = SchurMemorySampler if selection == "pairwise" else SchurConditionalMemorySampler
    return cls(model, k, selection=selection, order=order)


def _compact_marginal(distribution, physical_modes):
    masks = np.arange(len(distribution), dtype=np.int64)
    indices = np.zeros_like(masks)
    for bit, mode in enumerate(physical_modes):
        indices |= ((masks >> mode) & 1) << bit
    return np.bincount(indices, weights=distribution, minlength=1 << len(physical_modes))


def _validation_fixtures(sample_count):
    from .gaussian import covariance_from_interferometer, haar_unitary, local_unitary
    from .paired_stress import dense_target, paired_covariance

    fixtures = []

    def add(name, covariance, k, selection, *, order=None, expected=None,
            target_exact=False, stress=False, count=None, boundary_uniforms=False,
            enumerate_law=True):
        modes = len(covariance) // 2
        number = sample_count if count is None else count
        seed = 811000 + len(fixtures)
        uniforms = np.random.default_rng(seed).random((number, modes))
        if boundary_uniforms:
            uniforms[0] = 0.0
            uniforms[1] = np.nextafter(1.0, 0.0)
            if name.startswith("strict_tmsv"):
                # Opposite uniforms on A/B specifically probe forbidden
                # outcomes after a partner has been observed, not just RNG
                # endpoints along pair-consistent all-zero/all-one paths.
                half = modes // 2
                uniforms[0, :half] = np.nextafter(1.0, 0.0)
                uniforms[0, half:] = 0.0
                uniforms[1, :half] = 0.0
                uniforms[1, half:] = np.nextafter(1.0, 0.0)
                if number >= 4:
                    uniforms[2] = 0.0
                    uniforms[3] = np.nextafter(1.0, 0.0)
        fixtures.append(dict(name=name, covariance=covariance, k=k, selection=selection,
            order=tuple(range(modes)) if order is None else tuple(order),
            expected=expected, target_exact=target_exact, stress=stress,
            uniforms=uniforms, seed=seed, enumerate_law=enumerate_law))

    selections = ("pairwise", "click_influence")
    for selection in selections:
        expected = np.zeros(1 << 8); expected[0] = 1.0
        add("vacuum_8", np.eye(16), 4, selection, expected=expected,
            boundary_uniforms=True)
        photons = np.linspace(.25, 1.4, 8)
        covariance = np.diag(np.tile(1.0 + 2.0 * photons, 2))
        bits = (np.arange(1 << 8)[:, None] >> np.arange(8)) & 1
        p_one = photons / (1.0 + photons)
        expected = np.prod(np.where(bits, p_one, 1.0-p_one), axis=1)
        for k in (2, 3, 4):
            add("thermal_nonidentical_8", covariance, k, selection,
                order=(6, 0, 5, 1, 7, 2, 4, 3), expected=expected)
        for modes, seed, squeezing, eta in (
            (8, 8101, .8, .65),
            (10, 8102, np.linspace(.45, .95, 10), np.linspace(.3, .85, 10)),
        ):
            covariance = covariance_from_interferometer(
                haar_unitary(modes, np.random.default_rng(seed)), squeezing, eta)
            order = tuple(range(modes)) if modes == 8 else (7, 2, 9, 0, 5, 1, 8, 3, 6, 4)
            for k in (2, 3, 4):
                add(f"haar_{modes}_seed{seed}", covariance, k, selection, order=order)
        for modes, k in ((3, 2), (4, 3), (5, 4)):
            covariance = covariance_from_interferometer(
                haar_unitary(modes, np.random.default_rng(8200+modes)),
                np.linspace(.65, .95, modes), np.linspace(.5, .8, modes))
            add(f"fully_conditioned_{modes}", covariance, k, selection,
                order=tuple(reversed(range(modes))), target_exact=True)
        # Fixed regressions from the formal cold suite.  Sparse local optics
        # expose zero/weak unconditional MI ties whose retained candidates
        # have different predictive effects after other observations.
        for depth, squeezing, heterogeneous, identifier in (
            (2, .4, False, "b2b9ae4c30d7745a"),
            (2, .8, False, "a71ab6f032848177"),
            (4, .4, False, "22c3418bd5ab4f76"),
            (4, .8, True, "b7e6861062dd571a"),
        ):
            U = local_unitary(10, depth, np.random.default_rng(5101))
            r, eta = np.full(10, squeezing), np.full(10, .5)
            if heterogeneous:
                profile_rng = np.random.default_rng(5101+80123)
                r *= profile_rng.uniform(.55, 1.45, 10)
                eta = np.clip(eta+profile_rng.uniform(-.25, .25, 10), .05, .95)
            covariance = covariance_from_interferometer(U, r, eta)
            add(f"local{depth}_regression_{identifier}", covariance, 4, selection)
            # Force the old mode9/prefix466 disagreement, instead of relying
            # on a small random batch to happen to visit this rare history.
            prefix = (0, 1, 0, 0, 1, 0, 1, 1, 1)
            fixtures[-1]["uniforms"][0, :9] = [0.0 if bit else np.nextafter(1.0, 0.0)
                                                for bit in prefix]
            fixtures[-1]["uniforms"][0, 9] = .5
            fixtures[-1]["archived_case_id"] = identifier
        for k in (2, 4):
            add("strict_tmsv_4pairs", paired_covariance(4, .5), k, selection,
                expected=dense_target(4, .5), stress=True, boundary_uniforms=True)
        add("weak_thermal_adversarial", np.eye(8)*(1.0+2.0e-8), 2, selection,
            stress=True, count=max(2, min(sample_count, 8)), boundary_uniforms=True)
        # Cross the 64-bit mask boundary without requesting exponential output.
        photons = np.linspace(.2, .8, 65)
        covariance = np.diag(np.tile(1.0 + 2.0*photons, 2))
        add("thermal_65_sampling_only", covariance, 2, selection,
            count=8, enumerate_law=False, order=tuple(reversed(range(65))))
    add("vacuum_256_sampling_only", np.eye(512), 2, "pairwise",
        count=2, enumerate_law=False, boundary_uniforms=True)
    return fixtures


def validate_cpp_click(*, executable=None, sample_count=64, archive_directory=None):
    """Run a fixed, deterministic integration suite against one binary hash.

    Ordinary fixtures must produce valid outputs, agree with the Python
    model's full Q and identical-uniform trajectories, and have the correct
    physical first k+1 measurement marginals.  Fully conditioned, vacuum and
    thermal fixtures also have complete physical-target comparisons.

    Strict-support and weak-event cases are reported separately.  An explicit
    numerical guard is an observed limitation, never a successful probability
    computation; a returned invalid probability is a validation failure.
    """
    from .exact import NumericalProbabilityError, ThresholdModel

    count = _integer(sample_count, "sample_count", 2)
    path = find_cpp_executable(executable)
    if path is None:
        raise FileNotFoundError("build the click sampler explicitly before integration validation")
    binary_sha = _sha256(path.read_bytes())
    build_metadata = _read_build_metadata(path, binary_sha)
    archive = Path(archive_directory) if archive_directory is not None else None
    if archive is not None:
        if archive.exists() and any(archive.iterdir()):
            raise FileExistsError("refusing to overwrite a nonempty validation archive")
        archive.mkdir(parents=True, exist_ok=True)
    tolerances = {"full_q_tv": 2e-9, "full_q_max_abs": 2e-10,
                  "physical_prefix_max_abs": 2e-10, "analytic_target_tv": 2e-9,
                  "paired_bit_mismatches": 0}
    reference_sources = {name: _sha256((Path(__file__).parent / name).read_bytes()) for name in
        ("cpp_click.py", "schur_memory.py", "conditional_memory.py", "zero_memory.py",
         "exact.py", "gaussian.py", "paired_stress.py")}
    fixtures = _validation_fixtures(count)
    if archive is not None:
        snapshot = archive / "source_snapshot" / "gbslab"
        snapshot.mkdir(parents=True)
        for name, expected_sha in reference_sources.items():
            source = (Path(__file__).parent / name).read_bytes()
            if _sha256(source) != expected_sha:
                raise RuntimeError("Python validation source changed before snapshot")
            (snapshot / name).write_bytes(source)
        protocol = {"executable_sha256": binary_sha, "build_metadata": build_metadata,
            "started_utc": datetime.now(timezone.utc).isoformat(), "tolerances": tolerances,
            "python_reference_sources_sha256": reference_sources,
            "fixtures": [{"fixture": f["name"], "modes": len(f["covariance"])//2,
                "selection": f["selection"], "k": f["k"], "order": f["order"],
                "seed": f["seed"], "samples": len(f["uniforms"]), "stress": f["stress"],
                "covariance_sha256": _sha256(f["covariance"].tobytes()),
                "uniforms_sha256": _sha256(f["uniforms"].tobytes())} for f in fixtures]}
        (archive / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False)+"\n")
    rows = []
    start_all = perf_counter()
    for index, fixture in enumerate(fixtures):
        V, uniforms = fixture["covariance"], fixture["uniforms"]
        modes, number = len(V)//2, len(uniforms)
        row = {"fixture": fixture["name"], "modes": modes,
               "selection": fixture["selection"], "k": fixture["k"],
               "order": list(fixture["order"]), "seed": fixture["seed"],
               "samples": number, "stress_case": fixture["stress"],
               "enumerated_q": fixture["enumerate_law"],
               "archived_case_id": fixture.get("archived_case_id"),
               "input_covariance_sha256": _sha256(np.ascontiguousarray(V).tobytes()),
               "input_uniforms_sha256": _sha256(uniforms.tobytes())}
        arrays = {"covariance": V, "uniforms": uniforms,
                  "order": np.asarray(fixture["order"], dtype=int)}
        errors = []
        cpp = python_samples = python_q = None
        try:
            cpp = run_cpp_click(V, selection=fixture["selection"], max_clicks=fixture["k"],
                samples=number, seed=fixture["seed"], order=fixture["order"], uniforms=uniforms,
                enumerate_distribution=fixture["enumerate_law"], executable=path,
                trace_conditionals=True)
            if cpp["executable_sha256"] != binary_sha:
                raise RuntimeError("binary changed within integration validation")
            row["cpp_status"] = "ok"
            arrays["cpp_samples"] = np.asarray(cpp["raw_samples"], dtype=np.uint8)
            if fixture["enumerate_law"]:
                arrays["cpp_distribution"] = np.asarray(cpp["distribution"], dtype=float)
            # Large arrays live in the reproducible case artifact when requested.
            row["cpp_result"] = {key: value for key, value in cpp.items()
                                 if key not in ("distribution", "raw_samples", "histogram_counts")}
        except CppClickError as exc:
            row.update(cpp_status="failure", cpp_error=str(exc),
                       cpp_returncode=exc.returncode, cpp_error_payload=exc.payload,
                       cpp_stderr=exc.stderr)
            is_numerical_guard = (exc.returncode not in (None, 0)
                and isinstance(exc.payload, dict)
                and exc.payload.get("error_type") == "numerical_probability")
            row["cpp_numerical_guard"] = is_numerical_guard
            if not fixture["stress"] or not is_numerical_guard:
                errors.append("C++ failed or returned an invalid result")
        sampler = _python_sampler(V, fixture["selection"], fixture["k"], fixture["order"])
        stream = _UniformSequence(uniforms)
        try:
            python_samples = sampler.sample(number, stream)
            if stream.position != number * modes:
                raise AssertionError("Python did not consume exactly one uniform per mode")
            row["python_sampling_status"] = "ok"
            arrays["python_samples"] = python_samples
        except NumericalProbabilityError as exc:
            row.update(python_sampling_status="numerical_guard", python_sampling_error=str(exc),
                       python_uniforms_consumed=stream.position)
            if not fixture["stress"]:
                errors.append("ordinary Python fixture triggered a numerical guard")
        if fixture["enumerate_law"]:
            try:
                python_q = sampler.full_distribution()
                arrays["python_distribution"] = python_q
                row["python_enumeration_status"] = "ok"
            except NumericalProbabilityError as exc:
                row.update(python_enumeration_status="numerical_guard", python_enumeration_error=str(exc))
                if not fixture["stress"]:
                    errors.append("ordinary Python full Q triggered a numerical guard")
        row["python_diagnostics"] = sampler.cache_info()
        if cpp is not None and python_samples is not None:
            mismatches = int(np.count_nonzero(arrays["cpp_samples"] != python_samples))
            row["paired_bit_mismatches"] = mismatches
            if mismatches:
                errors.append("identical uniforms produced different sample bits")
        if cpp is not None and python_q is not None:
            difference = np.abs(arrays["cpp_distribution"] - python_q)
            row.update(full_q_tv=float(difference.sum()/2), full_q_max_abs=float(difference.max()))
            if row["full_q_tv"] > tolerances["full_q_tv"] or row["full_q_max_abs"] > tolerances["full_q_max_abs"]:
                errors.append("C++ and Python full Q disagree")
        if cpp is not None and fixture["enumerate_law"]:
            # The first k+1 outcomes cannot yet omit any preceding clicks.
            prefix = fixture["order"][:min(fixture["k"]+1, modes)]
            physical_prefix = ThresholdModel(V).marginal(prefix)
            q_prefix = _compact_marginal(arrays["cpp_distribution"], prefix)
            error = float(np.max(np.abs(q_prefix - physical_prefix)))
            row.update(physical_prefix_modes=list(prefix), physical_prefix_max_abs=error)
            if error > tolerances["physical_prefix_max_abs"]:
                errors.append("first k+1 measurement outcomes disagree with physical target")
            expected = fixture["expected"]
            if fixture["target_exact"]:
                expected = ThresholdModel(V).full_distribution()
            if expected is not None:
                arrays["physical_target"] = expected
                difference = float(np.abs(arrays["cpp_distribution"] - expected).sum()/2)
                row["physical_full_tv"] = difference
                row["off_target_support_probability"] = float(arrays["cpp_distribution"][expected == 0].sum())
                if difference > tolerances["analytic_target_tv"]:
                    errors.append("known exact full-target fixture disagrees")
                if modes <= 14:
                    masks = arrays["cpp_samples"].astype(np.int64) @ (1 << np.arange(modes))
                    unsupported = int(np.count_nonzero(expected[masks] == 0))
                    row["off_target_support_samples"] = unsupported
                    if unsupported:
                        errors.append("an exact-support fixture produced a forbidden sample")
        if cpp is not None and not fixture["enumerate_law"]:
            if fixture["name"].startswith("thermal"):
                photons = (np.diag(V)[:modes] - 1.0)/2.0
                physical_probabilities = photons/(1.0+photons)
                expected = np.zeros((number, modes), dtype=np.uint8)
                for position, mode in enumerate(fixture["order"]):
                    expected[:, mode] = uniforms[:, position] < physical_probabilities[mode]
            else:
                expected = np.zeros((number, modes), dtype=np.uint8)
            analytic_mismatches = int(np.count_nonzero(arrays["cpp_samples"] != expected))
            row["analytic_paired_bit_mismatches"] = analytic_mismatches
            if analytic_mismatches:
                errors.append("large-mode analytic trajectory disagrees")
        row["errors"] = errors
        row["status"] = ("failed" if errors else "numerical_guard_observed" if
            row.get("cpp_status") != "ok" or row.get("python_sampling_status") != "ok"
            or row.get("python_enumeration_status", "ok") != "ok" else "passed")
        if archive is not None:
            artifact = archive / f"case_{index:02d}_{fixture['name']}_{fixture['selection']}_k{fixture['k']}.npz"
            np.savez_compressed(artifact, **arrays)
            row.update(artifact_path=str(artifact.resolve()),
                       artifact_relative_path=str(artifact.relative_to(archive.parent)),
                       artifact_sha256=_sha256(artifact.read_bytes()))
            with (archive / "cases.jsonl").open("a") as journal:
                journal.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row)
    if _sha256(path.read_bytes()) != binary_sha:
        raise RuntimeError("binary changed during integration validation")
    failed = sum(row["status"] == "failed" for row in rows)
    return {
        "schema_version": 1, "passed": failed == 0,
        "status": "passed" if failed == 0 else "failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "executable_sha256": binary_sha, "build_metadata": build_metadata,
        "python": platform.python_version(), "numpy": np.__version__,
        "validator_sha256": _sha256(Path(__file__).read_bytes()),
        "python_reference_sources_sha256": reference_sources,
        "tolerances": tolerances, "sample_count_per_ordinary_case": count,
        "cases": rows, "case_count": len(rows), "failed_case_count": failed,
        "passed_case_count": sum(row["status"] == "passed" for row in rows),
        "ordinary_case_count": sum(not row["stress_case"] for row in rows),
        "ordinary_passed_case_count": sum(not row["stress_case"] and row["status"] == "passed" for row in rows),
        "stress_case_count": sum(row["stress_case"] for row in rows),
        "guarded_stress_case_count": sum(row["status"] == "numerical_guard_observed" for row in rows),
        "strict_support_certified": False,
        "elapsed_seconds": perf_counter() - start_all,
        "qualification": "Floating-point numerical equivalence tests, not an interval-arithmetic proof. Strict-support numerical guards are limitations, not successful evaluations. No generic physical-target equality is claimed for truncated k.",
        "cost_note": "Validation enables full-law enumeration and conditional tracing; these timings must not be used as sampling performance benchmarks.",
    }


def validate_cpp_support_stress(reference_validation, *, executable=None):
    """Inspect sampling independently when later full-Q guards hid its output.

    Reuses the exact covariance, order and uniforms in an existing validation
    archive.  These are deterministic diagnostic reruns, never replacement iid
    samples for a confidence bound.  The binary must be the same one validated
    in the supplied reference record.
    """
    reference_path = Path(reference_validation)
    reference = json.loads(reference_path.read_text())
    path = find_cpp_executable(executable)
    if path is None:
        raise FileNotFoundError("compiled click sampler is absent")
    binary_sha = _sha256(path.read_bytes())
    if binary_sha != reference["executable_sha256"]:
        raise ValueError("support diagnostic must use the reference validation binary")
    records = []
    for original in reference["cases"]:
        if not original["stress_case"]:
            continue
        original_artifact = Path(original["artifact_path"])
        relative = original.get("artifact_relative_path",
            str(Path(original_artifact.parent.name) / original_artifact.name))
        relocated_artifact = reference_path.parent / relative
        artifact = relocated_artifact if relocated_artifact.is_file() else original_artifact
        if _sha256(artifact.read_bytes()) != original["artifact_sha256"]:
            raise ValueError("reference case artifact hash changed")
        with np.load(artifact) as data:
            covariance, uniforms = data["covariance"], data["uniforms"]
            python_samples = data["python_samples"] if "python_samples" in data.files else None
        row = {key: original[key] for key in ("fixture", "modes", "selection", "k",
            "order", "seed", "samples", "artifact_path", "artifact_sha256",
            "python_sampling_status")}
        row["original_cpp_failure_phase"] = original.get("cpp_error_payload", {}).get("context", {}).get("phase")
        row["python_sampling_error"] = original.get("python_sampling_error")
        errors = []
        try:
            result = run_cpp_click(covariance, selection=original["selection"],
                max_clicks=original["k"], samples=original["samples"], seed=original["seed"],
                order=original["order"], uniforms=uniforms, executable=path,
                enumerate_distribution=False, trace_conditionals=True)
            row["cpp_status"] = "ok"
            row["cpp_result"] = result
            cpp_samples = np.asarray(result["raw_samples"], dtype=np.uint8)
            if python_samples is not None:
                row["paired_bit_mismatches"] = int(np.count_nonzero(cpp_samples != python_samples))
                # A difference in exact-support boundary cases is itself a
                # numeric limitation, not evidence about generic noisy inputs.
            if original["fixture"].startswith("strict_tmsv"):
                half = original["modes"]//2
                unsupported = np.flatnonzero(np.any(cpp_samples[:, :half] != cpp_samples[:, half:], axis=1))
                row["cpp_off_support_sample_indices"] = unsupported.tolist()
                row["cpp_off_support_samples"] = cpp_samples[unsupported].tolist()
                if len(unsupported):
                    errors.append("C++ returned a forbidden exact-support sample")
                if python_samples is not None:
                    python_unsupported = np.flatnonzero(np.any(
                        python_samples[:, :half] != python_samples[:, half:], axis=1))
                    row["python_off_support_sample_indices"] = python_unsupported.tolist()
                    row["python_off_support_samples"] = python_samples[python_unsupported].tolist()
            row["status"] = "failed" if errors else "sampling_passed"
        except CppClickError as exc:
            row.update(cpp_status="failure", cpp_error=str(exc),
                cpp_error_payload=exc.payload, cpp_returncode=exc.returncode)
            numerical = (exc.returncode not in (None, 0) and isinstance(exc.payload, dict)
                         and exc.payload.get("error_type") == "numerical_probability")
            if not numerical:
                errors.append("C++ returned an invalid result or nonnumerical failure")
            row["status"] = "failed" if errors else "numerical_guard_observed"
        row["errors"] = errors
        records.append(row)
    if _sha256(path.read_bytes()) != binary_sha:
        raise RuntimeError("binary changed during support diagnostics")
    failed = sum(row["status"] == "failed" for row in records)
    return {"schema_version": 1, "passed": failed == 0,
        "status": "passed" if failed == 0 else "failed",
        "reference_validation": str(reference_path.resolve()),
        "reference_validation_sha256": _sha256(reference_path.read_bytes()),
        "executable_sha256": binary_sha, "validator_sha256": _sha256(Path(__file__).read_bytes()),
        "cases": records, "case_count": len(records), "failed_case_count": failed,
        "sampling_passed_case_count": sum(row["status"] == "sampling_passed" for row in records),
        "guarded_stress_case_count": sum(row["status"] == "numerical_guard_observed" for row in records),
        "qualification": "Separate deterministic sampling-only diagnostics using the original archived uniforms; not replacement samples or a statistical certificate. Exact-zero support remains a numerical risk outside checked histories."}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--support-from", type=Path,
                      help="inspect sampling-only support on the stress cases of a prior validation JSON")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("refusing to overwrite an earlier validation artifact")
    support_snapshot = (args.output.with_name(args.output.stem + "_validator.py")
                        if args.output is not None and args.support_from is not None else None)
    if support_snapshot is not None and support_snapshot.exists():
        parser.error("refusing to overwrite an earlier support validator snapshot")
    archive = args.output.parent / (args.output.stem + "_cases") if args.output is not None else None
    if args.validate:
        result = validate_cpp_click(executable=args.executable, sample_count=args.samples,
                                    archive_directory=archive)
    else:
        result = validate_cpp_support_stress(args.support_from, executable=args.executable)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        if support_snapshot is not None:
            support_snapshot.write_bytes(Path(__file__).read_bytes())
        print(json.dumps({key: result[key] for key in ("status", "case_count", "failed_case_count",
            "passed_case_count", "sampling_passed_case_count", "guarded_stress_case_count",
            "executable_sha256", "elapsed_seconds") if key in result}))
    else:
        print(encoded, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
