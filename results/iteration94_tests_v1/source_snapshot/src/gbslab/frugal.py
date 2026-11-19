"""Local adapter for the pinned, official FrugalGBS CPU implementation.

The optional enumerated law is instrumentation, not an upstream API. Its
probabilities follow the upstream SIMD arithmetic and the saturating random
threshold rule. It is not the exact physical GBS target, nor an empirical
sample histogram. No target probability vector is passed to FrugalGBS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


def frugal_sample(
    model_or_covariance,
    cutoff: int,
    sample_count: int = 0,
    seed: int = 0,
    *,
    enumerate_distribution: bool = True,
    executable: str | Path | None = None,
    threads: int = 1,
    timeout: float = 600.0,
) -> dict:
    """Run official samples and optionally enumerate its induced law.

    The input is a ``ThresholdModel`` or a physical, zero-displacement Gaussian
    covariance in xxpp order, hbar=2 (vacuum covariance I). Mode zero is the
    least significant mask bit in both output arrays. Uniform and nonuniform
    loss are both supported through the covariance. The adapter uses no fit
    data beyond this covariance and the selected cumulant cutoff.

    ``histogram_counts`` always contains 2**m integer counts from unmodified
    upstream samples. ``distribution`` is either a length-2**m probability
    list or None. ``timings_seconds`` separates official layout, target and
    sampling work from enumeration, histogram construction and process I/O.
    The wrapper returns JSON-compatible types so results can be archived.

    Build once using ``python experiments/build_frugal.py``. The default
    binary supports m in {2,4,8,10,12}; pass --modes to the builder for other
    small dimensions. Building is deliberately outside timed experiments.
    """
    covariance = np.asarray(getattr(model_or_covariance, "V", model_or_covariance), dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or covariance.shape[0] == 0 or covariance.shape[0] % 2):
        raise ValueError("covariance must be a nonempty even-dimensional square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    modes = covariance.shape[0] // 2
    for name, value in (("cutoff", cutoff), ("sample_count", sample_count),
                        ("seed", seed), ("threads", threads)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer")
    if not 1 <= cutoff <= min(5, modes):
        raise ValueError("cutoff must lie in [1,min(5,modes)]")
    if sample_count < 0 or not 0 <= seed <= 2**32 - 1 or not 1 <= threads <= 1024:
        raise ValueError("invalid sample count, unsigned 32-bit seed, or thread count")
    if modes > 16:
        raise ValueError("this full-law research adapter supports at most 16 modes")
    if executable is None:
        executable = Path(__file__).resolve().parents[2] / ".cache/frugal-build/frugal_adapter"
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(
            f"FrugalGBS adapter is missing at {executable}; "
            "run python experiments/build_frugal.py first"
        )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gbslab-frugal-") as directory:
        path = Path(directory) / "covariance.csv"
        np.savetxt(path, covariance, delimiter=",", fmt="%.17g")
        command = [str(executable), "--input", str(path), "--modes", str(modes),
                   "--cutoff", str(cutoff), "--samples", str(sample_count),
                   "--seed", str(seed), "--threads", str(threads)]
        if enumerate_distribution:
            command.append("--enumerate")
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "FrugalGBS adapter failed")
    result = json.loads(completed.stdout)
    result["wrapper_wall_seconds"] = time.perf_counter() - started
    if result.get("modes") != modes or result.get("cutoff") != cutoff:
        raise RuntimeError("FrugalGBS result metadata does not match the request")
    if sum(result["histogram_counts"]) != sample_count:
        raise RuntimeError("FrugalGBS sample histogram has the wrong count")
    if enumerate_distribution:
        probability = np.asarray(result["distribution"], dtype=float)
        if (probability.shape != (2**modes,) or not np.all(np.isfinite(probability))
                or np.any(probability < 0.0) or abs(float(probability.sum()) - 1.0) > 1e-11):
            raise RuntimeError("FrugalGBS induced law failed basic probability checks")
        if result["trace"]["nonfinite_threshold_path_steps"]:
            raise FloatingPointError("FrugalGBS trace encountered a nonfinite branch threshold")
    return result


def validate_adapter(*, executable: str | Path | None = None,
                     sample_count: int = 200_000) -> dict:
    """Reproducible analytic fixtures and official-sampler/trace crosschecks.

    This is a validation utility, never a fitting step. Histogram agreement
    uses an explicit finite-sample TV bound; empirical TV is not presented as
    an exact comparison with the physical target.
    """
    from .exact import ThresholdModel
    from .gaussian import covariance_from_interferometer, haar_unitary

    if sample_count < 1:
        raise ValueError("validation sample_count must be positive")
    if executable is None:
        executable = Path(__file__).resolve().parents[2] / ".cache/frugal-build/frugal_adapter"
    executable = Path(executable).resolve()
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    build_metadata_path = executable.parent / "build.json"
    build_metadata = (json.loads(build_metadata_path.read_text())
                      if build_metadata_path.is_file() else {})
    build_cache_key = (build_metadata.get("cache_key")
                       if build_metadata.get("executable_sha256") == executable_sha256
                       else None)
    fixtures = []
    for modes, cutoff in ((2, 2), (4, 4), (8, 2), (8, 5), (10, 5), (12, 5)):
        expected = np.zeros(2**modes)
        expected[0] = 1.0
        fixtures.append((f"vacuum_{modes}", np.eye(2*modes), cutoff, 1024, expected))
    modes = 8
    photons = np.linspace(0.1, 1.3, modes)
    thermal = np.diag(np.tile(1.0 + 2.0 * photons, 2))
    masks = np.arange(2**modes)
    bits = (masks[:, None] >> np.arange(modes)) & 1
    p_one = photons / (1.0 + photons)
    thermal_law = np.prod(np.where(bits, p_one, 1.0-p_one), axis=1)
    for cutoff in (2, 5):
        fixtures.append(("thermal_nonidentical_8", thermal, cutoff,
                         sample_count, thermal_law))
    ordinary = covariance_from_interferometer(
        haar_unitary(8, np.random.default_rng(20032)), 0.8, loss=0.7)
    for cutoff in (2, 3, 4, 5):
        fixtures.append(("haar_8_r0.8_eta0.7_seed20032", ordinary, cutoff,
                         sample_count, None))
    strong = covariance_from_interferometer(
        haar_unitary(8, np.random.default_rng(20033)), 1.3, loss=1.0)
    fixtures.append(("haar_8_r1.3_eta1_seed20033", strong, 2, sample_count, None))
    clipped = covariance_from_interferometer(
        haar_unitary(8, np.random.default_rng(4001)), 1.0, loss=1.0)
    fixtures.append(("haar_8_clipping_seed4001", clipped, 5, sample_count, None))
    nonuniform = covariance_from_interferometer(
        haar_unitary(10, np.random.default_rng(20034)),
        np.linspace(0.35, 1.1, 10), loss=np.linspace(0.2, 0.9, 10))
    fixtures.append(("haar_10_nonuniform_seed20034", nonuniform, 5,
                     sample_count, None))
    rows = []
    delta = 1e-6
    for index, (name, covariance, cutoff, count, expected) in enumerate(fixtures):
        result = frugal_sample(covariance, cutoff, count, seed=8400+index,
                               executable=executable, threads=1)
        if "clipping" in name and not result["trace"]["saturated_threshold_path_steps"]:
            raise AssertionError("Clipping fixture did not exercise threshold saturation")
        q = np.asarray(result["distribution"])
        counts = np.asarray(result["histogram_counts"])
        empirical_tv = float(0.5 * np.abs(counts / count - q).sum())
        # E[TV(Qhat,Q)] <= 1/2 sum_x sqrt(Q(x)(1-Q(x))/count).
        # Changing one draw changes TV by <=1/count; McDiarmid adds
        # sqrt(log(1/delta)/(2 count)). This is deliberately conservative.
        expectation_bound = 0.5 * float(np.sqrt(q * (1.0-q) / count).sum())
        high_probability_bound = min(
            1.0, expectation_bound + np.sqrt(np.log(1.0/delta) / (2*count)))
        off_support = int(counts[q == 0.0].sum())
        if off_support or empirical_tv > high_probability_bound:
            raise AssertionError(f"Official samples disagree with trace: {name}, K={cutoff}")
        first_two = np.bincount(np.arange(q.size) & 3, weights=q, minlength=4)
        physical_first_two = ThresholdModel(covariance).marginal((0, 1))
        prefix_error = float(np.max(np.abs(first_two - physical_first_two)))
        if prefix_error > 1e-10:
            raise AssertionError(f"Two-mode prefix/covariance convention mismatch: {name}")
        row = {
            "fixture": name, "modes": covariance.shape[0]//2, "cutoff": cutoff,
            "sample_count": count, "seed": 8400+index,
            "q_sum": float(q.sum()), "q_min": float(q.min()),
            "empirical_tv_to_trace": empirical_tv,
            "empirical_tv_upper_bound": float(high_probability_bound),
            "bound_failure_probability": delta,
            "off_support_samples": off_support,
            "two_mode_prefix_max_error": prefix_error,
            "trace": result["trace"], "timings_seconds": result["timings_seconds"],
        }
        if expected is not None:
            expected_tv = float(0.5 * np.abs(q - expected).sum())
            row["analytic_law_tv"] = expected_tv
            if expected_tv > 1e-10:
                raise AssertionError(f"Analytic fixture failed: {name}, K={cutoff}")
        rows.append(row)
    if hashlib.sha256(executable.read_bytes()).hexdigest() != executable_sha256:
        raise RuntimeError("Executable changed during validation; rerun against a stable build")
    return {"passed": True, "validation_cases": rows,
            "executable_sha256": executable_sha256, "build_cache_key": build_cache_key,
            "histogram_bound": "multinomial expectation bound plus McDiarmid, per case",
            "sampling": "unmodified upstream generate_samples; one OpenMP thread",
            "trace": "generated upstream SIMD body, forced branch choices",
            "upstream_commit": "fa8b796f2afebd4845a2e335e457735952617ef8"}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true",
                        help="run analytic and official-sampler/trace checks")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.validate:
        parser.error("use --validate, or import frugal_sample from Python")
    result = validate_adapter(executable=args.executable, sample_count=args.samples)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"Passed {len(result['validation_cases'])} checks; wrote {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
