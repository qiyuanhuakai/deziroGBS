"""Point-probability Monte Carlo upper bounds for complete-distribution TV.

For normalized target P and sampler Q,

    TV(P,Q) = E_{X~Q}[(1 - P(X)/Q(X))_+].

This identity still holds when P has mass outside Q's support.  Replacing the
integrand by one on high-click or unresolved records is conservative.  A
Hoeffding margin yields a one-sided statistical bound for a sampler fixed
before iid validation samples are drawn.  No complete target enumeration is
needed.

The default Gaussian point oracle uses floating-point arithmetic.  Its values
are estimates, NOT verified lower bounds.  Consequently the statistical
statement is conditional on correct numerical probabilities.  Callers may
supply a target probability lower-bound callback, but this module still does
not implement outward-rounded log/ratio/confidence arithmetic.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import json
import math
from time import perf_counter
from typing import Callable

import numpy as np

from .exact import NumericalProbabilityError
from .schur_memory import GaussianVacuumConditioner, gaussian_vacuum_probability


_NUMERICAL_FAILURES = (ArithmeticError, ValueError, np.linalg.LinAlgError)


def tv_score_from_log_probabilities(log_p_lower: float, log_q: float) -> float:
    """Return (1 - p_lower/q)_+ on a point with q>0, using stable log ratios.

    A genuine target lower bound gives a conservative TV score.  Substituting
    a point estimate instead makes that guarantee depend on its correctness.
    This floating-point function is not an interval-arithmetic enclosure.
    """
    log_p_lower, log_q = float(log_p_lower), float(log_q)
    if not math.isfinite(log_q) or log_q > 0:
        raise ValueError("log_q must be finite and at most zero")
    if math.isnan(log_p_lower) or log_p_lower > 0:
        raise ValueError("log target lower probability must be at most zero")
    if log_p_lower == -math.inf:
        return 1.0
    if log_p_lower >= log_q:
        return 0.0
    return float(-math.expm1(log_p_lower - log_q))


class GaussianThresholdPointOracle:
    """Evaluate individual zero-mean Gaussian threshold pattern probabilities.

    On a pattern with click set C and zero set Z, condition the Gaussian state
    on Z=0 and evaluate P(C=1 | Z=0) by 2**|C| small vacuum terms.  Multiply by
    P(Z=0) in log space.  The original no-click factor requires one potentially
    large determinant; Gaussian conditioning has its own reported cost.

    The click cap prevents accidentally exponential work at high count.  The
    all-click inclusion-exclusion has an explicit relative cancellation guard.
    This guard is diagnostic, not a rigorous numerical error certificate.
    """

    def __init__(self, covariance, max_exact_clicks, *, max_relative_roundoff=1e-6,
                 covariance_cache_size=128, covariance_cache_bytes=64 * 1024**2):
        if (not isinstance(max_exact_clicks, (int, np.integer))
                or max_exact_clicks < 0 or max_exact_clicks > 20):
            raise ValueError("Gaussian point-oracle click cap must be between 0 and 20")
        if not np.isfinite(max_relative_roundoff) or max_relative_roundoff <= 0:
            raise ValueError("max_relative_roundoff must be positive finite")
        self.conditioner = GaussianVacuumConditioner(
            covariance, cache_size=covariance_cache_size,
            max_cache_bytes=covariance_cache_bytes)
        self.V = self.conditioner.V
        self.m = self.conditioner.m
        self.max_exact_clicks = int(max_exact_clicks)
        self.max_relative_roundoff = float(max_relative_roundoff)
        self.diagnostics = {
            "point_evaluations": 0,
            "large_no_click_determinants": 0,
            "maximum_large_determinant_dimension": 0,
            "small_vacuum_determinants": 0,
            "maximum_small_determinant_dimension": 0,
            "maximum_all_click_terms": 0,
            "maximum_relative_cancellation_indicator": 0.0,
            "log_probability_roundoff_corrections": 0,
        }

    def log_probability(self, sample) -> float:
        sample = np.asarray(sample)
        if sample.shape != (self.m,) or not np.all((sample == 0) | (sample == 1)):
            raise ValueError("sample must be a binary vector of length m")
        clicks = tuple(int(j) for j in np.flatnonzero(sample))
        if len(clicks) > self.max_exact_clicks:
            raise ValueError("pattern exceeds point-oracle click cap")
        zeros = tuple(int(j) for j in np.flatnonzero(1 - sample))
        self.diagnostics["point_evaluations"] += 1

        if zeros:
            indices = np.asarray(zeros + tuple(j + self.m for j in zeros))
            q = (self.V[np.ix_(indices, indices)] + np.eye(len(indices))) * 0.5
            sign, logdet = np.linalg.slogdet(q)
            self.diagnostics["large_no_click_determinants"] += 1
            self.diagnostics["maximum_large_determinant_dimension"] = max(
                self.diagnostics["maximum_large_determinant_dimension"], len(indices))
            if sign <= 0 or not np.isfinite(logdet):
                raise NumericalProbabilityError("nonpositive/nonfinite no-click determinant")
            log_vacuum = -0.5 * float(logdet)
            if log_vacuum > 2e-10:
                raise NumericalProbabilityError("no-click probability exceeds one")
            if log_vacuum > 0:
                log_vacuum = 0.0
                self.diagnostics["log_probability_roundoff_corrections"] += 1
        else:
            log_vacuum = 0.0
        if not clicks:
            return log_vacuum

        W = self.conditioner.subcovariance(zeros, clicks)
        count = 1 << len(clicks)
        terms = np.empty(count, dtype=np.longdouble)
        terms[0] = 1.0
        signs = np.fromiter(((-1 if mask.bit_count() % 2 else 1)
                             for mask in range(count)), dtype=np.int8, count=count)
        for mask in range(1, count):
            local = [j for j in range(len(clicks)) if mask & (1 << j)]
            quadratures = np.asarray(local + [j + len(clicks) for j in local])
            terms[mask] = gaussian_vacuum_probability(W[np.ix_(quadratures, quadratures)])
            self.diagnostics["small_vacuum_determinants"] += 1
            self.diagnostics["maximum_small_determinant_dimension"] = max(
                self.diagnostics["maximum_small_determinant_dimension"], len(quadratures))
        all_click = np.sum(signs * terms, dtype=np.longdouble)
        absolute_indicator = (8.0 * np.finfo(float).eps * max(1, self.m)
                              * np.sum(np.abs(terms), dtype=np.longdouble))
        self.diagnostics["maximum_all_click_terms"] = max(
            self.diagnostics["maximum_all_click_terms"], count)
        if all_click <= 0 or not np.isfinite(all_click):
            # Even a genuinely impossible target point can safely receive
            # worst score one.  Do not pretend to resolve a floating zero.
            raise NumericalProbabilityError("zero, negative or unresolved all-click event")
        relative_indicator = float(absolute_indicator / all_click)
        self.diagnostics["maximum_relative_cancellation_indicator"] = max(
            self.diagnostics["maximum_relative_cancellation_indicator"], relative_indicator)
        if relative_indicator > self.max_relative_roundoff:
            raise NumericalProbabilityError(
                f"all-click cancellation indicator {relative_indicator:.3g} exceeds "
                f"{self.max_relative_roundoff:.3g}")
        log_probability = log_vacuum + math.log(float(all_click))
        if not math.isfinite(log_probability) or log_probability > 2e-10:
            raise NumericalProbabilityError("invalid log threshold probability")
        if log_probability > 0:
            log_probability = 0.0
            self.diagnostics["log_probability_roundoff_corrections"] += 1
        return float(log_probability)

    def cache_info(self):
        return {**self.diagnostics, "gaussian_conditioner": self.conditioner.cache_info()}


def _binary_samples(samples, n, m):
    result = np.asarray(samples)
    if result.shape != (n, m) or not np.all((result == 0) | (result == 1)):
        raise ValueError(f"sampler must return binary shape ({n}, {m})")
    return result.astype(np.uint8, copy=True)


def _log_sequential_probability(sampler, sample, order):
    prefix = []
    terms = []
    for mode in order:
        p_one = float(sampler.conditional_click_probability(tuple(prefix)))
        if not math.isfinite(p_one) or p_one < 0 or p_one > 1:
            raise NumericalProbabilityError("invalid sampler conditional probability")
        bit = int(sample[mode])
        if bit:
            if p_one == 0:
                raise NumericalProbabilityError("sample has zero sequential Q probability")
            terms.append(math.log(p_one))
        else:
            if p_one == 1:
                raise NumericalProbabilityError("sample has zero sequential Q probability")
            terms.append(math.log1p(-p_one))
        prefix.append(bit)
    return math.fsum(terms)


def validate_tv(
    model, sampler, n, max_exact_clicks, alpha, rng=None, *,
    samples=None,
    probability_lower_bound: Callable[[np.ndarray], float] | None = None,
    keep_scores=False, pattern_cache_size=8192, max_relative_roundoff=1e-6,
    covariance_cache_size=128, covariance_cache_bytes=64 * 1024**2,
) -> dict:
    """Estimate a conservative one-sided global TV bound without full targets.

    Parameters beyond the required model/sampler/sample count:

    ``max_exact_clicks``:
        Scores of records above this cutoff are one; no target or Q density
        needs to be evaluated for those records.
    ``probability_lower_bound``:
        Optional callable of a physical-mode binary sample returning a valid
        lower bound on P(sample).  With this callback ``model`` may be None or
        omit V.  The caller owns the lower-bound claim.  Without it, Gaussian
        point estimates are used and numerical correctness is an assumption.
    ``samples``:
        Optional pre-generated array of n iid draws from this fixed Q.  Their
        iid provenance is the caller's responsibility.  No draws are retried.
    ``keep_scores``:
        Include per-record scores and click counts for diagnostics.

    A sampling failure returns ``status='sampling_failed', upper=None``.  It is
    not repaired by discarding or replacing failed draws.  Density-evaluation
    failures on successfully obtained samples receive the conservative score
    one and remain in the denominator n.

    ``upper`` has nominal coverage 1-alpha for this single fixed comparison,
    conditional on correct Q probabilities and correct target probabilities
    or lower bounds.  It is not simultaneous over many comparisons, does not
    validate post-selection of the best tested sampler, and does not certify
    floating-point arithmetic.  Bonferroni alpha allocation can give a fixed
    family of comparisons a simultaneous statistical guarantee.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(max_exact_clicks, (int, np.integer)) or max_exact_clicks < 0:
        raise ValueError("max_exact_clicks must be nonnegative")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not isinstance(pattern_cache_size, (int, np.integer)) or pattern_cache_size < 0:
        raise ValueError("pattern_cache_size must be nonnegative")
    m = int(sampler.m)
    if m <= 0:
        raise ValueError("sampler must have a positive number of modes")
    if model is not None and int(model.m) != m:
        raise ValueError("target and sampler mode counts differ")
    order = tuple(int(j) for j in getattr(sampler, "order", tuple(range(m))))
    if sorted(order) != list(range(m)):
        raise ValueError("sampler order must be a permutation")
    if not callable(getattr(sampler, "conditional_click_probability", None)):
        raise ValueError("sampler must expose its sequential conditional probabilities")
    if probability_lower_bound is not None and not callable(probability_lower_bound):
        raise ValueError("probability_lower_bound must be callable")
    n, max_exact_clicks = int(n), int(max_exact_clicks)
    start_total = perf_counter()
    record = {
        "status": "running", "n_requested": n, "n_observed": 0,
        "modes": m, "max_exact_clicks": max_exact_clicks,
        "alpha": float(alpha), "nominal_confidence": 1.0 - float(alpha),
        "confidence_scope": "one fixed target/sampler/configuration; not simultaneous",
        "provided_samples": samples is not None,
        "target_probability_mode": ("caller_supplied_lower_bound" if probability_lower_bound
                                    is not None else "floating_gaussian_point_estimate"),
        "strict_floating_point_certificate": False,
        "qualification": (
            "Statistical coverage assumes iid draws from fixed normalized Q and correct "
            "Q point probabilities, together with correct target probabilities or valid "
            "target lower bounds. No outward-rounded numerical arithmetic is provided."),
        "support_note": "P may have mass outside Q support; Q-sampled points must have positive Q mass.",
        "cutoff_count": 0, "q_probability_failure_count": 0,
        "target_probability_failure_count": 0, "successfully_evaluated_count": 0,
        "unique_q_probability_evaluations": 0,
        "unique_target_probability_evaluations": 0,
        "pattern_cache_hits": 0, "pattern_cache_evictions": 0,
        "sample_seconds": 0.0, "q_probability_seconds": 0.0,
        "target_probability_seconds": 0.0, "oracle_initialization_seconds": 0.0,
        "mean_y": None, "statistical_margin": math.sqrt(math.log(1.0 / alpha) / (2.0 * n)),
        "upper": None,
    }
    if probability_lower_bound is None:
        if model is None or not hasattr(model, "V"):
            raise ValueError("Gaussian point evaluation requires model.V or a lower-bound callback")
        start = perf_counter()
        oracle = GaussianThresholdPointOracle(
            model.V, min(max_exact_clicks, m), max_relative_roundoff=max_relative_roundoff,
            covariance_cache_size=covariance_cache_size,
            covariance_cache_bytes=covariance_cache_bytes)
        record["oracle_initialization_seconds"] = perf_counter() - start
    else:
        oracle = None
    if samples is None:
        rng = np.random.default_rng() if rng is None else rng
        if hasattr(rng, "bit_generator"):
            record["rng_state_before"] = json.loads(json.dumps(
                rng.bit_generator.state,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else int(value)))
        start = perf_counter()
        try:
            observed = _binary_samples(sampler.sample(n, rng), n, m)
        except _NUMERICAL_FAILURES as exc:
            record.update(status="sampling_failed", sample_seconds=perf_counter()-start,
                          sampling_error_type=type(exc).__name__, sampling_error=str(exc),
                          elapsed_seconds=perf_counter()-start_total)
            return record
        record["sample_seconds"] = perf_counter() - start
    else:
        observed = _binary_samples(samples, n, m)
    record["n_observed"] = n
    scores = np.ones(n, dtype=float)
    click_counts = observed.sum(axis=1).astype(int)
    cache = OrderedDict()
    reasons = Counter()

    for index, sample in enumerate(observed):
        if click_counts[index] > max_exact_clicks:
            record["cutoff_count"] += 1
            continue
        key = np.packbits(sample, bitorder="little").tobytes()
        previous = cache.get(key)
        if previous is not None:
            cache.move_to_end(key)
            record["pattern_cache_hits"] += 1
            value, kind, reason = previous
        else:
            kind, reason, value = "evaluated", None, 1.0
            start = perf_counter()
            record["unique_q_probability_evaluations"] += 1
            try:
                log_q = _log_sequential_probability(sampler, sample, order)
            except _NUMERICAL_FAILURES as exc:
                kind, reason = "q_failure", f"{type(exc).__name__}: {exc}"
            finally:
                record["q_probability_seconds"] += perf_counter() - start
            if kind == "evaluated":
                start = perf_counter()
                record["unique_target_probability_evaluations"] += 1
                try:
                    if oracle is not None:
                        log_p = oracle.log_probability(sample)
                    else:
                        lower = float(probability_lower_bound(sample))
                        if not math.isfinite(lower) or lower < 0 or lower > 1:
                            raise ValueError("target probability lower bound must be in [0,1]")
                        log_p = -math.inf if lower == 0 else math.log(lower)
                    value = tv_score_from_log_probabilities(log_p, log_q)
                except _NUMERICAL_FAILURES as exc:
                    kind, reason, value = "target_failure", f"{type(exc).__name__}: {exc}", 1.0
                finally:
                    record["target_probability_seconds"] += perf_counter() - start
            if pattern_cache_size:
                if len(cache) >= pattern_cache_size:
                    cache.popitem(last=False)
                    record["pattern_cache_evictions"] += 1
                cache[key] = (value, kind, reason)
        scores[index] = value
        if kind == "q_failure":
            record["q_probability_failure_count"] += 1
        elif kind == "target_failure":
            record["target_probability_failure_count"] += 1
        else:
            record["successfully_evaluated_count"] += 1
        if reason is not None:
            reasons[f"{kind}: {reason}"] += 1

    mean_y = float(np.mean(scores))
    record.update(
        status="ok", mean_y=mean_y,
        upper=min(1.0, mean_y + record["statistical_margin"]),
        score_min=float(scores.min()), score_max=float(scores.max()),
        pattern_cache_entries=len(cache), failure_reasons=dict(reasons),
        target_oracle_diagnostics=oracle.cache_info() if oracle is not None else None,
        elapsed_seconds=perf_counter()-start_total,
    )
    if keep_scores:
        record["scores"] = scores.tolist()
        record["click_counts"] = click_counts.tolist()
    return record
