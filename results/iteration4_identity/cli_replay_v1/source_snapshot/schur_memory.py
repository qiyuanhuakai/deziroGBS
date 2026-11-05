"""Equivalent retained-click probabilities after Gaussian vacuum conditioning.

The approximation is still exactly the one in ``zero_memory``: retain every
past zero outcome and a selected budget of past clicks.  This module changes
how its partial-event probabilities are computed, not which target it means.

For hbar=2, xxpp covariance V and zero-observed modes Z, the normalized state
of the remaining modes is Gaussian with covariance

    W = V_RR - V_RZ (V_ZZ + I)**(-1) V_ZR.

The common probability P(Z=0) cancels in every retained-event ratio.  Only
vacuum determinants on the retained clicks and next mode are then needed.
Cached one-zero-smaller states permit rank-two Gaussian updates.  Both these
updates and the Schur complement identity are established Gaussian/numerical
techniques; this implementation makes no novelty or speedup claim.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from .exact import NumericalProbabilityError
from .zero_memory import ZeroMemorySampler


def _covariance_array(value, *, allow_empty: bool) -> np.ndarray:
    original = np.asarray(value)
    if np.iscomplexobj(original) and np.any(np.imag(original) != 0):
        raise ValueError("quadrature covariance must be real")
    covariance = np.asarray(np.real(original), dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or covariance.shape[0] % 2 or (not allow_empty and not covariance.size)):
        raise ValueError("covariance must be an even-dimensional square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite")
    if covariance.size:
        scale = max(1.0, float(np.max(np.abs(covariance))))
        if not np.allclose(covariance, covariance.T, atol=1e-10 * scale, rtol=0):
            raise ValueError("covariance must be symmetric")
    return covariance


def gaussian_vacuum_probability(covariance, *, tolerance=2e-10) -> float:
    """Return the zero-mean Gaussian vacuum probability of all supplied modes.

    The input uses hbar=2 and xxpp order.  Empty covariance means no constraints
    and returns one.  This checks numerical shape/positivity but does not run a
    full physicality eigensolve; callers are responsible for physical input.
    """
    covariance = _covariance_array(covariance, allow_empty=True)
    if not covariance.size:
        return 1.0
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    return _vacuum_probability_validated(covariance, tolerance)


def _vacuum_probability_validated(covariance, tolerance=2e-10):
    """Inner kernel for already checked covariance/principal submatrices."""
    q = (covariance + np.eye(covariance.shape[0])) * 0.5
    sign, logdet = np.linalg.slogdet(q)
    if sign <= 0 or not np.isfinite(logdet):
        raise NumericalProbabilityError("conditioned vacuum determinant is not positive finite")
    probability = float(np.exp(-0.5 * logdet))
    if probability > 1.0 + tolerance:
        raise NumericalProbabilityError("conditioned vacuum probability exceeds one")
    return min(1.0, probability)


def conditional_click_from_covariance(
    covariance, *, max_relative_roundoff=1e-6, conditioning_size=0
) -> tuple[float, dict]:
    """Compute P(last mode clicks | all preceding supplied modes click).

    ``covariance`` is ordered as (retained clicks..., target) in xxpp form.
    It may already be conditioned on an arbitrary number of zero observations;
    their common event probability is absent from this normalized Gaussian.
    The function uses all 2**s vacuum subsets, s = retained clicks + 1.

    Returns ``(click_probability, diagnostics)``.  A zero or numerically
    unresolved retained event raises instead of manufacturing a conditional.
    ``conditioning_size`` accounts for earlier zero projections in a heuristic
    dimension factor for the cancellation guard.  The indicator is NOT a
    rigorous floating-point or distributional error bound.
    """
    covariance = _covariance_array(covariance, allow_empty=False)
    if (not np.isfinite(max_relative_roundoff) or max_relative_roundoff <= 0):
        raise ValueError("max_relative_roundoff must be finite and positive")
    if (not isinstance(conditioning_size, (int, np.integer)) or conditioning_size < 0):
        raise ValueError("conditioning_size must be a nonnegative integer")
    modes = covariance.shape[0] // 2
    if modes > 20:
        raise ValueError("retained-event enumeration exceeds 20 supplied modes")
    count = 1 << modes
    vacuum = np.empty(count, dtype=np.longdouble)
    vacuum[0] = 1.0
    max_dimension = 0
    for mask in range(1, count):
        indices = [j for j in range(modes) if mask & (1 << j)]
        quadratures = np.asarray(indices + [j + modes for j in indices], dtype=int)
        submatrix = covariance[np.ix_(quadratures, quadratures)]
        # This is a principal submatrix of the already validated covariance;
        # repeating shape/finiteness/symmetry scans would dominate tiny k.
        vacuum[mask] = _vacuum_probability_validated(submatrix)
        max_dimension = max(max_dimension, len(quadratures))
    half = count >> 1
    signs = np.fromiter(((-1 if mask.bit_count() % 2 else 1)
                         for mask in range(half)), dtype=np.int8, count=half)
    denominator = np.sum(vacuum[:half] * signs, dtype=np.longdouble)
    numerator = np.sum(vacuum[half:] * signs, dtype=np.longdouble)
    error_factor = (8.0 * np.finfo(float).eps
                    * max(1, int(conditioning_size) + modes))
    den_indicator = error_factor * np.sum(np.abs(vacuum[:half]), dtype=np.longdouble)
    num_indicator = error_factor * np.sum(np.abs(vacuum[half:]), dtype=np.longdouble)
    event_corrections = 0
    if numerator < 0:
        if -numerator > num_indicator:
            raise NumericalProbabilityError("negative conditioned event exceeds roundoff indicator")
        numerator = np.longdouble(0)
        event_corrections += 1
    if denominator <= 0:
        raise NumericalProbabilityError(
            "retained-click conditioning event has zero or numerically unresolved probability")
    p_zero = numerator / denominator
    indicator = float((num_indicator + abs(p_zero) * den_indicator) / denominator)
    if not np.isfinite(p_zero) or not np.isfinite(indicator):
        raise NumericalProbabilityError("nonfinite conditioned-event ratio")
    if indicator > max_relative_roundoff:
        raise NumericalProbabilityError(
            f"conditioned-event cancellation indicator {indicator:.3g} exceeds "
            f"{max_relative_roundoff:.3g}")
    probability_corrections = 0
    if p_zero < 0 or p_zero > 1:
        deviation = float(max(-p_zero, p_zero - 1))
        if deviation > indicator:
            raise NumericalProbabilityError("conditioned no-click probability outside [0,1]")
        p_zero = np.clip(p_zero, 0, 1)
        probability_corrections += 1
    return float(1.0 - p_zero), {
        "determinant_evaluations": count - 1,
        "vacuum_probability_calls": count,
        "maximum_determinant_dimension": max_dimension,
        "inclusion_terms_per_event": half,
        "roundoff_indicator": indicator,
        "conditioning_probability": float(denominator),
        "conditioning_probability_reference": "P(retained clicks | zero observations)",
        "event_corrections": event_corrections,
        "probability_corrections": probability_corrections,
    }


@dataclass(frozen=True)
class _ConditionedState:
    modes: tuple[int, ...]
    covariance: np.ndarray
    positions: np.ndarray

    @property
    def payload_bytes(self):
        return self.covariance.nbytes + self.positions.nbytes


class GaussianVacuumConditioner:
    """Bounded cache of Gaussian covariances conditioned on zero outcomes.

    The base covariance is always retained.  ``cache_size`` caps additional
    states and ``max_cache_bytes`` caps their NumPy payloads (covariances and
    index arrays); bounded Python key/tuple overhead is reported separately by
    the entry count, not represented as an exact process-memory measurement.

    A cached state differing by one zero observation permits an O(r**2)
    rank-two update for r remaining modes.  A cache miss without such a parent
    uses a Cholesky Schur complement of the original covariance.  The latter
    can cost O(m**3).  Arbitrary measurement orders are supported.
    """

    def __init__(self, covariance, *, cache_size=128, max_cache_bytes=64 * 1024**2):
        value = _covariance_array(covariance, allow_empty=False)
        if not isinstance(cache_size, (int, np.integer)) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        if not isinstance(max_cache_bytes, (int, np.integer)) or max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be a nonnegative integer")
        self.V = ((value + value.T) * 0.5).copy()
        self.V.setflags(write=False)
        self.m = self.V.shape[0] // 2
        self.cache_size = int(cache_size)
        self.max_cache_bytes = int(max_cache_bytes)
        self._base = self._state(tuple(range(self.m)), self.V)
        self._cache: OrderedDict[tuple[int, ...], _ConditionedState] = OrderedDict()
        self._cache_bytes = 0
        self.diagnostics = {
            "requests": 0, "base_returns": 0, "cache_hits": 0, "cache_misses": 0,
            "parent_cache_reuses": 0, "rank_two_updates": 0,
            "direct_factorizations": 0, "maximum_direct_factorization_dimension": 0,
            "cache_evictions": 0, "peak_cache_entries": 0,
            "peak_cache_payload_bytes": 0, "oversized_states_not_cached": 0,
            "maximum_asymmetry_correction": 0.0,
        }

    def _indices(self, values, *, name):
        values = tuple(values)
        if any(not isinstance(j, (int, np.integer)) for j in values):
            raise ValueError(f"{name} must contain integer mode indices")
        modes = tuple(int(j) for j in values)
        if len(set(modes)) != len(modes):
            raise ValueError(f"{name} cannot contain duplicate modes")
        if any(j < 0 or j >= self.m for j in modes):
            raise ValueError(f"{name} contains an out-of-range mode")
        return modes

    def _state(self, modes, covariance):
        covariance.setflags(write=False)
        positions = np.full(self.m, -1, dtype=np.intp)
        positions[list(modes)] = np.arange(len(modes))
        positions.setflags(write=False)
        return _ConditionedState(tuple(modes), covariance, positions)

    def _finish_covariance(self, value):
        if not np.all(np.isfinite(value)):
            raise NumericalProbabilityError("nonfinite Gaussian conditional covariance")
        asymmetry = float(np.max(np.abs(value - value.T))) if value.size else 0.0
        self.diagnostics["maximum_asymmetry_correction"] = max(
            self.diagnostics["maximum_asymmetry_correction"], asymmetry * 0.5)
        return (value + value.T) * 0.5

    def _from_parent(self, parent, removed):
        n = len(parent.modes)
        local = int(parent.positions[removed])
        if local < 0:
            raise ValueError("cannot project an already removed mode")
        remaining = tuple(mode for mode in parent.modes if mode != removed)
        self.diagnostics["rank_two_updates"] += 1
        if not remaining:
            return self._state((), np.empty((0, 0)))
        local_keep = [j for j in range(n) if j != local]
        kept = np.asarray(local_keep + [j + n for j in local_keep])
        measured = np.asarray([local, local + n])
        matrix = parent.covariance
        block = matrix[np.ix_(kept, kept)]
        cross = matrix[np.ix_(kept, measured)]
        measured_covariance = matrix[np.ix_(measured, measured)] + np.eye(2)
        try:
            factor = np.linalg.cholesky(measured_covariance)
            transformed = solve_triangular(factor, cross.T, lower=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise NumericalProbabilityError("Gaussian vacuum update is not positive definite") from exc
        conditional = self._finish_covariance(block - transformed.T @ transformed)
        return self._state(remaining, conditional)

    def _direct(self, zeros):
        zero_set = set(zeros)
        remaining = tuple(mode for mode in range(self.m) if mode not in zero_set)
        if not remaining:
            return self._state((), np.empty((0, 0)))
        retained = np.asarray(remaining + tuple(j + self.m for j in remaining))
        measured = np.asarray(zeros + tuple(j + self.m for j in zeros))
        block = self.V[np.ix_(retained, retained)]
        cross = self.V[np.ix_(retained, measured)]
        measured_covariance = self.V[np.ix_(measured, measured)] + np.eye(len(measured))
        try:
            factor = np.linalg.cholesky(measured_covariance)
            transformed = solve_triangular(factor, cross.T, lower=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise NumericalProbabilityError("Gaussian vacuum projection is not positive definite") from exc
        self.diagnostics["direct_factorizations"] += 1
        self.diagnostics["maximum_direct_factorization_dimension"] = max(
            self.diagnostics["maximum_direct_factorization_dimension"], len(measured))
        conditional = self._finish_covariance(block - transformed.T @ transformed)
        return self._state(remaining, conditional)

    def _remember(self, zeros, state):
        if not self.cache_size or not self.max_cache_bytes:
            return
        if state.payload_bytes > self.max_cache_bytes:
            self.diagnostics["oversized_states_not_cached"] += 1
            return
        while (len(self._cache) >= self.cache_size
               or self._cache_bytes + state.payload_bytes > self.max_cache_bytes):
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.payload_bytes
            self.diagnostics["cache_evictions"] += 1
        self._cache[zeros] = state
        self._cache_bytes += state.payload_bytes
        self.diagnostics["peak_cache_entries"] = max(
            self.diagnostics["peak_cache_entries"], len(self._cache))
        self.diagnostics["peak_cache_payload_bytes"] = max(
            self.diagnostics["peak_cache_payload_bytes"], self._cache_bytes)

    def _get(self, zeros):
        zeros = tuple(sorted(self._indices(zeros, name="zeros")))
        self.diagnostics["requests"] += 1
        if not zeros:
            self.diagnostics["base_returns"] += 1
            return self._base
        cached = self._cache.get(zeros)
        if cached is not None:
            self._cache.move_to_end(zeros)
            self.diagnostics["cache_hits"] += 1
            return cached
        self.diagnostics["cache_misses"] += 1
        parent = None
        removed = None
        # The most recently added zero in the common increasing order is last;
        # searching other deletions also reuses states for permuted orders.
        for position in range(len(zeros) - 1, -1, -1):
            parent_key = zeros[:position] + zeros[position + 1:]
            if not parent_key:
                parent, removed = self._base, zeros[position]
                break
            if parent_key in self._cache:
                parent, removed = self._cache[parent_key], zeros[position]
                self._cache.move_to_end(parent_key)
                self.diagnostics["parent_cache_reuses"] += 1
                break
        state = self._direct(zeros) if parent is None else self._from_parent(parent, removed)
        self._remember(zeros, state)
        return state

    def conditioned(self, zeros) -> tuple[tuple[int, ...], np.ndarray]:
        """Return (remaining global modes, read-only covariance in xxpp order)."""
        state = self._get(zeros)
        return state.modes, state.covariance

    def subcovariance(self, zeros, modes) -> np.ndarray:
        """Conditional covariance of ``modes`` in their requested order."""
        requested = self._indices(modes, name="modes")
        state = self._get(zeros)
        local = state.positions[list(requested)]
        if np.any(local < 0):
            raise ValueError("requested modes include a zero-projected mode")
        quadratures = np.concatenate((local, local + len(state.modes)))
        result = state.covariance[np.ix_(quadratures, quadratures)]
        result.setflags(write=False)
        return result

    def cache_info(self):
        return {
            **self.diagnostics,
            "cache_entries": len(self._cache), "cache_size_limit": self.cache_size,
            "cache_payload_bytes": self._cache_bytes,
            "cache_payload_bytes_limit": self.max_cache_bytes,
            "base_payload_bytes": self._base.payload_bytes,
        }


class SchurMemorySampler(ZeroMemorySampler):
    """Same recent/pairwise retained-click model, with small event determinants.

    ``conditioner`` is public so selection rules may reuse its zero-conditioned
    Gaussian covariances.  Full target probabilities are never requested by
    fitting or sampling.  The inherited dense ``full_distribution`` remains
    solely a small-instance evaluator.
    """

    def __init__(self, model, max_clicks, order=None, selection="recent", *,
                 cache_size=8192, max_relative_roundoff=1e-6,
                 covariance_cache_size=128, covariance_cache_bytes=64 * 1024**2):
        super().__init__(model, max_clicks, order=order, selection=selection,
                         cache_size=cache_size, max_relative_roundoff=max_relative_roundoff)
        self.conditioner = GaussianVacuumConditioner(
            model.V, cache_size=covariance_cache_size, max_cache_bytes=covariance_cache_bytes)
        self.diagnostics.update({
            "small_determinant_evaluations": 0,
            "maximum_small_determinant_dimension": 0,
            "conditioning_probability_reference": "P(retained clicks | zero observations)",
        })

    def _conditional(self, zeros, clicks, mode):
        zeros, clicks = tuple(sorted(zeros)), tuple(sorted(clicks))
        key = (zeros, clicks, int(mode))
        if key in self._conditional_cache:
            self._conditional_cache.move_to_end(key)
            self.diagnostics["conditional_cache_hits"] += 1
            return self._conditional_cache[key]
        if set(zeros).intersection(clicks) or mode in zeros or mode in clicks:
            raise ValueError("zero, retained-click and next-mode sets must be disjoint")
        small = self.conditioner.subcovariance(zeros, clicks + (int(mode),))
        probability, details = conditional_click_from_covariance(
            small, max_relative_roundoff=self.max_relative_roundoff,
            conditioning_size=len(zeros))
        self.diagnostics["conditional_evaluations"] += 1
        self.diagnostics["event_evaluations"] += 2
        self.diagnostics["vacuum_probability_calls"] += details["vacuum_probability_calls"]
        self.diagnostics["small_determinant_evaluations"] += details["determinant_evaluations"]
        self.diagnostics["maximum_small_determinant_dimension"] = max(
            self.diagnostics["maximum_small_determinant_dimension"],
            details["maximum_determinant_dimension"])
        self.diagnostics["maximum_inclusion_terms"] = max(
            self.diagnostics["maximum_inclusion_terms"], details["inclusion_terms_per_event"])
        self.diagnostics["maximum_conditional_roundoff_indicator"] = max(
            self.diagnostics["maximum_conditional_roundoff_indicator"], details["roundoff_indicator"])
        self.diagnostics["event_roundoff_corrections"] += details["event_corrections"]
        self.diagnostics["conditional_roundoff_corrections"] += details["probability_corrections"]
        previous = self.diagnostics["minimum_conditioning_probability"]
        self.diagnostics["minimum_conditioning_probability"] = (
            details["conditioning_probability"] if previous is None
            else min(previous, details["conditioning_probability"]))
        self._remember(self._conditional_cache, key, probability)
        return probability

    def cache_info(self):
        return {**super().cache_info(), "gaussian_conditioner": self.conditioner.cache_info()}
