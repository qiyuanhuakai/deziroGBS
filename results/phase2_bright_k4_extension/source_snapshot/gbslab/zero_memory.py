"""Exploratory sampler retaining all no-clicks and a budget of past clicks.

At each step, no-click observations and up to k prior clicks are retained;
other clicks are marginalized.  Retention uses either the most recent clicks
or the largest unconditional pairwise mutual information with the next mode.
The resulting conditionals define a normalized sequential probability model.
This module does not claim algorithmic novelty or a quantum speedup reversal.

The term "zero memory" refers to remembering zero outcomes, not zero storage.
All no-click modes are remembered and the wrapped ThresholdModel has its own
cache.  For larger-mode sampling, construct it with a finite
``vacuum_cache_size`` (or zero to disable that cache); the unlimited default is
intended for small reference calculations.  The extra caches owned here are
bounded.  No full exact-marginal cache is populated by this sampler.
The optional pairwise selector precomputes O(m**2) exact two-mode marginals
and an m by m score matrix; those small marginals are cached by ThresholdModel.

Each conditional uses O(2**k) vacuum-probability terms, with determinants
whose dimension can grow with the number of earlier no-click observations.
Consequently a fixed k does not imply constant per-sample work.
"""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter

import numpy as np

from .exact import NumericalProbabilityError


class ZeroMemorySampler:
    """Retain no-click observations and at most ``max_clicks`` prior clicks.

    Parameters
    ----------
    model : ThresholdModel-compatible object
        Provides ``m`` and ``vacuum_probability(modes)``.  The optional pairwise
        selector also uses ``marginal((i,j))``.  No full distribution is requested
        by this sampler, including during small-system model enumeration.
    max_clicks : int
        Retained click budget.  A budget at least m-1 is fully conditioned.
    order : iterable[int], optional
        Measurement order.  Returned samples and enumerated distributions
        always use original physical mode indices.
    selection : str
        ``recent`` keeps the most recent prior clicks.  ``pairwise`` ranks
        previously clicked modes by their exact unconditional threshold MI
        with the next mode, breaking ties by physical mode index.  This is a
        fixed covariance-derived criterion; conditional dependence after
        postselection can differ and improvement is not guaranteed.  A zero
        click budget skips all pairwise precomputation.
    cache_size : int
        Maximum entries in EACH of the conditional and event LRU caches.
        Independently set ``ThresholdModel(..., vacuum_cache_size=...)`` to cap
        the wrapped model's vacuum-probability cache for larger-mode sampling.
    max_relative_roundoff : float
        Fail if a conservative floating-point cancellation indicator for a
        conditional exceeds this value.  The indicator is a numerical guard,
        not a rigorous determinant error bound or statistical accuracy claim.
    """

    def __init__(self, model, max_clicks, order=None, selection="recent", *,
                 cache_size=8192, max_relative_roundoff=1e-6):
        if not isinstance(max_clicks, (int, np.integer)) or max_clicks < 0:
            raise ValueError("max_clicks must be a nonnegative integer")
        if not isinstance(cache_size, (int, np.integer)) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        if not np.isfinite(max_relative_roundoff) or max_relative_roundoff <= 0:
            raise ValueError("max_relative_roundoff must be finite and positive")
        if selection not in ("recent", "pairwise"):
            raise ValueError("selection must be 'recent' or 'pairwise'")
        self.model = model
        self.m = int(model.m)
        if self.m < 1:
            raise ValueError("model must contain at least one mode")
        order = tuple(range(self.m)) if order is None else tuple(order)
        if (len(order) != self.m
                or any(not isinstance(j, (int, np.integer)) for j in order)
                or set(order) != set(range(self.m))):
            raise ValueError("order must be a permutation of all mode indices")
        self.order = tuple(int(j) for j in order)
        self.max_clicks = int(max_clicks)
        self.selection = selection
        self.cache_size = int(cache_size)
        self.max_relative_roundoff = float(max_relative_roundoff)
        self._conditional_cache = OrderedDict()
        self._event_cache = OrderedDict()
        self.diagnostics = {
            "conditional_evaluations": 0,
            "conditional_cache_hits": 0,
            "event_evaluations": 0,
            "event_cache_hits": 0,
            "vacuum_probability_calls": 0,
            "maximum_inclusion_terms": 0,
            "maximum_conditional_roundoff_indicator": 0.0,
            "event_roundoff_corrections": 0,
            "conditional_roundoff_corrections": 0,
            "minimum_conditioning_probability": None,
            "pairwise_marginals_requested": 0,
            "pairwise_precompute_seconds": 0.0,
        }
        self.pairwise_mi = None
        if self.selection == "pairwise" and self.max_clicks:
            start = perf_counter()
            scores = np.zeros((self.m, self.m))
            for first in range(self.m):
                for second in range(first + 1, self.m):
                    joint = np.asarray(model.marginal((first, second)), dtype=float)
                    if (joint.shape != (4,) or not np.all(np.isfinite(joint))
                            or np.any(joint < 0)
                            or abs(float(np.sum(joint)) - 1.0) > 1e-10):
                        raise NumericalProbabilityError("invalid pair marginal for MI selector")
                    first_marginal = np.array([joint[0] + joint[2], joint[1] + joint[3]])
                    second_marginal = np.array([joint[0] + joint[1], joint[2] + joint[3]])
                    product = np.array([first_marginal[mask & 1] * second_marginal[mask >> 1]
                                        for mask in range(4)])
                    positive = joint > 0
                    mi = float(np.sum(joint[positive] * np.log(joint[positive] / product[positive])))
                    if mi < -1e-12:
                        raise NumericalProbabilityError(f"negative pairwise MI {mi:.3g}")
                    scores[first, second] = scores[second, first] = max(0.0, mi)
                    self.diagnostics["pairwise_marginals_requested"] += 1
            scores.setflags(write=False)
            self.pairwise_mi = scores
            self.diagnostics["pairwise_precompute_seconds"] = perf_counter() - start

    def _retained_clicks(self, clicks, mode):
        """Select a covariance-derived subset, with no full-target fitting."""
        if not self.max_clicks:
            return ()
        if len(clicks) <= self.max_clicks:
            return tuple(clicks)
        if self.selection == "recent":
            return tuple(clicks[-self.max_clicks:])
        return tuple(sorted(clicks, key=lambda past: (-self.pairwise_mi[mode, past], past))
                     [:self.max_clicks])

    def _remember(self, cache, key, value):
        if self.cache_size:
            cache[key] = value
            cache.move_to_end(key)
            if len(cache) > self.cache_size:
                cache.popitem(last=False)

    def _event_probability(self, zeros, clicks):
        """P(zeros=0, clicks=1), plus a floating-point cancellation indicator."""
        zeros, clicks = tuple(sorted(zeros)), tuple(sorted(clicks))
        key = (zeros, clicks)
        if key in self._event_cache:
            self.diagnostics["event_cache_hits"] += 1
            self._event_cache.move_to_end(key)
            return self._event_cache[key]
        n_terms = 1 << len(clicks)
        terms = np.empty(n_terms, dtype=np.longdouble)
        signs = np.empty(n_terms, dtype=np.int8)
        for mask in range(n_terms):
            subset = zeros + tuple(clicks[j] for j in range(len(clicks))
                                   if mask & (1 << j))
            terms[mask] = self.model.vacuum_probability(subset)
            signs[mask] = -1 if mask.bit_count() % 2 else 1
        value = np.sum(terms * signs, dtype=np.longdouble)
        abs_sum = np.sum(np.abs(terms), dtype=np.longdouble)
        # Determinants are evaluated in float64 by ThresholdModel.  This is a
        # deliberately conservative heuristic guard, not verified arithmetic.
        indicator = (8.0 * np.finfo(float).eps
                     * max(1, len(zeros) + len(clicks)) * abs_sum)
        self.diagnostics["event_evaluations"] += 1
        self.diagnostics["vacuum_probability_calls"] += n_terms
        self.diagnostics["maximum_inclusion_terms"] = max(
            self.diagnostics["maximum_inclusion_terms"], n_terms)
        if not np.isfinite(value) or not np.isfinite(indicator):
            raise NumericalProbabilityError("nonfinite inclusion-exclusion event probability")
        if value < 0:
            if -value > indicator:
                raise NumericalProbabilityError(
                    f"negative event probability {float(value):.3g} exceeds roundoff indicator")
            value = np.longdouble(0)
            self.diagnostics["event_roundoff_corrections"] += 1
        result = (value, indicator)
        self._remember(self._event_cache, key, result)
        return result

    def _conditional(self, zeros, clicks, mode):
        zeros = tuple(sorted(zeros))
        clicks = tuple(sorted(clicks))
        key = (zeros, clicks, int(mode))
        if key in self._conditional_cache:
            self.diagnostics["conditional_cache_hits"] += 1
            self._conditional_cache.move_to_end(key)
            return self._conditional_cache[key]
        denominator, den_indicator = self._event_probability(zeros, clicks)
        numerator, num_indicator = self._event_probability(zeros + (mode,), clicks)
        self.diagnostics["conditional_evaluations"] += 1
        if denominator <= 0:
            raise NumericalProbabilityError(
                "conditioning event has zero or numerically unresolved probability")
        p_zero = numerator / denominator
        indicator = float((num_indicator + abs(p_zero) * den_indicator) / denominator)
        self.diagnostics["maximum_conditional_roundoff_indicator"] = max(
            self.diagnostics["maximum_conditional_roundoff_indicator"], indicator)
        previous_min = self.diagnostics["minimum_conditioning_probability"]
        self.diagnostics["minimum_conditioning_probability"] = (
            float(denominator) if previous_min is None
            else min(previous_min, float(denominator)))
        if indicator > self.max_relative_roundoff:
            raise NumericalProbabilityError(
                f"conditional cancellation indicator {indicator:.3g} exceeds "
                f"{self.max_relative_roundoff:.3g}; reduce click budget or use higher precision")
        if p_zero < 0 or p_zero > 1:
            deviation = float(max(-p_zero, p_zero - 1))
            if deviation > indicator:
                raise NumericalProbabilityError(
                    f"conditional no-click probability {float(p_zero):.17g} is outside [0,1]")
            p_zero = np.clip(p_zero, 0, 1)
            self.diagnostics["conditional_roundoff_corrections"] += 1
        result = float(1.0 - p_zero)
        self._remember(self._conditional_cache, key, result)
        return result

    def conditional_click_probability(self, prefix) -> float:
        """P(next outcome=1) for a prefix in this sampler's measurement order.

        Prefixes with impossible retained events raise.  This method is useful
        for auditing when approximate and exact transitions first diverge.
        """
        prefix = tuple(prefix)
        if len(prefix) >= self.m:
            raise ValueError("prefix must leave at least one unmeasured mode")
        if any(value not in (0, 1) for value in prefix):
            raise ValueError("prefix entries must be binary")
        zeros = tuple(self.order[j] for j, value in enumerate(prefix) if value == 0)
        clicks = tuple(self.order[j] for j, value in enumerate(prefix) if value == 1)
        mode = self.order[len(prefix)]
        clicks = self._retained_clicks(clicks, mode)
        return self._conditional(zeros, clicks, mode)

    def full_distribution(self, max_modes=14) -> np.ndarray:
        """Enumerate this sequential model, independently of the full target.

        This routine is for small-scale verification, costs exponentially many
        branches, and is not used by ``sample``.  Index bit j always refers to
        physical mode j, even if the measurement order is permuted.
        """
        if self.m > max_modes:
            raise ValueError(f"enumeration of {self.m} modes exceeds cap {max_modes}")
        probabilities = np.zeros(1 << self.m)

        def visit(position, zeros, clicks, mask, weight):
            if position == self.m:
                probabilities[mask] = weight
                return
            mode = self.order[position]
            retained = self._retained_clicks(clicks, mode)
            p_one = self._conditional(zeros, retained, mode)
            if p_one < 1:
                visit(position + 1, zeros + (mode,), clicks, mask, weight * (1.0 - p_one))
            if p_one > 0:
                visit(position + 1, zeros, clicks + (mode,), mask | (1 << mode), weight * p_one)

        visit(0, (), (), 0, 1.0)
        if abs(float(np.sum(probabilities)) - 1.0) > 1e-11:
            raise NumericalProbabilityError("sequential model failed normalization")
        return probabilities

    def sample(self, n, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples; return uint8[n,m] in physical mode order.

        This first implementation uses Python loops and cached determinants.
        Timings must not be represented as an optimized GPU or C++ baseline.
        """
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        result = np.zeros((n, self.m), dtype=np.uint8)
        for sample_index in range(n):
            zeros, clicks = (), ()
            for mode in self.order:
                retained = self._retained_clicks(clicks, mode)
                p_one = self._conditional(zeros, retained, mode)
                if rng.random() < p_one:
                    result[sample_index, mode] = 1
                    clicks += (mode,)
                else:
                    zeros += (mode,)
        return result

    def cache_info(self) -> dict:
        """Diagnostics for this sampler; inspect model.cache_info separately."""
        return {
            **self.diagnostics,
            "conditional_cache_entries": len(self._conditional_cache),
            "event_cache_entries": len(self._event_cache),
            "cache_size_limit_each": self.cache_size,
        }


def click_tail_coupling_bound(target_distribution, max_clicks) -> float:
    """Conservative small-system TV bound P_target(N_click >= k+1).

    When the sampler and target share the same prefix with at most k clicks,
    their next-step conditional is identical.  A sequential coupling can
    therefore differ only on paths reaching k+1 clicks.  This is a general
    coupling observation, not a novelty claim.  Evaluating this function from
    a full target distribution is exponential and is NOT a scalable accuracy
    certificate.  A separately available click-count tail bound could be used
    instead.  The formula is deliberately loose for a final-step (k+1)st click.
    """
    if not isinstance(max_clicks, (int, np.integer)) or max_clicks < 0:
        raise ValueError("max_clicks must be a nonnegative integer")
    p = np.asarray(target_distribution, dtype=float)
    if p.ndim != 1 or p.size == 0 or p.size & (p.size - 1):
        raise ValueError("target_distribution length must be a power of two")
    if (not np.all(np.isfinite(p)) or np.min(p) < 0
            or not np.isclose(np.sum(p), 1, atol=1e-10, rtol=0)):
        raise ValueError("target_distribution must be normalized and nonnegative")
    return float(sum(probability for mask, probability in enumerate(p)
                     if mask.bit_count() >= max_clicks + 1))
