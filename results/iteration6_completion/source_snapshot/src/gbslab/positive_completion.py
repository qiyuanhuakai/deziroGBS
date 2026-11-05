"""Small-system research reference for completing omitted positive outcomes.

Let B retain every past zero and the selected past clicks R, and let F be the
omitted clicks.  For a fixed partition of F, this module uses

    s_a = P(X_i=a | B) prod_G P(X_G=1 | X_i=a, B),   q_a=s_a/(s_0+s_1).

``product`` uses singleton blocks; ``pair_block`` uses disjoint adjacent pairs
and a final singleton.  Adjacency is in the measurement order by default.
These are conditional-independence approximations, not exact Bayes updates
unless the block factorization holds (in particular, zero or one block).
No parameters are fitted to samples, no epsilon is added, and improvement in
TV is not guaranteed.  The existing retained-click selectors are reused.

The default Gaussian oracle reuses the direct vacuum inclusion-exclusion
kernel.  Its determinants still include all observed zeros.  This is a Python
quality/reference implementation, not a small-k native performance result.
Its cancellation indicators are numerical guards, not certified error bounds.
Only a separately certified zero may create an exactly zero score; unresolved
cancellation, two zero scores, and final positive-branch underflow all fail.
Enumeration includes a failure symbol and never conditions on success.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .conditional_memory import ConditionalMemorySampler
from .exact import NumericalProbabilityError
from .zero_memory import ZeroMemorySampler


@dataclass(frozen=True)
class EventEstimate:
    """Unconditional event mass and an absolute numerical-error indicator.

    ``exact_zero`` is a support certificate supplied by the oracle, not a
    tolerance test.  Gaussian floating cancellation does not provide one.
    The table oracle can certify an empty intersection with its stored support.
    A custom oracle must honor this contract; the sampler cannot prove it.
    """

    value: np.longdouble
    roundoff_indicator: np.longdouble = np.longdouble(0)
    exact_zero: bool = False
    provenance: str = "custom_event_oracle"


def _indices(modes, m, name):
    modes = tuple(modes)
    if any(not isinstance(j, (int, np.integer)) for j in modes):
        raise ValueError(f"{name} must contain integer mode indices")
    modes = tuple(int(j) for j in modes)
    if len(set(modes)) != len(modes) or any(j < 0 or j >= m for j in modes):
        raise ValueError(f"{name} must contain distinct valid mode indices")
    return modes


class TableEventOracle:
    """Independent, exponential full-table oracle for boundary/reference tests.

    Array index bit j denotes physical mode j.  A copy is normalized by its
    long-double total (which must already be within 1e-10 of one); this scalar
    normalization cannot change conditional ratios or support.  All sums use
    nonnegative entries, with no inclusion-exclusion.  Exact zeros refer to
    THIS supplied table, not a certificate for a floating Gaussian reference
    that may have produced it.  ``cache_info`` reports the normalization.

    Pass ``event_oracle=TableEventOracle(P)`` with the original Gaussian model
    to cross-check event evaluation while keeping the very same selector.
    For recent/pairwise boundary tests this oracle can also serve as ``model``.
    """

    def __init__(self, distribution, *, max_modes=20):
        raw = np.asarray(distribution)
        if np.iscomplexobj(raw):
            raise ValueError("distribution must be real")
        p = np.array(raw, dtype=np.longdouble, copy=True)
        if p.ndim != 1 or p.size < 2 or p.size & (p.size - 1):
            raise ValueError("distribution length must be a power of two >= 2")
        if not isinstance(max_modes, (int, np.integer)) or max_modes < 1:
            raise ValueError("max_modes must be a positive integer")
        self.m = p.size.bit_length() - 1
        if self.m > max_modes:
            raise ValueError("table oracle exceeds max_modes")
        total = np.sum(p, dtype=np.longdouble)
        if (not np.all(np.isfinite(p)) or np.any(p < 0)
                or not np.isfinite(total) or abs(total - 1) > 1e-10):
            raise ValueError("distribution must be finite, nonnegative and normalized")
        self.input_total = float(total)
        self._p = p / total
        self._p.setflags(write=False)
        self._masks = np.arange(p.size, dtype=np.uint64)
        self.diagnostics = {"event_calls": 0, "table_entries_examined": 0,
                            "exact_zero_events": 0, "event_seconds": 0.0,
                            "marginal_calls": 0}

    def event_probability(self, zeros, ones):
        zeros = _indices(zeros, self.m, "zeros")
        ones = _indices(ones, self.m, "ones")
        if set(zeros) & set(ones):
            raise ValueError("zero and one event constraints must be disjoint")
        start = perf_counter()
        self.diagnostics["event_calls"] += 1
        zero_mask = sum(1 << j for j in zeros)
        one_mask = sum(1 << j for j in ones)
        keep = ((self._masks & zero_mask) == 0) & ((self._masks & one_mask) == one_mask)
        values = self._p[keep]
        self.diagnostics["table_entries_examined"] += self._p.size
        value = np.sum(values, dtype=np.longdouble)
        exact_zero = not np.any(values > 0)
        self.diagnostics["exact_zero_events"] += int(exact_zero)
        # A positive sum has no cancellation.  This is still a roundoff
        # indicator, not interval arithmetic.  Empty constraints are exact.
        indicator = 8 * np.finfo(np.longdouble).eps * max(1, values.size) * value
        if not zeros and not ones:
            value, indicator = np.longdouble(1), np.longdouble(0)
        self.diagnostics["event_seconds"] += perf_counter() - start
        return EventEstimate(value, indicator, exact_zero, "nonnegative_reference_table_sum")

    def vacuum_probability(self, modes):
        return self.event_probability(modes, ()).value

    def marginal(self, modes):
        modes = _indices(modes, self.m, "modes")
        self.diagnostics["marginal_calls"] += 1
        result = np.zeros(1 << len(modes), dtype=np.longdouble)
        indices = np.zeros(self._p.size, dtype=np.intp)
        for bit, mode in enumerate(modes):
            indices |= ((self._masks >> mode) & 1).astype(np.intp) << bit
        np.add.at(result, indices, self._p)
        return np.asarray(result, dtype=float)

    def cache_info(self):
        return {**self.diagnostics, "kind": "reference_table_only",
                "modes": self.m, "input_total_before_normalization": self.input_total,
                "table_payload_bytes": self._p.nbytes + self._masks.nbytes,
                "table_support_is_not_a_gaussian_zero_certificate": True}


class PositiveCompletionSamplingError(NumericalProbabilityError):
    """One attempted sample failed; completed outputs are available, no redraw."""

    def __init__(self, cause, *, prefix, sample_index, successful_samples, mode):
        super().__init__(f"sample {sample_index}, prefix {prefix}, mode {mode}: {cause}")
        self.failed_prefix = tuple(prefix)
        self.position = len(prefix)
        self.mode = int(mode)
        self.sample_index = int(sample_index)
        self.completed_samples = int(sample_index)
        self.successful_samples = successful_samples.copy()
        self.successful_samples.setflags(write=False)


class PositiveCompletionSampler:
    """Fixed product or adjacent-disjoint-pair positive completion.

    ``selection`` is ``recent``, ``pairwise``, or ``click_influence``; the latter
    composes ConditionalMemorySampler without changing scores or tie rules.
    ``event_oracle=None`` uses Gaussian direct inclusion-exclusion.  A custom
    oracle supplies ``m`` and ``event_probability(zeros, ones) -> EventEstimate``.
    A legacy ``(value, indicator)`` pair is accepted for positive values, but
    cannot certify a zero.  Each event/conditional LRU has ``cache_size`` slots;
    the composed selector and model retain their own separately reported caches.

    ``max_relative_roundoff`` limits the propagated log-score indicator as
    well as individual relative event indicators.  This is deliberately
    stricter at support boundaries than the old sampler: no roundoff-clipped
    zero is accepted as an analytic zero, and no epsilon smoothing is used.
    """

    def __init__(self, model, max_clicks, *, completion="product", selection="recent",
                 order=None, block_order="measurement", cache_size=8192,
                 max_relative_roundoff=1e-6, event_oracle=None):
        start = perf_counter()
        if completion not in ("product", "pair_block"):
            raise ValueError("completion must be 'product' or 'pair_block'")
        if block_order not in ("measurement", "physical"):
            raise ValueError("block_order must be 'measurement' or 'physical'")
        if selection not in ("recent", "pairwise", "click_influence"):
            raise ValueError("selection must be recent, pairwise, or click_influence")
        if selection == "click_influence":
            if not hasattr(model, "V"):
                raise ValueError("click_influence requires the original Gaussian model")
            selector = ConditionalMemorySampler(model, max_clicks, order=order,
                selection=selection, cache_size=cache_size,
                max_relative_roundoff=max_relative_roundoff)
        else:
            selector = ZeroMemorySampler(model, max_clicks, order=order,
                selection=selection, cache_size=cache_size,
                max_relative_roundoff=max_relative_roundoff)
        self.model, self._selector = model, selector
        self.m, self.order = selector.m, selector.order
        self.max_clicks, self.cache_size = selector.max_clicks, selector.cache_size
        self.max_relative_roundoff = selector.max_relative_roundoff
        self.selection, self.completion, self.block_order = selection, completion, block_order
        # A table used as model is explicitly a table oracle, not a Gaussian
        # inclusion-exclusion path.  The Gaussian model can be kept separately.
        if event_oracle is None and isinstance(model, TableEventOracle):
            event_oracle = model
        self._custom_oracle = event_oracle
        self._isolated_vacuum_modes = frozenset()
        if event_oracle is None:
            self._event_backend = ZeroMemorySampler(model, 0, cache_size=0,
                max_relative_roundoff=max_relative_roundoff)
            self.oracle_kind = "direct_vacuum_inclusion_exclusion"
            V = getattr(model, "V", None)
            if V is not None:
                # Exact covariance structure only: no allclose, eigenvalue
                # threshold, or floating Schur-complement zero certificate.
                eye = np.eye(2 * self.m)
                self._isolated_vacuum_modes = frozenset(j for j in range(self.m)
                    if np.array_equal(V[[j, j + self.m], :], eye[[j, j + self.m], :])
                    and np.array_equal(V[:, [j, j + self.m]], eye[:, [j, j + self.m]]))
        else:
            if (getattr(event_oracle, "m", None) != self.m
                    or not callable(getattr(event_oracle, "event_probability", None))):
                raise ValueError("event oracle must have matching m and event_probability")
            self._event_backend = event_oracle
            self.oracle_kind = ("reference_table" if isinstance(event_oracle, TableEventOracle)
                                else "custom_event_oracle")
        self._conditional_cache = OrderedDict()
        self._event_cache = OrderedDict()
        self.diagnostics = {
            "conditional_requests": 0, "conditional_evaluations": 0,
            "conditional_cache_hits": 0, "conditional_failures": 0,
            "conditional_seconds": 0.0, "selector_calls": 0, "selector_seconds": 0.0,
            "event_requests": 0, "event_cache_hits": 0, "event_oracle_calls": 0,
            "event_oracle_seconds": 0.0, "structural_zero_returns": 0,
            "event_exact_zero_returns": 0, "analytic_zero_score_branches": 0,
            "maximum_event_relative_indicator": 0.0,
            "maximum_log_score_indicator": 0.0, "maximum_factor_count": 0,
            "conditional_cache_evictions": 0, "event_cache_evictions": 0,
            "sampling_attempts": 0, "sampling_failures": 0,
            "initialization_seconds": perf_counter() - start,
        }

    def _remember(self, cache, key, value, counter):
        if self.cache_size:
            cache[key] = value
            cache.move_to_end(key)
            if len(cache) > self.cache_size:
                cache.popitem(last=False)
                self.diagnostics[counter] += 1

    def _event(self, zeros, ones):
        zeros, ones = tuple(sorted(zeros)), tuple(sorted(ones))
        key = (zeros, ones)
        self.diagnostics["event_requests"] += 1
        if key in self._event_cache:
            self.diagnostics["event_cache_hits"] += 1
            self._event_cache.move_to_end(key)
            return self._event_cache[key]
        if self._isolated_vacuum_modes.intersection(ones):
            estimate = EventEstimate(np.longdouble(0), np.longdouble(0), True,
                                     "exact_isolated_vacuum_covariance")
            self.diagnostics["structural_zero_returns"] += 1
        else:
            start = perf_counter()
            self.diagnostics["event_oracle_calls"] += 1
            try:
                if self._custom_oracle is None:
                    value, indicator = self._event_backend._event_probability(zeros, ones)
                    estimate = EventEstimate(value, indicator, False,
                                             "gaussian_direct_float64_determinants")
                else:
                    estimate = self._event_backend.event_probability(zeros, ones)
                    if not isinstance(estimate, EventEstimate):
                        value, indicator = estimate
                        estimate = EventEstimate(value, indicator)
            finally:
                self.diagnostics["event_oracle_seconds"] += perf_counter() - start
        value, indicator = np.longdouble(estimate.value), np.longdouble(estimate.roundoff_indicator)
        if (not np.isfinite(value) or not np.isfinite(indicator)
                or indicator < 0 or value < 0 or value > 1):
            raise NumericalProbabilityError("invalid event mass or roundoff indicator")
        if estimate.exact_zero and (value != 0 or indicator != 0):
            raise NumericalProbabilityError("inconsistent analytic-zero certificate")
        if value == 0 and not estimate.exact_zero:
            raise NumericalProbabilityError(
                f"uncertified numerical zero for event zeros={zeros}, ones={ones}")
        if value > 0:
            relative = indicator / value
            self.diagnostics["maximum_event_relative_indicator"] = max(
                self.diagnostics["maximum_event_relative_indicator"], float(relative))
            if not np.isfinite(relative) or relative >= 1 or relative > self.max_relative_roundoff:
                raise NumericalProbabilityError(
                    f"relative event indicator {float(relative):.3g} exceeds guard "
                    f"{self.max_relative_roundoff:.3g}")
        self.diagnostics["event_exact_zero_returns"] += int(estimate.exact_zero)
        result = EventEstimate(value, indicator, estimate.exact_zero, estimate.provenance)
        self._remember(self._event_cache, key, result, "event_cache_evictions")
        return result

    @staticmethod
    def _log_indicator(estimate):
        # Every positive estimate already passed relative < 1.  The quantity
        # below propagates the supplied heuristic, not a verified interval.
        return -np.log1p(-estimate.roundoff_indicator / estimate.value)

    @staticmethod
    def _nested(numerator, denominator, label):
        if numerator.value > denominator.value:
            deviation = numerator.value - denominator.value
            combined = numerator.roundoff_indicator + denominator.roundoff_indicator
            raise NumericalProbabilityError(
                f"{label} violates event nesting (deviation {float(deviation):.3g}, "
                f"indicator {float(combined):.3g}); no clipping applied")

    def _conditional_record(self, prefix):
        prefix = tuple(prefix)
        if len(prefix) >= self.m or any(x not in (0, 1) for x in prefix):
            raise ValueError("prefix must be binary and leave at least one unmeasured mode")
        prefix = tuple(int(x) for x in prefix)
        self.diagnostics["conditional_requests"] += 1
        if prefix in self._conditional_cache:
            self.diagnostics["conditional_cache_hits"] += 1
            self._conditional_cache.move_to_end(prefix)
            return self._conditional_cache[prefix]
        start = perf_counter()
        self.diagnostics["conditional_evaluations"] += 1
        try:
            result = self._compute_conditional(prefix)
        except NumericalProbabilityError:
            self.diagnostics["conditional_failures"] += 1
            raise
        finally:
            self.diagnostics["conditional_seconds"] += perf_counter() - start
        self._remember(self._conditional_cache, prefix, result, "conditional_cache_evictions")
        return result

    def _compute_conditional(self, prefix):
        mode = self.order[len(prefix)]
        zeros = tuple(self.order[j] for j, value in enumerate(prefix) if value == 0)
        clicks = tuple(self.order[j] for j, value in enumerate(prefix) if value == 1)
        start = perf_counter()
        self.diagnostics["selector_calls"] += 1
        try:
            retained = self._selector._retained_clicks(clicks, mode)
        finally:
            self.diagnostics["selector_seconds"] += perf_counter() - start
        selected = set(retained)
        omitted = tuple(j for j in clicks if j not in selected)
        if self.block_order == "physical":
            omitted = tuple(sorted(omitted))
        size = 1 if self.completion == "product" else 2
        blocks = tuple(omitted[j:j + size] for j in range(0, len(omitted), size))
        self.diagnostics["maximum_factor_count"] = max(
            self.diagnostics["maximum_factor_count"], len(blocks))
        baseline = self._event(zeros, retained)
        if baseline.value == 0:
            raise NumericalProbabilityError("baseline conditioning event has zero support")
        branches = (self._event(zeros + (mode,), retained),
                    self._event(zeros, retained + (mode,)))
        for branch in branches:
            self._nested(branch, baseline, "baseline branch")
        consistency = abs(branches[0].value + branches[1].value - baseline.value)
        allowance = (baseline.roundoff_indicator + branches[0].roundoff_indicator
                     + branches[1].roundoff_indicator
                     + 8 * np.finfo(np.longdouble).eps * baseline.value)
        if consistency > allowance:
            raise NumericalProbabilityError("branch masses do not sum to baseline event mass")
        log_scores = [None, None]
        log_indicators = [0.0, 0.0]
        factors = [[], []]
        zero_reasons = [None, None]
        for outcome, branch in enumerate(branches):
            if branch.value == 0:
                zero_reasons[outcome] = "analytic_zero_baseline_branch"
                self.diagnostics["analytic_zero_score_branches"] += 1
                continue
            log_w = np.log(branch.value)
            log_score = log_w
            log_indicator = abs(1 - len(blocks)) * self._log_indicator(branch)
            branch_zeros = zeros + (mode,) if outcome == 0 else zeros
            branch_ones = retained if outcome == 0 else retained + (mode,)
            for group in blocks:
                numerator = self._event(branch_zeros, branch_ones + group)
                self._nested(numerator, branch, "completion factor")
                factors[outcome].append(float(numerator.value / branch.value))
                if numerator.value == 0:
                    zero_reasons[outcome] = "analytic_zero_completion_factor"
                    self.diagnostics["analytic_zero_score_branches"] += 1
                    break
                log_score += np.log(numerator.value) - log_w
                log_indicator += self._log_indicator(numerator)
            if zero_reasons[outcome] is not None:
                continue
            log_indicator += (16 * np.finfo(np.longdouble).eps
                              * (1 + abs(log_score) + len(blocks) * abs(log_w)))
            self.diagnostics["maximum_log_score_indicator"] = max(
                self.diagnostics["maximum_log_score_indicator"], float(log_indicator))
            if (not np.isfinite(log_score) or not np.isfinite(log_indicator)
                    or log_indicator > self.max_relative_roundoff):
                raise NumericalProbabilityError("completion log-score roundoff guard failed")
            log_scores[outcome] = log_score
            log_indicators[outcome] = float(log_indicator)
        if log_scores[0] is None and log_scores[1] is None:
            raise NumericalProbabilityError("both completion scores have zero support")
        if log_scores[0] is None:
            p_one = 1.0
        elif log_scores[1] is None:
            p_one = 0.0
        else:
            difference = log_scores[0] - log_scores[1]
            # Stable logistic; both mathematical branches are strictly positive.
            # Final float64 rounding to an endpoint is a failure, not a zero.
            with np.errstate(under="ignore"):
                if difference >= 0:
                    small = np.exp(-difference)
                    p_one = float(small / (1 + small))
                else:
                    small = np.exp(difference)
                    p_one = float(1 / (1 + small))
            if not np.isfinite(p_one) or not 0 < p_one < 1:
                raise NumericalProbabilityError(
                    "strictly positive completion branches rounded to a Bernoulli endpoint")
        return {
            "prefix": prefix, "mode": mode, "zeros": zeros, "clicks": clicks,
            "retained_clicks": retained, "omitted_clicks": omitted, "blocks": blocks,
            "baseline_probability": float(baseline.value),
            "branch_masses": tuple(float(e.value) for e in branches),
            "baseline_probabilities": tuple(float(e.value / baseline.value) for e in branches),
            "factor_probabilities": tuple(tuple(values) for values in factors),
            "log_scores_up_to_common_baseline": tuple(None if v is None else float(v)
                                                       for v in log_scores),
            "log_score_indicators": tuple(log_indicators),
            "zero_score_reasons": tuple(zero_reasons), "click_probability": p_one,
            "log_score_common_factor_omitted": "1/P(B)",
        }

    def conditional_click_probability(self, prefix):
        """Next-click probability; prefix follows the fixed measurement order."""
        return self._conditional_record(prefix)["click_probability"]

    def conditional_details(self, prefix):
        """Return a defensive copy of the selected events, factors, and scores."""
        return deepcopy(self._conditional_record(prefix))

    def enumerate_law(self, max_modes=14):
        """Enumerate Q on physical patterns plus failure, without renormalizing.

        On a failed reachable prefix its whole Q mass goes to the final failure
        symbol.  Other branches continue; there is no resampling or conditioning
        on successful outputs.  This is exponential and uses no full target P.
        """
        if not isinstance(max_modes, (int, np.integer)) or max_modes < 1 or self.m > max_modes:
            raise ValueError(f"enumeration of {self.m} modes exceeds valid cap {max_modes}")
        start = perf_counter()
        probabilities = np.zeros(1 << self.m, dtype=np.longdouble)
        failure_mass = np.longdouble(0)
        failures = []

        def visit(prefix, mask, weight):
            nonlocal failure_mass
            if len(prefix) == self.m:
                probabilities[mask] = weight
                return
            try:
                p_one = self.conditional_click_probability(prefix)
            except NumericalProbabilityError as error:
                failure_mass += weight
                failures.append({"prefix": prefix, "position": len(prefix),
                                 "mode": self.order[len(prefix)], "mass": float(weight),
                                 "error_type": type(error).__name__, "error": str(error)})
                return
            for value, probability in ((0, 1 - np.longdouble(p_one)),
                                       (1, np.longdouble(p_one))):
                if probability == 0:
                    continue
                child_weight = weight * probability
                if child_weight == 0:
                    raise NumericalProbabilityError("positive enumeration prefix mass underflowed")
                visit(prefix + (value,), mask | (value << self.order[len(prefix)]), child_weight)

        visit((), 0, np.longdouble(1))
        result = np.asarray(probabilities, dtype=float)
        fail = float(failure_mass)
        if (np.any((probabilities > 0) & (result == 0))
                or (failure_mass > 0 and fail == 0)):
            raise NumericalProbabilityError("positive enumerated mass is not representable in float64")
        normalization_error = float(abs(np.sum(probabilities, dtype=np.longdouble) + failure_mass - 1))
        if normalization_error > 1e-12:
            raise NumericalProbabilityError("extended sequential law failed normalization")
        return {"distribution": result, "failure_mass": fail,
                "extended_distribution": np.concatenate((result, [fail])),
                "failures": failures, "failure_count": len(failures),
                "normalization_error": normalization_error,
                "output_space": "binary_patterns_plus_failure",
                "conditional_on_success": False,
                "elapsed_seconds": perf_counter() - start}

    def full_distribution(self, max_modes=14):
        """Return Q only if enumeration has no failures; never renormalize it."""
        law = self.enumerate_law(max_modes=max_modes)
        if law["failures"]:
            raise NumericalProbabilityError(
                f"full_distribution refuses failure mass {law['failure_mass']:.17g}; "
                "use enumerate_law for the extended output space")
        return law["distribution"]

    def sample(self, n, rng):
        """Draw uint8 physical-mode samples, aborting at the first failure.

        The failure exception exposes the completed outputs and failed prefix.
        Every successful conditional consumes one uniform, including analytic
        deterministic outcomes.  A failed conditional consumes none; no retry
        occurs.  Use enumeration for an unconditional failure-mass calculation.
        """
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        result = np.zeros((n, self.m), dtype=np.uint8)
        for sample_index in range(n):
            self.diagnostics["sampling_attempts"] += 1
            prefix = ()
            for mode in self.order:
                try:
                    p_one = self.conditional_click_probability(prefix)
                except NumericalProbabilityError as error:
                    self.diagnostics["sampling_failures"] += 1
                    raise PositiveCompletionSamplingError(error, prefix=prefix,
                        sample_index=sample_index, successful_samples=result[:sample_index],
                        mode=mode) from error
                uniform = rng.random()
                if not np.isfinite(uniform) or not 0 <= uniform < 1:
                    raise ValueError("rng.random() must return a finite scalar in [0,1)")
                value = int(uniform < p_one)
                result[sample_index, mode] = value
                prefix += (value,)
        return result

    def cache_info(self):
        """Actual calls/costs, including independent selector/backend counters.

        Shared model counters may include work done by the caller.  Only the
        module's event_oracle_calls and nested backend counters attribute its
        own direct-event requests.  Timings are descriptive Python reference
        costs and cannot be compared as native-algorithm throughput claims.
        """
        return {**self.diagnostics, "oracle_kind": self.oracle_kind,
                "completion": self.completion, "selection": self.selection,
                "block_order": self.block_order, "max_clicks": self.max_clicks,
                "cache_size_limit_each": self.cache_size,
                "conditional_cache_entries": len(self._conditional_cache),
                "event_cache_entries": len(self._event_cache),
                "isolated_vacuum_modes": tuple(sorted(self._isolated_vacuum_modes)),
                "selector": self._selector.cache_info(),
                "event_backend": (self._event_backend.cache_info()
                                  if hasattr(self._event_backend, "cache_info") else None),
                "shared_model_totals": (self.model.cache_info()
                                        if hasattr(self.model, "cache_info") else None),
                "timing_scope": "descriptive_python_reference_only",
                "roundoff_indicators_are_rigorous_bounds": False}
