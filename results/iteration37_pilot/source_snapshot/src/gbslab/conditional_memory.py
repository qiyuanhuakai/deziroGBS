"""Exploratory click selection after conditioning on all no-click outcomes.

Only no-click outcomes are used to construct the Gaussian conditional state:

    W_R = V_R - V_RZ (V_Z + I)^(-1) V_ZR.

Conditioning additionally on clicked modes is non-Gaussian.  The
``conditional_mi`` selector deliberately uses pairwise threshold information
in W (averaged over the two click values), whereas ``click_influence`` ranks
the actual observed-one effect on the next click probability.  Neither rule
is claimed to minimize global approximation error or to be a novel method.

The direct and Schur sampler classes differ only in how the retained-event
conditionals are evaluated.  Both selectors use the shared small Gaussian
kernel and never access a complete target distribution.
"""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter

import numpy as np

from .exact import NumericalProbabilityError
from .schur_memory import (
    GaussianVacuumConditioner,
    SchurMemorySampler,
    conditional_click_from_covariance,
    gaussian_vacuum_probability,
)
from .zero_memory import ZeroMemorySampler


class _ConditionalSelector:
    """Selection mixin shared by the direct and Schur event evaluators."""

    def _initialize_selector(self, selection):
        if selection not in ("conditional_mi", "click_influence"):
            raise ValueError("selection must be 'conditional_mi' or 'click_influence'")
        self.selection = selection
        self._positions = {mode: position for position, mode in enumerate(self.order)}
        self._selection_cache = OrderedDict()
        self.diagnostics.update({
            "selection_requests": 0,
            "selection_computations": 0,
            "selection_cache_hits": 0,
            "selection_cache_evictions": 0,
            "selection_cache_peak_entries": 0,
            "selection_compute_seconds": 0.0,
            "selection_score_evaluations": 0,
            "selection_small_vacuum_calls": 0,
            "selection_small_determinant_evaluations": 0,
            "selection_conditional_kernel_calls": 0,
            "selection_max_candidate_count": 0,
            "selection_max_small_mode_count": 0,
            "selection_probability_corrections": 0,
            "selection_max_l1_probability_correction": 0.0,
            "selection_max_conditional_roundoff_indicator": 0.0,
            "selection_conditional_event_corrections": 0,
            "selection_conditional_probability_corrections": 0,
        })

    def _validate_indices(self, modes, name):
        modes = tuple(modes)
        if any(not isinstance(j, (int, np.integer)) for j in modes):
            raise ValueError(f"{name} must contain integer mode indices")
        modes = tuple(int(j) for j in modes)
        if len(set(modes)) != len(modes) or any(j < 0 or j >= self.m for j in modes):
            raise ValueError(f"{name} must contain distinct valid mode indices")
        return modes

    def conditional_covariance(self, zeros):
        """Return (remaining physical modes, xxpp covariance), conditioned on zeros.

        The returned covariance is read-only.  Its modes are in the order
        supplied by the shared conditioner (ascending physical mode index).
        This is NOT a covariance description of the state conditioned on any
        retained clicks.  General physical Gaussian covariances, including
        those with nonuniform output losses, are allowed.
        """
        zeros = tuple(sorted(self._validate_indices(zeros, "zeros")))
        return self.conditioner.conditioned(zeros)

    @staticmethod
    def _subcovariance(covariance, index, modes):
        n = covariance.shape[0] // 2
        positions = tuple(index[mode] for mode in modes)
        quadratures = positions + tuple(j + n for j in positions)
        return covariance[np.ix_(quadratures, quadratures)]

    def _checked_pair_distribution(self, q_first, q_second, q_pair):
        # Local bit 0 is the first mode.  q values refer to no-click events.
        a, b, c = map(np.longdouble, (q_first, q_second, q_pair))
        raw = np.array([c, b-c, a-c, 1-a-b+c], dtype=float)
        tolerance = getattr(self.model, "probability_tolerance", 2e-10)
        if not np.all(np.isfinite(raw)) or np.min(raw) < -tolerance:
            raise NumericalProbabilityError(
                "zero-conditioned pair probability is negative or nonfinite")
        corrected = np.maximum(raw, 0.0)
        corrected /= np.sum(corrected, dtype=np.longdouble)
        change = float(np.sum(np.abs(corrected - raw)))
        if change > 10 * tolerance:
            raise NumericalProbabilityError("zero-conditioned pair correction exceeds tolerance")
        if change:
            self.diagnostics["selection_probability_corrections"] += 1
            self.diagnostics["selection_max_l1_probability_correction"] = max(
                self.diagnostics["selection_max_l1_probability_correction"], change)
        return corrected

    def _mi_scores_in_state(self, covariance, index, mode, candidates):
        target_covariance = self._subcovariance(covariance, index, (mode,))
        q_first = gaussian_vacuum_probability(target_covariance)
        self.diagnostics["selection_small_vacuum_calls"] += 1
        self.diagnostics["selection_small_determinant_evaluations"] += 1
        scores = {}
        for candidate in candidates:
            q_second = gaussian_vacuum_probability(
                self._subcovariance(covariance, index, (candidate,)))
            q_pair = gaussian_vacuum_probability(
                self._subcovariance(covariance, index, (mode, candidate)))
            self.diagnostics["selection_small_vacuum_calls"] += 2
            self.diagnostics["selection_small_determinant_evaluations"] += 2
            self.diagnostics["selection_score_evaluations"] += 1
            self.diagnostics["selection_max_small_mode_count"] = max(
                self.diagnostics["selection_max_small_mode_count"], 2)
            joint = self._checked_pair_distribution(q_first, q_second, q_pair)
            first = np.array([joint[0] + joint[2], joint[1] + joint[3]])
            second = np.array([joint[0] + joint[1], joint[2] + joint[3]])
            mi = 0.0
            for mask in range(4):
                if joint[mask] > 0:
                    # Separate logs avoid underflow in a product of marginals.
                    mi += joint[mask] * (np.log(joint[mask])
                        - np.log(first[mask & 1]) - np.log(second[mask >> 1]))
            if not np.isfinite(mi) or mi < -2e-10:
                raise NumericalProbabilityError("invalid zero-conditioned mutual information")
            scores[candidate] = max(0.0, float(mi))
        return scores

    def conditional_pairwise_mi(self, zeros, mode, candidates):
        """Exact threshold I(next; candidate | every zero in Z observed).

        This conditions on a particular zero event, not the entropy-average
        I(next; candidate | X_Z) across all values of X_Z.  It also does not
        condition on the candidate having already been observed to click.
        Returned values are nats, with original physical-mode dictionary keys.
        """
        zeros = self._validate_indices(zeros, "zeros")
        modes = self._validate_indices((mode,) + tuple(candidates), "mode and candidates")
        if set(zeros).intersection(modes):
            raise ValueError("scored modes cannot also be conditioned to zero")
        remaining, covariance = self.conditional_covariance(zeros)
        index = {j: local for local, j in enumerate(remaining)}
        return self._mi_scores_in_state(covariance, index, modes[0], modes[1:])

    def _click_in_state(self, covariance, index, zeros, retained, mode):
        modes = tuple(retained) + (mode,)
        subcovariance = self._subcovariance(covariance, index, modes)
        probability, details = conditional_click_from_covariance(
            subcovariance, max_relative_roundoff=self.max_relative_roundoff,
            conditioning_size=len(zeros))
        self.diagnostics["selection_conditional_kernel_calls"] += 1
        self.diagnostics["selection_small_determinant_evaluations"] += details["determinant_evaluations"]
        self.diagnostics["selection_small_vacuum_calls"] += details["vacuum_probability_calls"]
        self.diagnostics["selection_max_conditional_roundoff_indicator"] = max(
            self.diagnostics["selection_max_conditional_roundoff_indicator"], details["roundoff_indicator"])
        self.diagnostics["selection_conditional_event_corrections"] += details["event_corrections"]
        self.diagnostics["selection_conditional_probability_corrections"] += details["probability_corrections"]
        self.diagnostics["selection_max_small_mode_count"] = max(
            self.diagnostics["selection_max_small_mode_count"], len(modes))
        return probability

    def _influence_scores_in_state(self, covariance, index, zeros, mode,
                                   candidates, retained=(), baseline=None):
        if baseline is None:
            baseline = self._click_in_state(covariance, index, zeros, retained, mode)
        scores, probabilities = {}, {}
        for candidate in candidates:
            p = self._click_in_state(covariance, index, zeros,
                                     tuple(retained) + (candidate,), mode)
            self.diagnostics["selection_score_evaluations"] += 1
            scores[candidate] = abs(p - baseline)
            probabilities[candidate] = p
        return baseline, scores, probabilities

    def click_influence_scores(self, zeros, mode, candidates, retained=()):
        """Return baseline, absolute click effects, and candidate probabilities.

        The effect of j is |P(X_mode=1 | Z0, retained1, j1)
        - P(X_mode=1 | Z0, retained1)|.  All retained and candidate bits in the
        conditional event are specified to be one.  An impossible or
        numerically unresolved conditioning event raises explicitly.
        """
        zeros = self._validate_indices(zeros, "zeros")
        retained = self._validate_indices(retained, "retained")
        candidates = self._validate_indices(candidates, "candidates")
        modes = self._validate_indices(retained + candidates + (mode,), "scored modes")
        if set(zeros).intersection(modes):
            raise ValueError("scored modes cannot also be conditioned to zero")
        remaining, covariance = self.conditional_covariance(zeros)
        index = {j: local for local, j in enumerate(remaining)}
        return self._influence_scores_in_state(covariance, index, zeros,
                                              int(mode), candidates, retained)

    def _retained_clicks(self, clicks, mode):
        self.diagnostics["selection_requests"] += 1
        if not self.max_clicks:
            return ()
        clicks = tuple(clicks)
        if len(clicks) <= self.max_clicks:
            return clicks
        past = self.order[:self._positions[mode]]
        click_set = set(clicks)
        if len(click_set) != len(clicks) or not click_set.issubset(past):
            raise ValueError("click history must contain distinct previously measured modes")
        zeros = tuple(sorted(j for j in past if j not in click_set))
        key = (self.selection, zeros, tuple(sorted(clicks)), mode)
        if key in self._selection_cache:
            self._selection_cache.move_to_end(key)
            self.diagnostics["selection_cache_hits"] += 1
            return self._selection_cache[key]
        start = perf_counter()
        remaining, covariance = self.conditional_covariance(zeros)
        index = {j: local for local, j in enumerate(remaining)}
        self.diagnostics["selection_computations"] += 1
        self.diagnostics["selection_max_candidate_count"] = max(
            self.diagnostics["selection_max_candidate_count"], len(clicks))
        if self.selection == "conditional_mi":
            scores = self._mi_scores_in_state(covariance, index, mode, clicks)
            selected = tuple(sorted(clicks, key=lambda j: (-scores[j], j))[:self.max_clicks])
        else:
            selected_list = []
            available = tuple(sorted(clicks))
            baseline = self._click_in_state(covariance, index, zeros, (), mode)
            for _ in range(self.max_clicks):
                _, scores, probabilities = self._influence_scores_in_state(
                    covariance, index, zeros, mode, available, selected_list, baseline)
                best = min(available, key=lambda j: (-scores[j], j))
                selected_list.append(best)
                baseline = probabilities[best]
                available = tuple(j for j in available if j != best)
            selected = tuple(selected_list)
        self.diagnostics["selection_compute_seconds"] += perf_counter() - start
        if self.cache_size:
            if len(self._selection_cache) >= self.cache_size:
                self.diagnostics["selection_cache_evictions"] += 1
            self._remember(self._selection_cache, key, selected)
            self.diagnostics["selection_cache_peak_entries"] = max(
                self.diagnostics["selection_cache_peak_entries"], len(self._selection_cache))
        return selected

    def cache_info(self):
        info = super().cache_info()
        info.update({"selection": self.selection,
            "selection_cache_entries": len(self._selection_cache),
            "selection_cache_size_limit": self.cache_size,
            "zero_conditioner": self.conditioner.cache_info()})
        return info


class ConditionalMemorySampler(_ConditionalSelector, ZeroMemorySampler):
    """Conditional selector with the original large-event ratio evaluator.

    This class isolates the selection-rule change from the Schur probability
    evaluator.  ``conditional_mi`` selects the k largest pairwise threshold MI
    scores after observing all zeros.  ``click_influence`` greedily selects the
    observed clicks producing the largest change in the next-click probability.
    Both are heuristics; neither is guaranteed to improve TV.
    """

    def __init__(self, model, max_clicks, order=None, selection="conditional_mi", *,
                 cache_size=8192, covariance_cache_size=128,
                 covariance_cache_bytes=64 * 1024**2, max_relative_roundoff=1e-6):
        if selection not in ("conditional_mi", "click_influence"):
            raise ValueError("selection must be 'conditional_mi' or 'click_influence'")
        ZeroMemorySampler.__init__(self, model, max_clicks, order, "recent",
            cache_size=cache_size, max_relative_roundoff=max_relative_roundoff)
        self.conditioner = GaussianVacuumConditioner(model.V,
            cache_size=covariance_cache_size, max_cache_bytes=covariance_cache_bytes)
        self._initialize_selector(selection)


class SchurConditionalMemorySampler(_ConditionalSelector, SchurMemorySampler):
    """Conditional selector sharing its Gaussian conditioner with Schur sampling.

    API and selected events match ConditionalMemorySampler.  Only the retained
    event probabilities are evaluated through the Schur small-mode kernel.
    Selection and sampling share the same bounded covariance cache.
    """

    def __init__(self, model, max_clicks, order=None, selection="conditional_mi", *,
                 cache_size=8192, covariance_cache_size=128,
                 covariance_cache_bytes=64 * 1024**2, max_relative_roundoff=1e-6):
        if selection not in ("conditional_mi", "click_influence"):
            raise ValueError("selection must be 'conditional_mi' or 'click_influence'")
        SchurMemorySampler.__init__(self, model, max_clicks, order, "recent",
            cache_size=cache_size, covariance_cache_size=covariance_cache_size,
            covariance_cache_bytes=covariance_cache_bytes,
            max_relative_roundoff=max_relative_roundoff)
        self._initialize_selector(selection)
