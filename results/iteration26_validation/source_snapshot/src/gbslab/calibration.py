"""Exact first-click-event calibration of a bounded-parent GBS surrogate.

For a fixed mode order, the partition consists of the event that the first
click occurs at position j (j = 0, ..., m-1), plus the all-zero event (index m).
Their exact probabilities require only m nested vacuum probabilities, rather
than enumeration of click patterns.  Rescaling the base distribution within
these events is a standard information projection, not a novelty claim.

If P is the target, Q is the base, and C labels this partition, the calibrated
distribution Q* satisfies

    KL(P || Q*) = KL(P || Q) - KL(P_C || Q_C).

The removed categorical KL is cheaply computable.  It does not bound the
remaining full-distribution error, and there is no general claim that total
variation distance must decrease.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .surrogates import BoundedParentSurrogate, distribution_metrics


class VacuumOracle(Protocol):
    m: int

    def vacuum_probability(self, modes_tuple: tuple[int, ...]) -> float: ...


def first_click_categories(m: int, order: Sequence[int]) -> np.ndarray:
    """Dense-validation helper: map each original-mode bitmask to its event.

    The returned array is exponentially large.  Neither fitting nor sampling
    calls this helper.
    """
    order = tuple(order)
    if sorted(order) != list(range(m)):
        raise ValueError("order must be a permutation of all modes")
    if m > 24:
        raise ValueError("dense first-click enumeration is limited to 24 modes")
    masks = np.arange(1 << m, dtype=np.uint64)
    categories = np.full(len(masks), m, dtype=np.intp)
    for position, mode in enumerate(order):
        first = (categories == m) & (((masks >> mode) & 1) != 0)
        categories[first] = position
    return categories


class CalibratedSurrogate:
    """A first-click-calibrated Bayesian network, with polynomial-cost sampling.

    Category j < m denotes the first click at ``order[j]``; category m denotes
    no clicks.  Fitting calls ``model.vacuum_probability`` m times on nested
    prefixes.  It never queries the full target or its threshold marginals.
    Sampling draws a category, fixes its prefix, then runs the base network's
    conditionals on the suffix.  Thus it does not use rejection sampling or
    enumerate an exponential distribution.
    """

    def __init__(self, model: VacuumOracle, fit: BoundedParentSurrogate):
        if int(model.m) != fit.m:
            raise ValueError("target and base surrogate mode counts differ")
        self.base = fit
        self.m = fit.m
        self.order = fit.order

        vacuum = np.ones(self.m + 1, dtype=float)
        for length in range(1, self.m + 1):
            vacuum[length] = model.vacuum_probability(self.order[:length])
        if not np.all(np.isfinite(vacuum)):
            raise ArithmeticError("nonfinite vacuum probability in calibration")
        if np.any(vacuum < -2e-10) or np.any(vacuum > 1.0 + 2e-10):
            raise ArithmeticError("vacuum probabilities lie outside [0, 1]")
        if np.any(np.diff(vacuum) > 2e-10):
            raise ArithmeticError("nested vacuum probabilities are not monotone")
        raw_target = np.concatenate((vacuum[:-1] - vacuum[1:], vacuum[-1:]))
        target = np.maximum(raw_target, 0.0)
        target /= target.sum()

        # Along the all-zero prefix, every parent's state is zero, regardless
        # of the selected parents or their ordering within the CPT.
        base_categories = np.zeros(self.m + 1, dtype=float)
        survival = 1.0
        for position, mode in enumerate(self.order):
            hazard = float(fit.conditional_tables[mode][0])
            base_categories[position] = survival * hazard
            survival *= 1.0 - hazard
        base_categories[self.m] = survival
        if not np.isclose(base_categories.sum(), 1.0, rtol=0, atol=2e-10):
            raise ArithmeticError("base first-click probabilities failed normalization")
        # Do not silently invent conditionals for events unsupported by Q.
        unsupported = (target > 0) & (base_categories == 0)
        if np.any(unsupported):
            raise ValueError(
                "target assigns positive mass to a category absent from the base "
                f"(indices {np.flatnonzero(unsupported).tolist()}); check support or underflow"
            )

        self.target_category_probabilities = target
        self.base_category_probabilities = base_categories
        self.prefix_vacuum_probabilities = vacuum
        self.target_category_probabilities.setflags(write=False)
        self.base_category_probabilities.setflags(write=False)
        self.prefix_vacuum_probabilities.setflags(write=False)
        category_kl = distribution_metrics(target, base_categories)["kl"]
        self.diagnostics = {
            "calibration": "first_click_event_information_projection",
            "oracle_vacuum_calls": self.m,
            "largest_vacuum_subset": self.m,
            "uses_full_target_distribution": False,
            "category_kl_nats": category_kl,
            "predicted_kl_reduction_nats": category_kl,
            "target_category_probabilities": target.tolist(),
            "base_category_probabilities": base_categories.tolist(),
            "category_l1_numerical_correction": float(np.abs(target - raw_target).sum()),
            "zero_target_categories": int(np.count_nonzero(target == 0)),
            "zero_base_categories": int(np.count_nonzero(base_categories == 0)),
        }

    def full_distribution(self, *, max_states: int = 1 << 24) -> np.ndarray:
        """Enumerate the calibrated distribution solely for small validation."""
        q = self.base.full_distribution(max_states=max_states)
        categories = first_click_categories(self.m, self.order)
        denominator = self.base_category_probabilities[categories]
        conditional = np.zeros_like(q)
        # Divide before multiplying to avoid overflow from a large P(C)/Q(C).
        np.divide(q, denominator, out=conditional, where=denominator > 0.0)
        calibrated = conditional * self.target_category_probabilities[categories]
        if not np.isclose(calibrated.sum(), 1.0, rtol=0.0, atol=2e-9):
            raise ArithmeticError("calibrated distribution failed normalization")
        return calibrated

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample the exact calibrated model in O(n m (k+1)) array operations."""
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        rng = np.random.default_rng() if rng is None else rng
        categories = rng.choice(self.m + 1, size=n, p=self.target_category_probabilities)
        samples = np.zeros((n, self.m), dtype=np.uint8)
        for position, mode in enumerate(self.order):
            samples[categories == position, mode] = 1
            active = np.flatnonzero(categories < position)
            if not len(active):
                continue
            parent_masks = np.zeros(len(active), dtype=np.intp)
            for local, parent in enumerate(self.base.parents[mode]):
                parent_masks |= samples[active, parent].astype(np.intp) << local
            probabilities = self.base.conditional_tables[mode][parent_masks]
            samples[active, mode] = rng.random(len(active)) < probabilities
        return samples
