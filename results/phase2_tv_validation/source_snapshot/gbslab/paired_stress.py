"""Analytic, physical stress tests for click-retention rules.

The target is a product of n two-mode squeezed-vacuum (TMSV) states.  The
physical mode order is A_0,...,A_(n-1),B_0,...,B_(n-1), with A_j paired to B_j.
Threshold clicks in a pair are identical Bernoulli(p) variables, p=tanh(r)^2.
This distribution is classically trivial; failure on it is an obstruction to
a particular retention rule, not evidence of GBS computational hardness.

The analytic oracle and TV formula deliberately avoid Gaussian determinants.
They separate the statistical approximation from floating-point difficulties
on exactly zero-probability outcomes.  Covariance-kernel failures are returned
explicitly by ``sampler_crosscheck`` and never silently dropped.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def _validate(n_pairs, p):
    if isinstance(n_pairs, (bool, np.bool_)) or not isinstance(n_pairs, (int, np.integer)) or n_pairs < 1:
        raise ValueError("n_pairs must be a positive integer")
    if not np.isscalar(p) or not np.isfinite(p) or not 0 <= p < 1:
        raise ValueError("p must be finite and in [0, 1); p=1 requires infinite squeezing")
    return int(n_pairs), float(p)


def paired_covariance(n_pairs: int, p: float = 0.5) -> np.ndarray:
    """Return the finite-energy product-TMSV covariance, hbar=2, xxpp.

    Using cosh(2r)=(1+p)/(1-p) and sinh(2r)=2 sqrt(p)/(1-p) avoids an
    unnecessary inverse-hyperbolic round trip.  These are still float64
    entries, so exact support must not be inferred from tiny numeric residues.
    """
    n, p = _validate(n_pairs, p)
    modes = 2 * n
    if modes > 4096:
        raise ValueError("covariance construction exceeds the 4096-mode resource guard")
    c = (1.0 + p) / (1.0 - p)
    s = 2.0 * np.sqrt(p) / (1.0 - p)
    V = np.eye(2 * modes) * c
    for j in range(n):
        V[j, n + j] = V[n + j, j] = s
        V[modes + j, modes + n + j] = V[modes + n + j, modes + j] = -s
    return V


def measurement_order(n_pairs: int, kind: str = "blocked") -> tuple[int, ...]:
    """Blocked order is all A then all B; interleaved pairs adjacent partners."""
    n, _ = _validate(n_pairs, 0.0)
    if kind == "blocked":
        return tuple(range(2 * n))
    if kind == "interleaved":
        return tuple(mode for j in range(n) for mode in (j, n + j))
    raise ValueError("kind must be 'blocked' or 'interleaved'")


def recent_tv(n_pairs: int, max_clicks: int, p: float = 0.5) -> float:
    """Exact TV of recent-k retention in BLOCKED measurement order.

    If L ~ Binomial(n,p) is the number of A clicks, the sampler's probability
    of reproducing all corresponding B clicks is one for L<=k and p**L
    otherwise.  Hence TV = E[1_{L>k} (1-p**L)].  The target support contains
    only pair-consistent strings, and Q(x)<=P(x) on that support.

    This O(n) formula is analytic evaluation of a special easy family.  It
    must not be reported as a large generic-GBS full-distribution calculation.
    """
    n, p = _validate(n_pairs, p)
    if isinstance(max_clicks, (bool, np.bool_)) or not isinstance(max_clicks, (int, np.integer)) or max_clicks < 0:
        raise ValueError("max_clicks must be a nonnegative integer")
    if max_clicks >= n or p == 0:
        return 0.0
    counts = np.arange(int(max_clicks) + 1, n + 1)
    mass = np.exp(binom.logpmf(counts, n, p))
    failure = -np.expm1(counts * np.log(p))
    answer = float(np.sum(mass.astype(np.longdouble) * failure.astype(np.longdouble)))
    if not -1e-12 <= answer <= 1 + 1e-12:
        raise ArithmeticError("analytic TV failed its probability bounds")
    return float(np.clip(answer, 0.0, 1.0))


def dense_target(n_pairs: int, p: float = 0.5, *, max_patterns: int = 1 << 20) -> np.ndarray:
    """Small-system full target with only 2**n of its 4**n patterns supported."""
    n, p = _validate(n_pairs, p)
    size = 1 << (2 * n)
    if size > max_patterns:
        raise ValueError("dense target exceeds max_patterns; use the analytic TV instead")
    result = np.zeros(size)
    for a in range(1 << n):
        count = a.bit_count()
        result[a | (a << n)] = p ** count * (1.0 - p) ** (n - count)
    if not np.isclose(result.sum(), 1.0, atol=1e-12, rtol=0):
        raise ArithmeticError("analytic target failed normalization")
    return result


class PairedThresholdOracle:
    """Analytic vacuum and marginal oracle for the same physical TMSV family.

    At p=.5 all returned probabilities are dyadic and exactly representable.
    This is a mathematics fixture, not a faster implementation for arbitrary
    Gaussian input.  It intentionally has no covariance or timing claims.
    """

    def __init__(self, n_pairs: int, p: float = 0.5):
        self.n_pairs, self.p = _validate(n_pairs, p)
        self.m = 2 * self.n_pairs
        self.vacuum_calls = 0
        self.marginal_calls = 0

    def _modes(self, modes):
        modes = tuple(modes)
        if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= self.m for i in modes):
            raise ValueError("invalid mode index")
        if len(set(modes)) != len(modes):
            raise ValueError("mode indices must not repeat")
        return tuple(int(i) for i in modes)

    def vacuum_probability(self, modes):
        modes = self._modes(modes)
        self.vacuum_calls += 1
        return (1.0 - self.p) ** len({i % self.n_pairs for i in modes})

    def marginal(self, modes):
        modes = self._modes(modes)
        if len(modes) > 20:
            raise ValueError("marginal fixture is limited to 20 requested modes")
        self.marginal_calls += 1
        probabilities = np.zeros(1 << len(modes))
        for mask in range(len(probabilities)):
            values = {}
            consistent = True
            for bit, mode in enumerate(modes):
                pair, value = mode % self.n_pairs, (mask >> bit) & 1
                if pair in values and values[pair] != value:
                    consistent = False
                    break
                values[pair] = value
            if consistent:
                probabilities[mask] = np.prod([
                    self.p if bit else 1.0-self.p for bit in values.values()])
        return probabilities

    def cache_info(self):
        return {"analytic_vacuum_calls": self.vacuum_calls,
                "analytic_marginal_calls": self.marginal_calls,
                "purpose": "physical analytic fixture, not generic-GBS speed measurement"}


def sampler_crosscheck(n_pairs: int, max_clicks: int, p: float = 0.5, *,
                      selection: str = "recent", order_kind: str = "blocked",
                      oracle_kind: str = "analytic") -> dict:
    """Compare the old sampler to the theorem and record numeric guards.

    ``oracle_kind='covariance'`` uses the ordinary Gaussian determinant code.
    A numerical guard is returned as status='numerical_guard', not counted as
    a successful confirmation.  Errors in configuration still raise.
    """
    from .exact import ThresholdModel, NumericalProbabilityError
    from .zero_memory import ZeroMemorySampler

    n, p = _validate(n_pairs, p)
    if selection not in ("recent", "pairwise"):
        raise ValueError("selection must be recent or pairwise")
    order = measurement_order(n, order_kind)
    target = dense_target(n, p, max_patterns=1 << 16)
    if oracle_kind == "analytic":
        model = PairedThresholdOracle(n, p)
        reference_tv = 0.0
    elif oracle_kind == "covariance":
        model = ThresholdModel(paired_covariance(n, p))
        reference_tv = float(np.abs(target - model.full_distribution()).sum() / 2)
    else:
        raise ValueError("oracle_kind must be analytic or covariance")
    if max_clicks == 0:
        theoretical = recent_tv(n, 0, p)
    elif selection == "pairwise" or order_kind == "interleaved":
        theoretical = 0.0
    else:
        theoretical = recent_tv(n, max_clicks, p)
    row = {"n_pairs": n, "modes": 2*n, "p": p, "k": max_clicks,
           "selection": selection, "order": order_kind, "oracle": oracle_kind,
           "theoretical_tv": theoretical, "covariance_reference_tv": reference_tv}
    sampler = ZeroMemorySampler(model, max_clicks, order=order, selection=selection)
    try:
        approximate = sampler.full_distribution(max_modes=16)
        actual = float(np.abs(target-approximate).sum() / 2)
        row.update(status="ok", observed_tv=actual,
                   absolute_theorem_error=abs(actual-theoretical))
    except NumericalProbabilityError as exc:
        row.update(status="numerical_guard", error_type=type(exc).__name__, error=str(exc))
    row["sampler_diagnostics"] = sampler.cache_info()
    row["oracle_diagnostics"] = model.cache_info()
    return row

