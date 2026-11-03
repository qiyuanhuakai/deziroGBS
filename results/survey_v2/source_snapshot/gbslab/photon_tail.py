"""Total detected-photon tails for squeezed inputs and uniform optical loss.

Assumptions: independent single-mode squeezed-vacuum inputs, an arbitrary
passive interferometer, a common output transmissivity eta, and no displacement,
dark counts, or thermal input noise.  The passive interferometer preserves total
photon number; common loss acts on that number by binomial thinning.  Consequently
the total detected-photon generating function is

    G(z) = product_j sech(r_j) /
           sqrt(1 - tanh(r_j)^2 * (1 - eta + eta*z)^2).

The first cutoff+1 coefficients can be computed by positive recurrences and
truncated convolutions in O(m * cutoff^2) arithmetic and O(cutoff) working memory.
For fixed cutoff this is polynomial in the number of inputs.  This module does
not enumerate the multimode Fock space or use click-pattern probabilities.

For ideal threshold detectors, N_click <= N_photon in every outcome.  Thus the
mathematical photon tail is an upper bound on the corresponding click tail.  It
can be loose when collisions are common.  Values returned here use ordinary
double precision, not outward-rounded interval arithmetic, and are not formal
numerical certificates.  No extension to an arbitrary loss vector is claimed.
"""

from __future__ import annotations

from math import fsum, log

import numpy as np


def _inputs(squeezing_array, transmissivity_scalar, cutoff):
    r = np.asarray(squeezing_array, dtype=float)
    if r.ndim == 0:
        r = r.reshape(1)
    if r.ndim != 1 or not np.all(np.isfinite(r)):
        raise ValueError("squeezing must be a finite one-dimensional array or scalar")
    eta_array = np.asarray(transmissivity_scalar, dtype=float)
    if eta_array.ndim != 0:
        raise ValueError("transmissivity must be a common scalar, not a loss vector")
    eta = float(eta_array)
    if not np.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("transmissivity must lie in [0, 1]")
    if isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, (int, np.integer)):
        raise ValueError("cutoff must be a nonnegative integer")
    if cutoff < 0:
        raise ValueError("cutoff must be a nonnegative integer")
    return r, eta, int(cutoff)


def _single_mode_prefix(r: float, eta: float, cutoff: int) -> np.ndarray:
    """Positive coefficient recurrence, with a stable scaled denominator.

    Writing a=1-t^2(1-eta)^2, b=2t^2 eta(1-eta), c=t^2 eta^2,
    t=tanh(r), differentiation of f(z)=sech(r)/sqrt(a-bz-cz^2)
    gives

        a(n+1) f[n+1] = b(n+1/2) f[n] + c*n*f[n-1].

    We calculate b/a, c/a and f[0] through w=sinh(r)^2 eta(2-eta)
    to avoid subtracting nearly equal terms in a or overflowing sinh(r).
    """
    out = np.zeros(cutoff + 1, dtype=float)
    r_abs = abs(float(r))
    if eta == 0.0 or r_abs == 0.0:
        out[0] = 1.0
        return out
    if r_abs < 20.0:
        log_nbar = 2.0 * log(float(np.sinh(r_abs)))
    else:
        log_nbar = 2.0 * (r_abs + float(np.log1p(-np.exp(-2.0 * r_abs))) - log(2.0))
    log_w = log_nbar + log(eta) + log(2.0 - eta)
    log_one_plus_w = float(np.logaddexp(0.0, log_w))
    out[0] = np.exp(-0.5 * log_one_plus_w)
    fraction = -float(np.expm1(-log_one_plus_w))  # w / (1+w), stable for small w
    b_over_a = 2.0 * (1.0 - eta) / (2.0 - eta) * fraction
    c_over_a = eta / (2.0 - eta) * fraction
    for n in range(cutoff):
        predecessor = out[n - 1] if n else 0.0
        out[n + 1] = (
            b_over_a * (n + 0.5) * out[n] + c_over_a * n * predecessor
        ) / (n + 1.0)
    return out


def photon_probabilities_prefix(
    squeezing_array, transmissivity_scalar: float, cutoff: int
) -> np.ndarray:
    """Return probabilities P(N_photon=n) for n=0,...,cutoff.

    The returned prefix is deliberately NOT normalized: its missing mass is
    the probability above the cutoff.  Signed squeezing values are supported;
    their signs do not affect total photon number.  An empty input array is
    interpreted as vacuum.  Uniform scalar loss is required.
    """
    squeezing, eta, cutoff = _inputs(squeezing_array, transmissivity_scalar, cutoff)
    probabilities = np.zeros(cutoff + 1, dtype=float)
    probabilities[0] = 1.0
    if eta == 0.0:
        return probabilities
    last_key = None
    last_prefix = None
    for r in squeezing:
        key = abs(float(r))
        if key != last_key:
            last_prefix = _single_mode_prefix(key, eta, cutoff)
            last_key = key
        probabilities = np.convolve(probabilities, last_prefix)[: cutoff + 1]
    mass = fsum(float(p) for p in probabilities)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ArithmeticError("photon-prefix recurrence produced invalid probabilities")
    if mass > 1.0 + 2e-10:
        raise ArithmeticError(f"photon-prefix probability exceeds one: {mass}")
    return probabilities


def photon_number_tail(
    squeezing_array, transmissivity_scalar: float, cutoff: int
) -> float:
    """Return P(N_photon >= cutoff+1) under the stated uniform-loss assumptions.

    This mathematical tail dominates P(N_click >= cutoff+1).  At extremely
    small tails, coefficient rounding can dominate subtraction from one; a
    tiny negative floating-point residual is clipped to zero.  The output is
    a numerical evaluation, not an outward-rounded upper-bound certificate.
    """
    probabilities = photon_probabilities_prefix(squeezing_array, transmissivity_scalar, cutoff)
    tail = fsum([1.0] + [-float(p) for p in probabilities])
    if tail < -2e-10 or tail > 1.0 + 2e-10:
        raise ArithmeticError(f"invalid total photon tail: {tail}")
    return float(min(1.0, max(0.0, tail)))
