"""Independent finite-Fock reference for lossy threshold Gaussian boson sampling.

This module deliberately does not use covariance determinants, hafnians, or the
threshold engine. It constructs pure-state Fock amplitudes by the creation-
operator recurrence, then applies independent optical loss and threshold
measurement. The photon cutoff truncates *total input/output photon number
before loss*. Its omitted mass is retained explicitly rather than silently
normalizing the result and treating it as exact.

Conventions: interferometers act as ``a_out = U a_in``; real input squeezings
have x-quadrature variance exp(-2*r), with vacuum covariance I (hbar=2).
Probability-array bit i (least significant bit first) denotes a click in mode
i. The method is an exponentially growing validation reference, not a scalable
classical sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, fsum, sqrt
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class FockThresholdResult:
    """Finite-photon reference and the exact omitted-mass accounting.

    ``probabilities`` is unnormalized: its sum is ``retained_mass``.
    ``normalized_distribution`` conditions the pure state on the retained
    photon sector before applying loss and threshold detection. Its total
    variation distance from the untruncated threshold distribution is at most
    ``omitted_mass``, up to floating point error.
    """

    probabilities: np.ndarray
    retained_mass: float
    omitted_mass: float
    normalized_distribution: np.ndarray
    max_photons: int
    fock_states_enumerated: int
    retained_mass_by_photon_number: dict[int, float]

    @property
    def unnormalized_probabilities(self) -> np.ndarray:
        """Descriptive alias for ``probabilities``."""
        return self.probabilities


def _vector(value: float | np.ndarray, m: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(m, float(arr))
    if arr.shape != (m,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite scalar or a vector of length {m}")
    return arr


def _validated_inputs(
    U: np.ndarray, squeezing: float | np.ndarray, max_photons: int, max_states: int
) -> tuple[np.ndarray, np.ndarray, int]:
    unitary = np.asarray(U, dtype=complex)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1] or not unitary.shape[0]:
        raise ValueError("U must be a nonempty square matrix")
    m = unitary.shape[0]
    if not np.all(np.isfinite(unitary)):
        raise ValueError("U must be finite")
    if not np.allclose(unitary.conj().T @ unitary, np.eye(m), rtol=1e-10, atol=1e-10):
        raise ValueError("U must be unitary")
    r = _vector(squeezing, m, "squeezing")
    if isinstance(max_photons, (bool, np.bool_)) or not isinstance(max_photons, (int, np.integer)):
        raise ValueError("max_photons must be a nonnegative integer")
    if max_photons < 0:
        raise ValueError("max_photons must be nonnegative")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    # A conservative guard includes odd sectors, which are skipped below.
    if comb(m + int(max_photons), m) > max_states:
        raise ValueError(
            "Requested Fock reference exceeds max_states; lower the cutoff or mode count"
        )
    return unitary, r, m


def _occupations(total: int, modes: int) -> Iterator[tuple[int, ...]]:
    if modes == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _occupations(total - first, modes - 1):
            yield (first,) + rest


def pure_fock_amplitudes(
    U: np.ndarray,
    squeezing: float | np.ndarray,
    max_photons: int,
    *,
    max_states: int = 2_000_000,
) -> dict[tuple[int, ...], complex]:
    """Return amplitudes in every even total-photon sector up to the cutoff.

    Odd total sectors are exactly zero and omitted from the returned mapping.
    For B = -U diag(tanh(r)) U.T and any i with n_i > 0, the recurrence is

        a[n] = sum_j B[i,j] sqrt(n_j - delta_ij) a[n-e_i-e_j] / sqrt(n_i).

    The vacuum amplitude is prod_i cosh(r_i)**(-1/2). Signed real squeezing
    values are supported, allowing an independent two-mode squeezed-state
    analytic test. ``max_states`` is a resource guard, not an approximation.
    """
    unitary, r, m = _validated_inputs(U, squeezing, max_photons, max_states)
    B = -(unitary * np.tanh(r)[None, :]) @ unitary.T
    log_cosh = np.logaddexp(r, -r) - np.log(2.0)
    a_vac = float(np.exp(-0.5 * np.sum(log_cosh)))
    if a_vac == 0.0:
        raise ValueError("Vacuum amplitude underflowed; use a lower squeezing/reference size")

    @lru_cache(maxsize=None)
    def amplitude(n: tuple[int, ...]) -> complex:
        if not any(n):
            return complex(a_vac)
        if sum(n) % 2:
            return 0.0j
        i = next(i for i, n_i in enumerate(n) if n_i)
        reduced = list(n)
        reduced[i] -= 1
        terms: list[complex] = []
        for j, n_j in enumerate(reduced):
            if n_j == 0:
                continue
            predecessor = reduced.copy()
            predecessor[j] -= 1
            terms.append(B[i, j] * sqrt(n_j) * amplitude(tuple(predecessor)))
        return sum(terms, 0.0j) / sqrt(n[i])

    return {
        n: amplitude(n)
        for total in range(0, int(max_photons) + 1, 2)
        for n in _occupations(total, m)
    }


def threshold_from_fock(
    U: np.ndarray,
    squeezing: float | np.ndarray,
    loss: float | np.ndarray = 1.0,
    max_photons: int = 12,
    *,
    max_states: int = 2_000_000,
) -> FockThresholdResult:
    """Apply optical loss and threshold detection to a truncated pure state.

    Here ``loss`` means transmissivity eta (1 is lossless, 0 is complete loss),
    matching the Gaussian constructor's API. For a Fock occupation n_i, the
    no-click probability after loss is (1-eta_i)**n_i. This diagonal
    measurement permits probability-level thinning even though the pre-loss
    state is coherent. No covariance or threshold-model code is called.
    """
    unitary = np.asarray(U, dtype=complex)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError("U must be square")
    m = unitary.shape[0]
    # Even the vacuum-only Fock sector has one threshold vector entry for
    # every bit pattern. A photon-sector guard alone would miss this cost.
    if (1 << m) > max_states:
        raise ValueError("Threshold pattern array exceeds max_states; lower the mode count")
    eta = _vector(loss, m, "loss/transmissivity")
    if np.any(eta < 0.0) or np.any(eta > 1.0):
        raise ValueError("loss/transmissivity must lie in [0, 1]")
    amplitudes = pure_fock_amplitudes(unitary, squeezing, max_photons, max_states=max_states)
    probabilities = np.zeros(1 << m, dtype=float)
    masses_by_total: dict[int, list[float]] = {}
    for occupation, amplitude in amplitudes.items():
        mass = float(abs(amplitude) ** 2)
        total = sum(occupation)
        masses_by_total.setdefault(total, []).append(mass)
        if mass == 0.0:
            continue
        no_click = np.power(1.0 - eta, np.asarray(occupation, dtype=int))
        patterns = np.ones(1, dtype=float)
        for vacuum_probability in no_click:
            # Appending the new mode puts its click bit above all previous bits.
            patterns = np.concatenate(
                (patterns * vacuum_probability, patterns * (1.0 - vacuum_probability))
            )
        probabilities += mass * patterns

    retained_by_total = {total: fsum(masses) for total, masses in masses_by_total.items()}
    retained_mass = fsum(retained_by_total.values())
    if retained_mass <= 0.0 or retained_mass > 1.0 + 1e-9:
        raise ArithmeticError(f"Invalid retained Fock probability mass: {retained_mass}")
    if not np.isclose(np.sum(probabilities), retained_mass, rtol=2e-12, atol=2e-14):
        raise ArithmeticError("Loss/threshold aggregation failed probability-mass conservation")
    return FockThresholdResult(
        probabilities=probabilities,
        retained_mass=retained_mass,
        omitted_mass=max(0.0, 1.0 - retained_mass),
        normalized_distribution=probabilities / retained_mass,
        max_photons=int(max_photons),
        fock_states_enumerated=len(amplitudes),
        retained_mass_by_photon_number=retained_by_total,
    )
