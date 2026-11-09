"""Small, explicit Gaussian-optics model builders.

Conventions throughout the project are hbar=2, vacuum covariance I, and
quadrature order (x_0, ..., x_{m-1}, p_0, ..., p_{m-1}).  All states in the
initial experiments have zero displacement.  These routines do not simulate
photon detection; that is handled by :mod:`gbslab.exact`.
"""

from __future__ import annotations

import numpy as np


def _mode_vector(value, m: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(m, float(array))
    if array.shape != (m,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite scalar or vector of length {m}")
    return array


def haar_unitary(m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an m by m Haar unitary by complex-normal QR decomposition."""
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer")
    normal = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(normal)
    diagonal = np.diag(r)
    phases = np.divide(diagonal, np.abs(diagonal), out=np.ones_like(diagonal),
                       where=np.abs(diagonal) > 0)
    return q * phases[np.newaxis, :]


def local_unitary(m: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a nearest-neighbor, alternating-pair interferometer.

    A depth is one layer, not a complete even/odd cycle.  Each layer has random
    single-mode phases followed by disjoint SU(2) rotations; pair offsets
    alternate between 0 and 1.  This ensemble is deliberately local and is not
    described as Haar-random over the entire m-mode unitary group.
    """
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer")
    if not isinstance(depth, (int, np.integer)) or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    result = np.eye(m, dtype=complex)
    for layer in range(depth):
        result *= np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, m))[:, None]
        for first in range(layer % 2, m - 1, 2):
            theta = rng.uniform(0.0, np.pi / 2.0)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            phase = np.exp(1j * phi)
            pair = np.array([[c, -phase * s], [phase.conjugate() * s, c]])
            result[[first, first + 1], :] = pair @ result[[first, first + 1], :]
    return result


def covariance_from_interferometer(U, squeezing, loss=1.0) -> np.ndarray:
    """Return output covariance for squeezed vacua, a passive U and loss.

    ``squeezing`` is the real squeeze parameter r, so an input mode has mean
    photon number sinh(r)**2.  ``loss`` is transmissivity eta in [0,1], despite
    the historical argument name: loss=1 is an ideal lossless output and
    loss=0 is vacuum.  Scalars and per-mode vectors are accepted.  Per-mode
    loss is applied after the interferometer.
    """
    U = np.asarray(U, dtype=complex)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or U.shape[0] == 0:
        raise ValueError("U must be a nonempty square matrix")
    if not np.all(np.isfinite(U)):
        raise ValueError("U must have finite entries")
    m = U.shape[0]
    if not np.allclose(U.conj().T @ U, np.eye(m), atol=1e-10, rtol=1e-10):
        raise ValueError("U is not unitary within tolerance")
    r = _mode_vector(squeezing, m, "squeezing")
    eta = _mode_vector(loss, m, "loss/transmissivity")
    if np.any(eta < 0.0) or np.any(eta > 1.0):
        raise ValueError("loss/transmissivity must lie in [0,1]")
    with np.errstate(over="raise", invalid="raise"):
        try:
            diagonal = np.concatenate((np.exp(-2.0 * r), np.exp(2.0 * r)))
        except FloatingPointError as error:
            raise ValueError("squeezing overflows float64 covariance") from error
    symplectic = np.block([[U.real, -U.imag], [U.imag, U.real]])
    V = (symplectic * diagonal[None, :]) @ symplectic.T
    eta_xp = np.concatenate((eta, eta))
    attenuation = np.sqrt(eta_xp)
    V = V * attenuation[:, None] * attenuation[None, :]
    V[np.diag_indices(2 * m)] += 1.0 - eta_xp
    return (V + V.T) / 2.0


def mean_photon_number(V) -> float:
    """Total mean photons for a zero-mean covariance in project conventions."""
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] % 2:
        raise ValueError("V must be an even-dimensional square covariance")
    return float((np.trace(V) - V.shape[0]) / 4.0)
