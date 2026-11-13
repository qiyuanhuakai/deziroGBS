"""Independent finite-Fock references for iteration 10.

This module never imports or invokes the native coherent-core implementation.
It constructs coefficients by a creation-operator recurrence, evaluates fixed
displacements by finite Fock contractions, and integrates Gaussian displacement
noise analytically by a Gaussian tilt and Wick moments.  The formulas are exact;
their NumPy/SciPy evaluations are floating-point references, not outward-rounded
certificates.  No Monte Carlo estimate is returned as a complete probability law.

Probability arrays use physical-mode click masks, with mode zero in bit zero.
Residual displacement covariance is real, in xxpp order and hbar=2, so that
alpha=(d_q+i*d_p)/2.  In particular the complex displacement is generally not
circular, even when its Hermitian covariance is proportional to the identity.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
from time import perf_counter

import numpy as np
from scipy.stats import nbinom


DEFAULT_MAX_PAIRS = 2_500_000
DEFAULT_MAX_MOMENTS = 100_000
DEFAULT_MAX_MODES = 12
DEFAULT_MAX_BASIS = 200_000
DEFAULT_NEGATIVE_ATOM_TOLERANCE = 1e-11
DEFAULT_SUM_ERROR_TOLERANCE = 1e-11
DEFAULT_CORRECTION_L1_TOLERANCE = 1e-9


class ReferenceResourceError(RuntimeError):
    """A declared reference-work limit was exhausted, without input substitution."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class ReferenceNumericalError(ArithmeticError):
    """The floating reference failed an explicit numerical acceptance condition."""


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _compositions(total, modes):
    if modes == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in _compositions(total - first, modes - 1):
                yield (first,) + rest


def occupation_basis(modes, cutoff, *, even_only=False, max_basis=DEFAULT_MAX_BASIS):
    """Total-budget basis, never a rectangular (K+1)**m tensor."""
    modes = _integer(modes, "modes", 1)
    cutoff = _integer(cutoff, "cutoff")
    maximum = _integer(max_basis, "max_basis", 1)
    totals = range(0, cutoff + 1, 2 if even_only else 1)
    count = sum(math.comb(modes + n - 1, n) for n in totals)
    if count > maximum:
        raise ReferenceResourceError("Total-photon basis exceeds its declared limit",
                                     {"basis_size": count, "max_basis": maximum})
    return np.asarray([n for total in totals for n in _compositions(total, modes)], dtype=np.int64)


def core_parameters(squeezing, loss):
    r, eta = float(squeezing), float(loss)
    if not math.isfinite(r) or r < 0 or not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("Require finite common squeezing >= 0 and common loss in [0,1]")
    a = 1.0 - eta + eta * math.exp(-2.0 * r)
    b = 1.0 - eta + eta * math.exp(2.0 * r)
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0:
        raise ValueError("Core parameters are not representable")
    s = -0.5 * math.log(a)
    # This identity avoids a subtractive negative variance at eta=0 or eta=1.
    delta = 4.0 * eta * (1.0 - eta) * math.sinh(r) ** 2 / a
    return {"squeezing": r, "loss": eta, "a": a, "b": b,
            "core_squeezing": s, "residual_variance": delta}


def core_state_reference(U, squeezing, loss, cutoff, *, max_basis=DEFAULT_MAX_BASIS):
    """Coherent normalized projection of the pure core onto total N<=cutoff.

    All total-budget occupations are returned; odd-total initial coefficients
    are exactly zero.  Initial coefficient recurrence is independent of the
    native code and is checked in tests against direct small hafnians.
    """
    start = perf_counter()
    U = np.asarray(U, dtype=np.complex128)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or not U.shape[0] or not np.all(np.isfinite(U)):
        raise ValueError("U must be a finite, nonempty square matrix")
    m = U.shape[0]
    if not np.allclose(U.conj().T @ U, np.eye(m), atol=1e-11, rtol=1e-11):
        raise ValueError("U must be unitary within the declared floating tolerance")
    cutoff = _integer(cutoff, "cutoff")
    pars = core_parameters(squeezing, loss)
    s = pars["core_squeezing"]
    B = -math.tanh(s) * (U @ U.T)
    occupations = occupation_basis(m, cutoff, max_basis=max_basis)
    @lru_cache(maxsize=None)
    def relative_coefficient(n):
        total = sum(n)
        if total == 0:
            return complex(1.0)
        if total % 2:
            return complex(0.0)
        i = next(j for j, nj in enumerate(n) if nj)
        answer = complex(0.0)
        for j in range(m):
            multiplicity = n[j] - int(j == i)
            if multiplicity > 0:
                previous = list(n)
                previous[i] -= 1
                previous[j] -= 1
                answer += B[i, j] * math.sqrt(multiplicity) * relative_coefficient(tuple(previous))
        return answer / math.sqrt(n[i])

    relative = np.asarray([relative_coefficient(tuple(n)) for n in occupations], dtype=np.complex128)
    norm2 = float(np.sum(np.abs(relative).astype(np.longdouble) ** 2, dtype=np.longdouble))
    if not math.isfinite(norm2) or norm2 <= 0:
        raise ReferenceNumericalError("Core coefficient normalization failed")
    amplitudes = relative / math.sqrt(norm2)
    vacuum_raw = math.exp(-0.5 * m * math.log(math.cosh(s)))
    if vacuum_raw == 0:
        raise ReferenceNumericalError("Untruncated vacuum amplitude underflowed")
    retained_from_coefficients = vacuum_raw ** 2 * norm2
    success = 1.0 / math.cosh(s) ** 2
    tail = float(nbinom.sf(cutoff // 2, m / 2.0, success))
    if not math.isfinite(tail) or not 0 <= tail <= 1:
        raise ReferenceNumericalError("Negative-binomial tail failed")
    O = np.block([[U.real, -U.imag], [U.imag, U.real]])
    W = (O * np.r_[np.full(m, pars["a"]), np.full(m, 1.0 / pars["a"])]) @ O.T
    # C=L L^T makes its possibly singular pushforward meaning explicit.
    L = math.sqrt(pars["residual_variance"]) * O[:, m:]
    C = L @ L.T
    V = (O * np.r_[np.full(m, pars["a"]), np.full(m, pars["b"])]) @ O.T
    return {"occupations": occupations, "amplitudes": amplitudes, "cutoff": cutoff,
            "modes": m, "even_initial_support": True, "B": B,
            "pure_covariance": W, "displacement_covariance": C,
            "displacement_real_factor": L, "target_covariance": V,
            "vacuum_amplitude_untruncated": vacuum_raw,
            "retained_weight": 1.0 - tail, "tail_probability": tail,
            "trace_distance_bound": math.sqrt(tail), "parameters": pars,
            "diagnostics": {"basis_size": len(occupations),
                "even_basis_size": int(np.sum(occupations.sum(axis=1) % 2 == 0)),
                "nonzero_coefficients": int(np.count_nonzero(amplitudes)),
                "retained_weight_from_coefficients": retained_from_coefficients,
                "retained_weight_absolute_residual": abs(retained_from_coefficients - (1.0 - tail)),
                "covariance_split_max_absolute_residual": float(np.max(np.abs(W + C - V))),
                "coefficient_recurrence_states": relative_coefficient.cache_info().currsize,
                "construction_seconds": perf_counter() - start,
                "outward_rounded_certificate": False}}


def hafnian_by_pairings(matrix):
    """Independent factorial-time pairing sum for small coefficient fixtures."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Hafnian input must be square")
    n = matrix.shape[0]
    if n % 2:
        return complex(0.0)
    if n > 12:
        raise ReferenceResourceError("Direct matching fixture is limited to order 12")
    if n == 0:
        return complex(1.0)
    answer = complex(0.0)
    for j in range(1, n):
        rest = [i for i in range(1, n) if i != j]
        answer += matrix[0, j] * hafnian_by_pairings(matrix[np.ix_(rest, rest)])
    return answer


def _state_arrays(state):
    occupations = np.asarray(state["occupations"])
    amplitudes = np.asarray(state["amplitudes"], dtype=np.complex128)
    if occupations.ndim != 2 or occupations.shape[1] < 1 or len(occupations) != len(amplitudes):
        raise ValueError("State occupations/amplitudes have incompatible shapes")
    if amplitudes.ndim != 1 or not np.all(np.isfinite(amplitudes)):
        raise ValueError("State amplitudes must be a finite vector")
    if not np.all(np.isfinite(occupations)) or np.any(occupations < 0) or np.any(occupations != np.floor(occupations)):
        raise ValueError("Occupations must be nonnegative integers")
    occupations = occupations.astype(np.int64)
    if len(set(map(tuple, occupations))) != len(occupations):
        raise ValueError("Duplicate occupations are not permitted")
    norm = float(np.sum(np.abs(amplitudes).astype(np.longdouble) ** 2, dtype=np.longdouble))
    if not math.isfinite(norm) or abs(norm - 1.0) > 5e-12:
        raise ValueError("Reference requires an explicitly normalized pure state")
    cutoff = _integer(state.get("cutoff", int(occupations.sum(axis=1).max(initial=0))), "cutoff")
    if np.any(occupations.sum(axis=1) > cutoff):
        raise ValueError("Input occupations exceed total cutoff")
    if state.get("even_initial_support", False) and np.any(amplitudes[occupations.sum(axis=1) % 2 == 1] != 0):
        raise ValueError("Claimed even initial support contains an odd coefficient")
    return occupations, amplitudes, cutoff


def _amplitude_norm(amplitudes):
    real = np.asarray(amplitudes).real.astype(np.longdouble)
    imag = np.asarray(amplitudes).imag.astype(np.longdouble)
    return np.sum(real * real + imag * imag, dtype=np.longdouble)


def _marginal_groups(occupations, modes):
    selected = set(modes)
    outside = tuple(j for j in range(occupations.shape[1]) if j not in selected)
    groups = defaultdict(list)
    for i, row in enumerate(occupations):
        groups[tuple(row[list(outside)])].append(i)
    return tuple(groups.values())


def reference_work_estimate(state, *, max_modes=DEFAULT_MAX_MODES,
                            max_pairs=DEFAULT_MAX_PAIRS):
    """Conservative count of compatible unordered coefficient pairs over all Z.

    For an even-initial core, all even basis occupations count, including exact
    zero coefficients.  No probability, random sample, or target-fit decision is
    used to select a smaller panel.  Rejection can stop once a lower bound has
    already exceeded the cap.
    """
    occupations, _, _ = _state_arrays(state)
    m = occupations.shape[1]
    maximum_modes = _integer(max_modes, "max_modes", 1)
    maximum_pairs = _integer(max_pairs, "max_pairs", 1)
    if m > maximum_modes:
        raise ReferenceResourceError("Complete threshold law exceeds mode cap",
                                     {"modes": m, "max_modes": maximum_modes})
    if state.get("even_initial_support", False):
        occupations = occupations[occupations.sum(axis=1) % 2 == 0]
    # A pair may occur in Z exactly when Z includes every differing mode.
    count = 0
    for i in range(len(occupations)):
        for j in range(i, len(occupations)):
            differences = int(np.count_nonzero(occupations[i] != occupations[j]))
            count += (1 << (m - differences)) - int(i == j)
            if count > maximum_pairs:
                raise ReferenceResourceError("Compatible amplitude-pair budget exceeded",
                    {"compatible_unordered_pairs_lower_bound": count,
                     "max_pairs": maximum_pairs, "basis_size_for_count": len(occupations),
                     "modes": m, "marginals": 1 << m})
    return {"compatible_unordered_pairs": count, "max_pairs": maximum_pairs,
            "basis_size_for_count": len(occupations), "modes": m,
            "marginals": 1 << m, "max_modes": maximum_modes}


def _law_from_marginals(marginals, *, negative_atom_tolerance=DEFAULT_NEGATIVE_ATOM_TOLERANCE,
                        sum_error_tolerance=DEFAULT_SUM_ERROR_TOLERANCE,
                        correction_l1_tolerance=DEFAULT_CORRECTION_L1_TOLERANCE):
    """Superset Möbius inversion, with raw values and every repair reported."""
    q = np.asarray(marginals, dtype=np.longdouble)
    count = len(q)
    if count < 2 or count & (count - 1) or not np.all(np.isfinite(q)):
        raise ValueError("Require finite no-click marginals for a positive mode count")
    if not math.isfinite(negative_atom_tolerance) or negative_atom_tolerance < 0:
        raise ValueError("Invalid negative-atom tolerance")
    if not math.isfinite(correction_l1_tolerance) or correction_l1_tolerance < 0:
        raise ValueError("Invalid correction tolerance")
    if not math.isfinite(sum_error_tolerance) or sum_error_tolerance < 0:
        raise ValueError("Invalid sum-error tolerance")
    transformed = q.copy()
    m = count.bit_length() - 1
    indices = np.arange(count)
    for bit in range(m):
        targets = indices[(indices & (1 << bit)) == 0]
        transformed[targets] -= transformed[targets | (1 << bit)]
    raw = transformed[(count - 1) ^ indices]
    raw_mass = np.sum(raw, dtype=np.longdouble)
    if not np.all(np.isfinite(raw)) or not np.isfinite(raw_mass):
        raise ReferenceNumericalError("Möbius inversion produced nonfinite values")
    negative_mass = -np.sum(raw[raw < 0], dtype=np.longdouble)
    minimum = np.min(raw)
    if abs(raw_mass - 1) > sum_error_tolerance:
        raise ReferenceNumericalError(f"Möbius mass error {float(abs(raw_mass - 1)):.17g} exceeds its cap")
    if minimum < -negative_atom_tolerance:
        raise ReferenceNumericalError(f"Möbius result has negative atom {float(minimum):.17g}")
    repaired = np.maximum(raw, np.longdouble(0))
    mass = np.sum(repaired, dtype=np.longdouble)
    if mass <= 0:
        raise ReferenceNumericalError("No positive probability mass remains")
    repaired /= mass
    correction = np.sum(np.abs(repaired - raw), dtype=np.longdouble)
    if correction > correction_l1_tolerance:
        raise ReferenceNumericalError(f"Möbius repair L1 {float(correction):.17g} exceeds its cap")
    repaired_output, raw_output = np.asarray(repaired, dtype=float), np.asarray(raw, dtype=float)
    exported_correction = np.sum(np.abs(repaired_output.astype(np.longdouble) - raw_output.astype(np.longdouble)),
                                dtype=np.longdouble)
    if exported_correction > correction_l1_tolerance:
        raise ReferenceNumericalError("Exported probability correction exceeds its cap")
    return {"distribution": repaired_output,
            "raw_distribution": raw_output,
            "vacuum_marginals": np.asarray(q, dtype=float),
            "diagnostics": {"raw_mass": float(raw_mass),
                "raw_sum_error": float(abs(raw_mass - 1)),
                "raw_minimum": float(minimum), "raw_negative_mass": float(negative_mass),
                "correction_l1": float(exported_correction),
                "internal_longdouble_correction_l1": float(correction),
                "correction_max_absolute": float(np.max(np.abs(repaired - raw))),
                "raw_export_conversion_l1": float(np.sum(np.abs(raw_output.astype(np.longdouble) - raw), dtype=np.longdouble)),
                "corrected_export_conversion_l1": float(np.sum(np.abs(repaired_output.astype(np.longdouble) - repaired), dtype=np.longdouble)),
                "negative_atom_tolerance": float(negative_atom_tolerance),
                "sum_error_tolerance": float(sum_error_tolerance),
                "correction_l1_tolerance": float(correction_l1_tolerance),
                "repair_rule": "zero negative atoms then divide by remaining mass; raw values retained",
                "outward_rounded_certificate": False}}


def fixed_displacement_law(state, alpha, *, max_modes=DEFAULT_MAX_MODES,
                           max_pairs=DEFAULT_MAX_PAIRS,
                           negative_atom_tolerance=DEFAULT_NEGATIVE_ATOM_TOLERANCE,
                           sum_error_tolerance=DEFAULT_SUM_ERROR_TOLERANCE,
                           correction_l1_tolerance=DEFAULT_CORRECTION_L1_TOLERANCE):
    """Direct finite-Fock vacuum contractions, independent of hidden channels."""
    start = perf_counter()
    occupations, amplitudes, cutoff = _state_arrays(state)
    m = occupations.shape[1]
    work = reference_work_estimate(state, max_modes=max_modes, max_pairs=max_pairs)
    alpha = np.asarray(alpha, dtype=np.complex128)
    if alpha.shape != (m,) or not np.all(np.isfinite(alpha)):
        raise ValueError("alpha must contain one finite complex displacement per mode")
    local = np.empty((m, cutoff + 1), dtype=np.complex128)
    local[:, 0] = np.exp(-0.5 * np.abs(alpha) ** 2)
    for n in range(1, cutoff + 1):
        local[:, n] = local[:, n - 1] * (-alpha) / math.sqrt(n)
    q = np.empty(1 << m, dtype=np.longdouble)
    q[0] = _amplitude_norm(amplitudes)
    active = amplitudes != 0
    occ, amp = occupations[active], amplitudes[active]
    for mask in range(1, 1 << m):
        modes = tuple(j for j in range(m) if mask >> j & 1)
        factors = np.ones(len(occ), dtype=np.complex128)
        for j in modes:
            factors *= local[j, occ[:, j]].conj()
        value = np.longdouble(0)
        for group in _marginal_groups(occ, modes):
            contracted = np.sum((amp[group] * factors[group]).astype(np.clongdouble), dtype=np.clongdouble)
            value += contracted.real ** 2 + contracted.imag ** 2
        q[mask] = value
    result = _law_from_marginals(q, negative_atom_tolerance=negative_atom_tolerance,
                               sum_error_tolerance=sum_error_tolerance,
                               correction_l1_tolerance=correction_l1_tolerance)
    result["diagnostics"].update(work)
    result["diagnostics"].update({"method": "finite_fock_vacuum_contraction",
                                   "input_state_norm_float": float(q[0]),
                                   "seconds": perf_counter() - start})
    return result


def fixed_displacement_direct_patterns(state, alpha, *, max_pairs=100_000):
    """Separate small-fixture reference from products of E0 and I-E0.

    Unlike fixed_displacement_law, this does not use vacuum marginals or a
    Möbius transform.  Its deliberately conservative D**2 * 2**m cap confines
    it to independent measurement gates, not the formal reference panel.
    """
    occupations, amplitudes, cutoff = _state_arrays(state)
    m = occupations.shape[1]
    work = len(occupations) ** 2 * (1 << m)
    if work > max_pairs:
        raise ReferenceResourceError("Direct tensor-POVM fixture exceeds its cap", {"work": work})
    alpha = np.asarray(alpha, dtype=complex)
    if alpha.shape != (m,) or not np.all(np.isfinite(alpha)):
        raise ValueError("Invalid displacement")
    effects = []
    for value in alpha:
        u = np.asarray([math.exp(-abs(value) ** 2 / 2) * (-value) ** n / math.sqrt(math.factorial(n))
                        for n in range(cutoff + 1)])
        e0 = np.outer(u, u.conj())
        effects.append((e0, np.eye(cutoff + 1) - e0))
    result = np.empty(1 << m)
    for mask in range(1 << m):
        total = np.clongdouble(0)
        for i, n in enumerate(occupations):
            for j, other in enumerate(occupations):
                term = np.clongdouble(amplitudes[i].conjugate() * amplitudes[j])
                for mode in range(m):
                    term *= effects[mode][(mask >> mode) & 1][n[mode], other[mode]]
                total += term
        if abs(total.imag) > 5e-12:
            raise ReferenceNumericalError("Direct pattern expectation is not real")
        result[mask] = float(total.real)
    return result


class _WickMoments:
    """Memoized Isserlis recurrence with complex-conjugate key canonicalization."""

    def __init__(self, covariance, modes, max_moments):
        self.G = np.asarray(covariance, dtype=np.clongdouble)
        self.modes = modes
        self.maximum = _integer(max_moments, "max_moments", 1)
        self.cache = {(): np.clongdouble(1)}
        self.hits = 0

    def moment(self, key):
        if len(key) % 2:
            return np.clongdouble(0)
        conjugate = tuple(sorted(i + self.modes if i < self.modes else i - self.modes for i in key))
        if conjugate < key:
            return self.moment(conjugate).conjugate()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        first, remaining = key[0], key[1:]
        result = np.clongdouble(0)
        index = 0
        while index < len(remaining):
            partner = remaining[index]
            end = index + 1
            while end < len(remaining) and remaining[end] == partner:
                end += 1
            rest = remaining[:index] + remaining[index + 1:]
            result += (end - index) * self.G[first, partner] * self.moment(rest)
            index = end
        if len(self.cache) >= self.maximum:
            raise ReferenceResourceError("Per-marginal Wick-moment cache cap exceeded",
                                         {"max_moments": self.maximum, "degree": len(key)})
        self.cache[key] = result
        return result

    def many(self, keys):
        """Vectorized degree 0/2/4, recursion only for higher-degree fixtures."""
        result = np.zeros(len(keys), dtype=np.clongdouble)
        lengths = np.fromiter(map(len, keys), dtype=int, count=len(keys))
        result[lengths == 0] = 1
        for degree in (2, 4):
            take = np.flatnonzero(lengths == degree)
            if not len(take):
                continue
            indices = np.asarray([keys[i] for i in take], dtype=int)
            if degree == 2:
                result[take] = self.G[indices[:, 0], indices[:, 1]]
            else:
                a, b, c, d = indices.T
                result[take] = self.G[a, b] * self.G[c, d] + self.G[a, c] * self.G[b, d] + self.G[a, d] * self.G[b, c]
        for i in np.flatnonzero((lengths > 4) & ((lengths % 2) == 0)):
            result[i] = self.moment(keys[i])
        return result


def _tilted_moments(C, modes, total_modes, max_moments):
    indices = list(modes) + [total_modes + j for j in modes]
    Sigma = C[np.ix_(indices, indices)] / 4.0
    matrix = np.eye(len(indices)) + 2.0 * Sigma
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0 or not math.isfinite(logdet):
        raise ReferenceNumericalError("Gaussian tilt determinant failed")
    T = np.linalg.solve(matrix, Sigma)
    residual = float(np.max(np.abs(T - T.T)))
    if not np.all(np.isfinite(T)) or residual > 1e-12:
        raise ReferenceNumericalError("Tilted covariance failed symmetry/representability guard")
    T = (T + T.T) / 2.0
    z = len(modes)
    A = T[:z, :z] - T[z:, z:] + 1j * (T[:z, z:] + T[z:, :z])
    H = T[:z, :z] + T[z:, z:] + 1j * (T[z:, :z] - T[:z, z:])
    G = np.block([[A, H], [H.T, A.conj()]])
    normalizer = math.exp(-0.5 * logdet)
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ReferenceNumericalError("Gaussian tilt normalizer lost representability")
    return normalizer, _WickMoments(G, z, max_moments), residual


def gaussian_averaged_law(state, displacement_covariance=None, *,
                          max_modes=DEFAULT_MAX_MODES, max_pairs=DEFAULT_MAX_PAIRS,
                          max_moments=DEFAULT_MAX_MOMENTS,
                          negative_atom_tolerance=DEFAULT_NEGATIVE_ATOM_TOLERANCE,
                          sum_error_tolerance=DEFAULT_SUM_ERROR_TOLERANCE,
                          correction_l1_tolerance=DEFAULT_CORRECTION_L1_TOLERANCE):
    """Complete threshold law averaged analytically over a real Gaussian shift.

    Gaussian tilt and finite Wick moments integrate all displacement variables;
    there is no quadrature grid or Monte Carlo noise.  The returned floating law
    records any accepted tiny-negative Möbius correction explicitly.  It is not
    an interval enclosure of the exact law or of its distance from a target.
    """
    start = perf_counter()
    occupations, amplitudes, cutoff = _state_arrays(state)
    m = occupations.shape[1]
    work = reference_work_estimate(state, max_modes=max_modes, max_pairs=max_pairs)
    if displacement_covariance is None:
        if "displacement_covariance" not in state:
            raise ValueError("Provide a displacement covariance for this state")
        displacement_covariance = state["displacement_covariance"]
    C = np.asarray(displacement_covariance, dtype=float)
    if C.shape != (2 * m, 2 * m) or not np.all(np.isfinite(C)):
        raise ValueError("Displacement covariance must be finite, real and in xxpp order")
    symmetry = float(np.max(np.abs(C - C.T)))
    scale = max(1.0, float(np.max(np.abs(C))))
    if symmetry > 1e-12 * scale:
        raise ValueError("Displacement covariance must be symmetric")
    C = (C + C.T) / 2.0
    minimum_eigenvalue = float(np.linalg.eigvalsh(C)[0])
    if minimum_eigenvalue < -1e-12 * scale:
        raise ValueError("Displacement covariance must be positive semidefinite")
    # A truly zero covariance needs no moments and has an independent Fock rule.
    if not np.any(C):
        result = fixed_displacement_law(state, np.zeros(m), max_modes=max_modes,
            max_pairs=max_pairs, negative_atom_tolerance=negative_atom_tolerance,
            sum_error_tolerance=sum_error_tolerance,
            correction_l1_tolerance=correction_l1_tolerance)
        result["diagnostics"].update({"method": "analytic_gaussian_tilt_wick_zero_covariance",
            "seconds": perf_counter() - start, "wick_cache_peak": 0,
            "max_moments": max_moments, "actual_compatible_pairs": 0})
        return result
    active = amplitudes != 0
    occ, amp = occupations[active], amplitudes[active]
    factorial = np.asarray([math.factorial(n) for n in range(cutoff + 1)], dtype=np.longdouble)
    q = np.empty(1 << m, dtype=np.longdouble)
    q[0] = _amplitude_norm(amplitudes)
    actual_pairs = cache_peak = cache_hits = 0
    maximum_tilt_asymmetry = maximum_diagonal_imaginary = 0.0
    for mask in range(1, 1 << m):
        modes = tuple(j for j in range(m) if mask >> j & 1)
        z = len(modes)
        normalizer, moments, asymmetry = _tilted_moments(C, modes, m, max_moments)
        maximum_tilt_asymmetry = max(maximum_tilt_asymmetry, asymmetry)
        local = occ[:, modes]
        sqrt_factorials = np.sqrt(np.prod(factorial[local], axis=1))
        # For bra coefficient c_n*, use alpha**n; for ket, alpha* ** n'.
        repeats = [tuple(j for j in range(z) for _ in range(int(row[j]))) for row in local]
        conjugates = [tuple(j + z for j in key) for key in repeats]
        first, second, keys = [], [], []
        for group in _marginal_groups(occ, modes):
            for place, i in enumerate(group):
                for j in group[place:]:
                    first.append(i)
                    second.append(j)
                    keys.append(repeats[i] + conjugates[j])
        left, right = np.asarray(first, dtype=int), np.asarray(second, dtype=int)
        actual_pairs += len(keys)
        values = moments.many(keys)
        coefficient = amp[left].conj().astype(np.clongdouble) * amp[right].astype(np.clongdouble)
        coefficient /= sqrt_factorials[left] * sqrt_factorials[right]
        # Odd total moments vanish.  The sign also permits arbitrary input states.
        parity = np.asarray([len(key) & 1 for key in keys])
        coefficient[parity == 1] *= -1
        terms = coefficient * values
        diagonal = left == right
        if np.any(diagonal):
            maximum_diagonal_imaginary = max(maximum_diagonal_imaginary,
                float(np.max(np.abs(terms[diagonal].imag))))
            if not math.isfinite(maximum_diagonal_imaginary) or maximum_diagonal_imaginary > 1e-11:
                raise ReferenceNumericalError("Diagonal expectation has a material imaginary part")
        terms_real = terms.real
        terms_real[~diagonal] *= 2
        q[mask] = np.longdouble(normalizer) * np.sum(terms_real, dtype=np.longdouble)
        cache_peak = max(cache_peak, len(moments.cache))
        cache_hits += moments.hits
    result = _law_from_marginals(q, negative_atom_tolerance=negative_atom_tolerance,
                               sum_error_tolerance=sum_error_tolerance,
                               correction_l1_tolerance=correction_l1_tolerance)
    result["diagnostics"].update(work)
    result["diagnostics"].update({"method": "analytic_gaussian_tilt_wick",
        "actual_compatible_pairs": actual_pairs, "max_moments": max_moments,
        "input_state_norm_float": float(q[0]),
        "wick_cache_peak": cache_peak, "wick_cache_hits": cache_hits,
        "input_covariance_symmetry_residual": symmetry,
        "input_covariance_minimum_eigenvalue_float": minimum_eigenvalue,
        "tilted_covariance_max_asymmetry_before_symmetrization": maximum_tilt_asymmetry,
        "tilted_covariance_asymmetry_tolerance": 1e-12,
        "diagonal_expectation_max_imaginary_residual": maximum_diagonal_imaginary,
        "diagonal_expectation_imaginary_tolerance": 1e-11,
        "maximum_moment_degree": 2 * cutoff,
        "exact_zero_coefficients_omitted": int(np.sum(~active)),
        "seconds": perf_counter() - start})
    return result
