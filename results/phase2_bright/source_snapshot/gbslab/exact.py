"""Exact small-mode threshold Gaussian boson sampling reference.

For a zero-mean Gaussian state with hbar=2 covariance V, the joint no-click
probability on a set S is det((V_S + I)/2)**(-1/2).  Inclusion-exclusion turns
these vacuum marginals into a threshold distribution.  This is exponential
in the number of requested modes and is a reference, not a proposed scalable
classical simulation algorithm.

Probability vector index x is a little-endian bit mask: bit j=1 means a click
in the j-th mode of the supplied ``modes`` tuple.  Arrays returned from the
cache are read-only to prevent accidental corruption.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class NumericalProbabilityError(ArithmeticError):
    """A probability computation has failed a numerical consistency check."""


class ThresholdModel:
    """Gaussian covariance plus cached exact threshold marginals.

    ``max_exact_modes`` caps each marginal enumeration, not the global state
    size.  Covariance physicality is checked once at construction using the
    uncertainty relation V + i Omega >= 0.  The correction diagnostics expose
    small negative probabilities clipped after floating-point cancellation;
    negativity beyond ``probability_tolerance`` raises instead of being hidden.
    ``vacuum_cache_size=None`` preserves unlimited caching for small exact
    reference work.  Set a nonnegative integer for an LRU cache of that maximum
    size during larger-state sampling; zero disables vacuum-result caching.
    The separate exact-marginal cache is not capped by this argument.
    """

    def __init__(self, V, *, probability_tolerance=2e-10,
                 physicality_tolerance=1e-9, max_exact_modes=20,
                 check_physicality=True, vacuum_cache_size=None):
        V = np.asarray(V, dtype=float)
        if (V.ndim != 2 or V.shape[0] != V.shape[1]
                or V.shape[0] == 0 or V.shape[0] % 2):
            raise ValueError("V must be a nonempty even-dimensional square matrix")
        if not np.all(np.isfinite(V)):
            raise ValueError("V must be finite")
        scale = max(1.0, float(np.max(np.abs(V))))
        if not np.allclose(V, V.T, atol=1e-10 * scale, rtol=0):
            raise ValueError("V must be symmetric")
        if probability_tolerance <= 0 or physicality_tolerance <= 0:
            raise ValueError("numerical tolerances must be positive")
        if not isinstance(max_exact_modes, (int, np.integer)) or max_exact_modes < 0:
            raise ValueError("max_exact_modes must be a nonnegative integer")
        if (vacuum_cache_size is not None
                and (not isinstance(vacuum_cache_size, (int, np.integer))
                     or vacuum_cache_size < 0)):
            raise ValueError("vacuum_cache_size must be None or a nonnegative integer")
        self.V = ((V + V.T) / 2.0).copy()
        self.V.setflags(write=False)
        self.m = V.shape[0] // 2
        self.probability_tolerance = float(probability_tolerance)
        self.max_exact_modes = int(max_exact_modes)
        self.vacuum_cache_size = (None if vacuum_cache_size is None
                                  else int(vacuum_cache_size))
        if check_physicality:
            eye = np.eye(self.m)
            zero = np.zeros_like(eye)
            omega = np.block([[zero, eye], [-eye, zero]])
            min_eig = float(np.linalg.eigvalsh(self.V + 1j * omega)[0])
            if min_eig < -physicality_tolerance * scale:
                raise ValueError(f"V violates Gaussian uncertainty relation: {min_eig:g}")
            self.minimum_uncertainty_eigenvalue = min_eig
        else:
            self.minimum_uncertainty_eigenvalue = None
        self._vacuum_cache = OrderedDict()
        if self.vacuum_cache_size != 0:
            self._vacuum_cache[()] = 1.0
        empty = np.array([1.0])
        empty.setflags(write=False)
        self._marginal_cache = {(): empty}
        self.diagnostics = {
            "determinant_evaluations": 0,
            "marginal_evaluations": 0,
            "negative_entries_clipped": 0,
            "largest_negative_probability": 0.0,
            "largest_normalization_error": 0.0,
            "largest_l1_numerical_correction": 0.0,
            "vacuum_cache_hits": 0,
            "vacuum_cache_evictions": 0,
            "vacuum_cache_peak_entries": len(self._vacuum_cache),
        }

    def _modes(self, modes) -> tuple[int, ...]:
        modes = tuple(modes)
        if any(not isinstance(j, (int, np.integer)) for j in modes):
            raise ValueError("mode indices must be integers")
        result = tuple(int(j) for j in modes)
        if len(set(result)) != len(result):
            raise ValueError("mode indices cannot repeat")
        if any(j < 0 or j >= self.m for j in result):
            raise ValueError("mode index out of range")
        return result

    def vacuum_probability(self, modes_tuple) -> float:
        """Probability of no clicks on every supplied mode, ignoring others."""
        modes = tuple(sorted(self._modes(modes_tuple)))
        cached = self._vacuum_cache.get(modes)
        if cached is not None:
            self._vacuum_cache.move_to_end(modes)
            self.diagnostics["vacuum_cache_hits"] += 1
            return cached
        if not modes:
            return 1.0
        indices = np.array(modes + tuple(j + self.m for j in modes), dtype=int)
        submatrix = self.V[np.ix_(indices, indices)]
        q_matrix = (submatrix + np.eye(len(indices))) / 2.0
        sign, logdet = np.linalg.slogdet(q_matrix)
        self.diagnostics["determinant_evaluations"] += 1
        if sign <= 0 or not np.isfinite(logdet):
            raise NumericalProbabilityError("vacuum probability determinant is not positive finite")
        probability = float(np.exp(-0.5 * logdet))
        if probability > 1.0 + self.probability_tolerance:
            raise NumericalProbabilityError(f"vacuum probability exceeds one: {probability}")
        probability = min(1.0, probability)
        if self.vacuum_cache_size != 0:
            if (self.vacuum_cache_size is not None
                    and len(self._vacuum_cache) >= self.vacuum_cache_size):
                self._vacuum_cache.popitem(last=False)
                self.diagnostics["vacuum_cache_evictions"] += 1
            self._vacuum_cache[modes] = probability
            self.diagnostics["vacuum_cache_peak_entries"] = max(
                self.diagnostics["vacuum_cache_peak_entries"], len(self._vacuum_cache))
        return probability

    def marginal(self, modes_tuple) -> np.ndarray:
        """Exact click distribution, in precisely the supplied mode order.

        Floating-point determinant evaluation and cancellation limit numerical
        accuracy.  Here 'exact' distinguishes complete inclusion-exclusion from
        model approximation; it does not mean arbitrary precision arithmetic.
        """
        modes = self._modes(modes_tuple)
        cached = self._marginal_cache.get(modes)
        if cached is not None:
            return cached
        k = len(modes)
        if k > self.max_exact_modes:
            raise ValueError(f"requested {k} modes exceeds exact cap {self.max_exact_modes}")
        count = 1 << k
        # Long double reduces cancellation in the Mobius transform, although
        # the determinant inputs remain float64 and bound the final accuracy.
        zeros = np.empty(count, dtype=np.longdouble)
        for mask in range(count):
            subset = tuple(modes[j] for j in range(k) if mask & (1 << j))
            zeros[mask] = self.vacuum_probability(subset)
        # Upper-set Mobius inversion: zeros[z] becomes P(exact zero set=z).
        for bit in range(k):
            stride = 1 << bit
            blocks = zeros.reshape(-1, 2 * stride)
            blocks[:, :stride] -= blocks[:, stride:]
        probabilities = np.asarray(zeros[::-1], dtype=float).copy()
        minimum = float(np.min(probabilities))
        raw_sum = float(np.sum(probabilities, dtype=np.longdouble))
        norm_error = abs(raw_sum - 1.0)
        self.diagnostics["marginal_evaluations"] += 1
        self.diagnostics["largest_negative_probability"] = max(
            self.diagnostics["largest_negative_probability"], max(0.0, -minimum))
        self.diagnostics["largest_normalization_error"] = max(
            self.diagnostics["largest_normalization_error"], norm_error)
        if minimum < -self.probability_tolerance:
            raise NumericalProbabilityError(
                f"negative probability {minimum:.3g} for marginal {modes}; "
                "inclusion-exclusion is numerically unstable at this precision")
        if norm_error > self.probability_tolerance:
            raise NumericalProbabilityError(f"marginal normalization error {norm_error:.3g}")
        original = probabilities.copy()
        negative = probabilities < 0
        self.diagnostics["negative_entries_clipped"] += int(np.count_nonzero(negative))
        probabilities[negative] = 0.0
        probabilities /= np.sum(probabilities, dtype=np.longdouble)
        correction = float(np.sum(np.abs(probabilities - original)))
        self.diagnostics["largest_l1_numerical_correction"] = max(
            self.diagnostics["largest_l1_numerical_correction"], correction)
        if correction > 10.0 * self.probability_tolerance:
            raise NumericalProbabilityError(
                f"aggregate L1 numerical correction {correction:.3g} exceeds tolerance; "
                "use fewer modes or a higher-precision reference")
        probabilities.setflags(write=False)
        self._marginal_cache[modes] = probabilities
        return probabilities

    def full_distribution(self) -> np.ndarray:
        """Enumerate all 2**m click patterns; intentionally exponential."""
        return self.marginal(tuple(range(self.m)))

    def cache_info(self) -> dict:
        """Return counters and persistent cache sizes for honest accounting."""
        return {
            **self.diagnostics,
            "vacuum_cache_entries": len(self._vacuum_cache),
            "vacuum_cache_size_limit": self.vacuum_cache_size,
            "marginal_cache_entries": len(self._marginal_cache),
            "marginal_cache_bytes": sum(v.nbytes for v in self._marginal_cache.values()),
        }
