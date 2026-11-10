"""Finite Gaussian Fock coefficients by annihilation recurrence.

Q uses exp(z.T @ Q @ z / 2). No normalizability or norm is assumed.
Downward closure makes every recursive predecessor available exactly.
"""
from functools import lru_cache
import math
import numpy as np
from .gaussian_schmidt_mps import _validate_basis


def pair_recurrence_tensor(Q, left_basis, dimension, right_basis, *, max_elements=4_000_000):
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 24:
        raise ValueError('Invalid physical dimension')
    groups = []
    for basis in (left_basis, right_basis):
        b = np.asarray(basis)
        if b.dtype.kind not in 'iu' or np.any(b > np.iinfo(np.int64).max):
            raise ValueError('Basis must have integer occupations')
        groups.append(_validate_basis(b)[0])
    left, right = groups
    shape = (len(left), dimension, len(right))
    if math.prod(shape) > max_elements:
        raise MemoryError('Local coefficient allocation cap')
    physical = left.shape[1]
    offset = physical + 1
    total = offset + right.shape[1]
    Q = np.asarray(Q, complex)
    if Q.shape != (total, total) or not np.all(np.isfinite(Q)) or not np.allclose(Q, Q.T, atol=1e-10, rtol=0):
        raise ValueError('Invalid local pairing kernel')
    tables, nonzero, vacuum, totals, roots = [], [], [], [], []
    for basis in groups:
        lookup = {tuple(n): i for i, n in enumerate(basis)}
        vacuum.append(lookup[(0,) * basis.shape[1]])
        pred = np.full(basis.shape, -1, int)
        nz = []
        for row, n in enumerate(basis):
            occupied = tuple(int(j) for j in np.flatnonzero(n))
            nz.append(occupied)
            for j in occupied:
                sub = n.copy(); sub[j] -= 1
                pred[row, j] = lookup[tuple(sub)]
        tables.append(pred); nonzero.append(nz)
        totals.append(basis.sum(axis=1)); roots.append(np.sqrt(basis))

    @lru_cache(None)
    def coefficient(a, p, b):
        if a == vacuum[0] and p == 0 and b == vacuum[1]:
            return 1. + 0j
        if nonzero[0][a]:
            j = nonzero[0][a][0]; pivot = j; divisor = roots[0][a, j]
            a = int(tables[0][a, j])
        elif p:
            pivot = physical; divisor = math.sqrt(p); p -= 1
        else:
            j = nonzero[1][b][0]; pivot = offset + j; divisor = roots[1][b, j]
            b = int(tables[1][b, j])
        value = 0j
        for j in nonzero[0][a]:
            value += Q[pivot, j] * roots[0][a, j] * coefficient(int(tables[0][a, j]), p, b)
        if p:
            value += Q[pivot, physical] * math.sqrt(p) * coefficient(a, p - 1, b)
        for j in nonzero[1][b]:
            value += Q[pivot, offset + j] * roots[1][b, j] * coefficient(a, p, int(tables[1][b, j]))
        return value / divisor

    value = np.zeros(shape, complex)
    for a in range(len(left)):
        for p in range(dimension):
            for b in range(len(right)):
                if (int(totals[0][a]) + p + int(totals[1][b])) % 2 == 0:
                    value[a, p, b] = coefficient(a, p, b)
    if not np.all(np.isfinite(value)):
        raise ArithmeticError('Nonfinite Gaussian coefficient')
    return value, dict(cache_entries=coefficient.cache_info().currsize, elements=value.size)
