"""Small coherent MPS prototype and untruncated Gaussian Schmidt diagnostics.

Dense-to-MPS construction is a validation tool, NOT a scalable Gaussian-to-MPS
algorithm. Displaced threshold effects include all output photons exactly on
the finite input space. Float64 diagnostics are not interval certificates.
"""
from __future__ import annotations

import heapq
import math
import numpy as np


def symplectic_spectrum(covariance):
    """Positive eigenvalues of i sqrt(V) Omega sqrt(V), vacuum = 1."""
    v = np.asarray(covariance, dtype=float)
    if (v.ndim != 2 or v.shape[0] != v.shape[1] or not len(v)
            or len(v) % 2 or not np.all(np.isfinite(v))
            or not np.allclose(v, v.T, atol=1e-11, rtol=0)):
        raise ValueError("Require finite symmetric even square covariance")
    w, q = np.linalg.eigh(v)
    if w[0] <= 0:
        raise ValueError("Covariance is not positive definite")
    root = (q * np.sqrt(w)) @ q.T
    n = len(v) // 2
    omega = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-np.eye(n), np.zeros((n, n))]])
    nu = np.linalg.eigvalsh(1j * root @ omega @ root)[n:]
    if nu[0] < 1 - 1e-9:
        raise ValueError("Covariance violates uncertainty within tolerance")
    return np.maximum(nu, 1.)


def schmidt_parameters(pure_covariance, left_modes):
    """Geometric parameters of squared Schmidt coefficients across one cut."""
    v = np.asarray(pure_covariance, dtype=float)
    symplectic = symplectic_spectrum(v)
    if np.max(np.abs(symplectic - 1)) > 1e-8:
        raise ValueError("Schmidt spectrum requires a pure Gaussian state")
    m = len(v) // 2
    left = list(left_modes)
    if (not left or len(left) >= m or len(set(left)) != len(left)
            or any(not isinstance(j, (int, np.integer)) or j < 0 or j >= m for j in left)):
        raise ValueError("Require a nonempty proper subset of modes")
    # The shorter side removes exactly pure spectator modes, not small weights.
    if len(left) > m // 2:
        left = [j for j in range(m) if j not in left]
    idx = left + [j + m for j in left]
    nu = symplectic_spectrum(v[np.ix_(idx, idx)])
    return (nu - 1) / (nu + 1)


def geometric_top_weights(parameters, max_rank):
    """Largest product-geometric weights without a Fock box or tail cutoff.

    A unique-parent tree enumerates each occupation once: increment coordinate
    j only if all coordinates below j are zero. Every child has <= parent weight.
    Heap size is bounded by max_rank * number_of_nonzero_parameters.
    """
    q = np.asarray(parameters, dtype=float)
    if q.ndim != 1 or not np.all(np.isfinite(q)) or np.any(q < 0) or np.any(q >= 1):
        raise ValueError("Geometric parameters must be in [0,1)")
    if isinstance(max_rank, bool) or not isinstance(max_rank, (int, np.integer)) or max_rank < 1:
        raise ValueError("max_rank must be a positive integer")
    q = q[q > 0]
    log0 = float(np.log1p(-q).sum())
    logs = np.log(q)
    heap = [(-log0, (0,) * len(q))]
    values = []
    while heap and len(values) < max_rank:
        energy, occ = heapq.heappop(heap)
        values.append(math.exp(-energy))
        for j in range(len(q)):
            child = list(occ)
            child[j] += 1
            heapq.heappush(heap, (energy - logs[j], tuple(child)))
            if occ[j]:
                break
    return np.asarray(values)


def dense_to_mps(tensor, max_bond, *, max_elements=2_000_000):
    """Sequential SVD with measured state error; returns right-canonical MPS."""
    x = np.asarray(tensor, dtype=np.complex128)
    if x.ndim < 1 or not x.size or x.size > max_elements or not np.all(np.isfinite(x)):
        raise ValueError("Invalid tensor or dense construction resource cap")
    if isinstance(max_bond, bool) or not isinstance(max_bond, (int, np.integer)) or max_bond < 1:
        raise ValueError("max_bond must be a positive integer")
    if abs(np.linalg.norm(x.ravel()) - 1) > 1e-10:
        raise ValueError("Input tensor must be normalized")
    work = x.copy()
    tensors, discarded = [], []
    rank = 1
    for d in x.shape[:-1]:
        u, s, vh = np.linalg.svd(work.reshape(rank * d, -1), full_matrices=False)
        keep = min(max_bond, len(s))
        discarded.append(float(np.sum(s[keep:] ** 2)))
        tensors.append(u[:, :keep].reshape(rank, d, keep))
        work = s[:keep, None] * vh[:keep]
        rank = keep
    norm = np.linalg.norm(work)
    if norm == 0:
        raise ArithmeticError("Compression lost the complete state")
    tensors.append(work.reshape(rank, x.shape[-1], 1) / norm)
    for j in range(len(tensors) - 1, 0, -1):
        a = tensors[j]
        q, r = np.linalg.qr(a.reshape(a.shape[0], -1).T, mode="reduced")
        tensors[j] = q.T.reshape(q.shape[1], a.shape[1], a.shape[2])
        tensors[j - 1] = np.tensordot(tensors[j - 1], r.T, axes=(-1, 0))
    reconstructed = mps_to_dense(tensors)
    overlap = np.vdot(x.ravel(), reconstructed.ravel())
    # Phase-aligned vector distance is stable when trace distance is tiny.
    phase = overlap / abs(overlap) if abs(overlap) else 1.
    distance2 = float(np.linalg.norm(x - reconstructed / phase) ** 2)
    trace_distance = math.sqrt(max(0., distance2 * (1 - distance2 / 4)))
    return tensors, {"discarded_weights": discarded, "discarded_sum": sum(discarded),
        "state_trace_distance": trace_distance,
        "fidelity": float(abs(overlap) ** 2),
        "bonds": [a.shape[2] for a in tensors[:-1]],
        "stored_complex_elements": sum(a.size for a in tensors),
        "dense_elements": x.size}


def mps_to_dense(tensors):
    x = tensors[0]
    for a in tensors[1:]:
        x = np.tensordot(x, a, axes=(-1, 0))
    return x[0, ..., 0]


def threshold_channels(alpha, dimension):
    """Rows K: vacuum bra then rows of sqrt(I - |v><v|), v_n=<-alpha|n>*.

    The hidden click row is marginalized, never exposed as marginal logQ.
    The stable coefficient 1/(1+sqrt(1-||v||^2)) avoids division by tiny ||v||.
    """
    if not np.isfinite(alpha) or dimension < 1:
        raise ValueError("Invalid displacement or dimension")
    v = np.empty(dimension, dtype=complex)
    v[0] = math.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, dimension):
        v[n] = v[n - 1] * (-alpha) / math.sqrt(n)
    norm2 = float(np.vdot(v, v).real)
    if norm2 > 1 + 1e-12:
        raise ArithmeticError("Vacuum effect norm exceeds one")
    click = np.eye(dimension) - np.outer(v, v.conj()) / (1 + math.sqrt(max(0., 1 - norm2)))
    return np.vstack((v.conj(), click))


def fixed_threshold_law(tensors, alpha, *, max_modes=12):
    """Full finite-input law via MPS effect contractions; exponential output."""
    alpha = np.asarray(alpha, dtype=complex)
    m = len(tensors)
    if m > max_modes or alpha.shape != (m,) or not np.all(np.isfinite(alpha)):
        raise ValueError("Invalid displacements or full-law resource cap")
    effects = []
    for a, z in zip(tensors, alpha):
        channels = threshold_channels(z, a.shape[1])
        effects.append([np.outer(channels[0].conj(), channels[0]),
                        channels[1:].conj().T @ channels[1:]])
    law = np.empty(1 << m)
    for mask in range(1 << m):
        env = np.ones((1, 1), dtype=complex)
        for j, a in enumerate(tensors):
            env = np.einsum("ab,anr,bms,mn->rs", env, a, a.conj(), effects[j][mask >> j & 1], optimize=True)
        if abs(env[0, 0].imag) > 1e-10 or env[0, 0].real < -1e-10:
            raise ArithmeticError("Invalid probability contraction")
        law[mask] = env[0, 0].real
    if abs(law.sum() - 1) > 1e-9:
        raise ArithmeticError("Threshold law normalization failed")
    return law


def sample_threshold(tensors, displacements, rng):
    """Sequential hidden rank-one channel sampling from right-canonical MPS.

    Each supplied displacement row is one sample; correlated Gaussian draws
    must be prepared by the caller. Any invalid conditional aborts the call.
    """
    alpha = np.asarray(displacements, dtype=complex)
    if alpha.ndim != 2 or alpha.shape[1] != len(tensors) or not np.all(np.isfinite(alpha)):
        raise ValueError("Require finite sample by mode displacement matrix")
    for a in tensors[1:]:
        flat = a.reshape(a.shape[0], -1)
        if not np.allclose(flat @ flat.conj().T, np.eye(len(flat)), atol=1e-10, rtol=0):
            raise ValueError("Require right-canonical MPS")
    out = np.zeros(alpha.shape, dtype=np.uint8)
    for i, row in enumerate(alpha):
        left = np.ones(1, dtype=complex)
        for j, (a, z) in enumerate(zip(tensors, row)):
            local = np.tensordot(left, a, axes=(0, 0))
            branches = threshold_channels(z, a.shape[1]) @ local
            weights = np.sum(np.abs(branches) ** 2, axis=1)
            total = float(weights.sum())
            if not np.all(np.isfinite(weights)) or abs(total - 1) > 1e-9:
                raise ArithmeticError(f"Conditional normalization failure at sample {i}, mode {j}")
            channel = int(rng.choice(len(weights), p=weights / total))
            if weights[channel] <= 0:
                raise ArithmeticError("Selected zero-probability branch")
            out[i, j] = channel != 0
            left = branches[channel] / math.sqrt(weights[channel])
    return out
