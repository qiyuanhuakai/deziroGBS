"""Scaled MPS overlaps and cancellation-resistant, phase-aligned distance.

All calculations use complex128; these are diagnostics, not interval bounds.
"""
import math
import numpy as np


def validate_mps(tensors):
    if not tensors:
        raise ValueError('Empty MPS')
    previous = 1
    for a in tensors:
        a = np.asarray(a)
        if a.ndim != 3 or min(a.shape) < 1 or a.shape[0] != previous or not np.all(np.isfinite(a)):
            raise ValueError('Invalid MPS tensor or bond')
        previous = a.shape[2]
    if previous != 1:
        raise ValueError('MPS right boundary is not scalar')


def scaled_overlap(left, right):
    """Return log(abs(<left|right>)) and phase, without forming huge norms."""
    validate_mps(left); validate_mps(right)
    if len(left) != len(right) or any(a.shape[1] != b.shape[1] for a, b in zip(left, right)):
        raise ValueError('Incompatible physical spaces')
    environment = np.ones((1, 1), complex); log_scale = 0.
    for a, b in zip(left, right):
        sa, sb = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
        if not sa or not sb:
            return dict(zero=True, log_abs=None, phase=[0., 0.])
        a, b = np.asarray(a, complex)/sa, np.asarray(b, complex)/sb
        temporary = (environment @ b.reshape(b.shape[0], -1)).reshape(a.shape[0]*a.shape[1], b.shape[2])
        environment = a.reshape(-1, a.shape[2]).conj().T @ temporary
        scale = float(np.linalg.norm(environment))
        if not scale:
            return dict(zero=True, log_abs=None, phase=[0., 0.])
        if not np.isfinite(scale):
            raise ArithmeticError('Nonfinite overlap environment')
        environment /= scale
        log_scale += math.log(sa) + math.log(sb) + math.log(scale)
    value = complex(environment[0, 0])
    return dict(zero=False, log_abs=log_scale+math.log(abs(value)), phase=[value.real/abs(value), value.imag/abs(value)])


def normalized_right_mps(tensors):
    """Exact QR, no truncation. Return unit-norm tensors and log input norm."""
    validate_mps(tensors)
    result = []; log_scale = 0.
    for a in tensors:
        scale = float(np.max(np.abs(a)))
        if not scale: raise ValueError('Zero MPS')
        result.append(np.asarray(a, complex)/scale); log_scale += math.log(scale)
    for i in range(len(result)-1, 0, -1):
        a = result[i]; l, d, r = a.shape
        q, carry = np.linalg.qr(a.reshape(l, d*r).T, mode='reduced')
        result[i] = q.T.reshape(q.shape[1], d, r)
        result[i-1] = np.tensordot(result[i-1], carry.T, axes=(2, 0))
        scale = float(np.max(np.abs(result[i-1])))
        if not scale: raise ValueError('Zero MPS')
        result[i-1] /= scale; log_scale += math.log(scale)
    norm = float(np.linalg.norm(result[0]))
    if not norm: raise ValueError('Zero MPS')
    result[0] /= norm; log_scale += math.log(norm)
    return result, log_scale


def difference_norm(left, right, phase):
    """Streaming QR of left - phase*right; input states must be normalized."""
    if len(left) == 1:
        return float(np.linalg.norm(left[0]-phase*right[0]))
    work = np.concatenate((left[0], -phase*right[0]), axis=2)
    _, carry = np.linalg.qr(work.reshape(-1, work.shape[2]), mode='reduced')
    for site in range(1, len(left)):
        a, b = left[site], right[site]
        first = np.tensordot(carry[:, :a.shape[0]], a, axes=(1, 0))
        second = np.tensordot(carry[:, a.shape[0]:], b, axes=(1, 0))
        if site == len(left)-1:
            return float(np.linalg.norm(first+second))
        work = np.concatenate((first, second), axis=2)
        _, carry = np.linalg.qr(work.reshape(-1, work.shape[2]), mode='reduced')


def compare_mps(left, right):
    aa, bb, ab = scaled_overlap(left, left), scaled_overlap(right, right), scaled_overlap(left, right)
    if aa['zero'] or bb['zero']: raise ValueError('Cannot normalize zero MPS')
    for norm in (aa, bb):
        if abs(complex(*norm['phase'])-1) > 1e-10: raise ArithmeticError('Nonpositive computed norm')
    log_fidelity = None if ab['zero'] else 2*ab['log_abs']-aa['log_abs']-bb['log_abs']
    if log_fidelity is not None and log_fidelity > 1e-9:
        raise ArithmeticError('Overlap exceeds Cauchy-Schwarz tolerance')
    phase = 1.+0j if ab['zero'] else complex(*ab['phase']).conjugate()
    a, log_a = normalized_right_mps(left); b, log_b = normalized_right_mps(right)
    delta = difference_norm(a, b, phase)
    if delta > math.sqrt(2)+1e-9: raise ArithmeticError('Phase-aligned distance exceeds sqrt(2)')
    distance = delta*math.sqrt(max(0., 1-delta*delta/4))
    return dict(log_norm_squared_left=aa['log_abs'], log_norm_squared_right=bb['log_abs'], raw_log_fidelity=log_fidelity,
        fidelity_from_overlap=0. if log_fidelity is None else math.exp(min(0., log_fidelity)),
        overlap_positive_log_clipped=bool(log_fidelity is not None and log_fidelity>0),
        phase_aligned_difference_norm=delta, pure_trace_distance_from_difference=distance,
        log_norm_qr_residual_left=2*log_a-aa['log_abs'], log_norm_qr_residual_right=2*log_b-bb['log_abs'],
        phase_alignment=[phase.real, phase.imag], arithmetic='complex128 diagnostics; no finite precision certificate')
