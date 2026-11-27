"""Coherent-state projection of a phase acting on one shared supermode of repeated cells."""
import argparse
import itertools
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits


def realify(u):
    return np.block([[u.real, -u.imag], [u.imag, u.real]])


def prepare(optical, bus, theta):
    u, r, eta = (optical[k] for k in ('unitary', 'squeezing', 'efficiency'))
    d = len(r)
    a, b = np.diag(np.sqrt(eta)), np.diag(np.sqrt(1-eta))
    passive = np.block([[a @ u, b], [-b @ u, a]])
    squeeze = np.r_[-r, np.zeros(d), r, np.zeros(d)]
    local = realify(passive) * np.exp(squeeze)
    observed = local[np.r_[0:d, 2*d:3*d]]
    assert np.max(abs((observed @ observed.T-np.eye(2*d))/2-optical['core_x'])) < 1e-12
    phase = np.eye(2*d, dtype=complex)
    phase[:d, :d] += np.expm1(1j*theta)*np.outer(bus, bus.conj())
    transform = np.linalg.solve(local, realify(phase) @ local)
    cov = transform @ transform.T
    vals, vec = np.linalg.eigh(cov)
    active = vals > 1+1e-11
    s = np.log(vals[active])/2
    axes = -1j*(vec[:2*d, active]+1j*vec[2*d:, active])
    assert len(s) <= 2
    basis = realify(axes)
    np.testing.assert_allclose(np.eye(4*d)+(basis*np.expm1(np.r_[-2*s, 2*s])) @ basis.T, cov, atol=2e-12)
    return s, axes, observed


def project(s, nodes):
    roots = np.polynomial.hermite.hermgauss(nodes)[0]
    def fit(scale):
        x = roots*scale
        gram = np.exp(-.5*(x[:, None]-x)**2)
        overlap = np.exp(-.5*(1-np.tanh(s))*x*x)/np.sqrt(np.cosh(s))
        weights = np.linalg.solve(gram, overlap)
        return 1-weights@overlap
    opt = minimize_scalar(fit, bounds=(.3, 2.4), method='bounded')
    x = roots*opt.x
    with mp.workdps(60):
        points = list(map(mp.mpf, x.tolist()))
        gram = mp.matrix([[mp.exp(-(a-b)**2/2) for a in points] for b in points])
        overlap = mp.matrix([mp.exp(-(1-mp.tanh(s))*a*a/2)/mp.sqrt(mp.cosh(s)) for a in points])
        coeff = mp.lu_solve(gram, overlap)
        norm = (coeff.T*overlap)[0]
        return x, np.array([float(c/mp.sqrt(norm)) for c in coeff]), float(1-norm)


def branches(s, axes, m):
    quadratures = [project(v, m) for v in s]
    positions = np.array(list(itertools.product(*[p[0] for p in quadratures]))).reshape(m**len(s), len(s))
    coeff = np.array([math.prod(t) for t in itertools.product(*[p[1] for p in quadratures])])
    loss = -np.expm1(sum(np.log1p(-p[2]) for p in quadratures))
    gram = np.exp(-.5*((positions[:, None]-positions)**2).sum(-1))
    assert abs(coeff@gram@coeff-1)<2e-12
    return 1j*positions@axes.T, coeff, gram, math.sqrt(loss)


def dyads(zeta, observed, cells):
    ket = zeta[:, None, :]/np.sqrt(cells)
    bra = zeta.conj()[None, :, :]/np.sqrt(cells)
    return np.concatenate([(ket+bra)/2, (ket-bra)/(2j)], axis=-1) @ observed.T


def click_table(x, disp):
    d = len(x)//2
    vac = np.ones((2**d, *disp.shape[:2]), complex)
    for mask in range(1, 2**d):
        bits = np.flatnonzero(mask & (1 << np.arange(d)))
        select = np.r_[bits, bits+d]
        q = np.eye(len(select))+x[np.ix_(select, select)]
        v = disp[..., select]
        vac[mask] = np.exp(-np.sum((v@np.linalg.inv(q))*v, axis=-1))/np.sqrt(np.linalg.det(q))
    table = vac[::-1].copy()
    for bit in range(d):
        chunks = table.reshape(-1, 2, 1 << bit, *disp.shape[:2])
        chunks[:, 1] -= chunks[:, 0]
    return table


def complete_clicks(table, weights):
    # Contract two cells per side so that all 65,536 probabilities share one matrix product.
    pair = (table[:, None]*table[None, :]).reshape(256, -1)
    law = ((pair*weights.ravel())@pair.T).real.ravel()
    assert min(law)>-1e-13 and abs(law.sum()-1)<1e-11
    return law


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    optical = np.load(root/'iteration110_collective_v1/optical.npz')
    reference = np.load(root/'iteration110_bus_v1/laws.npz')
    summary = []
    for angle, index in ((.1, 2), (1., 4), (np.pi, 5)):
        s, axes, observed = prepare(optical, reference['bus'], angle)
        for m in (2, 3, 4, 5, 6, 8):
            zeta, c, gram, bound = branches(s, axes, m)
            tab = click_table(optical['core_x'], dyads(zeta, observed, 4))
            p = complete_clicks(tab, np.outer(c,c)*gram)
            tv = np.abs(p-reference[f'truth{index}']).sum()/2
            assert tv < bound+1e-11
            row = dict(angle=angle, nodes_per_axis=m, squeezings=s.tolist(), branches=len(c),
                       whole_state_tv_bound=bound, click_model_tv=tv,
                       rejection_acceptance=float(1/sum(abs(c))**2))
            summary.append(row)
            if angle == 1 and m in (3, 6):
                np.savez_compressed(out/f'click{m}.npz', law=p)
            print(f'{angle:.3f} {m}: TV={tv:.8g} bound={bound:.8g} acceptance={row["rejection_acceptance"]:.5f}', flush=True)
    (out/'projection.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        run(Path(__file__).resolve().parents[1]/'results', args.output)
