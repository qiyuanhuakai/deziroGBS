"""Match spatial photon covariances and contract complete two-group count and click laws."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import ROOT, Tree, Native, ORDER, realify, split_joint, H, save, manifest
from iteration100_projection_v1 import Projector

DATA = ROOT / 'results'
BASE = DATA / 'iteration100_spatial_v1'
OLD = DATA / 'iteration100_residual_v1'
K = 56


def features(p):
    p = p / p.sum()
    # Use survival probabilities for the high-count tail to avoid 1-p cancellation.
    left = np.cumsum(p)[:-1]
    right = np.cumsum(p[::-1])[::-1][1:]
    t = np.where(left < .5, ndtri(left.clip(0, 1)), -ndtri(right.clip(0, 1)))
    keep = np.isfinite(t)
    x = t[keep]
    h = np.exp(-x*x/2) / np.sqrt(2*np.pi)
    previous = np.zeros_like(h)
    coefficients = np.zeros((len(p), K+1))
    coefficients[:, 0] = p
    for k in range(1, K+1):
        boundary = np.zeros(len(t))
        boundary[keep] = h / np.sqrt(k)
        coefficients[:, k] = np.diff(np.r_[0., -boundary, 0.])
        previous, h = h, (x*h - np.sqrt(k-1)*previous)/np.sqrt(k)
    n = np.arange(len(p))
    return t, coefficients, n @ coefficients


def fit(target, moments):
    a, b = np.triu_indices(len(moments), 1)
    weights = moments[a, 1:] * moments[b, 1:]
    desired = target[a, b]
    def evaluate(r):
        v = np.zeros_like(r)
        for col in weights.T[::-1]:
            v = (v+col)*r
        return v
    lo = np.full(len(a), -.999)
    hi = -lo
    assert np.all(evaluate(lo) <= desired) and np.all(evaluate(hi) >= desired)
    for _ in range(52):
        mid = (lo+hi)/2
        above = evaluate(mid) >= desired
        lo = np.where(above, lo, mid)
        hi = np.where(above, mid, hi)
    corr = np.eye(len(moments))
    corr[a, b] = corr[b, a] = (lo+hi)/2
    e, v = np.linalg.eigh(corr)
    if e[0] < 1e-10:
        corr = (v*e.clip(1e-10)) @ v.T
        d = np.sqrt(corr.diagonal())
        corr /= d[:, None]*d
    error = evaluate(corr[a,b])-desired
    norm = np.sqrt(target[a,a]*target[b,b])
    tail = abs(corr[a,b])**(K+1)*norm
    return corr, dict(pairs=len(a), raw_min_eigenvalue=float(e[0]),
        final_min_eigenvalue=float(np.linalg.eigvalsh(corr)[0]),
        max_rho=float(abs(corr[a,b]).max()), max_cov_error=float(abs(error).max()),
        max_cov_series_bound=float(tail.max()), rms_cov_error=float(np.sqrt(np.mean(error**2))))


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    with np.load(BASE/'prepared.npz') as f:
        model = Tree(None, dict(f))
    proj = Projector(model)
    groups = model.leaves
    parts = [features(proj.totals[g]) for g in range(groups)]
    moments = np.array([p[2] for p in parts])
    target = np.load(OLD/'covariance.npz')['optical']
    corr, diag = fit(target, moments)
    cuts = np.full((groups, max(map(lambda x: len(x[0]), parts))), np.inf)
    for i, (thresholds, _, _) in enumerate(parts):
        cuts[i, :len(thresholds)] = thresholds
    a = moments[:,1]
    sigma = np.sqrt(a@corr@a)
    direction = corr@a/sigma
    assert min(direction)>0
    np.savez_compressed(out/'fit.npz', corr=corr, cuts=cuts, direction=direction, score=a/sigma,
                        mean=moments[:,0].sum(), sigma=sigma)
    rows = []
    frozen = json.loads((OLD/'summary.json').read_text())['rows']
    saved = DATA/'iteration100_projection_v1/panel_4.npz'
    with np.load(saved) as f:
        ports = f['ports']
        assert np.array_equal(ports.reshape(2,8)%8,np.tile(np.arange(8),(2,1)))
        sibling = (ports.reshape(2,8)[:,0]//8).tolist()
        sibling_truth = f['truth']
    for j, (g,h) in enumerate([r['spatial'] for r in frozen]+[sibling]):
        if j<len(frozen):
            with np.load(OLD/f'pair_{j}.npz') as f:
                truth, ref, old = f['truth'], f['count_reference'], f['click']
        else:
            truth = sibling_truth
            z = np.load(ORDER)
            v = z['v'][np.argsort(z['order'])]
            m = [realify(v[i*8:i*8+8]) for i in (g,h)]
            ref = split_joint(Native(BASE/'spatial.dll'),m[0].T@m[0],m[1].T@m[1],z['x'],
                              int(model.caps[g]),int(model.caps[h]))
            old = proj.clicks(ports)
        power = corr[g,h]**np.arange(K+1)
        joint = (parts[g][1]*power)@parts[h][1].T
        assert joint.min()>-2e-13 and abs(joint.sum()-1)<1e-12
        bits = (proj.mask[h].T@joint.T@proj.mask[g]).ravel()
        count_tv = abs(joint-ref).sum()/2
        click_tv = abs(bits-truth).sum()/2
        # Parseval + Cauchy-Schwarz for each categorical indicator's omitted series.
        bound = abs(corr[g,h])**(K+1)*np.sqrt(joint.shape[0]*joint.shape[1])/2
        rows.append(dict(groups=[g,h],rho=float(corr[g,h]), count_tv=count_tv,click_tv=click_tv,
                         tree_click_tv=abs(old-truth).sum()/2,series_tv_bound=bound))
        np.savez_compressed(out/f'pair_{j}.npz', counts=joint, reference=ref, bits=bits, truth=truth)
    save(out,'summary.json',dict(fit=diag,rows=rows,groups=groups,order=K,
        inputs={str(p.relative_to(ROOT)):H(p) for p in [BASE/'prepared.npz',OLD/'covariance.npz',saved,ORDER]}))
    manifest(out)
    print(diag,rows,flush=True)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        main(ap.parse_args().output.resolve())
