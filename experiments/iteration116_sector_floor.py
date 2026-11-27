"""Distance from the entire class of total-only reweightings on a bank-word statistic."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

from iteration115_banks_v1 import bank_model
from iteration115_projection_review_v3 import factor, project
from iteration115_visible_v1 import DATA, hardware

# First four distinct bank pairs in the previously archived panel list.
BANKS = [34, 35, 0, 1, 17, 89, 45, 98]
GRID = 512
OUT = DATA / "iteration116_sectors"


def condition_dense(x, selected):
    if not len(selected):
        return 0., eigh(x, eigvals_only=True)
    keep = np.ones(len(x), bool)
    keep[selected] = False
    d = np.eye(len(selected))+x[np.ix_(selected, selected)]
    cross = x[np.ix_(selected, keep)]
    reduced = x[np.ix_(keep, keep)] - cross.T @ np.linalg.solve(d, cross)
    return -.5*np.linalg.slogdet(d)[1], eigh(reduced, eigvals_only=True)


def condition_factors(model, selected):
    u, e = model
    if not len(selected):
        return 0., e
    b = u[selected]
    cross = b*e
    d = np.eye(len(b))+cross@b.T
    small = np.diag(e)-cross.T@np.linalg.solve(d,cross)
    chol = np.linalg.cholesky(np.eye(len(e))-b.T@b)
    return -.5*np.linalg.slogdet(d)[1], eigh(chol.T@small@chol, eigvals_only=True)


def tables(name, model, condition, axes):
    z = np.exp(-2j*np.pi*np.arange(GRID//2+1)/GRID)
    void = np.empty((1<<len(BANKS), GRID))
    for mask in range(len(void)):
        indices = [bank for j,bank in enumerate(BANKS) if mask & (1<<j)]
        selected = axes(np.array(indices, int))
        terms = [condition(component, selected) for component in model]
        e = np.concatenate([v for _,v in terms])
        pgf = np.exp(sum(lp for lp,_ in terms)-.5*np.log1p((1-z[:,None])*e).sum(1))
        void[mask] = np.fft.irfft(pgf, n=GRID)
        if not mask:
            radii = np.linspace(1.001, 1+.95/e.max(), 200)
            tail = np.exp(min(-GRID*np.log(r)-.5*np.log1p((1-r)*e).sum() for r in radii))
    # Index is the set of empty banks; sum over supersets gives the joint void.
    words = void.copy()
    for j in range(len(BANKS)):
        for mask in range(len(void)):
            if mask & (1<<j) == 0:
                words[mask] -= words[mask | (1<<j)]
    print(name, "tail", tail, "min", words.min(), flush=True)
    assert abs(words.sum()-1)<1e-12 and words.min()>-1e-12
    np.testing.assert_allclose(words.sum(0),void[0],atol=5e-14)
    return words.T, float(tail)


def floor(p, q):
    # Ignore sparse tails in the witness by making it the same constant there.
    pn, qn = p.sum(1), q.sum(1)
    use = (pn > 1e-7) & (qn > 1e-7) & (np.arange(len(p)) < 340)
    pp = p[use]
    conditional = q[use] / qn[use,None]
    assert np.all(conditional>0) and np.all(pp>0)
    ratios = pp / conditional
    index = np.argsort(-ratios,axis=1)
    cs = np.cumsum(np.take_along_axis(conditional,index,axis=1),axis=1)
    slopes = np.take_along_axis(ratios,index,axis=1)
    ps = np.cumsum(np.take_along_axis(pp,index,axis=1),axis=1)
    rows = np.arange(len(pp))

    def evaluate(b):
        j = np.minimum((cs<b).sum(1),cs.shape[1]-1)
        threshold = slopes[rows,j]
        power = ps[rows,j]+(b-cs[rows,j])*threshold
        return power.sum()-b*pn[use].sum(), threshold

    opt = minimize_scalar(lambda b:-evaluate(b)[0],bounds=(0,1),method="bounded",
                          options={"xatol":1e-13})
    lower, scale = evaluate(opt.x)
    stops=np.minimum((cs<opt.x).sum(1),cs.shape[1]-1)
    before=np.where(stops>0,cs[rows,np.maximum(0,stops-1)],0.)
    atom=index[rows,stops]
    test=np.zeros_like(pp)
    np.put_along_axis(test,index,(np.arange(index.shape[1])<stops[:,None]).astype(float),axis=1)
    test[rows,atom]=(opt.x-before)/conditional[rows,atom]
    assert test.min()>=-1e-12 and test.max()<=1+1e-12
    size_error=float(abs((conditional*test).sum(1)-opt.x).max())
    assert size_error<1e-12 and abs((pp*test).sum()-opt.x*pp.sum()-lower)<1e-12
    # A feasible reweighting supplies an independent primal upper bound for this statistic.
    scale = scale*(pn[use].sum()/scale.sum())
    upper = .5*np.abs(pp-scale[:,None]*conditional).sum()+pn[~use].clip(0).sum()
    assert 0 <= lower <= upper+2e-12
    return dict(lower=float(lower), upper=float(upper), common_size=float(opt.x), max_size_residual=size_error,
                active_n=np.flatnonzero(use).tolist(), target_outside=float(1-pn[use].sum()),
                model_outside=float(1-qn[use].sum()))


def run():
    OUT.mkdir(exist_ok=True)
    t, ss = hardware()
    truth = factor(t,ss)
    _, projected, _ = project(t,ss,1)
    a,e,_,_ = bank_model(t,ss,8)[0]
    base = []
    for j in range(2):
        f = a.reshape(len(a),144,4)[:,:,2*j:2*j+2].reshape(len(a),-1)
        base.append(2*f.T@f+np.diag(e[:,2*j:2*j+2].ravel()))
    def full_axes(b):
        ports=(8*b[:,None]+np.arange(8)).ravel()
        return np.r_[ports,1152+ports]
    start = perf_counter()
    p,tp = tables("optical", truth, condition_factors, full_axes)
    q,tq = tables("base", base, condition_dense, lambda b:(2*b[:,None]+np.arange(2)).ravel())
    r,tr = tables("projected", projected, condition_factors, lambda b:np.r_[b,b+144])
    # Cross-check the old independent fixed-global-N calculation on the pressure pair.
    old=json.loads((DATA/"iteration115_review_v3/projection.json").read_text())
    pair=(np.arange(p.shape[1]) & 3)==3
    errors = []
    for dist,key in [(p,"optical"),(q,"base"),(r,"projected")]:
        observed = (dist[:,pair].sum(1)[old["total_photons"]]/dist.sum(1)[old["total_photons"]])
        errors.append(float(np.max(abs(observed-old["panels"][0][key+"_fixed_N_vacuum"]))))
    assert max(errors)<2e-11
    fit=json.loads((DATA/"iteration115_tilt_v1/summary.json").read_text())
    n=np.arange(len(p))
    # Normalizers from Gaussian eigenvalues, independent of this FFT grid.
    ev=np.concatenate([eigh(x,eigvals_only=True) for x in base])
    lz=np.array(fit["log_tilts"])
    lg=-.5*np.log1p((1-np.exp(lz[:,None]))*ev).sum(1)
    rn=np.exp(n[:,None]*lz-lg)@fit["weights"]
    # Do not magnify roundoff-only FFT slots by the positive tilts.
    use=(n>=30)&(n<=270)
    def compared_tv(x,y):
        active=.5*abs(x[use]-y[use]).sum()
        tails=max(0,1-x[use].sum())+max(0,1-y[use].sum())
        return [float(active), float(active+.5*tails)]
    result=dict(banks=BANKS, grid=GRID, optical_alias=tp, base_alias=tq, projected_alias=tr,
        sector_floor=floor(p,q), tilt_observable_tv=compared_tv(p,q*rn[:,None]),
        projected_observable_tv=compared_tv(p,r), crosscheck_errors=errors,
        min_raw_joint_probability=[float(x.min()) for x in (p,q,r)], elapsed_seconds=perf_counter()-start)
    np.savez_compressed(OUT/"joint.npz", optical=p,base=q,projected=r)
    (OUT/"floor.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(result["sector_floor"]["lower"], result["sector_floor"]["upper"], result["tilt_observable_tv"],
          result["projected_observable_tv"], flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=3):
        run()
