"""Design a bounded multi-bank event, freeze it, and test on disjoint streams."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import beta
from threadpoolctl import threadpool_limits

import iteration121_goodman as g
from iteration114_rank_v1 import cov

DESIGN = 262144
TEST = 2097152
K = 12
ROOT = g.OUT


def bank_law(t,ss,banks):
    sites = (8*np.asarray(banks)[:,None]+np.arange(8)).ravel()
    n = len(sites)
    perm = np.stack([np.arange(n),np.arange(n)+n],axis=1).ravel()
    probs = np.ones(1<<len(banks))
    for s in ss:
        q = np.eye(2*n)+cov(t[sites],s)[np.ix_(perm,perm)]
        logs = np.zeros_like(probs)
        def visit(a,bit,mask,log):
            if not len(a):
                logs[mask] = log
                return
            rest = a[16:,16:]
            visit(rest,bit+1,mask,log)
            L = cholesky(a[:16,:16],lower=True,check_finite=False)
            cross = solve_triangular(L,a[:16,16:],lower=True,check_finite=False)
            visit(rest-cross.T@cross,bit+1,mask|(1<<bit),log-np.log(L.diagonal()).sum())
        visit(q,0,0,0.)
        probs *= np.exp(logs)
    out = probs[::-1].copy()
    for j in range(len(banks)):
        v = out.reshape(-1,2,1<<j)
        v[:,1] -= v[:,0]
    assert np.all(out>0) and abs(out.sum()-1)<1e-11
    return out


def setup(route):
    torch.set_float32_matmul_precision('highest')
    t,ss = g.hardware()
    mean,covariance = g.moments(*g.actual_input(),route)
    fields = g.Fields(t,ss)
    layers,_ = g.train(fields,route,mean,covariance,g.TRAIN)
    hashes = [hashlib.sha256(b''.join(z.cpu().numpy().tobytes() for z in layer)).hexdigest() for layer in layers]
    data = json.loads((ROOT/(route+'.json')).read_bytes())
    assert hashes==data['layer_sha256'],'fitted maps differ from the benchmark'
    return fields,layers,t,ss


def vacuum(fields,layers,route,rng,indices=None):
    y = g.clip(fields.raw(rng,route),route)
    for mat,offset in layers:
        y = g.clip(y@mat+offset,route)
    v = y.reshape(-1,144,8)
    if indices is not None:
        v = v[:,indices,:]
    return torch.exp(-v.double().sum(2)) if route=='pnr' else v.double().prod(2)


def design(route):
    fields,layers,t,ss = setup(route)
    rng = g.generator(route,4,0)
    means = torch.zeros(144,device=g.DEVICE,dtype=torch.float64)
    for _ in range(DESIGN//g.BATCH):
        means += vacuum(fields,layers,route,rng).sum(0)
    with np.load(g.DATA/'iteration120_contractions/calibration_predictions.npz') as f:
        reference = f['published_multiplicity'][:,0]
    discrepancy = means.cpu().numpy()/DESIGN-reference
    banks = np.sort(np.argsort(-abs(discrepancy),kind='stable')[:K])
    law = bank_law(t,ss,banks)
    # Rao-Blackwellize over all readout noise on the design fold.
    rng = g.generator(route,5,0)
    hist = torch.zeros(1<<K,device=g.DEVICE,dtype=torch.float64)
    for _ in range(DESIGN//g.BATCH):
        p = vacuum(fields,layers,route,rng,banks)
        probs = torch.ones((len(p),1),device=g.DEVICE,dtype=torch.float64)
        for j in range(K):
            q = p[:,j:j+1]
            probs = torch.cat([probs*q,probs*(1-q)],dim=1)
        hist += probs.sum(0)
    approx = hist.cpu().numpy()/DESIGN
    event = approx>law
    p = ROOT/(route+'_protocol.npz')
    if p.exists():
        raise FileExistsError(p)
    np.savez_compressed(p,banks=banks,optical=law,design=approx,event=event,bank_vacuum_error=discrepancy)
    print(route,'banks',banks,'design difference',sum((approx-law)[event]),flush=True)


def test(route):
    protocol = ROOT/(route+'_protocol.npz')
    with np.load(protocol) as f:
        banks,event,optical = [f[k] for k in ('banks','event','optical')]
    fields,layers,_,_ = setup(route)
    rng,draw = [g.generator(route,6,i) for i in (0,1)]
    powers = torch.tensor(1<<np.arange(K),device=g.DEVICE)
    counts = np.zeros(1<<K,np.int64)
    for _ in range(TEST//g.BATCH):
        p = vacuum(fields,layers,route,rng,banks)
        # Each tested word has exactly the marginal law of a full record.
        code = ((torch.rand(p.shape,generator=draw,device=g.DEVICE,dtype=torch.float64)>p).long()*powers).sum(1)
        counts += np.bincount(code.cpu().numpy(),minlength=len(counts))
    n = counts.sum()
    hits = int(counts[event].sum())
    target = float(optical[event].sum())
    # This second experiment allocates 0.005 per route. No interim looks.
    lower = beta.ppf(.005,hits,n-hits+1)
    record = dict(test=int(n),hits=hits,optical=target,empirical=hits/n,binomial_lower=float(lower),
                  full_tv_lower99=float(max(0,lower-target-1e-8)),optical_numerical_allowance=1e-8,
                  protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                  random_stream=[121,int(route=='pnr'),6],tested_projection='occupied bank word (one binary digit per bank)')
    (ROOT/(route+'_witness.json')).write_bytes((json.dumps(record,indent=2)+'\n').encode())
    np.save(ROOT/(route+'_counts.npy'),counts)
    print(route,record,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['design','test'])
    parser.add_argument('route',choices=['pnr','click'])
    args = parser.parse_args()
    with threadpool_limits(limits=6),torch.inference_mode():
        (design if args.stage=='design' else test)(args.route)
