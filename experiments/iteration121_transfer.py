"""Optimize a separate 14-bank test for each large fit, using fresh design folds."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import beta
from threadpoolctl import threadpool_limits

import iteration121_goodman as g
from iteration121_precision import N
from iteration121_witness import ROOT,bank_law,vacuum

K = 14
SCREEN = 524288
FOLD = 262144
TEST = 2097152


def model(route,cache):
    t,ss = g.hardware()
    fields = g.Fields(t,ss,wide=True)
    path = cache/(route+'.npz')
    if path.exists():
        with np.load(path) as f:
            layers = [[torch.tensor(f[f'{i}_{j}'],device=g.DEVICE) for j in (0,1)] for i in range(10)]
    else:
        mean,cov = g.moments(*g.actual_input(),route)
        layers,_ = g.train(fields,route,mean,cov,N,stage=8)
        cache.mkdir(exist_ok=True,parents=True)
        np.savez(path,**{f'{i}_{j}':a.cpu().numpy() for i,l in enumerate(layers) for j,a in enumerate(l)})
    ref = json.loads((ROOT/(route+'_fp64.json')).read_bytes())
    assert ref['layer_sha256']==[hashlib.sha256(b''.join(z.cpu().numpy().tobytes() for z in l)).hexdigest() for l in layers]
    return fields,layers,t,ss


def integrate(fields,layers,route,rng,banks):
    h = torch.zeros(1<<len(banks),device=g.DEVICE,dtype=torch.float64)
    for _ in range(FOLD//g.BATCH):
        p = vacuum(fields,layers,route,rng,banks)
        for block in p.split(512):
            word = torch.ones((len(block),1),device=g.DEVICE,dtype=torch.float64)
            for j in range(len(banks)):
                q = block[:,j:j+1]
                word = torch.cat([word*q,word*(1-q)],1)
            h += word.sum(0)
    return h.cpu().numpy()/FOLD


def design(route,cache):
    fields,layers,t,ss = model(route,cache)
    rng = g.generator(route,10,0)
    sums = torch.zeros(144,device=g.DEVICE,dtype=torch.float64)
    for _ in range(SCREEN//g.BATCH):
        sums += vacuum(fields,layers,route,rng).sum(0)
    with np.load(g.DATA/'iteration120_contractions/calibration_predictions.npz') as f:
        v = f['published_multiplicity'][:,0]
    delta = sums.cpu().numpy()/SCREEN-v
    banks = np.sort(np.argsort(-abs(delta)/np.sqrt(v*(1-v)),kind='stable')[:K])
    opt = bank_law(t,ss,banks)
    first,second = [integrate(fields,layers,route,g.generator(route,11,i),banks) for i in (0,1)]
    masks = {'unrestricted':first>opt}
    bits = ((np.arange(1<<K)[:,None] >> np.arange(K))&1).astype(float)
    bits -= opt@bits
    basis = np.column_stack([np.ones(len(bits)),bits])
    w = np.sqrt(opt)
    y = np.log(first/opt)
    for name in ('linear','quadratic'):
        if name=='quadratic':
            i,j = np.triu_indices(K,1)
            basis = np.column_stack([basis,bits[:,i]*bits[:,j]])
        coef = np.linalg.lstsq(basis*w[:,None],y*w,rcond=None)[0]
        masks[name] = basis@coef>0
    scores = {k:float((second-opt)[mask].sum()) for k,mask in masks.items()}
    chosen = max(scores,key=scores.get)
    path = ROOT/(route+'_large_protocol.npz')
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path,optical=opt,banks=banks,event=masks[chosen],design1=first,design2=second,
                        standardized_bank_error=delta/np.sqrt(v*(1-v)),chosen=chosen,
                        **{'candidate_'+k:mask for k,mask in masks.items()})
    print(route,banks,chosen,scores,flush=True)


def test(route,cache):
    path = ROOT/(route+'_large_protocol.npz')
    with np.load(path) as f:
        banks,mask,opt = [f[k] for k in ('banks','event','optical')]
    fields,layers,_,_ = model(route,cache)
    rng,draw = [g.generator(route,12,i) for i in (0,1)]
    counts = np.zeros(len(mask),np.int64)
    powers = 1<<np.arange(len(banks))
    digest = hashlib.sha256()
    examples,times = [],[]
    for off in range(0,TEST,g.BATCH):
        torch.cuda.synchronize()
        begin = perf_counter()
        y = g.clip(fields.raw(rng,route),route)
        for mat,shift in layers:
            y = g.clip(y@mat+shift,route)
        a = (torch.poisson(y,generator=draw).int().cpu().numpy() if route=='pnr'
             else (torch.rand(y.shape,dtype=y.dtype,device=g.DEVICE,generator=draw)>y).cpu().numpy())
        times.append(perf_counter()-begin)
        assert a.min()>=0 and a.max()<256
        rows = a.astype(np.uint8,copy=False)
        digest.update(rows)
        if off<4096:
            examples.append(rows.copy())
        counts += np.bincount((rows.reshape(-1,144,8)[:,banks,:]!=0).any(2)@powers,minlength=len(mask))
    hits = int(counts[mask].sum())
    ref = float(opt[mask].sum())
    result = dict(rows=TEST,hits=hits,optical=ref,empirical=hits/TEST,full_tv_lower99=float(max(0.,
                  beta.ppf(.01/8,hits,TEST-hits+1)-ref-1e-8)),alpha=.01/8,
                  records_sha256=digest.hexdigest(),host_us_per_record=1e6*sum(times)/TEST,
                  protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (ROOT/(route+'_large.json')).write_bytes((json.dumps(result,indent=2)+'\n').encode())
    np.savez_compressed(ROOT/(route+'_large.npz'),counts=counts,first4096=np.vstack(examples),batch_seconds=times)
    print(route,result,flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage',choices=['design','test'])
    ap.add_argument('route',choices=['pnr','click'])
    ap.add_argument('--cache',type=Path,required=True)
    args = ap.parse_args()
    with threadpool_limits(limits=6),torch.inference_mode():
        (design if args.stage=='design' else test)(args.route,args.cache)
