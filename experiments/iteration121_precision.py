"""Independent million-trajectory FP64 fits tested on previously frozen events."""
import argparse
import hashlib
import json
from time import perf_counter

import numpy as np
import torch
from scipy.stats import beta
from threadpoolctl import threadpool_limits

import iteration121_goodman as g
from iteration121_witness import ROOT

N = 1048576


def run(route):
    path = ROOT/(route+'_protocol.npz')
    with np.load(path) as src:
        banks,mask,law = [src[k] for k in ('banks','event','optical')]
    t,ss = g.hardware()
    mean,cov = g.moments(*g.actual_input(),route)
    fields = g.Fields(t,ss,wide=True)
    start = perf_counter()
    layers,diag = g.train(fields,route,mean,cov,N,stage=8)
    torch.cuda.synchronize()
    fit_s = perf_counter()-start
    rng,draw = [g.generator(route,9,i) for i in (0,1)]
    blocks,hist,examples = [],np.zeros(len(mask),np.int64),[]
    powers = 1<<np.arange(len(banks))
    for start in range(0,N,g.BATCH):
        torch.cuda.synchronize()
        begin = perf_counter()
        y = g.clip(fields.raw(rng,route),route)
        for mat,offset in layers:
            y = g.clip(y@mat+offset,route)
        rows = (torch.poisson(y,generator=draw).int().cpu().numpy() if route=='pnr'
                else (torch.rand(y.shape,dtype=y.dtype,device=g.DEVICE,generator=draw)>y).cpu().numpy())
        blocks.append(perf_counter()-begin)
        word = (rows.reshape(-1,144,8)[:,banks,:]!=0).any(2) @ powers
        hist += np.bincount(word,minlength=len(hist))
        if start<4096:
            assert rows.min()>=0 and rows.max()<256
            examples.append(rows.astype(np.uint8))
    hits = int(hist[mask].sum())
    # Report a single family covering all six confirmatory tests.
    lower = float(beta.ppf(.01/6,hits,N-hits+1))
    opt = float(law[mask].sum())
    result = dict(route=route,rows=N,width=len(t),train=N,training_stage=8,test_stage=9,precision='FP64',
                  hits=hits,empirical=hits/N,optical=opt,binomial_lower=lower,alpha=.01/6,
                  full_tv_lower99=float(max(0.,lower-opt-1e-8)),fit_s=fit_s,diagnostics=diag,
                  host_us_per_record=1e6*sum(blocks)/N,protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  layer_sha256=[hashlib.sha256(b''.join(z.cpu().numpy().tobytes() for z in l)).hexdigest() for l in layers])
    np.savez_compressed(ROOT/(route+'_fp64.npz'),first4096=np.vstack(examples),counts=hist,batch_seconds=blocks)
    (ROOT/(route+'_fp64.json')).write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(route,hits,result['full_tv_lower99'],result['host_us_per_record'],flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('route',choices=['click','pnr'])
    with threadpool_limits(limits=6),torch.inference_mode():
        run(ap.parse_args().route)
