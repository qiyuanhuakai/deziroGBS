"""Confirm both frozen bank witnesses on complete records from the timed sampler."""
import argparse
import hashlib
import json
from time import perf_counter

import numpy as np
import torch
from scipy.stats import beta
from threadpoolctl import threadpool_limits

import iteration121_goodman as g
from iteration121_witness import ROOT,TEST,setup


def run(route):
    protocol = ROOT/(route+'_protocol.npz')
    with np.load(protocol) as f:
        banks,event,optical = [f[k] for k in ('banks','event','optical')]
    fields,layers,_,_ = setup(route)
    rng,draw = [g.generator(route,7,i) for i in (0,1)]
    counts = np.zeros(len(event),np.int64)
    powers = 1<<np.arange(len(banks))
    examples,latencies = [],[]
    digest = hashlib.sha256()
    for start in range(0,TEST,g.BATCH):
        torch.cuda.synchronize()
        clock = perf_counter()
        y = g.clip(fields.raw(rng,route),route)
        for mat,offset in layers:
            y = g.clip(y@mat+offset,route)
        if route=='pnr':
            a = torch.poisson(y,generator=draw).to(torch.int32).cpu().numpy()
        else:
            a = (torch.rand(y.shape,device=g.DEVICE,generator=draw)>y).cpu().numpy()
        latencies.append(perf_counter()-clock)
        if a.min()<0 or a.max()>255:
            raise ArithmeticError('Cannot store record as uint8')
        rows = a.astype(np.uint8,copy=False)
        digest.update(rows)
        if start<4096:
            examples.append(rows.copy())
        occupied = (rows.reshape(-1,144,8)[:,banks,:]!=0).any(2)
        counts += np.bincount(occupied@powers,minlength=len(event))
    n = int(counts.sum())
    hits = int(counts[event].sum())
    # Four confirmatory tests (two routes, two ways to generate the word).
    lower = float(beta.ppf(.0025,hits,n-hits+1))
    target = float(optical[event].sum())
    res = dict(route=route,records=n,width=fields.width,test_hits=hits,
               empirical=hits/n,optical=target,binomial_lower=lower,
               full_tv_lower_simultaneous99=max(0.,lower-target-1e-8),
               alpha_per_test=.0025,random_stream=[121,int(route=='pnr'),7],
               sha256_uint8_records=digest.hexdigest(),
               protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
               delivered_us_per_record=1e6*sum(latencies)/n,batch_size=g.BATCH,
               timing_scope='End-to-end generation into host memory, excludes fixed training and validation/hashing.')
    np.savez_compressed(ROOT/(route+'_records.npz'),first4096=np.vstack(examples),counts=counts,batch_seconds=latencies)
    (ROOT/(route+'_records.json')).write_bytes((json.dumps(res,indent=2)+'\n').encode())
    print(route,res,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('route',choices=['pnr','click'])
    with threadpool_limits(limits=6),torch.inference_mode():
        run(p.parse_args().route)
