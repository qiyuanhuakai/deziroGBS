"""Verify probabilities of frozen bank events from the optical input using Arb."""
import argparse
import json
from pathlib import Path

import numpy as np
from flint import arb, arb_mat, acb, ctx
from iteration120_interval import INPUT,exact

OUT = INPUT.parents[1]/'iteration121_positive_p'


def initial(raw,s):
    # Paired sources are first changed to independent squeezed inputs. Rows
    # are interleaved quadratures so each bank occupies a contiguous block.
    n,m = raw.shape
    scale = 1/arb(2).sqrt()
    rows = []
    for row in raw:
        t = []
        for a,b in zip(row[::2],row[1::2]):
            x = acb(exact(a.real),exact(a.imag))
            y = acb(exact(b.real),exact(b.imag))
            t.extend((acb(0,scale)*(x+y),scale*(x-y)))
        rows.extend(([v.real for v in t]+[-v.imag for v in t],
                     [v.imag for v in t]+[v.real for v in t]))
    diag = [((sign*2*exact(z)).exp()-1)/2 for sign in (-1,1) for z in s]
    Q = arb_mat([[v*w for v,w in zip(r,diag)] for r in rows])*arb_mat(rows).transpose()
    for i in range(2*n):
        Q[i,i] += 1
    return Q


def all_void(Q,k):
    out = [None]*(1<<k)
    def visit(a,bit,mask,d):
        if bit==k:
            out[mask] = 1/d.sqrt()
            return
        A = arb_mat([r[:16] for r in a[:16]])
        det = A.det()
        assert det>0
        if bit==k-1:
            out[mask],out[mask|(1<<bit)] = 1/d.sqrt(),1/(d*det).sqrt()
            return
        C = [r[16:] for r in a[16:]]
        visit(C,bit+1,mask,d)
        B = arb_mat([r[16:] for r in a[:16]])
        S = arb_mat(C)-B.transpose()*A.solve(B)
        visit(S.tolist(),bit+1,mask|(1<<bit),d*det)
    visit(Q.tolist(),0,0,arb(1))
    return out


def run(route):
    ctx.prec = 160
    ctx.threads = 2
    with np.load(OUT/(route+'_protocol.npz')) as f:
        banks,event,old = (f[k] for k in ('banks','event','optical'))
    with np.load(INPUT) as f:
        raw = f['T'][np.argsort(f['order'])][(8*banks[:,None]+np.arange(8)).ravel()]
        ss = f['squeezing'].T.repeat(2,axis=1)
    a,b = [all_void(initial(raw,s),len(banks)) for s in ss]
    law = [x*y for x,y in zip(reversed(a),reversed(b))]
    for j in range(len(banks)):
        step = 1<<j
        for start in range(0,len(law),2*step):
            for i in range(start,start+step):
                law[i+step] -= law[i]
    assert all(v>0 for v in law) and sum(law).contains(1)
    target = sum(v for keep,v in zip(event,law) if keep)
    tv = sum(abs(v-exact(w)) for v,w in zip(law,old))/2
    assert tv < arb('1e-8') and abs(target-exact(old[event].sum()))<arb('1e-8')
    result = dict(precision_bits=ctx.prec,cells=len(law),optical_event_ball=str(target),
                  nominal_table_tv_error_ball=str(tv),input='binary transfer and squeezing values')
    (OUT/(route+'_interval.json')).write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(route,result,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('route',choices=['pnr','click'])
    run(p.parse_args().route)
