"""Full-output L2 bound for products of the true optical single-cell marginals."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from iteration111_memory_v1 import prepare,branches,dyads,click_table
from iteration111_sampling_v1 import DATA
from iteration112_coarse_v1 import combinations,fourier
from iteration112_dilution_v1 import optics,marginal_overflow


def certify(q,local,weight,cells):
    # local is one representative per Hermitian and inversion orbit.
    # A conjugate-pair orbit contributes its real part to the first marginal.
    diff=local-q[:,None]
    single=(diff.real@weight)
    first=math.sqrt(cells*np.dot(single/q,single))
    t=(abs(diff)**2/q[:,None]).sum(0)
    # The binomial sum for orders >=2 is nonnegative; start with its first term
    # to avoid cancellation of 1 and C*t at small t. Accumulate in long double.
    t=t.astype(np.longdouble)
    term=cells*(cells-1)*t*t/2
    tail=term.copy()
    j=min(cells,2)
    for j in range(3,min(cells,2048)+1):
        term=term*(cells-j+1)*t/j
        tail+=term
        if np.all(term<=np.maximum(np.longdouble(1e-60),tail)*1e-20):break
    else:
        if j<cells:raise ArithmeticError('Binomial tail did not converge')
    # Account for the remaining geometric tail using monotonically declining ratios.
    ratio=(cells-j)*t/(j+1)
    if np.any(ratio>=1):raise ArithmeticError('Unbounded numerical remainder')
    tail+=term*ratio/(1-ratio)
    return first,float(np.dot(abs(weight),np.sqrt(tail))),float(t.max())


def run(out,pnr):
    out.mkdir(parents=True,exist_ok=True)
    optical=np.load(DATA/'iteration110_collective_v1/optical.npz')
    bus=np.load(DATA/'iteration110_bus_v1/laws.npz')['bus']
    x,bright,_,_=optics(1)
    s,axes,observed=prepare(optical,bus,1.)
    rows=[];arrays={}
    for m in ((8,10) if pnr else (8,10,12)):
        z,c,g,bound=branches(s,axes,m)
        pos=(z@axes.conj()).imag
        pair,w=combinations(pos,c)
        for cells in ((32768,) if pnr else (256,2048,32768,262144)):
            start=perf_counter()
            marginal=x+(bright-x)/cells
            reg=0.
            if pnr:
                radix=26
                q0=fourier(marginal,np.zeros((1,8),complex),radix)[:,0].real
                order=np.argsort(-q0)
                keep=order[:np.searchsorted(np.cumsum(q0[order],dtype=np.longdouble),1-2e-9)+1]
                q=q0[keep].copy();omitted=max(0.,float(1-np.sum(q,dtype=np.longdouble)));q[0]+=1-q.sum()
                reg=cells*(omitted+marginal_overflow(marginal,radix))
                # All dyads use the same Fourier forms. Stream the alphabet
                # blocks after inversion, so only one K-pair panel occupies RAM.
                displacement=dyads(z,observed,cells)[pair[0],pair[1]]
                first_sum=np.zeros(len(q));tail=0.;tmax=0.
                for off in range(0,len(w),180):
                    sl=slice(off,off+180)
                    table=fourier(x,displacement[sl],radix,keep)
                    first_sum+=(table.real-q[:,None])@w[sl]
                    _,part,high=certify(q,table,w[sl],cells)
                    tmax=max(tmax,high);tail+=part
                first=math.sqrt(cells*np.dot(first_sum/q,first_sum))
                arrays['keep']=keep
            else:
                q=click_table(marginal,np.zeros((1,1,8),complex))[:,0,0].real
                table=click_table(x,dyads(z,observed,cells))[:,pair[0],pair[1]]
                first,tail,tmax=certify(q,table,w,cells)
            # The first-order term and all higher orders are orthogonal in L2.
            full=min(1.,bound+.5*math.hypot(first,tail)+reg)
            rows.append(dict(nodes=m,branches=len(c),readout='pnr' if pnr else 'click',cells=cells,
                ports=4*cells,projection=bound,first_order_l2=first,higher_order_l2_upper=tail,
                register=reg,full_tv_budget=full,max_chi_local=tmax,calculation_s=perf_counter()-start))
            arrays[f'q{cells}']=q
            print(m,cells,full,flush=True)
            np.savez_compressed(out/'marginals.npz',**arrays)
            (out/'bounds.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--pnr',action='store_true')
    args=ap.parse_args()
    with threadpool_limits(limits=3):run(args.output.resolve(),args.pnr)
