"""Sparse photon symbols at 131,072 detectors, with an optical marginal error budget."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration111_sampling_v1 import (DATA,Native,Readout,prepare,branches,dyads,pnr_table,
                                       realify,marginal_tail)


def run(out,m,radix,shots):
    out.mkdir(parents=True,exist_ok=True)
    native=Native(out)
    optical=np.load(DATA/'iteration110_collective_v1/optical.npz')
    ref=np.load(DATA/'iteration110_bus_v1/laws.npz')
    cells=32768
    x=optical['core_x']
    bus=ref['bus']
    w=realify(np.eye(4)+np.expm1(1j)*np.outer(bus,bus.conj()))
    local=x+(w@x@w.T-x)/cells
    start=perf_counter()
    target=pnr_table(local,np.zeros((1,1,8),complex),radix)[:,0,0].real
    assert target.min()>-1e-14
    order=np.argsort(-target,kind='stable')
    end=np.searchsorted(np.cumsum(target[order],dtype=np.longdouble),1-1e-8)+1
    keep=order[:end]
    assert keep[0]==0
    omitted=float(np.sum(target[order[end:]],dtype=np.longdouble))
    wrap=marginal_tail(x,bus,cells,radix)
    # The alternative physical-number contour uses a different period and the
    # same explicitly selected Fock outcomes. Its excess mass is bounded by wrap.
    other=radix+3
    digits=np.stack([keep//radix**i%radix for i in range(4)])
    alternative=pnr_table(local,np.zeros((1,1,8),complex),other)[:,0,0].real
    omitted2=float(1-np.sum(alternative[(other**np.arange(4))@digits],dtype=np.longdouble))
    wrap2=marginal_tail(x,bus,cells,other)
    assert abs(omitted-omitted2)*cells<=wrap+wrap2+1e-7
    s,axes,observed=prepare(optical,bus,1.)
    z,c,gram,bound=branches(s,axes,m)
    print('preparing',end,'symbols for',len(c),'branches',flush=True)
    tab=pnr_table(x,dyads(z,observed,cells),radix,keep)
    model=Readout(native,tab,c,gram,cells,radix,keep)
    del tab
    prep=perf_counter()-start
    thread_us={}
    for nt in (1,8):
        us=[]
        for rep in range(6):
            start=perf_counter()
            model.sample(8,111500100+100*m+20*nt+rep,nt)
            if rep:us.append((perf_counter()-start)*1e6/8)
        thread_us[str(nt)]=us
    a,trials=model.sample(shots,111500000+m,8)
    a0,_=model.sample(2,111500000+m,1,2)
    assert np.array_equal(a0,a[2:4])
    n=a.sum(1,dtype=np.int64)
    expected=cells*np.trace(x)/2
    # Across records: mirrors physical columns exhaustively, never treats sites as iid.
    rows=dict(modes=a.shape[1],cells=cells,radix=radix,nodes=m,shots=shots,kept=len(keep),
              setup_s=prep,state_bound=bound,overflow_bound=wrap,discarded_marginal_mass=omitted,
              second_grid_omitted=omitted2,second_grid_overflow_bound=wrap2,
              total_tv_budget=min(1.,bound+wrap+cells*omitted),us_by_threads=thread_us,
              mean_photons=float(n.mean()),target_mean=float(expected),sample_mean_se=float(n.std(ddof=1)/np.sqrt(shots)),
              max_photons_in_port=int(a.max()),observed_acceptance=float(shots/trials.sum()),
              predicted_acceptance=float(1/sum(abs(c))**2))
    np.savez_compressed(out/'counts.npz',first=a[:8],totals=n,attempts=trials,symbols=keep)
    (out/'wide.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
    print(rows,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    ap.add_argument('--nodes',type=int,default=6)
    ap.add_argument('--radix',type=int,default=24)
    ap.add_argument('--shots',type=int,default=128)
    args=ap.parse_args()
    with threadpool_limits(limits=2):run(args.output.resolve(),args.nodes,args.radix,args.shots)
