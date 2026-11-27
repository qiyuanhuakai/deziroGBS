"""Finite-sample diagnostics for new runs, distinct from analytical TV certificates."""
import json
import numpy as np
from iteration122_powers import OUT,json_write


def run():
    stats={}
    # DKW and two-sided Hoeffding with 0.01 divided across six diagnostics.
    for case in ('unequal_loss','bright'):
        report=json.loads((OUT/case/'records.json').read_bytes());tests=[]
        with np.load(OUT/case/'records.npz') as f:
            ports,law=f['ports'],f['optical_mod4096']
            assert abs(law.sum()-1)<1e-12 and law.min()>-1e-12
            for r,route in enumerate(('pnr','click')):
                a=f[route];n=len(a);row=report['readouts'][r]
                vacuum=(a[:,ports]==0).all(1).mean()
                radius=np.sqrt(np.log(12/.01)/(2*n))
                bound=float(report['quartic_full_tv_upper'])
                assert abs(vacuum-row['observed_vacuum'])<1e-14
                delta=abs(vacuum-row['optical_vacuum'])
                assert delta<bound+radius
                tests.append(dict(route=route,test='joint vacuum',observed_discrepancy=delta,
                                  sampling_allowance=radius,model_allowance=bound))
                if route=='pnr':
                    hist=np.bincount((a.sum(1)%4096).astype(int),minlength=4096)
                    ks=abs(np.cumsum(hist/n-law)).max()
                    assert abs(ks-row['mod4096_ks'])<1e-12 and ks<bound+radius
                    tests.append(dict(route=route,test='total photon number modulo 4096',observed_discrepancy=ks,
                                      sampling_allowance=radius,model_allowance=bound))
        stats[case]=tests
    with np.load(OUT/'memory.npz') as f:
        ref=f['optical_clicks'];n=len(f['pnr'])
        rng=np.random.default_rng([122,992]);words=1<<np.arange(6)
        population=np.concatenate([abs(rng.multinomial(n,ref,size=1000)/n-ref).sum(1)/2 for _ in range(100)])
        stats['memory']=dict(records=n,exact_multinomial_null_replicates=len(population),
            null_tv_quantiles=np.quantile(population,[.005,.5,.995]).tolist(),
            empirical_full_click_tv={k:float(abs(np.bincount((f[k]>0)@words,minlength=64)/n-ref).sum()/2)
                                     for k in ('pnr','click')})
    json_write(OUT/'sample_checks.json',stats)
    print(stats,flush=True)


if __name__=='__main__':run()
