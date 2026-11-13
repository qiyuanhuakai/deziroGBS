"""Independent propagation and small full-law checks; run AFTER isolated timings."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import mpmath as mp
import numpy as np
from iteration14_run import save
from iteration10_reference import core_parameters

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.local_mps import local_circuit, build_local_core_layerwise
from gbslab.local_displacement import LocalPropagator, iter_local_displacements
from gbslab.sampling_audit import high_precision_law


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    jobs=[dict(modes=m,depth=depth,seed=seed) for m in (2,4,128,512,1024) for depth in (4,12) for seed in (140801,140802)]
    save(out/'plan.json',{'jobs':jobs,'normal_rows':17,'batch_size':7,'profiles':[{'r':.6,'eta':.35}],
        'small_full_law':'m2/4, d4 chi4, first displacement, 80 decimal digits'})
    rows=[]; variance=core_parameters(.6,.35)['residual_variance']
    for index,job in enumerate(jobs):
        m=job['modes']; seed=job['seed']; row=dict(job,index=index)
        try:
            gates,u=local_circuit(m,job['depth'],np.random.default_rng(seed))
            local,none=local_circuit(m,job['depth'],np.random.default_rng(seed),materialize_unitary=False)
            equal=len(gates)==len(local) and all(a[0]==b[0] and all(np.array_equal(x,y) for x,y in zip(a[1:],b[1:])) for a,b in zip(gates,local))
            prop=LocalPropagator(m,local)
            z=np.random.default_rng(seed+10000).normal(size=(17,m))
            dense=.5j*np.sqrt(variance)*(z@u.T)
            direct=np.concatenate(list(iter_local_displacements(prop,variance,17,np.random.default_rng(seed+10000),batch_size=7)))
            residual=float(np.max(np.abs(direct-dense)))
            row.update(gates_exactly_equal=equal,unitary_not_materialized=none is None,alpha_max_abs_error=residual,
                alpha_relative_l2_error=float(np.linalg.norm(direct-dense)/np.linalg.norm(dense)),
                norm_preservation_relative_error=float(np.max(np.abs(np.sum(abs(direct)**2,axis=1)-variance/4*np.sum(z*z,axis=1)))/(variance/4*np.max(np.sum(z*z,axis=1)))),
                compiled_array_bytes=prop.array_bytes,dense_unitary_bytes=u.nbytes)
            if m<=4:
                transfer=.5j*np.sqrt(variance)*prop.apply(np.eye(m)).T
                row['normal_covariance_max_error']=float(np.max(np.abs(transfer@transfer.conj().T-variance/4*np.eye(m))))
                row['anomalous_covariance_max_error']=float(np.max(np.abs(transfer@transfer.T+variance/4*(u@u.T))))
                core=core_parameters(.6,.35)['core_squeezing']
                tensors,_=build_local_core_layerwise(m,core,4,4,gates)
                law_dense=high_precision_law(tensors,dense[0],digits=80)
                law_local=high_precision_law(tensors,direct[0],digits=80)
                with mp.workdps(100):
                    row['conditional_ideal_law_tv']=mp.nstr(sum(abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(law_dense['law_decimal'],law_local['law_decimal']))/2,40)
                save(out/f'law_{index:02d}.json',{'dense':law_dense,'local':law_local})
            np.savez_compressed(out/f'input_{index:02d}.npz',alpha_dense=dense,alpha_local=direct)
            row['status']='complete'
        except Exception as e: row.update(status='failed',error=repr(e))
        rows.append(row); save(out/'rows.json',rows)
        print(index,m,row['status'],row.get('alpha_max_abs_error'),flush=True)
    files=['experiments/iteration14_quality.py','src/gbslab/local_displacement.py','src/gbslab/local_mps.py','src/gbslab/sampling_audit.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'source_hashes.json',{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    save(out/'completion.json',{'planned':len(jobs),'complete':sum(r['status']=='complete' for r in rows),'failures':sum(r['status']!='complete' for r in rows)})


if __name__=='__main__': main()
