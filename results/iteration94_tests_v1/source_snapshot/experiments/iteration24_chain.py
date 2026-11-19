"""Frozen known-state, historical-map and external Gaussian chain audit panel."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import zipfile
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian import haar_unitary
from gbslab.local_mps import local_circuit
from gbslab.gaussian_schmidt_mps import decompose_gaussian
from gbslab.gaussian_kernel_chain import fold_kernel_chain,reverse_kernel_chain,GaussianContractionError
from gbslab.banded_gaussian_mps import pairing_log_normalization,pairing_covariance
from iteration22_external import ARCHIVE_SHA,PREFIX,digest,load_entry,save


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    sources=sorted({'experiments/iteration24_chain.py','experiments/iteration22_external.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))})
    hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        target=out/'source_snapshot'/f; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,target)
    shutil.copy2(ROOT/'research/iteration24_plan.md',out/'frozen_plan.md')
    known=[dict(panel='known',modes=m,ensemble=e,seed=s,squeezing=r) for m in (4,8,16,32) for e in ('local','haar') for s in (240801,240802) for r in (.2,.6,1.2)]
    old=json.loads((ROOT/'results/iteration21_panel/rows.json').read_text()); historical=[dict(panel='historical',modes=r['modes'],ensemble=r['ensemble'],seed=r['seed'],old_sequence=r['sequence']) for r in old if r['panel'] in ('original','confirmation')]
    assert len(historical)==8
    jobs=known+historical+[dict(panel='external',modes=144)]; assert len(jobs)==57
    for i,job in enumerate(jobs): job['sequence']=i
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,archive_sha256=ARCHIVE_SHA,margin=1e-12,tolerance=1e-9,timing_scope='Diagnostic only; no isolated performance claims'))
    assert digest(args.archive)==ARCHIVE_SHA
    frozen=json.loads((ROOT/'results/iteration21_manifest.json').read_text())['artifacts']; dependencies={}; rows=[]
    for job in jobs:
        row=dict(job); i=job['sequence']; arrays={}; started=perf_counter()
        try:
            if job['panel']=='known':
                m=job['modes']; rng=np.random.default_rng(job['seed']); U=haar_unitary(m,rng) if job['ensemble']=='haar' else local_circuit(m,12,rng)[1]
                original=-math.tanh(job['squeezing'])*(U@U.T); kernels,records,c,_=decompose_gaussian(original)
                bonds=list(range(m))+[0]; scales=[math.log(r['kernel_scale']) for r in records]+[np.log(c)]
                arrays.update(U=U,original=original)
            elif job['panel']=='historical':
                seq=job['old_sequence']; names=[f'results/iteration21_panel/run_{seq:02d}.maps.npz',f'results/iteration21_panel/run_{seq:02d}.json']
                for name in names:
                    sha=digest(ROOT/name); assert sha==frozen[name]; dependencies[name]=sha
                records=json.loads((ROOT/names[1]).read_text())['info']['cuts']; m=job['modes']
                with np.load(ROOT/names[0],allow_pickle=False) as a:
                    kernels=[a[f'K{j}'] for j in range(m)]; original=a['pairing']; scales=[math.log(r['kernel_scale']) for r in records]+[np.log(complex(a['final_scalar']))]
                bonds=list(range(m))+[0]; arrays['original']=original
                row['dependencies']={name:dependencies[name] for name in names}
            else:
                entries=[]; kernels=[]; bonds=[0]
                with zipfile.ZipFile(args.archive) as z:
                    for j in range(144):
                        K=load_entry(z,PREFIX+f'Q/{j}.npy',entries).astype(complex); K[np.diag_indices_from(K)]*=2; kernels.append(K)
                        if j<143: bonds.append(load_entry(z,PREFIX+f'List_vst/{j}.npy',entries).shape[1])
                bonds.append(0); scales=np.zeros(144,complex); original=None
                save(out/'external_inputs.json',entries); row['scalar_convention']='Local vacuum coefficients set to one, matching supplied coefficient kernels; no inferred physical normalization constant'
            row['bond_modes']=bonds; directions={}
            for direction in ('forward','reverse'):
                try:
                    ks,bs,ss=(kernels,bonds,scales) if direction=='forward' else reverse_kernel_chain(kernels,bonds,scales)
                    B,c,info=fold_kernel_chain(ks,bs,log_scales=ss)
                    if direction=='reverse': B=B[::-1,::-1]
                    arrays[direction]=B; arrays[direction+'_log_scalar']=np.asarray(c)
                    record=dict(status='reconstructed',**info)
                    if original is not None:
                        record.update(original_pairing_max_error=float(np.max(np.abs(B-original))),original_log_normalization_error=abs(c.real-pairing_log_normalization(original)),original_phase_error=abs(np.exp(1j*c.imag)-1))
                    directions[direction]=record
                except Exception as e:
                    directions[direction]=dict(status='failed',error=repr(e),diagnostics=e.diagnostics if isinstance(e,GaussianContractionError) else None,traceback=traceback.format_exc())
            row['directions']=directions
            success=all(r['status']=='reconstructed' for r in directions.values())
            if success:
                cf,cr=complex(arrays['forward_log_scalar']),complex(arrays['reverse_log_scalar'])
                row.update(bidirectional_pairing_max_error=float(np.max(np.abs(arrays['forward']-arrays['reverse']))),bidirectional_log_magnitude_error=abs(cf.real-cr.real),bidirectional_phase_error=abs(np.exp(1j*(cf.imag-cr.imag))-1))
                success=max(row[k] for k in ('bidirectional_pairing_max_error','bidirectional_log_magnitude_error','bidirectional_phase_error'))<1e-9
                if original is not None: success=success and all(max(r[k] for k in ('original_pairing_max_error','original_log_normalization_error','original_phase_error'))<1e-9 for r in directions.values())
                else:
                    V=pairing_covariance(arrays['forward']); arrays['covariance']=V; m=144; omega=np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]])
                    row.update(covariance_min_eigenvalue=float(np.linalg.eigvalsh(V)[0]),covariance_purity_residual=float(np.max(np.abs(V@omega@V-omega))),mean_photons=float((np.trace(V)-2*m)/4))
                    success=success and row['covariance_min_eigenvalue']>0 and row['covariance_purity_residual']<1e-9
            row['status']='passed' if success else 'failed'
        except Exception as e: row.update(status='failed',error=repr(e),traceback=traceback.format_exc())
        row['seconds']=perf_counter()-started
        if arrays:
            target=out/f'case_{i:02d}.npz'; np.savez_compressed(target,**arrays); row['output_sha256']=digest(target)
        save(out/f'case_{i:02d}.json',row); rows.append(row); save(out/'rows.json',rows)
        print(i,job['panel'],row['status'],round(row['seconds'],3),flush=True)
    assert hashes=={f:digest(ROOT/f) for f in sources}
    save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),passed=sum(r['status']=='passed' for r in rows),failed=sum(r['status']!='passed' for r in rows),unchanged_sources=True,dependencies=dependencies))


if __name__=='__main__': main()
