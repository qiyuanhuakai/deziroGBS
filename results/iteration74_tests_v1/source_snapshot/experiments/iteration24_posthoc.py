"""Post-reconstruction global algebra and low-photon full-state diagnostics."""
import argparse
import json
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
from gbslab.mps_overlap import normalized_right_mps
from gbslab.banded_gaussian_mps import pairing_log_normalization
from iteration24_reference import global_elimination,low_sector_mps,low_sector_gaussian,partial_state_bounds
from iteration22_external import ARCHIVE_SHA,PREFIX,digest,load_entry,save
from iteration22_global import load_family


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    sources=sorted({'experiments/iteration24_posthoc.py','experiments/iteration24_reference.py','experiments/iteration22_external.py','experiments/iteration22_global.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))})
    hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        dst=out/'source_snapshot'/f; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dst)
    shutil.copy2(ROOT/'research/iteration24_posthoc_plan.md',out/'frozen_plan.md')
    inputs=['results/iteration24_chain/case_56.npz','results/iteration24_chain/case_56.json','results/iteration22_global/rows.json']
    dependencies={name:digest(ROOT/name) for name in inputs}
    save(out/'plan.json',dict(source_hashes=hashes,dependencies=dependencies,archive_sha256=ARCHIVE_SHA,families=['own_recurrence','author_pco','author_hafnian'],global_virtual_cap=20000,scope='Post-hoc after kernel reconstruction, not a performance benchmark or experimental target certificate'))
    assert digest(args.archive)==ARCHIVE_SHA
    chain=json.loads((ROOT/inputs[1]).read_text()); assert chain['status']=='passed' and dependencies[inputs[0]]==chain['output_sha256']
    with np.load(ROOT/inputs[0],allow_pickle=False) as a: B=a['forward']; c=complex(a['forward_log_scalar'])
    started=perf_counter(); entries=[]; kernels=[]
    with zipfile.ZipFile(args.archive) as z:
        for i in range(144):
            K=load_entry(z,PREFIX+f'Q/{i}.npy',entries).astype(complex); K[np.diag_indices_from(K)]*=2; kernels.append(K)
    save(out/'global_inputs.json',entries)
    try:
        G,info=global_elimination(kernels,chain['bond_modes'])
        info.update(pairing_max_error=float(np.max(np.abs(G-B))),log_determinant_magnitude_error=abs(info['log_determinant_abs']+2*c.real),determinant_phase_error=abs(complex(*info['determinant_phase'])-np.exp(-2j*c.imag)),seconds=perf_counter()-started)
        info['status']='passed' if info['pairing_max_error']<1e-9 and max(info['log_determinant_magnitude_error'],info['determinant_phase_error'])<1e-8 else 'mismatch'
        np.savez_compressed(out/'global_pairing.npz',pairing=G)
    except Exception: info=dict(status='failed',traceback=traceback.format_exc(),seconds=perf_counter()-started)
    save(out/'global.json',info); print('global',info['status'],info.get('pairing_max_error'),flush=True)
    previous={}
    for r in json.loads((ROOT/inputs[2]).read_text()):
        left,right=r['pair'].split(':'); previous[left]=r['log_norm_squared_left']; previous[right]=r['log_norm_squared_right']
    rows=[]
    for family in ('own_recurrence','author_pco','author_hafnian'):
        started=perf_counter(); row=dict(family=family)
        try:
            tensors,entries=load_family(family,ROOT/'results/iteration22_external',args.archive)
            save(out/(family+'_inputs.json'),entries)
            if family.startswith('own_'):
                frozen=json.loads((ROOT/'results/iteration22_manifest.json').read_text())['artifacts']
                for entry in entries:
                    path='results/iteration22_external/'+entry['name']; assert entry['sha256']==frozen[path]; dependencies[path]=entry['sha256']
            tensors,lognorm=normalized_right_mps(tensors); low,pairs=low_sector_mps(tensors)
            gauss=low_sector_gaussian(B,pairing_log_normalization(B),pairs)
            row.update(partial_state_bounds(low,gauss),raw_log_norm_squared=2*lognorm,prior_norm_error=abs(2*lognorm-previous[family]),gaussian_two_photon_formula_error=abs(float(np.sum(np.abs(gauss[1:])**2))-.5*abs(gauss[0])**2*float(np.linalg.norm(B)**2)))
            row['status']='passed' if row['prior_norm_error']<1e-9 and row['gaussian_two_photon_formula_error']<1e-12 else 'mismatch'
            target=out/(family+'.npz'); np.savez_compressed(target,mps_low=low,gaussian_low=gauss,pairs=pairs)
            row['output_sha256']=digest(target); del tensors
        except Exception: row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-started; save(out/(family+'.json'),row); rows.append(row); save(out/'rows.json',rows)
        print(family,row['status'],row.get('pure_trace_lower_estimate'),row.get('vacuum_probability_gap'),flush=True)
    assert hashes=={f:digest(ROOT/f) for f in sources}
    save(out/'summary.json',dict(global_status=info['status'],families=len(rows),passed=sum(r['status']=='passed' for r in rows),unchanged_sources=True,dependencies=dependencies))


if __name__=='__main__': main()
