"""Frozen exploratory resource searches; no successful optimizer is silently accepted."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian_resource import weighted_pure,search_resource,omega
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from iteration23_support import save,digest


def main():
    out=ROOT/'results/iteration28_pilot'; out.mkdir(exist_ok=False)
    files=['experiments/iteration28_pilot.py','experiments/iteration23_support.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]]
    hashes={f:digest(ROOT/f) for f in files}
    for f in files:
        target=out/'source_snapshot'/f; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,target)
    source=ROOT.parent/'external_sources/zenodo18045305/filtration_selected_20261109/jiuzhang2_mixed_cov.npy'
    expected=next(r['sha256'] for r in json.loads((ROOT/'results/iteration27_source/selected.json').read_text()) if r['local_name']==source.name)
    assert digest(source)==expected
    M=np.load(source,allow_pickle=False); m=len(M)//2
    N=(M[:m,:m]+M[m:,m:]-2*np.eye(m)+1j*(M[m:,:m]-M[:m,m:]))/4; ev,U=np.linalg.eigh(N)
    save(out/'plan.json',dict(source_hashes=hashes,external_sha256=expected,max_iterations=1000,objective='Exploratory: validate actual remainder and gap; do not accept optimizer flag alone',photon_normal_spectrum=ev))
    cases=[('small',covariance_from_interferometer(haar_unitary(4,np.random.default_rng(281001)),[.2,.4,.6,.8],[.2,.4,.6,.8])) ,('source_full',M)]
    rows=[]
    for name,V in cases:
        start=perf_counter(); print('starting',name,flush=True)
        baseline,_=weighted_pure(np.linalg.inv(V)); O=omega(len(V)//2)
        base=dict(mean_photons=float((np.trace(baseline)-len(V))/4),minimum_remainder=float(np.linalg.eigvalsh(V-baseline)[0]),purity_residual=float(np.max(abs(baseline@O@baseline-O))))
        def progress(row):
            if row['iteration']%50==0: print(name,row,flush=True)
        result=search_resource(V,max_iterations=1000,callback=progress)
        np.savez_compressed(out/(name+'.npz'),pure=result['pure_covariance'],dual=result['dual_multiplier'],remainder=result['remainder'],williamson_pure=baseline)
        row=dict(name=name,baseline=base,diagnostics=result['diagnostics'],optimizer_success=result['optimizer_success'],optimizer_message=result['optimizer_message'],iterations=result['iterations'],evaluations=result['evaluations'],seconds=perf_counter()-start)
        save(out/(name+'.history.json'),result['history']); rows.append(row); save(out/'rows.json',rows); print(json.dumps(row),flush=True)
    assert hashes=={f:digest(ROOT/f) for f in files}; save(out/'summary.json',dict(unchanged_sources=True,finished=len(rows)))


if __name__=='__main__': main()
