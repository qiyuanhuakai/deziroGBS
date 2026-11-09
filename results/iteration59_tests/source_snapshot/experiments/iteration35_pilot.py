import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil,traceback
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.set_int_max_str_digits(0)
import numpy as np
from gbslab.certified_gaussian import ExactPairingModel,certify_uncertainty
from iteration22_external import save,digest
from iteration33_run import encode

def main():
    out=ROOT/'results/iteration35_pilot';out.mkdir(exist_ok=False);names=['experiments/iteration35_pilot.py','experiments/iteration22_external.py','experiments/iteration33_run.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    deps={p:digest(ROOT/p) for p in ('results/iteration28_panel/case_14.npz','results/iteration31_panel/large_input.npz')};save(out/'plan.json',dict(source_hashes=hashes,dependencies=deps));data=np.load(ROOT/'results/iteration28_panel/case_14.npz');V=np.load(ROOT/'results/iteration31_panel/large_input.npz')['covariance'];rows=[]
    for bits in (64,128):
        start=perf_counter();r=dict(bits=bits)
        try:
            model=ExactPairingModel(data['full_pairing'],data['full_noise_factor'],bits=bits);r.update(status='completed',core=model.core.diagnostics(),noise=model.noise_certificate.diagnostics(),vacuum=model.vacuum_interval,comparison=model.compare_covariance(V),transfers={str(e):model.noise_transfer_bound(V,V+2.**e*np.eye(len(V))) for e in (-52,-50,-46,-40)})
        except Exception:r.update(status='failed',traceback=traceback.format_exc())
        r['seconds']=perf_counter()-start;rows.append(r);save(out/'rows.json',encode(rows));print('model',bits,r['status'],r['seconds'],flush=True)
    start=perf_counter();r=certify_uncertainty(V+2.**-40*np.eye(len(V)));cert=r.pop('certificate',None)
    if cert:r['certificate']=cert.diagnostics()
    r['seconds']=perf_counter()-start;save(out/'shifted_physicality.json',encode(r));print('shifted',r['status'],r['seconds'],flush=True)
    assert hashes=={n:digest(ROOT/n) for n in names};save(out/'summary.json',dict(unchanged_sources=True,completed=sum(r['status']=='completed' for r in rows),shifted_status=r['status']))
if __name__=='__main__':main()
