"""Fixed finite-input controls with rigorous nonzero-displacement targets."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil,time,traceback
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from iteration46_displaced_trace import enumerate_trace,public_calibration
from gbslab.mps_probe import dense_to_mps
from gbslab.displaced_fock_interval import displaced_finite_law
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration47_displacement';out.mkdir(exist_ok=False);names=['research/iteration47_displacement_plan.md',*[f'experiments/{n}.py' for n in ('iteration47_displacement','iteration46_displaced_trace','iteration22_external','iteration33_run')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    single2=np.array([0,0,1],complex);single3=np.array([0,0,0,1],complex);noon=np.zeros((3,3),complex);noon[2,0]=1;noon[0,2]=1j;general=np.array([[(-1)**(i+j)*(i+1)/(j+2)+1j*(i-j)/8 for j in range(4)] for i in range(4)]);paired=np.diag([.5**k for k in range(6)]).astype(complex);ghz=np.zeros((3,3,3),complex);ghz[0,0,0]=ghz[1,1,1]=1;ghz[2,2,2]=1j;states=[single2,single3,noon,general,paired,ghz];alphas=[(0,0,0),(.125,.25j,-.125),(.5+.25j,-.25+.5j,.25),(1,1j,.5)];jobs=[dict(state=s,displacement=a,modes=states[s].ndim) for s in range(6) for a in range(4)];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies={},numpy_version=np.__version__));rows=[]
    for i,job in enumerate(jobs):
        start=time.perf_counter();row=dict(sequence=i,**job)
        try:
            vector=states[job['state']];built,_=dense_to_mps(vector/np.linalg.norm(vector),16);tensors=[np.ascontiguousarray(a) for a in built];alpha=np.array(alphas[job['displacement']][:job['modes']],complex);np.savez_compressed(out/f'case_{i:02d}.npz',alpha=alpha,**{f'tensor_{j}':a for j,a in enumerate(tensors)});target=displaced_finite_law(tensors,alpha);sampling=enumerate_trace(tensors,alpha);calibration=public_calibration(tensors,alpha,sampling['trace'],473000+i);value=physical_tv_interval(sampling['word_law'],target['intervals']+[(F(0),F(0))]);save(out/f'case_{i:02d}.json',encode(dict(job=job,stored_strides=[a.strides for a in tensors],target=target,sampling=sampling,calibration=calibration,TV_to_exact_measurement_interval=value)));row.update(status='completed',TV=[float(x) for x in value],target_width=max(float(b-a) for a,b in target['intervals']),failure=float(sampling['word_law'][-1]))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('displacement',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=24,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True))
if __name__=='__main__':main()
