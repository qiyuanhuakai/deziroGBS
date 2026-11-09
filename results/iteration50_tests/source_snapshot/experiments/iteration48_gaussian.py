"""Fixed exact Gaussian targets, finite MPS budgets, and displacement audits."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,time,traceback,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from iteration46_displaced_trace import enumerate_trace,public_calibration
from gbslab.mps_probe import dense_to_mps
from gbslab.gaussian_fock_bridge import hermite_box,gaussian_fidelity,gaussian_displaced_law
from gbslab.displaced_fock_interval import displaced_finite_law
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def inputs():
    result=[]
    for family in ('single','paired','complex'):
        for strength,a in (('moderate',F(1,2)),('strong',F(3,4))):
            if family=='single':r=[[a]];i=[[F(0)]]
            elif family=='paired':r=[[F(0),a],[a,F(0)]];i=[[F(0),F(0)],[F(0),F(0)]]
            else:
                b=F(1,4) if strength=='moderate' else F(1,2);r=[[(a-b)/2,F(0)],[F(0),(b-a)/2]];i=[[F(0),(a+b)/2],[(a+b)/2,F(0)]]
            result.append(dict(family=family,strength=strength,real=r,imag=i,modes=len(r),parameter=a))
    return result


def main():
    out=ROOT/'results/iteration48_gaussian';out.mkdir(exist_ok=False);names=['research/iteration48_gaussian_plan.md',*[f'experiments/{n}.py' for n in ('iteration48_gaussian','iteration46_displaced_trace','iteration22_external','iteration33_run')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    targets=inputs();jobs=[dict(target=s,modes=t['modes'],family=t['family'],strength=t['strength'],dimension=d,bond=b,displacement=z) for s,t in enumerate(targets) for d in (4,6,8) for b in ((d,) if t['modes']==1 else (2,d)) for z in (0,1)];save(out/'plan.json',encode(dict(jobs=jobs,targets=targets,source_hashes=hashes,dependencies={},numpy_version=np.__version__)));rows=[];cache={}
    for index,job in enumerate(jobs):
        start=time.perf_counter();row=dict(sequence=index,**job)
        try:
            t=targets[job['target']];d=job['dimension'];n=t['modes'];key=(job['target'],d,job['bond'])
            if key not in cache:
                occupations,H=hermite_box(t['real'],t['imag'],[d]*n);vector=np.array([complex(float(a),float(b))/math.sqrt(math.prod(math.factorial(k) for k in o)) for o,(a,b) in zip(occupations,H)]).reshape((d,)*n);vector/=np.linalg.norm(vector);built,info=dense_to_mps(vector,job['bond']);tensors=[np.ascontiguousarray(a) for a in built];bridge=gaussian_fidelity(tensors,t['real'],t['imag']);cache[key]=(tensors,info,bridge)
            tensors,info,bridge=cache[key];alpha=np.array(([0,0] if job['displacement']==0 else [.5+.25j,-.25+.5j])[:n],complex);np.savez_compressed(out/f'case_{index:02d}.npz',alpha=alpha,**{f'tensor_{j}':a for j,a in enumerate(tensors)});finite=displaced_finite_law(tensors,alpha);gaussian=gaussian_displaced_law(t['real'],t['imag'],alpha);sampling=enumerate_trace(tensors,alpha);calibration=public_calibration(tensors,alpha,sampling['trace'],483000+index);arithmetic=physical_tv_interval(sampling['word_law'],finite['intervals']+[(F(0),F(0))]);actual=physical_tv_interval(sampling['word_law'],gaussian['intervals']+[(F(0),F(0))]);total_upper=min(F(1),bridge['full_trace_interval'][1]+arithmetic[1]);assert actual[1]<=total_upper;save(out/f'case_{index:02d}.json',encode(dict(job=job,input=t,stored_strides=[a.strides for a in tensors],construction=info,bridge=bridge,finite_target=finite,gaussian_target=gaussian,sampling=sampling,calibration=calibration,arithmetic_TV=arithmetic,full_Gaussian_TV=actual,total_TV_upper=total_upper)));row.update(status='completed',tail_mass=[float(x) for x in bridge['tail_mass_interval']],state_trace=[float(x) for x in bridge['full_trace_interval']],compression_trace=[float(x) for x in bridge['compression_trace_interval']],arithmetic_TV=[float(x) for x in arithmetic],Gaussian_TV=[float(x) for x in actual],total_TV_upper=float(total_upper),failure=float(sampling['word_law'][-1]))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('gaussian',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,source_hashes=hashes,dependencies={}))
if __name__=='__main__':main()
