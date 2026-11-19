"""Complete displaced threshold programs on actual Gaussian-SVD finite tensors."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,traceback
from fractions import Fraction as F
import numpy as np
from iteration66_support import ROOT,snapshot,save,digest,constructions,displacement
from verify_iteration34 import decode
from iteration46_displaced_trace import enumerate_trace,public_calibration
from gbslab.extended_displaced_fock import displaced_finite_law
from gbslab.gaussian_fock_bridge import gaussian_displaced_law
from gbslab.interval_completion import physical_tv_interval

def main():
    out=ROOT/'results/iteration66_program';hashes=snapshot(out);deps={}
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    s=read('results/iteration66_gsvd/summary.json');assert s['completed']==s['planned']==s['finished']==20
    v=read('results/iteration66_gsvd_validation/validation.json');assert v['count']==v['passed'] and v['constructions']==20
    jobs=[dict(construction=i,displacement=z,**j) for i,j in enumerate(constructions()) for z in range(3)];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps.copy()));rows=[]
    for index,job in enumerate(jobs):
        start=time.perf_counter()
        try:
            ci=job['construction'];name=f'results/iteration66_gsvd/case_{ci:02d}.json';r=read(name);assert deps[name]==v['dependencies'][name];name=f'results/iteration66_gsvd/case_{ci:02d}.npz';deps[name]=digest(ROOT/name);assert deps[name]==v['dependencies'][name];z=np.load(ROOT/name);tensors=[z[f'tensor_{j}'] for j in range(job['modes'])];alpha=displacement(job['displacement'],job['modes']);finite=displaced_finite_law(tensors,alpha);target=gaussian_displaced_law(job['real'],job['imag'],alpha);sampling=enumerate_trace(tensors,alpha);calibration=public_calibration(tensors,alpha,sampling['trace'],660000+index);arithmetic=physical_tv_interval(sampling['word_law'],finite['intervals']+[(F(0),F(0))]);actual=physical_tv_interval(sampling['word_law'],target['intervals']+[(F(0),F(0))]);upper=min(F(1),r['bridge']['full_trace_interval'][1]+arithmetic[1]);assert actual[1]<=upper
            save(out/f'case_{index:02d}.json',dict(job=job,alpha_hex=[(a.real.hex(),a.imag.hex()) for a in alpha],finite_target=finite,gaussian_target=target,sampling=sampling,calibration=calibration,arithmetic_TV=arithmetic,full_Gaussian_TV=actual,total_TV_upper=upper));row=dict(case=index,construction=ci,family=job['family'],action=job['action'],displacement=job['displacement'],status='completed',nodes=sampling['nodes'],failure=float(sampling['word_law'][-1]),actual_TV=[float(x) for x in actual],total_upper=float(upper),arithmetic_upper=float(arithmetic[1]))
        except Exception:row=dict(case=index,job=job,status='failed',traceback=traceback.format_exc());save(out/f'case_{index:02d}.failed.json',row)
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('program',index,row['status'],flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=60,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
