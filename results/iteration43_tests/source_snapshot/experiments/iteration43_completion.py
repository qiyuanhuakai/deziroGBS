"""Exact feasible moment completions and entropy certificates on a fixed panel."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,shutil,time,traceback
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.pairwise_completion import feature_masks,character,fit_candidate,exact_moment_projection,entropy_certificate
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/iteration43_completion';out.mkdir(exist_ok=False)
    names=['research/iteration43_panel_plan.md','src/gbslab/pairwise_completion.py',*[f'experiments/{n}.py' for n in ('iteration43_completion','iteration22_external','iteration33_run','verify_iteration34')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    jobs=[dict(round=41,case=i,path=f'results/iteration41_panel/case_{i:02d}.audit.json') for i in (6,8,21,23,26,28,34,35,36,37)]+[dict(round=42,case=i,path=f'results/iteration42_fresh/case_{i:02d}.json') for i in range(15)]
    for job in jobs:
        sha=digest(ROOT/job['path']);assert sha==json.loads((ROOT/f"results/iteration{job['round']}_manifest.json").read_text())['artifacts'][job['path']];deps[job['path']]=sha
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps));rows=[]
    for i,job in enumerate(jobs):
        start=time.perf_counter();row=dict(sequence=i,source=job);print('start completion',i,flush=True)
        try:
            old=decode(json.loads((ROOT/job['path']).read_text()));n=len(old['job']['weights']);target=old['target'][:1<<n];masks=feature_masks(n);mu=[sum((p*character(a,m) for a,p in enumerate(target)),F(0)) for m in masks]
            fit=fit_candidate(mu,n);projection=exact_moment_projection(fit['probabilities'],mu,n);certificate=entropy_certificate(projection['law'],mu,fit['theta'],n);tv=sum((abs(x-y) for x,y in zip(projection['law'],target)),F(0))/2
            baseline=old['real_sampler_TV'] if job['round']==41 else old['official_formula_real_TV'];result=dict(job=job,physical_job=old['job'],modes=n,target=target,moments=mu,fit=fit,projection=projection,certificate=certificate,physical_TV=tv,reference_TV=baseline);save(out/f'case_{i:02d}.json',encode(result));row.update(status='completed',modes=n,physical_TV=float(tv),reference_TV=float(baseline),entropy_gap_upper=float(certificate['entropy_gap_upper']),maxent_TV_upper=float(certificate['TV_to_exact_maximum_entropy_upper']),minimum_probability=float(projection['minimum_probability']),optimizer=fit['numerical'])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('end completion',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=len(jobs),completed=sum(r['status']=='completed' for r in rows),finished=len(rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
