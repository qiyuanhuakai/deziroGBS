"""Fixed cross-family second/third/fourth-order moment completion panel."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,time,traceback
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.moment_hierarchy import feature_masks,moments_from_vacua,fit_candidate,project,bridge
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
def main():
    out=ROOT/'results/iteration45_hierarchy';out.mkdir(exist_ok=False);names=['research/iteration45_panel_plan.md',*[f'experiments/{n}.py' for n in ('iteration45_hierarchy','iteration22_external','iteration33_run','verify_iteration34')],*[f'src/gbslab/{n}.py' for n in ('moment_hierarchy','pairwise_completion','interval_completion')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    inputs=[dict(round=41,case=i,path=f'results/iteration41_panel/case_{i:02d}.audit.json') for i in (6,8,21,23)]+[dict(round=44,case=i,path=f'results/iteration44_panel/case_{i:02d}.json') for i in (0,4,6,10,12,16,18,22)];records=[];jobs=[]
    for i,source in enumerate(inputs):
        sha=digest(ROOT/source['path']);assert sha==json.loads((ROOT/f"results/iteration{source['round']}_manifest.json").read_text())['artifacts'][source['path']];deps[source['path']]=sha;old=decode(json.loads((ROOT/source['path']).read_text()));n=len(old['job']['weights']) if source['round']==41 else old['job']['modes'];records.append(old)
        for order in ((2,3,4) if n==4 else (2,3)):jobs.append(dict(input=i,source=source,modes=n,order=order))
    assert len(jobs)==30;save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps));rows=[]
    for i,job in enumerate(jobs):
        start=time.perf_counter();row=dict(sequence=i,**job);print('start hierarchy',i,flush=True)
        try:
            old=records[job['input']];n=job['modes'];order=job['order']
            if job['source']['round']==41:
                weights=old['job']['weights'];a=old['job']['intensity'];p=[1/(1+a*sum(weights[j]**2 for j in range(n) if mask>>j&1)) for mask in range(1<<n)];vac=[(x,x) for x in p];target=[(x,x) for x in old['target'][:1<<n]]
            else:vac=old['target']['vacuum_intervals'];target=old['target']['target_intervals']
            intervals=moments_from_vacua(vac,n,order);midpoints=[(a+b)/2 for a,b in intervals];fit=fit_candidate(midpoints,n,order);projection=project(fit['probabilities'],midpoints,n,order);certificate=bridge(projection['law'],intervals,fit['theta'],n,order);tv=physical_tv_interval(projection['law'],target);epsilon=certificate['TV_to_true_maxent_upper'];true_tv=(max(F(0),tv[0]-epsilon),min(F(1),tv[1]+epsilon));save(out/f'case_{i:02d}.json',encode(dict(job=job,target=target,vacua=vac,moment_intervals=intervals,moment_midpoints=midpoints,fit=fit,projection=projection,bridge=certificate,physical_TV_interval=tv,true_maxent_physical_TV_interval=true_tv)));row.update(status='completed',features=len(intervals),physical_TV=[float(x) for x in tv],true_maxent_TV_upper=float(epsilon),minimum_probability=float(min(projection['law'])),optimizer=fit['numerical'])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('end hierarchy',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=30,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
