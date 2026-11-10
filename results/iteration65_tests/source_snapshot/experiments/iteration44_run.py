"""Fixed nonclassical physical cases with interval-aware completion and tree."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,time,traceback
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from iteration44_physical import physical_input,exact_target
from gbslab.pairwise_completion import fit_candidate,exact_moment_projection
from gbslab.interval_completion import moment_bridge,physical_tv_interval
from gbslab.rational_tree import tree_control
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
def main():
    out=ROOT/'results/iteration44_panel';out.mkdir(exist_ok=False);names=['research/iteration44_panel_plan.md',*[f'experiments/{n}.py' for n in ('iteration44_run','iteration44_physical','iteration22_external','iteration33_run')],*[f'src/gbslab/{n}.py' for n in ('pairwise_completion','interval_completion','rational_tree','rational_gaussian')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    jobs=[dict(modes=n,strength=s,transmission=eta,order=list(range(n)) if order=='forward' else list(reversed(range(n))),order_label=order) for n in (4,8) for s in ('moderate','strong') for eta in (F(1),F(3,4),F(1,4)) for order in ('forward','reverse')];save(out/'plan.json',encode(dict(jobs=jobs,source_hashes=hashes,dependencies={})));rows=[]
    for i,job in enumerate(jobs):
        start=time.perf_counter();row=dict(sequence=i,modes=job['modes'],strength=job['strength'],transmission=str(job['transmission']),order=job['order_label']);print('start physical',i,flush=True)
        try:
            physical=physical_input(job['modes'],job['strength'],job['transmission'],job['order']);target=exact_target(physical['covariance']);mu=[(a+b)/2 for a,b in target['moment_intervals']];fit=fit_candidate(mu,job['modes']);projection=exact_moment_projection(fit['probabilities'],mu,job['modes']);bridge=moment_bridge(projection['law'],target['moment_intervals'],fit['theta'],job['modes']);tree=tree_control(mu,job['modes']);tv=physical_tv_interval(projection['law'],target['target_intervals']);tree_tv=physical_tv_interval(tree['law'],target['target_intervals']);epsilon=bridge['TV_to_true_maxent_upper'];maxent_physical=(max(F(0),tv[0]-epsilon),min(F(1),tv[1]+epsilon))
            result=dict(job=job,physical=physical,target=target,moment_midpoints=mu,fit=fit,projection=projection,bridge=bridge,tree=tree,physical_TV_interval=tv,tree_TV_interval=tree_tv,true_maxent_physical_TV_interval=maxent_physical);save(out/f'case_{i:02d}.json',encode(result));row.update(status='completed',physical_TV=[float(x) for x in tv],tree_TV=[float(x) for x in tree_tv],true_maxent_TV_upper=float(epsilon),true_moment_bridge_TV_upper=float(bridge['TV_to_true_feasible_upper']),nonclassical_rayleigh=float(physical['nonclassical_rayleigh']),optimizer=fit['numerical'])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('end physical',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=24,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True))
if __name__=='__main__':main()
