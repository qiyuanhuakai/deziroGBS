"""Frozen new exact physical cases for parity repair; real reference only."""
from pathlib import Path
from fractions import Fraction as F
import json,shutil,time
from iteration22_external import digest,save
from iteration33_run import encode
from iteration41_physical import coefficients,target_law
from iteration41_ablation import enumerate_laws
from gbslab.parity_repair import shift,feasible_correction,oracle_tv_projection,first_negative_marginal
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/iteration42_fresh';out.mkdir(exist_ok=False);names=['research/iteration42_repair_plan.md','src/gbslab/parity_repair.py',*[f'experiments/{n}.py' for n in ('iteration42_fresh','iteration22_external','iteration33_run','iteration41_physical','iteration41_ablation')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    jobs=[]
    for a in (F(3,16),F(3,8),F(3,4),F(3),F(12)):jobs.append(dict(family='equal',intensity=a,weights=[F(1)]*8,order=list(range(8))))
    w=[F(x,4) for x in (4,3,2,1)*2]
    for a in (F(3,16),F(3,8),F(3,4),F(3),F(12)):
        for order in (list(range(8)),list(reversed(range(8)))):jobs.append(dict(family='weighted',intensity=a,weights=[w[j] for j in order],order=order))
    save(out/'plan.json',encode(dict(jobs=jobs,source_hashes=hashes,scope='New exact real reference cases; not new official floating sampler output')));rows=[]
    for i,job in enumerate(jobs):
        start=time.perf_counter();p,c=coefficients(job['weights'],job['intensity']);control=enumerate_laws(p,c);base=control['unclamped'];target=target_law(job['weights'],job['intensity']);delta=control['matching'][0]-base[0];assert tuple(control['matching'])==shift(base,delta);correction=feasible_correction(base,delta);oracle=oracle_tv_projection(base,target);negative=first_negative_marginal(base);tv=lambda law:None if law is None else sum((abs(x-y) for x,y in zip(law,target)),F(0))/2
        result=dict(case=i,job=job,physical_pvac=p,physical_covariance=c,controls=control,base=base,target=target,desired_delta=delta,correction=correction,oracle=oracle,first_negative_proper_marginal=negative,base_is_probability=min(base)>=0,matching_is_probability=min(control['matching'])>=0,base_TV=tv(base) if min(base)>=0 else None,correction_TV=tv(correction['law']),matching_half_L1=tv(control['matching']),official_formula_real_TV=tv(control['sampler']));save(out/f'case_{i:02d}.json',encode(result))
        row=dict(case=i,family=job['family'],intensity=str(job['intensity']),order=job['order'],feasible=correction['interval']['feasible'],base_TV=None if result['base_TV'] is None else float(result['base_TV']),corrected_TV=None if result['correction_TV'] is None else float(result['correction_TV']),oracle_TV=None if oracle['TV'] is None else float(oracle['TV']),real_sampler_TV=float(result['official_formula_real_TV']),negative_proper_order=None if negative is None else negative['order'],seconds=time.perf_counter()-start);rows.append(row);save(out/'rows.json',rows);print(json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=len(jobs),completed=len(rows),unchanged_sources=True))
if __name__=='__main__':main()
