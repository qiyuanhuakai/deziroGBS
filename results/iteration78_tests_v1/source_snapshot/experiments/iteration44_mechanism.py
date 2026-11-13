"""Post-inspection control: same low moments, different higher-order choices."""
import json,shutil,sys
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration41_ablation import enumerate_laws
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
def main():
    out=ROOT/'results/iteration44_mechanism';out.mkdir(exist_ok=False);names=['research/iteration44_mechanism_plan.md',*[f'experiments/{n}.py' for n in ('iteration44_mechanism','iteration22_external','iteration33_run','verify_iteration34','iteration41_ablation')],*[f'src/gbslab/{n}.py' for n in ('interval_completion','pairwise_completion')]];hashes={n:digest(ROOT/n) for n in names};deps={};rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    for i in range(12):
        name=f'results/iteration44_panel/case_{i:02d}.json';deps[name]=digest(ROOT/name);r=decode(json.loads((ROOT/name).read_text()));n=r['job']['modes'];assert n==4;mu=r['moment_midpoints'];p=[(1+x)/2 for x in mu[:n]];pairs=[(i,j) for j in range(n) for i in range(j)];cov={(i,j):(mu[n+k]-mu[i]*mu[j])/4 for k,(i,j) in enumerate(pairs)};controls=enumerate_laws(p,cov);law=controls['matching'];valid=min(law)>0 and sum(law)==1;tv=physical_tv_interval(law,r['target']['target_intervals']) if valid else None;rho=r['bridge']['atom_radius'];bridge=r['bridge']['TV_to_true_feasible_upper'];true_feasible=valid and min(law)>rho;upper=None if not true_feasible else min(F(1),tv[1]+bridge);gap=None if upper is None else r['true_maxent_physical_TV_interval'][0]-upper
        result=dict(case=i,pvac=p,covariance=cov,controls=controls,strictly_positive=valid,matching_TV=tv,true_moment_feasible=true_feasible,true_feasible_physical_TV_upper=upper,true_maxent_excess_lower=gap);save(out/f'case_{i:02d}.json',encode(result));row=dict(case=i,positive=valid,matching_equals_unclamped=law==controls['unclamped'],matching_equals_sampler=law==controls['sampler'],matching_TV=None if tv is None else [float(x) for x in tv],true_maxent_excess_lower=None if gap is None else float(gap));rows.append(row);print(json.dumps(row),flush=True)
    save(out/'plan.json',dict(cases=list(range(12)),source_hashes=hashes,dependencies=deps,scope='Exploratory post-inspection control'));save(out/'rows.json',rows);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=12,completed=12,positive=sum(r['positive'] for r in rows),maxent_worse=sum(r['true_maxent_excess_lower'] is not None and r['true_maxent_excess_lower']>0 for r in rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
