"""Freeze the axis policy before executing candidate complete programs."""
import json,sys,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.joint_displacement_coupling import joint_coupling
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
def main():
    out=ROOT/'results/iteration60_selection';out.mkdir(exist_ok=False);names=['research/iteration60_budget_plan.md',*[f'experiments/{n}.py' for n in ('iteration60_selection','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    manifest=read('results/iteration59_manifest.json');valid=read('results/iteration59_noise_validation/validation.json');assert valid['count']==valid['passed'] and deps['results/iteration59_noise_validation/validation.json']==manifest['artifacts']['results/iteration59_noise_validation/validation.json'];records=[];jobs=[]
    for baseline in (1,3,5):
        name=f'results/iteration59_noise/case_{baseline:02d}.json';r=read(name);assert deps[name]==manifest['artifacts'][name]==valid['dependencies'][name];job=r['job'];candidates=[]
        for axis in range(4):
            cells=[16 if i==axis else 4 for i in range(4)];bound=joint_coupling(r['target']['generator_gram'],[dict(sigma=s,cells=N,radius={4:2,16:3}[N]) for s,N in zip(job['sigmas'],cells)]);candidates.append(dict(axis=axis,cells=cells,nodes=3888,bound=bound))
        selected=min(candidates,key=lambda c:(c['bound']['best_upper'],c['axis']));records.append(dict(baseline_case=baseline,family=job['family'],candidates=candidates,selected_axis=selected['axis'],selected_cells=selected['cells'],selection_upper=selected['bound']['best_upper']))
        common=dict(job);common.update(holdout=False,baseline_case=baseline,prior_holdout=job['holdout']);jobs.append(dict(common,action='grid',dimension=4,cells=selected['cells'],selected_axis=selected['axis']));jobs.append(dict(common,action='input',dimension=6,cells=[4]*4,selected_axis=None))
    save(out/'selection.json',encode(records));save(out/'jobs.json',encode(jobs));assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,baselines=3,candidates=12,new_jobs=6,new_nodes=15552,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True));print('selected axes',[r['selected_axis'] for r in records],flush=True)
if __name__=='__main__':main()
