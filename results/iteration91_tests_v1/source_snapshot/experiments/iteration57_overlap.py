"""Combine certified complete-output masses into full-law overlap bounds."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.complete_path_overlap import selected_overlap
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration57_overlap';out.mkdir(exist_ok=False)
    names=['research/iteration57_overlap_plan.md','research/iteration57_interval_note.md','src/gbslab/complete_path_overlap.py','src/gbslab/frugal_law.py',*[f'experiments/{n}.py' for n in ('iteration57_overlap','iteration22_external','iteration33_run','verify_iteration34')]];hashes={n:digest(ROOT/n) for n in names};deps={};rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    for n in ('paths','target'):
        v=read(f'results/iteration57_{n}_validation/validation.json');assert v['count']==v['passed']>0
    physical=read('results/iteration57_target/target.json');oldlocal=read('results/iteration37_local/target_000.json');oldlocal_audit=read('results/iteration39_local/case_03.audit.json');oldglobal=read('results/iteration56_prefix_validation/bounds.json');oldstrong=read('results/iteration56_events_validation/rows.json')
    for it,files in ((37,['results/iteration37_local/target_000.json']),(39,['results/iteration39_local/case_03.audit.json']),(56,['results/iteration56_prefix_validation/bounds.json','results/iteration56_events_validation/rows.json'])):
        manifest=read(f'results/iteration{it}_manifest.json')
        for n in files:assert deps[n]==manifest['artifacts'][n]
    for label in ('local_4','global_1','global_2','global_3','global_4'):
        audit=read('results/iteration57_paths/'+label+'.audit.json');n=audit['job']['modes'];k=audit['job']['cutoff'];laws=audit['laws'];outputs=[r['output'] for r in laws];target=oldlocal['law']['intervals'] if n==10 else [r['probability_interval'] for r in physical['targets']]
        if n==144:assert outputs==[r['output'] for r in physical['targets']]
        models={}
        for model in ('uint64','continuous'):
            q=[r['models'][model]['probability'] for r in laws];bound=selected_overlap(outputs,q,target)
            if n==10:prior=oldlocal_audit['TV_intervals'][model]
            else:prior=(next(r for r in oldstrong if r['cutoff']==k)['full144_lower'] if model=='uint64' else next(r for r in oldglobal if r['cutoff']==k)['prefix_TV_intervals'][model][0],F(1))
            combined=(max(prior[0],bound['full_TV_interval'][0]),min(prior[1],bound['full_TV_interval'][1]));assert combined[0]<=combined[1]
            models[model]=dict(overlap=bound,prior_interval=prior,combined_full_TV_interval=combined,shifted_full_TV_interval=(max(F(0),combined[0]-physical['transfer_upper']),min(F(1),combined[1]+physical['transfer_upper'])) if n==144 else None)
        atoms=[dict(output=s,target_interval=p,probabilities={model:r['models'][model]['probability'] for model in models}) for s,p,r in zip(outputs,target,laws)]
        row=dict(label=label,modes=n,cutoff=k,selected_outputs=len(outputs),models=models);rows.append(row);save(out/(label+'.atoms.json'),encode(atoms));print(label,'gap',float(models['uint64']['overlap']['upper_gap_from_one']),'upper',float(models['uint64']['combined_full_TV_interval'][1]),flush=True)
    save(out/'rows.json',encode(rows));assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,jobs=5,global_outputs_per_cutoff=5,local_outputs=1024,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))


if __name__=='__main__':main()
