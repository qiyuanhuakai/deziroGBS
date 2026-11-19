"""Frozen48 mixture errors separated into cancellation and branch imbalance."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.mixture_geometry import cancellation_geometry
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration48_mixture';out.mkdir(exist_ok=False);names=['research/iteration48_mixture_plan.md',*[f'experiments/{n}.py' for n in ('iteration48_mixture','iteration22_external','iteration33_run','verify_iteration34')],*[f'src/gbslab/{n}.py' for n in ('mixture_geometry','interval_completion','pairwise_completion')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    manifest=read('results/iteration47_manifest.json');validation=read('results/iteration47_mixture_validation/validation.json');assert validation['count']==validation['passed'];rows=[]
    for index in range(48):
        name=f'results/iteration47_mixture/case_{index:02d}.json';old=read(name);assert deps[name]==manifest['artifacts'][name];job=old['job'];it=job['round'];first=job['first'];folder='panel' if it==44 else 'inputs';source=read(f'results/iteration{it}_{folder}/case_{first:02d}.json');target=source['target']['target_intervals']+[(F(0),F(0))]*(1024-(1<<old['modes']));q=old['mixture']['aligned_reverse'];p=[2*m-r for m,r in zip(old['mixture']['law'],q)];geometry=cancellation_geometry(p,q,target);assert geometry['forward_TV']==tuple(old['forward_TV']) and geometry['reverse_TV']==tuple(old['reverse_TV']) and geometry['mixture_TV']==tuple(old['mixture_TV']);save(out/f'case_{index:02d}.json',encode(dict(job=job,modes=old['modes'],geometry=geometry)));rows.append(dict(sequence=index,**job,cancellation=[float(x) for x in geometry['cancellation_interval']],branch_imbalance=[float(x) for x in geometry['branch_imbalance_interval']],improvement_over_best=[float(x) for x in geometry['improvement_over_best_interval']]));print('geometry',index,flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'rows.json',rows);save(out/'summary.json',dict(planned=48,completed=48,unchanged_sources=True,unchanged_dependencies=True,source_hashes=hashes,dependencies=deps))
if __name__=='__main__':main()
