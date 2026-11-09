"""Fixed equal mixture on historical diagnostics and a newly frozen holdout."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.order_mixture import reversal_mixture
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration47_mixture';out.mkdir(exist_ok=False);names=['research/iteration47_mixture_plan.md',*[f'experiments/{n}.py' for n in ('iteration47_mixture','iteration22_external','iteration33_run','verify_iteration34')],*[f'src/gbslab/{n}.py' for n in ('order_mixture','interval_completion','pairwise_completion')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n,round_number=None):
        deps[n]=digest(ROOT/n)
        if round_number:
            manifest=json.loads((ROOT/f'results/iteration{round_number}_manifest.json').read_text());assert deps[n]==manifest['artifacts'][n]
        return decode(json.loads((ROOT/n).read_text()))
    jobs=[dict(round=it,first=i,cutoff=k,held_out=it==47) for it in (44,46,47) for i in range(0,24 if it==44 else 8,2) for k in ((2,4) if it==44 else (2,3,4))];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies={},material_TV_threshold='1e-10'));rows=[]
    for index,job in enumerate(jobs):
        it=job['round'];first=job['first'];k=job['cutoff'];bind=it if it<47 else None;folder='panel' if it==44 else 'inputs';source=read(f'results/iteration{it}_{folder}/case_{first:02d}.json',bind);other=read(f'results/iteration{it}_{folder}/case_{first+1:02d}.json',bind);n=source['job']['modes'];a=read(f'results/iteration{it}_native/case_{first:02d}_k{k}.audit.json',bind);b=read(f'results/iteration{it}_native/case_{first+1:02d}_k{k}.audit.json',bind);p=a['laws']['uint64']['distribution'];q=b['laws']['uint64']['distribution'];mixed=reversal_mixture(p,q,n);target=source['target']['target_intervals']+[(F(0),F(0))]*(1024-(1<<n));reverse_target=other['target']['target_intervals'];assert all(target[m]==reverse_target[sum(((m>>j)&1)<<(n-1-j) for j in range(n))] for m in range(1<<n));ptv=physical_tv_interval(p,target);qtv=physical_tv_interval(mixed['aligned_reverse'],target);mtv=physical_tv_interval(mixed['law'],target);average=((ptv[0]+qtv[0])/2,(ptv[1]+qtv[1])/2);gain=(max(F(0),average[0]-mtv[1]),max(F(0),average[1]-mtv[0]));assert mtv[1]<=average[1];record=dict(job=job,modes=n,mixture=mixed,forward_TV=ptv,reverse_TV=qtv,mixture_TV=mtv,average_TV=average,cancellation_gain_interval=gain,strict_better_both=mtv[1]<min(ptv[0],qtv[0]),material_better_both=mtv[1]+F(1,10**10)<min(ptv[0],qtv[0]),material_worse_best=mtv[0]>min(ptv[1],qtv[1])+F(1,10**10));save(out/f'case_{index:02d}.json',encode(record));rows.append(dict(sequence=index,**job,forward_TV=float(ptv[0]),reverse_TV=float(qtv[0]),mixture_TV=float(mtv[0]),material_better_both=record['material_better_both'],material_worse_best=record['material_worse_best'],cancellation_gain=[float(x) for x in gain]));print('mixture',index,'heldout',job['held_out'],'TV',float(mtv[0]),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'rows.json',rows);save(out/'summary.json',dict(planned=48,completed=48,unchanged_sources=True,unchanged_dependencies=True,source_hashes=hashes,dependencies=deps,heldout_better_both=sum(r['held_out'] and r['material_better_both'] for r in rows),heldout_worse_best=sum(r['held_out'] and r['material_worse_best'] for r in rows)))
if __name__=='__main__':main()
