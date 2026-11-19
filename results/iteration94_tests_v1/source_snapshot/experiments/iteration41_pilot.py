"""Preserved exploratory exact-rational physical controls, no statistical claims."""
import json,shutil,time
from pathlib import Path
from fractions import Fraction as F
from iteration41_ablation import common_thermal,common_thermal_law,enumerate_laws
from iteration22_external import digest,save
from iteration33_run import encode
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/iteration41_pilot';out.mkdir(exist_ok=False)
    names=['experiments/iteration41_pilot.py','experiments/iteration41_ablation.py','experiments/iteration22_external.py','experiments/iteration33_run.py']
    hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    jobs=[dict(modes=n,intensity=a) for n in (4,6,8) for a in (F(1,16),F(1,4),F(1),F(4),F(16))]
    save(out/'plan.json',encode(dict(jobs=jobs,source_hashes=hashes,scope='Exploratory physical family; exact real controls, not floating sampler')));rows=[]
    for i,job in enumerate(jobs):
        start=time.perf_counter();n=job['modes'];a=job['intensity'];p,c=common_thermal(n,a);laws=enumerate_laws(p,c);target=common_thermal_law(n,a)
        negative={k:sum((-x for x in laws[k] if x<0),F(0)) for k in ('matching','unclamped','clamped')}
        differences={k:sum((abs(x-y) for x,y in zip(laws[k],laws['matching'])),F(0))/2 for k in ('unclamped','clamped','sampler')}
        tv=sum((abs(x-y) for x,y in zip(laws['sampler'],target)),F(0))/2
        save(out/f'case_{i:02d}.json',encode(dict(job=job,laws=laws,target=target,negative_mass=negative,half_L1_to_matching=differences,sampler_TV=tv)))
        row=dict(sequence=i,modes=n,intensity=str(a),negative_mass={k:float(v) for k,v in negative.items()},half_L1_to_matching={k:float(v) for k,v in differences.items()},sampler_TV=float(tv),saturated_prefixes=len(laws['saturated']),seconds=time.perf_counter()-start);rows.append(row);save(out/'rows.json',rows);print(json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=len(jobs),completed=len(rows),unchanged_sources=True))
if __name__=='__main__':main()
