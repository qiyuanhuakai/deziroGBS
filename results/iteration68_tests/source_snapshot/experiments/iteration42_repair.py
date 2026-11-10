"""Exact parity-fiber feasibility and target-aware oracle controls on frozen cases."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,shutil,time
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.parity_repair import shift,feasible_correction,oracle_tv_projection,first_negative_marginal
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'results/iteration42_repair';out.mkdir(exist_ok=False);names=['experiments/iteration42_repair.py','src/gbslab/parity_repair.py','experiments/iteration22_external.py','experiments/iteration33_run.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names};deps={};read=lambda p:decode(json.loads(p.read_text()))
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    old=read(ROOT/'results/iteration41_manifest.json');jobs=[20,21,22,23,24,34,35,36,37]
    for i in jobs:
        n=f'results/iteration41_panel/case_{i:02d}.audit.json';sha=digest(ROOT/n);assert old['artifacts'][n]==sha;deps[n]=sha
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps,scope='Exploratory analysis of published round41 cases; target-aware optimizer is an oracle control, not a deployable sampler'))
    rows=[]
    for i in jobs:
        start=time.perf_counter();r=read(ROOT/f'results/iteration41_panel/case_{i:02d}.audit.json');control=r['controls'];base=control['unclamped'];target=r['target'][:256];delta=control['matching'][0]-base[0];assert tuple(control['matching'])==shift(base,delta)
        correction=feasible_correction(base,delta);oracle=oracle_tv_projection(base,target);negative=first_negative_marginal(base);tv=lambda law:None if law is None else sum((abs(x-y) for x,y in zip(law,target)),F(0))/2
        result=dict(case=i,job=r['job'],base=base,target=target,desired_delta=delta,correction=correction,oracle=oracle,first_negative_proper_marginal=negative,base_is_probability=min(base)>=0,matching_is_probability=min(control['matching'])>=0,base_TV=tv(base) if min(base)>=0 else None,correction_TV=tv(correction['law']),matching_half_L1=tv(control['matching']),official_real_TV=r['real_sampler_TV']);save(out/f'case_{i:02d}.json',encode(result))
        row=dict(case=i,intensity=str(r['job']['intensity']),family=r['job']['family'],feasible=correction['interval']['feasible'],base_TV=None if result['base_TV'] is None else float(result['base_TV']),corrected_TV=None if result['correction_TV'] is None else float(result['correction_TV']),oracle_TV=None if oracle['TV'] is None else float(oracle['TV']),official_real_TV=float(r['real_sampler_TV']),negative_proper_order=None if negative is None else negative['order'],seconds=time.perf_counter()-start);rows.append(row);save(out/'rows.json',rows);print(json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=len(jobs),completed=len(rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
