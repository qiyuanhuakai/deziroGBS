"""Ideal local cutoff references for four exact eight-mode physical inputs."""
import sys,json,shutil,time
from pathlib import Path
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.gaussian_box_reference import gaussian_box_reference
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration52_box';out.mkdir(exist_ok=False);names=['research/iteration52_mps_plan.md',*[f'experiments/{n}.py' for n in ('iteration52_box','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={};rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    jobs=[dict(source_round=it,case=c,dimension=d) for it in (46,47) for c in (4,6) for d in (3,4)];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes))
    for i,j in enumerate(jobs):
        started=time.perf_counter();v=read(f"results/iteration{j['source_round']}_input_validation/validation.json");assert v['count']==v['passed'];n=f"results/iteration{j['source_round']}_inputs/case_{j['case']:02d}.json";r=read(n);assert deps[n]==v['dependencies'][n];p=r['physical'];box=gaussian_box_reference(p['pairing_real'],p['pairing_imag'],[j['dimension']]*8);assert box['covariance']==p['covariance'];tv=physical_tv_interval(box['distribution'],r['target']['target_intervals']);assert tv[1]<=box['tail_trace_interval'][1];save(out/f'case_{i:02d}.json',encode(dict(job=j,reference=box,physical_TV=tv)));row=dict(case=i,job=j,TV=[float(x) for x in tv],projection=[float(x) for x in box['projection_probability_interval']],tail_trace_upper=float(box['tail_trace_interval'][1]),seconds=time.perf_counter()-started);rows.append(row);save(out/'rows.json',rows);print('box',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=8,completed=8,unchanged_sources=True,unchanged_dependencies=True,source_hashes=hashes,dependencies=deps))
if __name__=='__main__':main()
