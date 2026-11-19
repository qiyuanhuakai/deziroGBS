"""Predeclared direct banded MPS construction panel; preserve all failed jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from iteration14_run import save
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); jobs=[]
    large=[dict(modes=m,seed=s,depth=12,ensemble='local',panel='original_unresolved') for m in (32,128) for s in (150801,150802)]
    large.extend(dict(modes=32,seed=s,depth=2,ensemble='local',panel='shallow_control') for s in (200801,200802))
    for case in large:
        for d in (4,6):
            for w in (1,2,3): jobs.append(dict(case,dimension=d,width=w,rank=128))
    for case in large[:4]: jobs.append(dict(case,dimension=6,width=4,rank=128,panel='wider_resource_probe'))
    for s in (200801,200802):
        for e in ('local','haar'):
            for d in (4,6):
                for w in (1,3): jobs.append(dict(modes=4,seed=s,depth=12,ensemble=e,panel='quality',dimension=d,width=w,rank=16))
        for depth in (2,12): jobs.append(dict(modes=6,seed=s,depth=depth,ensemble='local',panel='quality',dimension=4,width=3 if depth==2 else 5,rank=32))
    for job in jobs: job.update(target=.1,max_elements=40000000,max_seconds=120.)
    random.Random(200899).shuffle(jobs)
    save(out/'plan.json',dict(jobs=jobs,case_count=len(jobs),worker_timeout_seconds=240,scope='Explicit frontier MPS, exponential in bandwidth. All constructed states and failed budgets retained. 40m-complex-element conservative allocation cap is not a process-RSS guarantee.'))
    files=['experiments/iteration20_run.py','experiments/iteration20_worker.py','src/gbslab/banded_gaussian_mps.py','src/gbslab/local_mps.py','src/gbslab/mps_probe.py','src/gbslab/gaussian.py','experiments/iteration10_reference.py','experiments/iteration12_reference.py','tests/test_banded_gaussian_mps.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'metadata.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}))
    rows=[]; env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for i,job in enumerate(jobs):
        spec=out/f'job_{i:02d}.json'; result=out/f'run_{i:02d}.json'; save(spec,job); print('starting',i,job,flush=True)
        try:
            child=subprocess.run([sys.executable,str(ROOT/'experiments/iteration20_worker.py'),str(spec.resolve()),str(result.resolve())],cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
            if child.returncode: raise RuntimeError(f'Worker return {child.returncode}: {child.stderr}')
            row=json.loads(result.read_text())
        except Exception as e: row=dict(job,status='worker_failed',target_met=False,error=repr(e)); save(result,row)
        row['sequence']=i; rows.append(row); save(out/'rows.json',rows); print('finished',i,row['status'],row.get('construction_trace_budget'),row.get('total_seconds'),flush=True)
    save(out/'completion.json',dict(planned=len(jobs),finished=len(rows),accepted=sum(r['status']=='accepted' for r in rows),budget_missed=sum(r['status']=='budget_missed' for r in rows),other=sum(r['status'] not in ('accepted','budget_missed') for r in rows)))

if __name__=='__main__': main()
