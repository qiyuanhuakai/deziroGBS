"""Freeze pilot or main panel before constructing any Gaussian-basis MPS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from iteration21_worker import save
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--phase',choices=['pilot','panel'],required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    if args.phase=='pilot':
        jobs=[dict(modes=32,seed=150801,depth=12,ensemble='local',panel='pilot_original',dimension=10,rank=128,max_seconds=240.)]
    else:
        jobs=[dict(modes=m,seed=s,depth=12,ensemble='local',panel='original' if s<190000 else 'confirmation',dimension=10,rank=128,max_seconds=480.) for m in (32,128) for s in (150801,150802,190801,190802)]
        jobs.extend(dict(modes=m,seed=150801,depth=12,ensemble='haar',panel='haar_control',dimension=10,rank=128,max_seconds=480.) for m in (32,128))
        jobs.extend(dict(modes=m,seed=s,depth=12,ensemble=e,panel='quality',dimension=d,rank=rank,max_seconds=120.) for m in (4,6) for s in (210901,210902) for e in ('local','haar') for d in (4,6) for rank in (2,8,32))
    for job in jobs: job.update(target=.1,samples=16)
    random.Random(210999).shuffle(jobs)
    save(out/'plan.json',dict(phase=args.phase,jobs=jobs,case_count=len(jobs),worker_timeout_seconds=900,
        scope='Original fixed Schmidt projectors followed by global physical box. Ideal float64 error estimates, not finite-precision sampler TV. All successes and failures retained.'))
    files=['experiments/iteration21_run.py','experiments/iteration21_worker.py','src/gbslab/gaussian_schmidt_mps.py','src/gbslab/gaussian_mps_feasibility.py','src/gbslab/banded_gaussian_mps.py','src/gbslab/local_mps.py','src/gbslab/mps_probe.py','src/gbslab/gaussian.py','experiments/iteration10_reference.py','experiments/iteration12_reference.py','tests/test_gaussian_schmidt_mps.py','tests/test_banded_gaussian_mps.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'metadata.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}))
    rows=[]; env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for i,job in enumerate(jobs):
        spec=out/f'job_{i:02d}.json'; result=out/f'run_{i:02d}.json'; save(spec,job); print('starting',i,job,flush=True)
        try:
            child=subprocess.run([sys.executable,str(ROOT/'experiments/iteration21_worker.py'),str(spec.resolve()),str(result.resolve())],cwd=ROOT,env=env,capture_output=True,text=True,timeout=900)
            if child.returncode: raise RuntimeError(f'Worker return {child.returncode}: {child.stderr}')
            row=json.loads(result.read_text())
        except Exception as e: row=dict(job,status='worker_failed',target_met=False,error=repr(e)); save(result,row)
        row['sequence']=i; rows.append(row); save(out/'rows.json',rows); print('finished',i,row['status'],row.get('construction_trace_budget'),row.get('total_seconds'),row.get('error'),flush=True)
    save(out/'completion.json',dict(planned=len(jobs),finished=len(rows),accepted=sum(r['status']=='accepted' for r in rows),budget_missed=sum(r['status']=='budget_missed' for r in rows),other=sum(r['status'] not in ('accepted','budget_missed') for r in rows)))


if __name__=='__main__': main()
