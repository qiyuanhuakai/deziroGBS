"""Revisit four unresolved inputs under matched expanded resources; two ablations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import numpy as np
from iteration14_run import save

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); jobs=[]
    for m in (4,32,128):
        for seed in (150801,150802):
            for target in ((.03,.1) if m==4 else (.1,)):
                for method in (('restart','checkpoint_rolling','checkpoint_equal') if m==32 else ('restart','checkpoint_rolling')):
                    jobs.append(dict(panel='quality' if m==4 else 'unresolved',modes=m,depth=12,seed=seed,target=target,
                        method=method,dimensions=[10,12],caps=[8,16,32,64,128],max_pair_elements=8000000,
                        max_checkpoint_elements=150000000,max_attempt_seconds=60.))
    random.Random(160899).shuffle(jobs)
    save(out/'plan.json',{'jobs':jobs,'case_count':len(jobs),'worker_timeout_seconds':600,
        'scope':'Exploratory one-run matched caps/dimensions/attempt limits. Time cap checks occur between operations. Checkpoint method also caps clone size. Compare against rerun restart, not old smaller-resource failures. No certified finite precision output TV.'})
    files=['src/gbslab/checkpoint_mps.py','src/gbslab/adaptive_mps.py','src/gbslab/local_mps.py','src/gbslab/local_displacement.py',
        'src/gbslab/mps_probe.py','tests/test_checkpoint_mps.py','experiments/iteration16_run.py','experiments/iteration16_worker.py',
        'experiments/iteration12_reference.py','experiments/iteration10_reference.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'metadata.json',{'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}})
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'); rows=[]
    for i,job in enumerate(jobs):
        spec=out/f'job_{i:02d}.json'; result=out/f'run_{i:02d}.json'; save(spec,job)
        print('starting',i,job['modes'],job['seed'],job['method'],flush=True)
        try:
            child=subprocess.run([sys.executable,str(ROOT/'experiments/iteration16_worker.py'),str(spec.resolve()),str(result.resolve())],cwd=ROOT,env=env,capture_output=True,text=True,timeout=600)
            if child.returncode: raise RuntimeError(f'Worker return {child.returncode}: {child.stderr}')
            row=json.loads(result.read_text())
        except Exception as e: row=dict(job,status='worker_failed',target_met=False,error=repr(e)); save(result,row)
        row['sequence']=i; rows.append(row); save(out/'rows.json',rows)
        print('finished',i,row['status'],row.get('construction_trace_budget'),round(row.get('search_seconds',0),3),flush=True)
    save(out/'completion.json',{'planned':len(jobs),'finished':len(rows),'accepted':sum(r['status']=='accepted' for r in rows),
        'other':sum(r['status']!='accepted' for r in rows)})


if __name__=='__main__': main()
