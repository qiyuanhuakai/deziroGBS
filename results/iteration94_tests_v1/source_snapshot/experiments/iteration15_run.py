"""Predeclared adaptive pilot; all unsuccessful attempts and fixed baselines retained."""
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
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    settings=[]
    profiles=[('lossy',.6,.35),('bright',.9,.65)]
    for panel in ('quality','scaling'):
        for m in ([4] if panel=='quality' else [32,128,512]):
            for depth in ([4,12] if m!=512 else [4]):
                for name,r,eta in (profiles if m!=512 else profiles[:1]):
                    for seed in (150801,150802):
                        for target in ([.1,.03] if panel=='quality' else [.1]):
                            settings.append(dict(panel=panel,modes=m,depth=depth,profile=name,r=r,eta=eta,seed=seed,target=target))
    jobs=[dict(setting,method=method,dimensions=[4,6,8,10,12],caps=[8,16,32],
        max_pair_elements=500000,max_attempt_seconds=20.) for setting in settings for method in ('adaptive','fixed_d8_chi16')]
    random.Random(150899).shuffle(jobs)
    save(out/'plan.json',{'jobs':jobs,'case_count':len(jobs),'worker_timeout_seconds':180,
        'scope':'One exploratory run per method/input, not a timing study. Adaptive cost includes rejected/rebuilt attempts. Stop search on first resource or numerical failure. No finite precision TV certificate.'})
    files=['src/gbslab/adaptive_mps.py','src/gbslab/local_mps.py','src/gbslab/local_displacement.py','src/gbslab/mps_probe.py',
        'tests/test_adaptive_mps.py','experiments/iteration15_worker.py','experiments/iteration15_run.py',
        'experiments/iteration12_reference.py','experiments/iteration10_reference.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'metadata.json',{'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}})
    rows=[]; env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for i,job in enumerate(jobs):
        spec=out/f'job_{i:03d}.json'; result=out/f'run_{i:03d}.json'; save(spec,job)
        try:
            child=subprocess.run([sys.executable,str(ROOT/'experiments/iteration15_worker.py'),str(spec.resolve()),str(result.resolve())],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
            if child.returncode: raise RuntimeError(f'Worker failed {child.returncode}: {child.stderr}')
            row=json.loads(result.read_text())
        except Exception as e: row=dict(job,status='worker_failed',target_met=False,error=repr(e)); save(result,row)
        row['sequence']=i; rows.append(row); save(out/'rows.json',rows)
        print(i,job['panel'],job['modes'],job['profile'],job['depth'],job['method'],row['status'],
            row.get('dimension'),row.get('cap'),row.get('construction_trace_budget'),flush=True)
    save(out/'completion.json',{'planned':len(jobs),'finished':len(rows),
        'accepted':sum(r['status']=='accepted' for r in rows),'other':sum(r['status']!='accepted' for r in rows)})


if __name__=='__main__': main()
