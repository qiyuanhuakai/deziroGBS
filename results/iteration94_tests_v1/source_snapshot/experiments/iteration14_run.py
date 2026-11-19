"""Serial randomized isolated-process benchmark, fixed panel before execution."""
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
import psutil

ROOT=Path(__file__).resolve().parents[1]


def save(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    methods=['dense_full','dense_stream','local_stream']; jobs=[]
    for panel in ('noise','pipeline'):
        for m in ((128,1024) if panel=='noise' else (128,512)):
            for depth in ((4,12) if panel=='noise' else (4,)):
                for count in ((128,2048) if panel=='noise' else (128,)):
                    for seed in (140801,140802):
                        for repeat in range(3):
                            for method in methods:
                                jobs.append(dict(panel=panel,modes=m,depth=depth,samples=count,seed=seed,repeat=repeat,
                                    method=method,batch_size=64,r=.6,eta=.35,d=6,chi=8))
    random.Random(140899).shuffle(jobs)
    save(out/'plan.json',{'jobs':jobs,'job_count':len(jobs),'nominal_construction_threshold':.1,
        'scope':'All cases retained. Timing repeats are not independent circuits. Noise-only speedups are not sampler speedups. Pipeline includes construction, noise, sampling and output creation; excludes imports and disk IO. Peak memory includes imports.'})
    files=['src/gbslab/local_mps.py','src/gbslab/local_displacement.py','src/gbslab/mps_probe.py',
        'tests/test_local_displacement.py','experiments/iteration14_run.py','experiments/iteration14_worker.py','experiments/iteration10_reference.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'metadata.json',{'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'psutil':psutil.__version__,
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}})
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    rows=[]
    for i,job in enumerate(jobs):
        spec=out/f'job_{i:03d}.json'; result=out/f'run_{i:03d}.json'; save(spec,job)
        try:
            process=subprocess.run([sys.executable,str(ROOT/'experiments/iteration14_worker.py'),str(spec.resolve()),str(result.resolve())],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
            if process.returncode!=0: raise RuntimeError(f'Worker return code {process.returncode}: {process.stderr}')
            row=json.loads(result.read_text())
        except Exception as e: row=dict(job,status='worker_failed',error=repr(e)); save(result,row)
        row['sequence']=i; rows.append(row); save(out/'rows.json',rows)
        print(i,job['panel'],job['modes'],job['method'],row['status'],round(row.get('end_to_end_seconds',0),4),flush=True)
    save(out/'completion.json',{'planned':len(jobs),'finished':len(rows),'complete':sum(r['status']=='complete' for r in rows),
        'failures':sum(r['status']!='complete' for r in rows)})


if __name__=='__main__': main()
