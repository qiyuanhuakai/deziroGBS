"""Frozen sixteen-case quality and four-rank candidate construction panel."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from iteration23_support import digest,save
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    jobs=[dict(panel='quality',modes=m,seed=s,dimension=d,rank=r,max_seconds=120.) for m in (4,6) for s in (250801,250802) for d in (4,6) for r in (4,16)]
    jobs+=[dict(panel='candidate',modes=144,dimension=4,rank=r,max_seconds=600.) for r in (32,64,128,200)]
    for i,job in enumerate(jobs): job['sequence']=i
    files=sorted({'experiments/iteration25_run.py','experiments/iteration25_worker.py','experiments/iteration24_reference.py','experiments/iteration23_support.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))}); hashes={f:digest(ROOT/f) for f in files}
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    deps=['results/iteration24_chain/case_56.npz','results/iteration25_spectra/spectra.npz']; dependencies={f:digest(ROOT/f) for f in deps}
    old=json.loads((ROOT/'results/iteration24_manifest.json').read_text())['artifacts']; assert dependencies[deps[0]]==old[deps[0]]
    spectrum=json.loads((ROOT/'results/iteration25_spectra/summary.json').read_text()); assert dependencies[deps[1]]==spectrum['output_sha256']
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependencies,worker_timeout_seconds=900,timing_scope='Single diagnostic construction per case, not repeated isolated speed evidence'))
    shutil.copy2(ROOT/'research/iteration25_plan.md',out/'frozen_plan.md'); env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'); rows=[]
    for job in jobs:
        i=job['sequence']; spec=out/f'job_{i:02d}.json'; target=out/f'run_{i:02d}.json'; save(spec,job); start=perf_counter(); print('starting',i,job,flush=True)
        try:
            child=subprocess.run([sys.executable,str(ROOT/'experiments/iteration25_worker.py'),str(spec),str(target)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=900)
            (out/f'run_{i:02d}.stdout.log').write_text(child.stdout,encoding='utf-8'); (out/f'run_{i:02d}.stderr.log').write_text(child.stderr,encoding='utf-8')
            if child.returncode: raise RuntimeError(child.stderr)
            row=json.loads(target.read_text())
        except Exception as e:
            if target.exists(): shutil.copy2(target,target.with_suffix('.partial.json'))
            row=dict(job,status='worker_failed',error=repr(e))
        row['process_wall_seconds']=perf_counter()-start; save(target,row); rows.append(row); save(out/'rows.json',rows)
        print('finished',i,row['status'],row.get('construction_trace_budget'),row.get('error'),flush=True)
    assert hashes=={f:digest(ROOT/f) for f in files}
    save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),accepted=sum(r['status']=='accepted' for r in rows),budget_missed=sum(r['status']=='budget_missed' for r in rows),other=sum(r['status'] not in ('accepted','budget_missed') for r in rows),unchanged_sources=True))


if __name__=='__main__': main()
