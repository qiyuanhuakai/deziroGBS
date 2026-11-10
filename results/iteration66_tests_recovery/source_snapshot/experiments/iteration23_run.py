"""Frozen paired-quality or isolated serial construction/sampling panel."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from time import perf_counter
from iteration23_support import digest,load_mps,mps_fingerprint,save
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['quality','performance'],required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    if args.phase=='quality':
        cases=[dict(modes=m,seed=230801,ensemble=e,dimension=d,rank=r,panel='quality',max_seconds=120.) for m in (4,6) for e in ('local','haar') for d in (4,6) for r in (2,8,32)]
        repeats=1
    else:
        cases=[dict(modes=m,seed=s,ensemble='local',dimension=10,rank=128,panel='performance',max_seconds=480.) for m in (32,128) for s in (150801,150802,190801,190802)]
        repeats=3
    jobs=[dict(case,backend=backend,repetition=repeat,samples=16,depth=12,target=.1) for case in cases for backend in ('pco','recurrence') for repeat in range(repeats)]
    random.Random(230999 if args.phase=='performance' else 230998).shuffle(jobs)
    for i,job in enumerate(jobs): job['sequence']=i
    files=sorted({Path(__file__).relative_to(ROOT).as_posix(),'experiments/iteration23_support.py','experiments/iteration23_worker.py','experiments/iteration10_reference.py','experiments/iteration12_reference.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))})
    hashes={f:digest(ROOT/f) for f in files}
    for f in files:
        target=out/'source_snapshot'/f; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,target)
    shutil.copy2(ROOT/'research/iteration23_plan.md',out/'frozen_plan.md')
    save(out/'plan.json',dict(phase=args.phase,jobs=jobs,count=len(jobs),worker_timeout_seconds=900,source_hashes=hashes,
        timing_scope='Pipeline: input, construction including budget, displacement, and 16 samples. Excludes saving, post-timing audits and startup; worker wall time also retained.',
        shared_randomness='Same displacement and sampling seeds for both backends and all repetitions; each process independently draws them.',deduplication='Canonical full array fingerprints only; no approximate equality.'))
    known={}
    if args.phase=='performance':
        baseline=json.loads((ROOT/'results/iteration21_panel/rows.json').read_text()); manifest=json.loads((ROOT/'results/iteration21_manifest.json').read_text())['artifacts']
        for row in baseline:
            if row['panel'] not in ('original','confirmation'): continue
            path=ROOT/f"results/iteration21_panel/run_{row['sequence']:02d}.npz"; name=path.relative_to(ROOT).as_posix()
            assert digest(path)==manifest[name]
            known[mps_fingerprint(load_mps(path))]=dict(path=name,file_sha256=manifest[name])
    save(out/'initial_known_states.json',known)
    rows=[]; env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for job in jobs:
        i=job['sequence']; spec=out/f'job_{i:02d}.json'; result=out/f'run_{i:02d}.json'; save(spec,dict(job,known_states=known))
        print('starting',i,job,flush=True); started=perf_counter()
        try:
            child=subprocess.run([sys.executable,str(ROOT/'experiments/iteration23_worker.py'),str(spec),str(result)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=900)
            (out/f'run_{i:02d}.stdout.log').write_text(child.stdout,encoding='utf-8'); (out/f'run_{i:02d}.stderr.log').write_text(child.stderr,encoding='utf-8')
            if child.returncode: raise RuntimeError(f'Worker returned {child.returncode}: {child.stderr}')
            row=json.loads(result.read_text())
        except Exception as e:
            row=dict(job,status='worker_failed',error=repr(e),target_met=False)
            if result.exists(): shutil.copy2(result,result.with_suffix('.partial.json'))
        row['process_wall_seconds']=perf_counter()-started; save(result,row)
        if 'state' in row:
            record=row['state']; known[record['array_fingerprint']]=dict(path=record['path'],file_sha256=record['file_sha256'])
        rows.append(row); save(out/'rows.json',rows)
        print('finished',i,row['status'],row.get('pipeline_seconds'),row.get('error'),flush=True)
    assert hashes=={f:digest(ROOT/f) for f in files}
    save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),accepted=sum(r['status']=='accepted' for r in rows),budget_missed=sum(r['status']=='budget_missed' for r in rows),other=sum(r['status'] not in ('accepted','budget_missed') for r in rows),unchanged_sources=True))


if __name__=='__main__': main()
