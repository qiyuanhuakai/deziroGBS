"""Recheck all external references, saved arrays, provenance and summaries."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
import scipy
import psutil
from iteration22_external import ARCHIVE_SHA, PREFIX, digest, save
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    source_hash=digest(__file__); shutil.copy2(__file__,out/Path(__file__).name)
    save(out/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,cpu_count=psutil.cpu_count(),physical_cpu_count=psutil.cpu_count(logical=False),total_memory=psutil.virtual_memory().total,threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    checks=[]
    def check(name,passed): checks.append(dict(name=name,passed=bool(passed)))
    check('archive SHA256',digest(args.archive)==ARCHIVE_SHA)
    panel=ROOT/'results/iteration22_external'; rows=json.loads((panel/'rows.json').read_text()); plan=json.loads((panel/'plan.json').read_text())
    check('all 144 sites exactly once',len(rows)==144 and sorted(r['site'] for r in rows)==list(range(144)))
    check('all workers terminal success',all(r['status']=='passed' for r in rows))
    entry_hashes={}
    with zipfile.ZipFile(args.archive) as z:
        for row in rows:
            i=row['site']; tag=f'site {i}'
            path=panel/f'site_{i:03d}.npz'
            check(tag+' saved hash',digest(path)==row['output_sha256'])
            with np.load(path,allow_pickle=False) as data: pco=data['pco']; rec=data['recurrence']
            reference=np.load(io.BytesIO(z.read(PREFIX+f'ts/D200/{i}.npy')),allow_pickle=False)
            check(tag+' shapes and finite',list(pco.shape)==row['shape'] and pco.shape==rec.shape==reference.shape and np.isfinite(pco).all() and np.isfinite(rec).all())
            for name,a,b in [('pco_reference',pco,reference),('recurrence_reference',rec,reference),('recurrence_pco',rec,pco)]:
                actual=dict(max_abs=float(np.max(np.abs(a-b))),frobenius=float(np.linalg.norm(a-b)),relative_frobenius=float(np.linalg.norm(a-b)/np.linalg.norm(b)))
                check(tag+' '+name+' metrics',all(abs(actual[k]-row['errors'][name][k])<1e-20 for k in actual))
                check(tag+' '+name+' threshold',actual['max_abs']<=plan['local_max_abs_tolerance'])
            for entry in row['entries']:
                if entry['name'] not in entry_hashes: entry_hashes[entry['name']]=hashlib.sha256(z.read(entry['name'])).hexdigest()
                check(tag+' external entry '+entry['name'],entry_hashes[entry['name']]==entry['sha256'])
        global_dir=ROOT/'results/iteration22_global'
        for f in sorted(global_dir.glob('*_input_hashes.json')):
            for entry in json.loads(f.read_text()):
                if f.name.startswith('own_'): value=digest(panel/entry['name'])
                else:
                    if entry['name'] not in entry_hashes: entry_hashes[entry['name']]=hashlib.sha256(z.read(entry['name'])).hexdigest()
                    value=entry_hashes[entry['name']]
                check(f.name+' '+entry['name'],value==entry['sha256'])
    global_rows=json.loads((global_dir/'rows.json').read_text())
    check('four global pairs complete',len(global_rows)==4 and [r['pair'] for r in global_rows]==plan['global_pairs'] and all(r['status']=='complete' for r in global_rows))
    for row in global_rows:
        pair=row['pair']; delta=row['phase_aligned_difference_norm']; distance=row['pure_trace_distance_from_difference']; logf=row['raw_log_fidelity']
        check(pair+' independent QR norms',abs(row['log_norm_qr_residual_left'])<1e-9 and abs(row['log_norm_qr_residual_right'])<1e-9)
        check(pair+' difference formula',abs(distance-delta*math.sqrt(1-delta*delta/4))<1e-15)
        check(pair+' overlap consistency',abs(distance**2-(1. if logf is None else -math.expm1(min(0.,logf))))<1e-9)
    bench=ROOT/'results/iteration22_benchmark'; br=json.loads((bench/'rows.json').read_text()); bp=json.loads((bench/'plan.json').read_text())
    check('all 30 benchmark repetitions',len(br)==30 and [dict(site=r['site'],algorithm=r['algorithm'],repetition=r['repetition']) for r in br]==bp['jobs'])
    for i,r in enumerate(br): check(f'benchmark {i} full coefficient quality',r['passed'] and r['max_abs_error']<=1e-10 and r['seconds']>0)
    provenance=[(panel,plan['source_hashes'],panel/'sources'),(global_dir,json.loads((global_dir/'source_hashes.json').read_text()),global_dir/'sources'),(bench,json.loads((bench/'source_hashes.json').read_text()),bench/'sources')]
    for directory,hashes,snapshots in provenance:
        for name,sha in hashes.items():
            check(directory.name+' frozen '+name,digest(snapshots/Path(name).name)==sha and digest(ROOT/name)==sha)
    testdir=ROOT/'results/iteration22_tests'; tests=json.loads((testdir/'validation.json').read_text())
    for name,sha in json.loads((testdir/'source_hashes_at_start.json').read_text()).items():
        check('test source '+name,digest(testdir/'source_snapshot'/name)==sha and digest(ROOT/name)==sha)
    check('eleven new tests and inherited failures',tests['new_tests_passed']==11 and tests['unchanged_failure_ids'] and tests['unchanged_sources'])
    check('verifier unchanged',digest(__file__)==source_hash)
    summary=dict(count=len(checks),passed=sum(c['passed'] for c in checks),external_entries=len(entry_hashes),sites=len(rows),global_pairs=len(global_rows),benchmark_runs=len(br),verifier_sha256=source_hash,checks=checks)
    save(out/'validation.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
    if not all(c['passed'] for c in checks): raise AssertionError('Preserve failed audit output')


if __name__=='__main__': main()
