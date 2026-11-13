"""Serial paired local-coefficient timing; not end-to-end sampler timing."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
import zipfile
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import numpy as np
import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gbslab.gaussian_pair_recurrence import pair_recurrence_tensor
from gbslab.gaussian_schmidt_mps import projected_pair_tensor
from iteration22_external import ARCHIVE_SHA, digest, load_site, save


def main():
    p = argparse.ArgumentParser(); p.add_argument('--archive', type=Path, required=True); p.add_argument('--panel', type=Path, required=True); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    out=args.output; out.mkdir(parents=True, exist_ok=False)
    sources=['experiments/iteration22_benchmark.py', 'experiments/iteration22_external.py', 'src/gbslab/gaussian_pair_recurrence.py', 'src/gbslab/gaussian_schmidt_mps.py']
    hashes={f:digest(ROOT/f) for f in sources}; (out/'sources').mkdir()
    for f in sources: shutil.copy2(ROOT/f, out/'sources'/Path(f).name)
    save(out/'source_hashes.json', hashes)
    assert json.loads((args.panel/'summary.json').read_text())['completed']==144
    assert digest(args.archive)==ARCHIVE_SHA
    spec=json.loads((args.panel/'plan.json').read_text())['isolated_benchmark']
    jobs=[dict(site=site, algorithm=algo, repetition=repeat) for site in spec['sites'] for algo in ('pco','recurrence') for repeat in range(spec['repetitions'])]
    random.Random(spec['seed']).shuffle(jobs); save(out/'plan.json', dict(jobs=jobs, seed=spec['seed'], warmup='none; all three repetitions retained', threads=1, timing_scope='coefficient call only, excludes input loading and comparison'))
    rows=[]
    for job in jobs:
        with zipfile.ZipFile(args.archive) as z: Q,left,right,reference,entries=load_site(z,job['site'])
        function=projected_pair_tensor if job['algorithm']=='pco' else pair_recurrence_tensor
        started=perf_counter(); actual,info=function(Q,left,4,right); seconds=perf_counter()-started
        error=float(np.max(np.abs(actual-reference))); memory=psutil.Process().memory_info()
        row=dict(**job, seconds=seconds, max_abs_error=error, passed=error<=1e-10, info=info, process_peak_working_set_bytes=getattr(memory,'peak_wset',memory.rss), entries=entries)
        rows.append(row); save(out/'rows.json',rows)
        print(len(rows),job,seconds,error,flush=True)
    assert hashes=={f:digest(ROOT/f) for f in sources}
    save(out/'summary.json',dict(completed=len(rows),passed=sum(r['passed'] for r in rows),unchanged_sources=True))


if __name__=='__main__': main()
