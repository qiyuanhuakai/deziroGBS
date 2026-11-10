"""All-site external coefficient comparison, no upstream code execution."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import zipfile
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gbslab.gaussian_pair_recurrence import pair_recurrence_tensor
from gbslab.gaussian_schmidt_mps import projected_pair_tensor

ARCHIVE_SHA = 'bd9422c2a194f8f0109a7463291089288d2b0156db771386633824fb6bb8f8a9'
PREFIX = 'PCO mapping/data/J2_P65_5_L144_thermal10/'
SOURCES = ['experiments/iteration22_external.py', 'src/gbslab/gaussian_pair_recurrence.py', 'src/gbslab/gaussian_schmidt_mps.py']


def digest(path):
    with open(path, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load_entry(z, name, entries):
    raw = z.read(name)
    a = np.load(io.BytesIO(raw), allow_pickle=False)
    entries.append(dict(name=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), shape=list(a.shape), dtype=str(a.dtype)))
    if not np.all(np.isfinite(a)): raise ValueError('Nonfinite external input')
    return a


def load_site(z, site):
    entries = []
    Q = load_entry(z, PREFIX + f'Q/{site}.npy', entries).astype(complex)
    Q[np.diag_indices_from(Q)] *= 2
    left = load_entry(z, PREFIX + f'List_vst/{site-1}.npy', entries)[:200] if site else np.zeros((1, 0), int)
    right = load_entry(z, PREFIX + f'List_vst/{site}.npy', entries)[:200] if site < 143 else np.zeros((1, 0), int)
    reference = load_entry(z, PREFIX + f'ts/D200/{site}.npy', entries)
    return Q, left, right, reference, entries


def worker(archive, output, site):
    started = perf_counter(); row = dict(site=site)
    try:
        with zipfile.ZipFile(archive) as z: Q, left, right, reference, entries = load_site(z, site)
        t = perf_counter(); recurrence, info = pair_recurrence_tensor(Q, left, 4, right)
        row['recurrence_seconds'] = perf_counter() - t
        t = perf_counter(); pco, pco_info = projected_pair_tensor(Q, left, 4, right)
        row['pco_seconds'] = perf_counter() - t
        row['pco_info'] = pco_info
        if reference.shape != pco.shape: raise ValueError('Reference shape mismatch')
        errors = {}
        for name, a, b in [('recurrence_reference', recurrence, reference), ('pco_reference', pco, reference), ('recurrence_pco', recurrence, pco)]:
            errors[name] = dict(max_abs=float(np.max(np.abs(a-b))), frobenius=float(np.linalg.norm(a-b)), relative_frobenius=float(np.linalg.norm(a-b)/np.linalg.norm(b)))
        target = Path(output) / f'site_{site:03d}.npz'
        np.savez_compressed(target, recurrence=recurrence, pco=pco)
        row.update(status='passed' if all(v['max_abs'] <= 1e-10 for v in errors.values()) else 'mismatch', errors=errors, entries=entries, shape=list(pco.shape), recurrence_info=info, output_sha256=digest(target))
    except Exception:
        row.update(status='error', traceback=traceback.format_exc())
    memory = psutil.Process().memory_info()
    row.update(seconds=perf_counter()-started, peak_working_set_bytes=getattr(memory, 'peak_wset', memory.rss))
    save(Path(output) / f'site_{site:03d}.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--archive', type=Path, required=True); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    hashes = {p: digest(ROOT / p) for p in SOURCES}
    source_dir = args.output / 'sources'; source_dir.mkdir()
    for p in SOURCES: shutil.copy2(ROOT / p, source_dir / Path(p).name)
    shutil.copy2(ROOT / 'research/iteration22_plan.md', args.output / 'frozen_plan.md')
    plan = dict(sites=list(range(144)), dimension=4, rank_cap=200, workers=args.workers, local_max_abs_tolerance=1e-10, archive_sha256=ARCHIVE_SHA, source_hashes=hashes, schedule='all sites once; concurrent quality timings are not isolated performance measurements', global_pairs=['own_pco:own_recurrence', 'own_recurrence:author_pco', 'own_recurrence:author_hafnian', 'author_pco:author_hafnian'], isolated_benchmark=dict(sites=[0, 35, 71, 107, 143], repetitions=3, seed=220902))
    save(args.output / 'plan.json', plan)
    if digest(args.archive) != ARCHIVE_SHA: raise ValueError('Unexpected archive checksum')
    rows = []; started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, str(args.archive), str(args.output), site) for site in range(144)]
        for future in as_completed(futures):
            row = future.result(); rows.append(row); rows.sort(key=lambda x: x['site'])
            save(args.output / 'rows.json', rows)
            print(len(rows), row['site'], row['status'], round(row['seconds'], 2), flush=True)
    if hashes != {p: digest(ROOT / p) for p in SOURCES}: raise RuntimeError('Running sources changed')
    save(args.output / 'summary.json', dict(completed=len(rows), passed=sum(r['status']=='passed' for r in rows), seconds=perf_counter()-started, unchanged_sources=True))


if __name__ == '__main__': main()
