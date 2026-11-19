"""Gauge-independent comparison of four complete 144-site MPS families."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import zipfile
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gbslab.mps_overlap import compare_mps
from iteration22_external import ARCHIVE_SHA, PREFIX, digest, load_entry, save


def load_family(family, panel, archive):
    result = []; entries = []
    if family.startswith('own_'):
        field = family.removeprefix('own_')
        for i in range(144):
            path = panel / f'site_{i:03d}.npz'
            with np.load(path, allow_pickle=False) as data: result.append(data[field])
            entries.append(dict(name=path.name, sha256=digest(path)))
        return result, entries
    with zipfile.ZipFile(archive) as z:
        for i in range(144):
            if family == 'author_pco':
                a = load_entry(z, PREFIX + f'ts/D200/{i}.npy', entries).astype(complex)
            elif family == 'author_hafnian':
                base = 'PCO mapping/data/d_4_chi_200/'
                a = load_entry(z, base + f'Gamma_{i}.npy', entries).astype(complex).transpose(0, 2, 1)
                if i < 143:
                    lam = load_entry(z, base + f'Lambda_{i}.npy', entries).astype(float)
                    a = a * lam[None, None, :]
                if i == 0: a = a[:1, :, :]
                if i == 143: a = a[:, :, :1]
            else: raise ValueError(family)
            result.append(a)
    return result, entries


def main():
    p = argparse.ArgumentParser(); p.add_argument('--archive', type=Path, required=True); p.add_argument('--panel', type=Path, required=True); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    sources = ['experiments/iteration22_global.py', 'experiments/iteration22_external.py', 'src/gbslab/mps_overlap.py']
    hashes = {f: digest(ROOT/f) for f in sources}
    (out/'sources').mkdir()
    for f in sources: shutil.copy2(ROOT/f, out/'sources'/Path(f).name)
    save(out/'source_hashes.json', hashes)
    assert digest(args.archive) == ARCHIVE_SHA
    panel_rows = json.loads((args.panel/'rows.json').read_text())
    assert len(panel_rows)==144 and {r['site'] for r in panel_rows}==set(range(144))
    assert all(r['status']=='passed' and digest(args.panel/f"site_{r['site']:03d}.npz")==r['output_sha256'] for r in panel_rows)
    plan = json.loads((args.panel/'plan.json').read_text()); rows = []
    for pair in plan['global_pairs']:
        started = perf_counter(); row = dict(pair=pair)
        try:
            left, right = pair.split(':')
            a, ea = load_family(left, args.panel, args.archive)
            b, eb = load_family(right, args.panel, args.archive)
            save(out/(left+'_input_hashes.json'), ea); save(out/(right+'_input_hashes.json'), eb)
            row.update(compare_mps(a, b)); row['status']='complete'
            del a, b
        except Exception: row.update(status='error', traceback=traceback.format_exc())
        row['seconds']=perf_counter()-started; rows.append(row)
        save(out/'rows.json', rows); print(pair, row.get('pure_trace_distance_from_difference'), row['status'], row['seconds'], flush=True)
    assert hashes == {f: digest(ROOT/f) for f in sources}
    save(out/'summary.json', dict(pairs=len(rows), completed=sum(r['status']=='complete' for r in rows), unchanged_sources=True))


if __name__ == '__main__': main()
