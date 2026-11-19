"""Full finite-MPS comparisons, separate from Gaussian-target error bounds."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.mps_overlap import compare_mps
from iteration23_support import digest,save,load_mps
from iteration22_external import ARCHIVE_SHA
from iteration22_global import load_family


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    sources=sorted({'experiments/iteration25_compare.py','experiments/iteration23_support.py','experiments/iteration22_external.py','experiments/iteration22_global.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))})
    hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    panel=ROOT/'results/iteration25_panel_retry'; runs=json.loads((panel/'rows.json').read_text(encoding='utf-8'))
    assert json.loads((panel/'summary.json').read_text(encoding='utf-8'))['finished']==20
    candidates=[r for r in runs if r['panel']=='candidate']; assert len(candidates)==4 and all(r['validation_status']=='passed' for r in candidates)
    families=['own_recurrence','author_pco','author_hafnian']; dependencies={r['state']['path']:r['state']['sha256'] for r in candidates}
    for f,s in dependencies.items(): assert digest(ROOT/f)==s
    assert digest(args.archive)==ARCHIVE_SHA
    save(out/'plan.json',dict(source_hashes=hashes,dependencies=dependencies,archive_sha256=ARCHIVE_SHA,candidate_ranks=[r['rank'] for r in candidates],families=families,reference_pairs=12,rank_pairs=6,scope='Full normalized finite-state distances only; not their Gaussian-target errors or experimental output distances. No speed comparison.'))
    states={r['rank']:load_mps(ROOT/r['state']['path']) for r in candidates}; rows=[]
    def compare(label,a,b):
        start=perf_counter(); row=dict(pair=label)
        try:
            row.update(compare_mps(a,b)); row['status']='passed'
            assert abs((1-row['pure_trace_distance_from_difference']**2)-row['fidelity_from_overlap'])<1e-9
            assert max(abs(row['log_norm_qr_residual_left']),abs(row['log_norm_qr_residual_right']))<1e-9
        except Exception: row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-start; rows.append(row); save(out/'rows.json',rows)
        print(label,row['status'],row.get('pure_trace_distance_from_difference'),flush=True)
    for family in families:
        reference,entries=load_family(family,ROOT/'results/iteration22_external',args.archive); save(out/(family+'_inputs.json'),entries)
        if family=='own_recurrence':
            old=json.loads((ROOT/'results/iteration22_manifest.json').read_text(encoding='utf-8'))['artifacts']
            for entry in entries:
                name='results/iteration22_external/'+entry['name']; assert old[name]==entry['sha256']; dependencies[name]=entry['sha256']
        for rank,state in states.items(): compare(f'new_{rank}:{family}',state,reference)
        del reference
    for a,b in itertools.combinations(states,2): compare(f'new_{a}:new_{b}',states[a],states[b])
    assert hashes=={f:digest(ROOT/f) for f in sources}
    save(out/'summary.json',dict(planned=18,finished=len(rows),passed=sum(r['status']=='passed' for r in rows),unchanged_sources=True,dependencies=dependencies))
    assert len(rows)==18 and all(r['status']=='passed' for r in rows)


if __name__=='__main__': main()
