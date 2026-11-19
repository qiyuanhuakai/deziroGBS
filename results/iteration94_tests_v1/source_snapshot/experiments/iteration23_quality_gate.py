"""Pre-performance paired-state gate on every held-out small system."""
import json
import os
from pathlib import Path
import shutil
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.mps_overlap import compare_mps
from iteration23_support import digest,load_mps,save


def main():
    out=ROOT/'results/iteration23_quality_gate'; out.mkdir(exist_ok=False)
    sources=['experiments/iteration23_quality_gate.py','src/gbslab/mps_overlap.py','experiments/iteration23_support.py']
    hashes={f:digest(ROOT/f) for f in sources}; (out/'sources').mkdir()
    for f in sources: shutil.copy2(ROOT/f,out/'sources'/Path(f).name)
    save(out/'source_hashes.json',hashes)
    panel=ROOT/'results/iteration23_quality'; rows=json.loads((panel/'rows.json').read_text()); assert len(rows)==48
    assert all(r['status'] in ('accepted','budget_missed') and r['validation_status']=='passed' and r['actual_bound_check'] for r in rows)
    grouped={}
    for row in rows:
        key=(row['modes'],row['ensemble'],row['dimension'],row['rank']); grouped.setdefault(key,{})[row['backend']]=row
    comparisons=[]
    for key,group in sorted(grouped.items()):
        a,b=group['pco'],group['recurrence']; left=load_mps(ROOT/a['state']['path']); right=load_mps(ROOT/b['state']['path'])
        result=compare_mps(left,right)
        with np.load(panel/f"run_{a['sequence']:02d}.samples.npz",allow_pickle=False) as pa,np.load(panel/f"run_{b['sequence']:02d}.samples.npz",allow_pickle=False) as pb:
            samples_equal=np.array_equal(pa['samples'],pb['samples']); displacement_equal=np.array_equal(pa['alpha'],pb['alpha'])
        passed=result['pure_trace_distance_from_difference']<1e-10 and abs(a['info']['projection_success']-b['info']['projection_success'])<1e-12 and a['construction_trace_budget']==b['construction_trace_budget'] and samples_equal and displacement_equal
        comparisons.append(dict(case=list(key),comparison=result,passed=passed,samples_equal=samples_equal,displacement_equal=displacement_equal))
    assert hashes=={f:digest(ROOT/f) for f in sources}
    summary=dict(pairs=len(comparisons),passed=sum(r['passed'] for r in comparisons),max_distance=max(r['comparison']['pure_trace_distance_from_difference'] for r in comparisons),comparisons=comparisons)
    save(out/'validation.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'},indent=2))
    assert len(comparisons)==24 and all(r['passed'] for r in comparisons)


if __name__=='__main__': main()
