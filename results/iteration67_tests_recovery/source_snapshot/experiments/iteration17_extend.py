"""Post-panel exploratory degree-eight extension on EVERY m8/m12 saved input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import numpy as np
from iteration14_run import save

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.meixner_high_order import projection_prefixes


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); panel=ROOT/'results/iteration17_panel'
    jobs=[r for r in json.loads((panel/'inputs.json').read_text()) if r['modes']>=8]
    save(out/'plan.json',{'input_indices':[r['index'] for r in jobs],'max_degree':8,'max_coefficients':200000,'max_wick_states':4000000,
        'scope':'Post-panel exploratory extension after degree-six results. All 24 m8/m12 inputs included, not a new confirmation set.'})
    rows=[]; inputs=[]
    for job in jobs:
        i=job['index']; mark=perf_counter(); item=dict(index=i,modes=job['modes'])
        try:
            with np.load(panel/f'input_{i:02d}.npz') as data: u=data['U']; target=data['target']
            result=projection_prefixes(u,job['r'],job['eta'],8,max_coefficients=200000,max_states=4000000)
            np.savez_compressed(out/f'input_{i:02d}_coefficients.npz',indices=result['indices'],coefficients=result['coefficients'])
            for d,(metric,law) in enumerate(zip(result['rows'],result['laws'])):
                row=dict(index=i,modes=job['modes'],ensemble=job['ensemble'],seed=job['seed'],profile=job['profile'],**metric,
                    status='complete',tv_repaired=float(.5*np.abs(law['distribution']-target).sum()),half_l1_signed=float(.5*np.abs(law['raw']-target).sum()))
                if d<=6:
                    with np.load(panel/f'input_{i:02d}_D{d}.npz') as previous:
                        row['prefix_regression_error']=float(np.max(np.abs(law['raw']-previous['raw'])))
                rows.append(row)
                np.savez_compressed(out/f'input_{i:02d}_D{d}.npz',raw=law['raw'],repaired=law['distribution'],subset_coefficients=law['subset_coefficients'])
            item.update(status='complete',wick_states=result['wick_states'],seconds=perf_counter()-mark)
        except Exception as e:
            item.update(status='failed',error=repr(e),seconds=perf_counter()-mark)
            rows.extend(dict(index=i,degree=d,status='failed',error=repr(e)) for d in range(9))
        inputs.append(item); save(out/'inputs.json',inputs); save(out/'rows.json',rows)
        print(i,item,flush=True)
    files=['experiments/iteration17_extend.py','src/gbslab/meixner_high_order.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'source_hashes.json',{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})


if __name__=='__main__': main()
