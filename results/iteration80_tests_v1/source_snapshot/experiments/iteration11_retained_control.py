"""Post-exploration stronger quality control on the saved Meixner input panel.

Python click-influence retained conditionals, including explicit failure mass.
This is not a native-speed benchmark and not an independent confirmation panel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian import covariance_from_interferometer
from gbslab.exact import ThresholdModel
from gbslab.conditional_memory import ConditionalMemorySampler
from iteration6_completion import extended_law


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((args.input/'plan.json').read_text())
    (args.output/'plan.json').write_text(json.dumps({'input':str(args.input),'k':[2,4],
        'selector':'click_influence','status':'post-exploration stronger control',
        'failure_handling':'extended law on bitmasks plus failure symbol'},indent=2))
    hashes={}
    for f in ['experiments/iteration11_retained_control.py','experiments/iteration6_completion.py',
              'src/gbslab/conditional_memory.py','src/gbslab/zero_memory.py','src/gbslab/schur_memory.py','src/gbslab/exact.py']:
        dst=args.output/'source_snapshot'/f
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/f,dst)
        hashes[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    (args.output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    rows=[]
    for m in plan['modes']:
        for seed in plan['seeds']:
            for p in plan['profiles']:
                label=f"m{m}_{p['name']}_s{seed}"
                data=np.load(args.input/(label+'_input.npz'))
                v=covariance_from_interferometer(data['U'],p['squeezing'],p['transmissivity'])
                for k in (2,4):
                    row={'input':label,'modes':m,'profile':p['name'],'seed':seed,'k':k}
                    try:
                        model=ThresholdModel(v,max_exact_modes=14,vacuum_cache_size=8192)
                        sampler=ConditionalMemorySampler(model,k,selection='click_influence')
                        q,failure,records,elapsed=extended_law(sampler,m)
                        row.update(status='complete',failure_mass=failure,failure_records=records,
                            tv_extended=float(.5*(np.abs(q-data['target']).sum()+failure)),enumeration_seconds=elapsed)
                        np.savez_compressed(args.output/(label+f'_k{k}.npz'),distribution=q,target=data['target'])
                    except Exception as e:
                        row.update(status='failed',error=repr(e))
                    rows.append(row)
                    (args.output/'rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
                    print(label,k,row['status'],flush=True)


if __name__=='__main__':
    main()
