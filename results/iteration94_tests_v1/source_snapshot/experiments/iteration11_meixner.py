"""Degree-0/1/2 threshold diagnostics and signed-error resource budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian import haar_unitary, covariance_from_interferometer
from gbslab.exact import ThresholdModel
from gbslab.meixner_probe import threshold_projection, signed_degree_budget


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'configs/iteration11_mps_probe.json').read_text())
    plan = {'modes':[4,8,12], 'seeds':config['seeds'], 'profiles':config['profiles'],
        'degrees':[0,1,2], 'budget_modes':[8,32,64,128,1024], 'l1_errors':[.1,.01],
        'haar_failure':.05, 'scope':'Exploratory degree-two threshold projection; not the paper randomized rejection sampler',
        'source':'https://arxiv.org/html/2611.08444v1', 'equations':['G.12','G.28','G.38','G.40']}
    save(args.output/'plan.json',plan)
    hashes = {}
    for f in ['src/gbslab/meixner_probe.py','experiments/iteration11_meixner.py','tests/test_meixner_probe.py',
              'src/gbslab/gaussian.py','src/gbslab/exact.py']:
        dst=args.output/'source_snapshot'/f
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/f,dst)
        hashes[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    save(args.output/'source_hashes.json',hashes)
    rows=[]
    for m in plan['modes']:
        for seed in plan['seeds']:
            u=haar_unitary(m,np.random.default_rng(seed))
            for profile in plan['profiles']:
                r,eta=profile['squeezing'],profile['transmissivity']
                label=f"m{m}_{profile['name']}_s{seed}"
                target=ThresholdModel(covariance_from_interferometer(u,r,eta)).full_distribution()
                np.savez_compressed(args.output/(label+'_input.npz'),U=u,target=target)
                # Matched-mean thermal baseline differs from the paper reference.
                mean=eta*np.sinh(r)**2
                bits=(np.arange(1<<m)[:,None]>>np.arange(m))&1
                thermal=np.prod(np.where(bits,mean/(1+mean),1/(1+mean)),axis=1)
                for degree in plan['degrees']:
                    start=perf_counter()
                    result=threshold_projection(u,r,eta,degree)
                    row={'input':label,'modes':m,'profile':profile['name'],'seed':seed,'degree':degree,
                        'tv_repaired':float(.5*np.abs(result['distribution']-target).sum()),
                        'half_l1_signed':float(.5*np.abs(result['raw']-target).sum()),
                        'negative_mass':result['negative_mass'],'repair_l1':result['repair_l1'],
                        'matched_mean_thermal_tv':float(.5*np.abs(thermal-target).sum()),
                        'enumeration_seconds':perf_counter()-start,'status':'complete'}
                    rows.append(row)
                    np.savez_compressed(args.output/(label+f'_D{degree}.npz'),
                        raw=result['raw'],repaired=result['distribution'],thermal=thermal)
                save(args.output/'rows.json',rows)
                print(label,flush=True)
    budgets=[]
    for m in plan['budget_modes']:
        for profile in plan['profiles']:
            for error in plan['l1_errors']:
                row=signed_degree_budget(m,profile['squeezing'],profile['transmissivity'],error,plan['haar_failure'])
                row.update(modes=m,profile=profile['name'])
                budgets.append(row)
    save(args.output/'budgets.json',budgets)
    save(args.output/'completion.json',{'quality_rows':len(rows),'budget_rows':len(budgets)})


if __name__=='__main__':
    main()
