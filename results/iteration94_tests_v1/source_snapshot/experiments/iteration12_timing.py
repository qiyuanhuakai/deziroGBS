"""Serial paired timing of two preselected nominal-construction-budget choices.

Representations chosen after the exploratory panel. New circuit seeds fixed
before this timing run. This does not compare complete finite-precision TV.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
from iteration12_run import run_case, save, ROOT


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plan={'modes':128,'depth':4,'profile':{'name':'lossy','r':.6,'eta':.35},
        'circuit_seeds':[120901,120902],'repeats':3,'samples':128,'nominal_construction_budget':.1,
        'methods':[{'builder':'gatewise','d':8,'chi':16},{'builder':'layerwise','d':6,'chi':8}],
        'ordering_seed':120999,'scope':'Serial full compute pipeline. Two new circuits; repetitions are timing replicates, not independent quality instances. No global finite-precision sampler certificate.'}
    jobs=[dict(method,seed=seed,repeat=rep) for seed in plan['circuit_seeds']
        for rep in range(plan['repeats']) for method in plan['methods']]
    random.Random(plan['ordering_seed']).shuffle(jobs)
    save(args.output/'plan.json',plan); save(args.output/'jobs.json',jobs)
    files=['src/gbslab/local_mps.py','src/gbslab/mps_probe.py','experiments/iteration12_run.py',
        'experiments/iteration12_timing.py','experiments/iteration10_reference.py']
    hashes={}
    for f in files:
        dst=args.output/'source_snapshot'/f; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/f,dst); hashes[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    save(args.output/'source_hashes.json',hashes)
    rows=[]
    for i,job in enumerate(jobs):
        folder=args.output/f'run_{i:02d}'; folder.mkdir()
        cfg={'samples':plan['samples'],'builder':job['builder']}
        row=run_case(cfg,folder,plan['modes'],job['seed'],plan['profile'],plan['depth'],job['d'],job['chi'],False)
        row.update(repeat=job['repeat'],sequence=i,
            nominal_budget_met=row.get('construction_trace_budget',1)<=plan['nominal_construction_budget'])
        rows.append(row); save(args.output/'rows.json',rows)
    save(args.output/'completion.json',{'planned':len(jobs),'finished':len(rows),
        'complete':sum(r['status']=='complete' for r in rows),
        'nominal_budget_met':sum(r['nominal_budget_met'] for r in rows)})


if __name__=='__main__': main()
