"""Check ledgers, saved MPS/reference overlaps, failed searches and full regressions."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import unittest
import numpy as np
from iteration14_run import save
from verify_iteration11 import RecordingResult

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.mps_probe import mps_to_dense


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); checks=[]
    def check(label,value): checks.append({'check':label,'passed':bool(value)})
    panel=ROOT/'results/iteration15_pilot'
    plan=json.loads((panel/'plan.json').read_text()); rows=json.loads((panel/'rows.json').read_text())
    check('all predeclared cases recorded',len(rows)==plan['case_count']==68)
    for file,digest in json.loads((panel/'metadata.json').read_text())['source_sha256'].items():
        check('source '+file,hashlib.sha256((panel/'source_snapshot'/file).read_bytes()).hexdigest()==digest)
    for i,(row,job) in enumerate(zip(rows,plan['jobs'])):
        check(f'{i} planned identity',all(row[k]==v for k,v in job.items()))
        path=panel/f'run_{i:03d}.attempts.json'
        trials=json.loads(path.read_text()) if path.exists() else []
        check(f'{i} explicit terminal status',row['status'] in ('accepted','budget_missed','search_unaccepted','failed','sampling_failed','worker_failed'))
        if row['status']=='accepted':
            check(f'{i} accepted full budget',row['target_met'] and row['construction_trace_budget']<=row['target'])
        if row['status']=='search_unaccepted':
            check(f'{i} unsuccessful trials retained',bool(trials) and all(t['status']!='accepted' for t in trials))
        if row['method']=='adaptive':
            for j,t in enumerate(trials):
                label=f'{i} trial {j}'
                if t['status']=='input_budget_rejected':
                    check(label+' input excludes additive target',t['input_trace_budget']>row['target']); continue
                projection=0.; rounding=0.
                for k,layer in enumerate(t['layers']):
                    joint=-math.expm1(sum(math.log1p(-p) for p in layer['conditional_losses']))
                    check(label+f' layer {k} joint loss',abs(joint-layer['joint_tail'])<1e-15)
                    check(label+f' layer {k} dimensions',len(layer['ranks'])==row['modes']-1 and all(1<=r<=t['cap'] for r in layer['ranks']))
                    if not layer['rank_cap_misses']:
                        check(label+f' layer {k} allowance',joint<=layer['joint_allowance']+1e-14)
                    projection+=math.sqrt(layer['projection_tail']); rounding+=math.sqrt(joint)
                check(label+' sums',abs(projection-t['projection_trace_sum'])<1e-14 and abs(rounding-t['rounding_trace_sum'])<1e-14)
                if t['status'] in ('accepted','budget_missed'):
                    total=t['input_trace_budget']+projection+rounding
                    check(label+' total bound',abs(total-t['unclipped_construction_trace_budget'])<1e-14)
                    check(label+' acceptance', (total<=row['target'])==(t['status']=='accepted'))
                else: check(label+' incomplete has no final bound','construction_trace_budget' not in t)
        state_path=panel/f'run_{i:03d}.npz'
        if state_path.exists():
            with np.load(state_path) as data: tensors=[data[f'A{k}'] for k in range(row['modes'])]
            check(f'{i} saved coefficient count',sum(a.size for a in tensors)==row['stored_complex_elements'])
            check(f'{i} actual bond count',max(a.shape[2] for a in tensors)==row['max_bond_actual'])
            check(f'{i} normalized center',abs(np.linalg.norm(tensors[0])-1)<1e-10)
            check(f'{i} right canonical',all(np.allclose(a.reshape(a.shape[0],-1)@a.reshape(a.shape[0],-1).conj().T,np.eye(a.shape[0]),atol=1e-10,rtol=0) for a in tensors[1:]))
            if row['panel']=='quality':
                with np.load(panel/f'run_{i:03d}.reference.npz') as ref:
                    actual=mps_to_dense(tensors)
                    check(f'{i} saved reference tensor',np.max(abs(actual-ref['mps_tensor']))<1e-14)
                    distance=math.sqrt(max(0,1-abs(np.vdot(ref['gaussian_box'].ravel(),actual.ravel()))**2))
                check(f'{i} actual state distance',abs(distance-row['state_trace_distance'])<1e-12)
                check(f'{i} full Gaussian bound',distance<=row['construction_trace_budget']+1e-10)
        sample_path=panel/f'run_{i:03d}.samples.npz'
        if sample_path.exists():
            with np.load(sample_path) as data: samples=data['samples']
            check(f'{i} sample count',len(samples)==row['completed_samples'])
            check(f'{i} binary samples',samples.shape[1]==row['modes'] and np.all((samples==0)|(samples==1)))
    save(out/'artifact_checks.json',{'count':len(checks),'passed':sum(x['passed'] for x in checks),'checks':checks})
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    result={'tests_run':r.testsRun,'passed':len(r.successes),'failures':len(r.failures),'errors':len(r.errors),
        'skipped_cases':sum('setUpClass' not in t.id() for t,_ in r.skipped),'skipped_classes':sum('setUpClass' in t.id() for t,_ in r.skipped),
        'adaptive_tests_passed':sum(s.startswith('test_adaptive_mps.') for s in r.successes),
        'failure_details':[{'test':t.id(),'traceback':msg} for t,msg in r.failures],
        'error_details':[{'test':t.id(),'traceback':msg} for t,msg in r.errors],
        'skip_details':[{'test':t.id(),'reason':msg} for t,msg in r.skipped],
        'full_gate_passed':r.wasSuccessful() and not r.skipped}
    old=json.loads((ROOT/'results/iteration14_validation/validation.json').read_text())
    result['unchanged_failure_ids']=set(x['test'] for k in ('failure_details','error_details') for x in old[k])==set(x['test'] for k in ('failure_details','error_details') for x in result[k])
    save(out/'validation.json',result)
    for file in ('experiments/verify_iteration15.py','tests/test_adaptive_mps.py','src/gbslab/adaptive_mps.py'):
        dest=out/'source_snapshot'/file; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/file,dest)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))
    print('Artifact checks:',sum(x['passed'] for x in checks),'/',len(checks))
    return 0 if r.wasSuccessful() and all(x['passed'] for x in checks) else 1


if __name__=='__main__': main()
