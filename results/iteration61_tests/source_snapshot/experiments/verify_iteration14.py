"""Validate complete panels, sample records, source hashes and full regressions."""
import argparse
from collections import defaultdict
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
import numpy as np
from iteration14_run import save
from verify_iteration11 import RecordingResult

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); checks=[]
    def check(label,result): checks.append({'check':label,'passed':bool(result)})
    panel=ROOT/'results/iteration14_cost'; rows=json.loads((panel/'rows.json').read_text()); plan=json.loads((panel/'plan.json').read_text())
    check('planned cost cases all present',len(rows)==plan['job_count']==180)
    hashes=json.loads((panel/'metadata.json').read_text())['source_sha256']
    for path,digest in hashes.items(): check('cost source '+path,hashlib.sha256((panel/'source_snapshot'/path).read_bytes()).hexdigest()==digest)
    groups=defaultdict(list)
    for i,(r,job) in enumerate(zip(rows,plan['jobs'])):
        check(f'cost {i} planned identity',all(r[k]==v for k,v in job.items()))
        check(f'cost {i} complete',r['status']=='complete' and r.get('completed_samples')==r['samples'])
        if r['status']!='complete': continue
        check(f'cost {i} memory ordering',r['peak_wset_bytes']>=r['baseline_peak_wset_bytes']>=r['baseline_rss_bytes'])
        check(f'cost {i} method storage',r['dense_unitary_bytes']==(0 if r['method']=='local_stream' else r['modes']**2*16))
        check(f'cost {i} batch cap',r['max_batch_rows']<=(r['samples'] if r['method']=='dense_full' else r['batch_size']))
        if r['panel']=='pipeline':
            with np.load(panel/f'run_{i:03d}.npz') as data: samples=data['samples']
            check(f'cost {i} samples',samples.shape==(r['samples'],r['modes']) and np.all((samples==0)|(samples==1)))
            check(f'cost {i} sample hash',hashlib.sha256(samples.tobytes()).hexdigest()==r['sample_sha256'])
            check(f'cost {i} clicks',int(samples.sum())==r['total_clicks'])
        groups[tuple(r[k] for k in ('panel','modes','depth','samples','seed'))].append(r)
    for key,group in groups.items():
        check(str(key)+' all methods/repeats',len(group)==9 and len({(r['method'],r['repeat']) for r in group})==9)
        norms=[r['alpha_abs2_sum'] for r in group]
        check(str(key)+' alpha norm summaries',max(norms)-min(norms)<=1e-12*max(norms))
        if key[0]=='pipeline':
            check(str(key)+' identical observed outputs',len({r['sample_sha256'] for r in group})==1)
            check(str(key)+' identical construction budgets',len({r['construction_trace_budget'] for r in group})==1)
    quality=ROOT/'results/iteration14_quality'; qrows=json.loads((quality/'rows.json').read_text())
    check('quality cases all present',len(qrows)==20)
    for path,digest in json.loads((quality/'source_hashes.json').read_text()).items():
        check('quality source '+path,hashlib.sha256((quality/'source_snapshot'/path).read_bytes()).hexdigest()==digest)
    for r in qrows:
        i=r['index']; check(f'quality {i} complete',r['status']=='complete')
        if r['status']!='complete': continue
        check(f'quality {i} exact gates',r['gates_exactly_equal'] and r['unitary_not_materialized'])
        with np.load(quality/f'input_{i:02d}.npz') as data:
            error=float(np.max(np.abs(data['alpha_dense']-data['alpha_local'])))
        check(f'quality {i} recorded error',error==r['alpha_max_abs_error'])
        check(f'quality {i} numerical agreement',error<1e-13 and r['alpha_relative_l2_error']<1e-13 and r['norm_preservation_relative_error']<1e-13)
        if r['modes']<=4:
            check(f'quality {i} moments',r['normal_covariance_max_error']<1e-13 and r['anomalous_covariance_max_error']<1e-13)
            check(f'quality {i} conditional law',float(r['conditional_ideal_law_tv'])<1e-13)
    save(out/'artifact_checks.json',{'count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks})
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    result={'tests_run':r.testsRun,'passed':len(r.successes),'failures':len(r.failures),'errors':len(r.errors),
        'skipped_cases':sum('setUpClass' not in t.id() for t,_ in r.skipped),'skipped_classes':sum('setUpClass' in t.id() for t,_ in r.skipped),
        'local_displacement_passed':sum(s.startswith('test_local_displacement.') for s in r.successes),
        'failure_details':[{'test':t.id(),'traceback':msg} for t,msg in r.failures],
        'error_details':[{'test':t.id(),'traceback':msg} for t,msg in r.errors],
        'skip_details':[{'test':t.id(),'reason':msg} for t,msg in r.skipped],
        'full_gate_passed':r.wasSuccessful() and not r.skipped}
    old=json.loads((ROOT/'results/iteration13_validation/validation.json').read_text())
    result['unchanged_failure_ids']=set(x['test'] for k in ('failure_details','error_details') for x in old[k])==set(x['test'] for k in ('failure_details','error_details') for x in result[k])
    save(out/'validation.json',result)
    for file in ('experiments/verify_iteration14.py','tests/test_local_displacement.py','src/gbslab/local_displacement.py','src/gbslab/local_mps.py'):
        dest=out/'source_snapshot'/file; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/file,dest)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))
    print('Artifact checks:',sum(c['passed'] for c in checks),'/',len(checks))
    return 0 if r.wasSuccessful() and all(c['passed'] for c in checks) else 1


if __name__=='__main__': raise SystemExit(main())
