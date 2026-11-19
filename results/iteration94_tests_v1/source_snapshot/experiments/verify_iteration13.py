"""Regression, exact artifact invariants, input hashes and reference recomputation."""
import argparse
import contextlib
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
import mpmath as mp
import numpy as np
from verify_iteration11 import RecordingResult
from iteration13_audit import compare, save

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.sampling_audit import enumerate_floating_law


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    panel=ROOT/'results/iteration13_conditional_audit'
    rows=json.loads((panel/'rows.json').read_text()); checks=[]
    def check(label,condition):
        checks.append({'check':label,'passed':bool(condition)})
    metadata=json.loads((panel/'metadata.json').read_text())
    for file,digest in metadata['input_sha256'].items():
        check('input hash '+file,hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest)
    for file,digest in metadata['source_sha256'].items():
        check('frozen source hash '+file,hashlib.sha256((panel/'source_snapshot'/file).read_bytes()).hexdigest()==digest)
    for row in rows:
        i=row['index']; check(f'{i} complete',row['status']=='complete')
        if row['status']!='complete': continue
        law=json.loads((panel/f'law_{i:03d}.json').read_text())
        den=int(law['grid_denominator']); nums=list(map(int,law['grid_numerators']))
        real=list(map(Fraction,law['real_cdf_law'])); fail=Fraction(law['real_failure'])
        check(f'{i} exact continuous mass',sum(real)+fail==1)
        check(f'{i} exact grid mass',sum(nums)+int(law['grid_failure_numerator'])==den)
        check(f'{i} nonnegative',min(nums)>=0 and min(real)>=0 and fail>=0)
        with np.load(panel/f'input_{i:03d}.npz') as data:
            tensors=[data[f'A{j}'] for j in range(len(data['alpha']))]
            rerun=enumerate_floating_law(tensors,data['alpha'])
        check(f'{i} rerun rational law',real==rerun['real_cdf_law'] and nums==rerun['grid_numerators'])
        check(f'{i} rerun failure',fail==rerun['real_failure'] and int(law['grid_failure_numerator'])==rerun['grid_failure_numerator'])
        check(f'{i} quantization coupling bound',Fraction(law['quantization_tv'])<=sum(Fraction(a.shape[1],2**53) for a in tensors))
        metrics=compare(rerun,law['reference80'],law['reference120'])
        check(f'{i} exact metric regeneration',all(row[k]==v for k,v in metrics.items()))
        with mp.workdps(140):
            ref=list(map(mp.mpf,law['reference120']['law_decimal']))
            check(f'{i} reference mass',abs(sum(ref)-1)<mp.mpf('1e-115'))
            check(f'{i} reference sign',min(ref)>=-mp.mpf('1e-115'))
            check(f'{i} reference convergence',mp.mpf(metrics['reference_convergence_max'])<mp.mpf('1e-75'))
    save(out/'artifact_checks.json',{'count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks})
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    details={'tests_run':result.testsRun,'passed':len(result.successes),'failures':len(result.failures),'errors':len(result.errors),
        'skipped_cases':sum('setUpClass' not in t.id() for t,_ in result.skipped),
        'skipped_classes':sum('setUpClass' in t.id() for t,_ in result.skipped),
        'audit_tests_passed':sum(s.startswith('test_sampling_audit.') for s in result.successes),
        'local_mps_passed':sum(s.startswith('test_local_mps.') for s in result.successes),
        'failure_details':[{'test':t.id(),'traceback':msg} for t,msg in result.failures],
        'error_details':[{'test':t.id(),'traceback':msg} for t,msg in result.errors],
        'skip_details':[{'test':t.id(),'reason':msg} for t,msg in result.skipped],
        'full_gate_passed':result.wasSuccessful() and not result.skipped}
    save(out/'validation.json',details)
    for f in ('experiments/verify_iteration13.py','tests/test_sampling_audit.py','src/gbslab/sampling_audit.py','src/gbslab/local_mps.py'):
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    print(json.dumps({k:v for k,v in details.items() if not isinstance(v,list)},indent=2))
    print('Artifact checks:',sum(c['passed'] for c in checks),'/',len(checks))
    return 0 if result.wasSuccessful() and all(c['passed'] for c in checks) else 1


if __name__=='__main__': raise SystemExit(main())
