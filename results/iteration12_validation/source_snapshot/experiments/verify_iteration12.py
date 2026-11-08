"""Current-platform full regression and source-bound local-MPS test record."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import platform
import sys
import unittest
import numpy as np
from verify_iteration11 import RecordingResult

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (args.output/'tests.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    files=['src/gbslab/local_mps.py','tests/test_local_mps.py','experiments/iteration12_reference.py','experiments/verify_iteration12.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        dest=args.output/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/f).read_bytes())
    result={'date':'2026-11-08','platform':platform.platform(),'numpy':np.__version__,
        'tests_run':r.testsRun,'passed':len(r.successes),'failures':len(r.failures),'errors':len(r.errors),
        'skipped_test_cases':sum('setUpClass' not in t.id() for t,_ in r.skipped),
        'skipped_classes':sum('setUpClass' in t.id() for t,_ in r.skipped),
        'local_mps_passed':sum(s.startswith('test_local_mps.') for s in r.successes),
        'local_mps_failures':sum(t.id().startswith('test_local_mps.') for t,_ in r.failures+r.errors),
        'failure_details':[{'test':t.id(),'traceback':msg} for t,msg in r.failures],
        'error_details':[{'test':t.id(),'traceback':msg} for t,msg in r.errors],
        'skip_details':[{'test':t.id(),'reason':msg} for t,msg in r.skipped],
        'source_sha256':hashes,'full_gate_passed':r.wasSuccessful() and not r.skipped}
    (args.output/'validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))},indent=2))
    return 0 if r.wasSuccessful() else 1


if __name__=='__main__': raise SystemExit(main())
