"""Record current-platform failures/skips without reusing historical gates."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class RecordingResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.successes=[]
    def addSuccess(self,test):
        super().addSuccess(test)
        self.successes.append(test.id())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (args.output/'tests.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    names=['mps_probe','meixner_probe']
    new=lambda s:any('test_'+n+'.' in s for n in names)
    tracked=subprocess.check_output(['git','diff','--name-only','HEAD','--','src','experiments','tests','configs'],cwd=ROOT,text=True).splitlines()
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'src/gbslab/mps_probe.py',ROOT/'src/gbslab/meixner_probe.py',
                ROOT/'tests/test_mps_probe.py',ROOT/'tests/test_meixner_probe.py']}
    summary={'date':'2026-11-08','platform':platform.platform(),'python':sys.version,
        'numpy':np.__version__,'longdouble_bits':int(np.finfo(np.longdouble).bits),
        'run':result.testsRun,'passed':len(result.successes),'failures':len(result.failures),
        'errors':len(result.errors),'skipped':len(result.skipped),
        'skipped_test_cases':sum('setUpClass' not in t.id() for t,_ in result.skipped),
        'skipped_classes':sum('setUpClass' in t.id() for t,_ in result.skipped),
        'new_passed':sum(new(s) for s in result.successes),
        'new_failures':sum(new(t.id()) for t,_ in result.failures+result.errors),
        'failure_details':[{'test':t.id(),'traceback':msg} for t,msg in result.failures],
        'error_details':[{'test':t.id(),'traceback':msg} for t,msg in result.errors],
        'skip_details':[{'test':t.id(),'reason':msg} for t,msg in result.skipped],
        'tracked_source_changes':tracked,'source_sha256':hashes,
        'full_gate_passed':result.wasSuccessful() and not result.skipped,
        'scope':'New module tests plus unchanged historical tests in this Windows environment; no native rebuild.'}
    (args.output/'validation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['failure_details','error_details','skip_details','source_sha256']},indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
