"""Preserve full-suite results including inherited platform failures/skips."""
import contextlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import unittest
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from iteration22_external import digest,save


class RecordingResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs); self.successes=[]
    def addSuccess(self,test): self.successes.append(test.id()); super().addSuccess(test)


def main():
    os.environ["GBSLAB_WIRE_FOREST_LIBRARY"]=str(ROOT/"results/iteration31_build/libwire_forest.dll")
    out=ROOT/'results/iteration54_tests'; out.mkdir(exist_ok=False)
    sources=[*sorted((ROOT/'experiments').glob('*.py')),*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'tests').rglob('*.py'))]
    hashes={f.relative_to(ROOT).as_posix():digest(f) for f in sources}
    save(out/'source_hashes_at_start.json',hashes)
    for f in sources:
        target=out/'source_snapshot'/f.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,target)
    save(out/'environment.json',dict(python=sys.version,platform=platform.platform(),processor=platform.processor()))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    report=dict(tests_run=result.testsRun,passed=len(result.successes),failures=len(result.failures),errors=len(result.errors),
        skipped_cases=sum('setUpClass' not in t.id() for t,_ in result.skipped),skipped_classes=sum('setUpClass' in t.id() for t,_ in result.skipped),
        new_tests_passed=sum(s.startswith(('test_rank_one_noise.',)) for s in result.successes),
        failure_details=[dict(test=t.id(),traceback=msg) for t,msg in result.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in result.errors],
        skip_details=[dict(test=t.id(),reason=msg) for t,msg in result.skipped],full_gate_passed=result.wasSuccessful() and not result.skipped)
    old=json.loads((ROOT/'results/iteration25_tests/validation.json').read_text(encoding='utf-8'))
    report['resolved_failure_ids']=[x['test'] for x in old['failure_details'] if x['test'] in result.successes]
    report['previously_unimportable_module_tests_passed']=sum(s.startswith('test_iteration5_quality.') for s in result.successes)
    report['executed_gate_passed']=result.wasSuccessful()
    save(out/'passed_test_ids.json',result.successes)
    report['unchanged_sources']=hashes=={f.relative_to(ROOT).as_posix():digest(f) for f in sources}
    save(out/'validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)},indent=2))
    assert report['unchanged_sources']


if __name__=='__main__': main()
