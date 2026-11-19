"""Freeze and run the exact-transport and inherited native numerical suites."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,contextlib,shutil,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from iteration23_support import digest,save
from iteration30_tests import RecordingResult

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);os.environ['GBSLAB_WIRE_FOREST_LIBRARY']=str(a.library.resolve())
    names=['experiments/iteration31_native_tests.py','experiments/iteration30_tests.py','tests/test_wire_forest.py','tests/test_batched_forest_numerics.py',*[x.relative_to(ROOT).as_posix() for x in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    save(out/'plan.json',dict(source_hashes=hashes,library=str(a.library.resolve()),library_sha256=digest(a.library)))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_wire_forest.py')
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    report=dict(run=result.testsRun,passed=len(result.successes),failures=[dict(test=t.id(),traceback=s) for t,s in result.failures],errors=[dict(test=t.id(),traceback=s) for t,s in result.errors],skipped=[dict(test=t.id(),reason=s) for t,s in result.skipped],unchanged_sources=hashes=={n:digest(ROOT/n) for n in names},passed_ids=result.successes)
    save(out/'validation.json',report);print({k:v for k,v in report.items() if k!='passed_ids'});assert report['unchanged_sources']
if __name__=='__main__':main()
