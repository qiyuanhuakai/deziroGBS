"""Focused, lightweight RNG regression tests after the domain repair.
No sampler benchmark or heavy calculation. Full repository suite is separate.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,argparse,unittest,contextlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from iteration70_tests import RecordingResult
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
dll=ROOT/'results/iteration84_rng_gates_v1/rng.dll';os.environ['GBSLAB_RNG84_LIBRARY']=str(dll);os.environ['GBSLAB_NATIVE_RUNTIME_DIRS']=str(Path(shutil.which('g++')).parent);files=[Path(__file__),ROOT/'tests/test_native_rng84.py',ROOT/'native/gaussian_rng_domain84_v1.hpp',ROOT/'native/gaussian_rng_probe84_v1.cpp',dll];hashes={p.relative_to(ROOT).as_posix():H(p) for p in files};save('plan.json',dict(scope=__doc__,hashes=hashes));suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_native_rng84.py')
with (OUT/'tests.log').open('x',encoding='utf-8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):result=unittest.TextTestRunner(stream=f,verbosity=2,resultclass=RecordingResult).run(suite)
report=dict(passed=result.wasSuccessful() and result.testsRun==7 and not result.skipped,tests_run=result.testsRun,successes=len(result.successes),failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),failure_details=[dict(test=t.id(),traceback=msg) for t,msg in result.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in result.errors],unchanged_sources=all(H(ROOT/name)==digest for name,digest in hashes.items()));save('validation.json',report);save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print(report,flush=True);assert report['passed']
