"""Freeze and run new native correctness gates before quality sampling."""
import argparse,contextlib,sys,unittest
from iteration92_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration92_build_v1/core.dll';os.environ['GBSLAB_COHERENT92_LIBRARY']=str(dll)
files=[Path(__file__),ROOT/'native/gaussian_coherent_core92_v1.cpp',ROOT/'experiments/iteration92_common_v1.py',ROOT/'tests/test_native_coherent92.py',dll];save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}));suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_native_coherent92.py')
with (out/'tests.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
report=dict(passed=r.wasSuccessful(),tests=r.testsRun,skipped=len(r.skipped),failures=[(str(t),s) for t,s in r.failures],errors=[(str(t),s) for t,s in r.errors]);save(out,'validation.json',report);manifest(out);print(json.dumps(report));assert r.wasSuccessful()
