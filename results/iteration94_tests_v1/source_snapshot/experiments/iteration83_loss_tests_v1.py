"""Small loss-map API gates only; does not replace the pending full suite."""
from pathlib import Path
import os,sys,unittest,argparse,json,hashlib,shutil
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
library=ROOT/'results/iteration83_full_quality_v1/loss.dll';os.environ['GBSLAB_LOSS83_LIBRARY']=str(library);os.environ['GBSLAB_NATIVE_RUNTIME_DIRS']=str(Path(shutil.which('g++')).parent);sources=[Path(__file__),ROOT/'tests/test_native_loss83.py',ROOT/'src/gbslab/native_loss_map.py',library];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources}))
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_native_loss83.py')
with (OUT/'tests.log').open('x',encoding='utf-8') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
row=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=[dict(test=t.id(),traceback=s) for t,s in result.failures],errors=[dict(test=t.id(),traceback=s) for t,s in result.errors],skipped=[dict(test=t.id(),reason=s) for t,s in result.skipped]);save('validation.json',row);save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print(json.dumps(row),flush=True);sys.exit(0 if row['passed'] else 1)
