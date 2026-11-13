"""Complete suite with historical backends and both iteration71 APIs enabled."""
from pathlib import Path
import os,json,hashlib,shutil,sys,unittest,contextlib,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from iteration70_tests import RecordingResult
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
old=json.loads((ROOT/'results/iteration70_tests/native_libraries.json').read_text())
for key,row in old.items():
    assert H(Path(row['path']))==row['sha256'];os.environ[key]=row['path']
os.environ['GBSLAB_GAUSSIAN_RESET_BLAS']=os.environ['GBSLAB_NATIVE_BLAS_LIBRARY']
gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent
os.environ['GBSLAB_NATIVE_RUNTIME_DIRS']=os.pathsep.join(map(str,[gcc,cuda,cuda/'x64']))
os.environ['GBSLAB_BRANCH_CLICK_LIBRARY']=str(ROOT/'results/iteration71_branch_pilot_v1/branches.dll')
os.environ['GBSLAB_BRANCH_PNR_LIBRARY']=str(ROOT/'results/iteration71_pnr_branch_pilot_v2/pnr_branches.dll')
sources=sorted(p for base in ('experiments','src','tests','native') for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.cu','.h','.hpp'))
hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources};save('source_hashes.json',hashes)
for p in sources:
    target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
save('libraries.json',{key:dict(path=value,sha256=H(Path(value))) for key,value in os.environ.items() if key.startswith('GBSLAB_') and Path(value).is_file()})
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
with (OUT/'tests.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,successes=len(result.successes),failures=len(result.failures),errors=len(result.errors),branch_click_passed=sum(s.startswith('test_native_click_branches.') for s in result.successes),branch_pnr_passed=sum(s.startswith('test_native_pnr_branches.') for s in result.successes),skipped_cases=sum('setUpClass' not in t.id() for t,_ in result.skipped),skipped_classes=sum('setUpClass' in t.id() for t,_ in result.skipped),failure_details=[dict(test=t.id(),traceback=msg) for t,msg in result.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in result.errors],skip_details=[dict(test=t.id(),reason=msg) for t,msg in result.skipped],unchanged_sources=all(H(ROOT/name)==h for name,h in hashes.items()))
save('passed_test_ids.json',result.successes);save('validation.json',report);print(json.dumps(report,indent=2),flush=True)
