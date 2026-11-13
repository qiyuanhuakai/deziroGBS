"""Archive the complete suite with both new native backends enabled."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,contextlib,json,shutil,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from iteration68_build import digest,save
class RecordingResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.successes=[]
    def addSuccess(self,test):self.successes.append(test.id());super().addSuccess(test)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--build',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);build=args.build.resolve()
    cuda=Path(shutil.which('nvcc')).parent;gcc=Path(shutil.which('g++')).parent
    os.environ['GBSLAB_NATIVE_CPU_LIBRARY']=str(build/'native_cpu.dll');os.environ['GBSLAB_NATIVE_CUDA_LIBRARY']=str(build/'native_cuda.dll');os.environ['GBSLAB_NATIVE_RUNTIME_DIRS']=os.pathsep.join(map(str,[gcc,cuda,cuda/'x64']));os.environ['GBSLAB_NATIVE_BLAS_LIBRARY']=str(next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll')))
    sources=sorted({p for base in ('experiments','src','tests') for p in (ROOT/base).rglob('*.py')}|set((ROOT/'native').glob('*')));hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sources if p.is_file()}
    save(out/'source_hashes_at_start.json',hashes)
    for name in hashes:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    save(out/'native_libraries.json',{name:dict(path=os.environ[name],sha256=digest(os.environ[name])) for name in ('GBSLAB_NATIVE_CPU_LIBRARY','GBSLAB_NATIVE_CUDA_LIBRARY','GBSLAB_NATIVE_BLAS_LIBRARY')})
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    report=dict(tests_run=result.testsRun,passed=len(result.successes),failures=len(result.failures),errors=len(result.errors),skipped_cases=sum('setUpClass' not in t.id() for t,_ in result.skipped),skipped_classes=sum('setUpClass' in t.id() for t,_ in result.skipped),new_native_tests_passed=sum(s.startswith('test_native_displaced_sampling.') for s in result.successes),failure_details=[dict(test=t.id(),traceback=msg) for t,msg in result.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in result.errors],skip_details=[dict(test=t.id(),reason=msg) for t,msg in result.skipped],executed_gate_passed=result.wasSuccessful(),unchanged_sources=hashes=={n:digest(ROOT/n) for n in hashes})
    save(out/'passed_test_ids.json',result.successes);save(out/'validation.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
