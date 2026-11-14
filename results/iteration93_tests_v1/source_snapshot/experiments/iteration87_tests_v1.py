"""Complete suite with historical backends and iteration71/72/73/74/75/76/77/78/79/80 APIs enabled, including RNG84, tree85, shared86, optical86, blocktree87 and guarded87."""
from pathlib import Path
import os,json,hashlib,shutil,sys,unittest,contextlib,argparse,gc
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
os.environ['GBSLAB_CUDA_BRANCH_LIBRARY']=str(ROOT/'results/iteration72_cuda_pilot_v1/gpu.dll')
os.environ['GBSLAB_CLICK_REPLAY_LIBRARY']=str(ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll')
os.environ['GBSLAB_NEWTON_PNR_LIBRARY']=str(ROOT/'results/iteration72_newton_pilot_v1/newton.dll')
os.environ['GBSLAB_POLYNOMIAL_PNR_LIBRARY']=str(ROOT/'results/iteration73_polynomial_pilot_v2/polynomial.dll')
os.environ['GBSLAB_PHILOX_CLICK_LIBRARY']=str(ROOT/'results/iteration73_philox_pilot_v1/philox.dll')
os.environ['GBSLAB_STRUCTURED_PNR_LIBRARY']=str(ROOT/'results/iteration74_structured_pilot_v1/structured.dll')
os.environ['GBSLAB_CLASSICAL_TAIL_PNR_LIBRARY']=str(ROOT/'results/iteration74_classical_pilot_v1/classical.dll')
os.environ['GBSLAB_FUSED_CLICK_LIBRARY']=str(ROOT/'results/iteration74_fused_pilot_v2/philox.dll')
os.environ['GBSLAB_DEEP_PNR_LIBRARY']=str(ROOT/'results/iteration75_deep_pilot_v1/deep.dll')
os.environ['GBSLAB_RESPLIT_PNR_LIBRARY']=str(ROOT/'results/iteration75_resplit_pilot_v1/resplit.dll')
os.environ['GBSLAB_PREFIX_PNR_LIBRARY']=str(ROOT/'results/iteration75_prefix_pilot_v1/prefix.dll')
os.environ['GBSLAB_PROJECTED_PNR_LIBRARY']=str(ROOT/'results/iteration76_projected_pilot_v3/projected.dll')
os.environ['GBSLAB_NUMBER_PNR_LIBRARY']=str(ROOT/'results/iteration77_number_pilot_v2/number.dll')
os.environ['GBSLAB_BUDGET_NUMBER_PNR_LIBRARY']=str(ROOT/'results/iteration77_budget_pilot_v1/budget.dll')
os.environ['GBSLAB_SPHERE_PNR_LIBRARY']=str(ROOT/'results/iteration78_pilot_v2/sphere.dll')
os.environ['GBSLAB_ALIAS_CLICK_LIBRARY']=str(ROOT/'results/iteration78_alias_pilot_v1/table.dll')
os.environ['GBSLAB_JOINT_PNR_LIBRARY']=str(ROOT/'results/iteration79_pilot_v1/joint.dll')
os.environ['GBSLAB_BLOCK_CLICK_LIBRARY']=str(ROOT/'results/iteration79_pilot_v1/block.dll')
os.environ['GBSLAB_BLOCK_PHILOX_LIBRARY']=str(ROOT/'results/iteration79_gpu_pilot_v1/block.dll')
os.environ['GBSLAB_CACHED_TRANSPORT_LIBRARY']=str(ROOT/'results/iteration80_cached_pilot_v1/cached.dll')
os.environ['GBSLAB_ORDER80_LIBRARY']=str(ROOT/'results/iteration80_order_pilot_v1/order.dll')
os.environ['GBSLAB_FACTOR81_LIBRARY']=str(ROOT/'results/iteration81_factor_pilot_v2/factor.dll')
os.environ['GBSLAB_JZ81_HIST_LIBRARY']=str(ROOT/'results/iteration81_jiuzhang_accuracy_v3/hist.dll')
os.environ['GBSLAB_FACTOR82_LIBRARY']=str(ROOT/'results/iteration82_factor_accuracy_v1/factor.dll')
os.environ['GBSLAB_STABLE82_LIBRARY']=str(ROOT/'results/iteration82_stable_pilot_v1/stable.dll')
os.environ['GBSLAB_LOSS83_LIBRARY']=str(ROOT/'results/iteration83_full_quality_v1/loss.dll')
os.environ['GBSLAB_CORE83_LIBRARY']=str(ROOT/'results/iteration83_core_energy_native_v1/core.dll')
os.environ['GBSLAB_RNG84_LIBRARY']=str(ROOT/'results/iteration84_rng_gates_v1/rng.dll')
os.environ['GBSLAB_EXACTSUM84_LIBRARY']=str(ROOT/'results/iteration84_exactsum_pilot_v1/exactsum.dll')
os.environ['GBSLAB_TREE85_LIBRARY']=str(ROOT/'results/iteration85_tree_pilot_v1/tree.dll')
os.environ['GBSLAB_SHARED86_LIBRARY']=str(ROOT/'results/iteration86_shared_gates_v4/shared.dll')
os.environ['GBSLAB_BLOCKTREE87_LIBRARY']=str(ROOT/'results/iteration87_blocktree_pilot_v1/blocktree.dll')
os.environ['GBSLAB_GUARDED87_LIBRARY']=str(ROOT/'results/iteration87_guarded_gates_v2/guarded.dll')
sources=sorted(p for base in ('experiments','src','tests','native') for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.cu','.h','.hpp'))
hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources};save('source_hashes.json',hashes)
for p in sources:
    target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
save('libraries.json',{key:dict(path=value,sha256=H(Path(value))) for key,value in os.environ.items() if key.startswith('GBSLAB_') and Path(value).is_file()})
# Windows accumulates duplicate AddDllDirectory registrations until its
# aggregate search-path limit. Retain actual handles during each test, then
# explicitly close only registrations created inside that test. Class setup
# registrations, numerical calls, source bindings and assertions are unchanged.
original_add_dll_directory=os.add_dll_directory
test_directory_handles=[]
def tracked_add_dll_directory(path):
    handle=original_add_dll_directory(path)
    test_directory_handles.append(handle)
    return handle
os.add_dll_directory=tracked_add_dll_directory
class CollectedResult(RecordingResult):
    def startTest(self,test):
        gc.collect()
        self.directory_checkpoint=len(test_directory_handles)
        super().startTest(test)
    def stopTest(self,test):
        for handle in test_directory_handles[self.directory_checkpoint:]:
            if handle.path is not None:handle.close()
        del test_directory_handles[self.directory_checkpoint:]
        super().stopTest(test)

suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
with (OUT/'tests.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=CollectedResult).run(suite)
report=dict(loss83_passed=sum(s.startswith('test_native_loss83.') for s in result.successes),sampling82_passed=sum(s.startswith('test_native_sampling82.') for s in result.successes),factor81_passed=sum(s.startswith('test_native_factor81.') for s in result.successes),passed=result.wasSuccessful(),tests_run=result.testsRun,successes=len(result.successes),failures=len(result.failures),errors=len(result.errors),branch_click_passed=sum(s.startswith('test_native_click_branches.') for s in result.successes),branch_pnr_passed=sum(s.startswith('test_native_pnr_branches.') for s in result.successes),cuda_click_passed=sum(s.startswith('test_native_cuda_click_branches.') for s in result.successes),newton_pnr_passed=sum(s.startswith('test_native_pnr_newton.') for s in result.successes),polynomial_pnr_passed=sum(s.startswith('test_native_polynomial_pnr.') for s in result.successes),philox_click_passed=sum(s.startswith('test_native_philox_click.') for s in result.successes),structured_pnr_passed=sum(s.startswith('test_native_structured_pnr.') for s in result.successes),classical_tail_passed=sum(s.startswith('test_native_classical_tail_pnr.') for s in result.successes),fused_click_passed=sum(s.startswith('test_native_fused_click.') for s in result.successes),deep_pnr_passed=sum(s.startswith('test_native_deep_pnr.') for s in result.successes),resplit_pnr_passed=sum(s.startswith('test_native_resplit_pnr.') for s in result.successes),skipped_cases=sum('setUpClass' not in t.id() for t,_ in result.skipped),skipped_classes=sum('setUpClass' in t.id() for t,_ in result.skipped),failure_details=[dict(test=t.id(),traceback=msg) for t,msg in result.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in result.errors],skip_details=[dict(test=t.id(),reason=msg) for t,msg in result.skipped],unchanged_sources=all(H(ROOT/name)==h for name,h in hashes.items()))
report['blocktree87_passed']=sum(s.startswith('test_native_blocktree87.') for s in result.successes)
report['guarded87_passed']=sum(s.startswith('test_native_guarded87.') for s in result.successes)
report['shared86_passed']=sum(s.startswith('test_native_shared86.') for s in result.successes)
report['optical86_passed']=sum(s.startswith('test_optical_pnr_reference86.') for s in result.successes)
report['tree85_passed']=sum(s.startswith('test_native_tree85.') for s in result.successes)
report['prefix_pnr_passed']=sum(s.startswith('test_native_prefix_pnr.') for s in result.successes)
report['projected_pnr_passed']=sum(s.startswith('test_native_projected_pnr.') for s in result.successes)
report['number_pnr_passed']=sum(s.startswith('test_native_number_pnr.') for s in result.successes)
report['sphere_pnr_passed']=sum(s.startswith('test_native_sphere_pnr.') for s in result.successes)
report['alias_click_passed']=sum(s.startswith('test_native_alias_click.') for s in result.successes)
report['joint_pnr_passed']=sum(s.startswith('test_native_joint_pnr.') for s in result.successes)
report['block_click_passed']=sum(s.startswith('test_native_block_click.') for s in result.successes)
report['block_philox_passed']=sum(s.startswith('test_native_block_philox_click.') for s in result.successes)
report['transport80_passed']=sum(s.startswith('test_native_transport80.') for s in result.successes)
report['core83_passed']=sum(s.startswith('test_native_core83.') for s in result.successes)
report['rng84_passed']=sum(s.startswith('test_native_rng84.') for s in result.successes)
report['exactsum84_passed']=sum(s.startswith('test_native_exactsum84.') for s in result.successes)
os.add_dll_directory=original_add_dll_directory
save('runtime_cleanup.json',dict(scope='Close only still-open actual DLL-directory handles created inside each test; native wrappers may already close their own registrations. No numerical source or assertion modifications.',retained_class_setup_handles=len(test_directory_handles)))
save('passed_test_ids.json',result.successes);save('validation.json',report);print(json.dumps(report,indent=2),flush=True)
