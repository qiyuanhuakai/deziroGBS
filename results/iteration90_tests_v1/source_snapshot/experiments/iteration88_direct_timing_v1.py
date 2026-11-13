"""Three serial paired4/64-record block/project16/64 timings on frozen85 addresses.
Includes model creation, uniform generation, sampling, OR and destruction;
excludes upstream input preparation, initial DLL loads and artifact I/O.
Repeated timing rows are not independent quality samples.
"""
import argparse,statistics
from iteration88_project_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
gates=ROOT/'results/iteration88_project_gates_v1';replay=gates;assert read(gates/'summary.json')['passed'];paths=dict(block16=ROOT/'results/iteration87_guarded_gates_v2/guarded.dll',project16=ROOT/'results/iteration88_project_pilot_v2/projected.dll');paths['block64']=paths['block16'];paths['project64']=paths['project16'];inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),ROOT/'experiments/iteration87_guarded_common_v1.py',ROOT/'experiments/iteration88_project_common_v1.py',inp,*paths.values(),gates/'summary.json',replay/'summary.json'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},records=[4,64],experiment=0x85001002,global_offset=0,purpose=5,sectors=[0,1],budgets=dict(block16=16,project16=16,block64=64,project64=64),threads=16,repetitions=3,intentional_replay=True));runtime=os.add_dll_directory(str(runtime_dirs()[0]));loaded=[C.CDLL(str(path)) for path in paths.values()];loaded.append(C.CDLL(str(ROOT/'results/iteration84_rng_gates_v1/rng.dll')));d=np.load(inp);v,x=d['v'],d['x'];rows=[];results={}
# All four implementations are loaded before the timed region.
for rep in range(3):
 for count in (4,64):
  order=['block16','project16','block64','project64'];order=order[rep:]+order[:rep]
  for method in order:
   budget=16 if method.endswith('16') else 64
   raw=[];meta=[];start=time.perf_counter()
   for sector in range(2):
    draws=uniforms(0x85001002,sector,0,count,len(v))
    with NativeStableClickSampler(v,x[sector],paths[method],branch_budget=budget,runtime_dirs=runtime_dirs()) as model:r,s=projected(model,draws,budget) if method.startswith('project') else sample(model,draws,budget)
    raw.append(r);meta.append(s)
   combined=np.bitwise_or(raw[0]['clicks'],raw[1]['clicks']);seconds=time.perf_counter()-start;row=dict(repetition=rep,method=method,budget=budget,records=count,seconds=seconds,success=all(r['returncode']==0 for r in meta),sectors=meta);rows.append(row);save(out,f'row_{rep}_{method}_{count}.json',row)
   for sector,r in enumerate(raw):np.savez_compressed(out/f'raw_{rep}_{method}_{count}_{sector}.npz',**r)
   results[rep,method,count]=raw;assert row['success'],row;print('formal click',rep,method,count,seconds,flush=True)
checks=[];medians=[]
for count in (4,64):
 for method in paths:
  equal=all(np.array_equal(results[0,method,count][s][k],results[rep,method,count][s][k],equal_nan=True) for rep in (1,2) for s in range(2) for k in results[0,method,count][s]);assert equal;(checks.append(dict(method=method,records=count,identical_repetitions=equal)));medians.append(dict(method=method,records=count,median_seconds=statistics.median(r['seconds'] for r in rows if r['method']==method and r['records']==count)))
save(out,'summary.json',dict(passed=True,rows=rows,medians=medians,checks=checks,scope=__doc__));manifest(out)
