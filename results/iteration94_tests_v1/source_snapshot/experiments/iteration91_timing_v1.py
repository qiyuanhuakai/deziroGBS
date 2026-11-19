"""Three strictly serial rotations, optical90 versus compact91, cold and warm.
Same frozen control addresses and 16 sampling threads. Cold includes creation
and destruction of one shared suffix cache; warm preparation reported separately.
Models, uniforms, dual-sector sampling/OR and model destruction included. Disk,
first DLL load and artifact serialization excluded. No fresh quality samples.
"""
import argparse,statistics
from iteration91_common_v1 import *
from iteration90_common_v1 import optical_cache as old_cache,optical_sample as old_sample
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dlls={'optical90':ROOT/'results/iteration90_build_v1/optical.dll','compact91':ROOT/'results/iteration91_build_v2/optical.dll'};inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),ROOT/'experiments/iteration91_common_v1.py',ROOT/'experiments/iteration90_common_v1.py',*dlls.values(),inp];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},experiment=hex(0x85001002),records=[4,64],repetitions=3,threads=16,budget=16));runtime=os.add_dll_directory(str(runtime_dirs()[0]));libs=[C.CDLL(str(p)) for p in dlls.values()];z=np.load(inp);warm={};rows=[];replays={};setups={};setup_trials=[]
for method,dll in dlls.items():
 with NativeStableClickSampler(z['v'],z['x'][0],dll,branch_budget=16,runtime_dirs=runtime_dirs()) as model:
  cache_fn=old_cache if method=='optical90' else optical_cache;warm[method]=cache_fn(model);setups[method]=warm[method][2];setup_trials.append(dict(method=method,repetition=0,**warm[method][2]))
  for rep in (1,2):
   h,d,meta=cache_fn(model);d(h);setup_trials.append(dict(method=method,repetition=rep,**meta))
save(out,'warm_setup.json',setups);save(out,'setup_trials.json',dict(rows=setup_trials,medians={method:statistics.median(r['seconds'] for r in setup_trials if r['method']==method) for method in dlls}))
try:
 for rep in range(3):
  order=[('optical90','cold'),('compact91','cold'),('optical90','warm'),('compact91','warm')];order=order[rep:]+order[:rep]
  for N in (4,64):
   for method,mode in order:
    cache_fn=old_cache if method=='optical90' else optical_cache;sample_fn=old_sample if method=='optical90' else optical_sample;h=warm[method][0] if mode=='warm' else None;parts=[];metas=[];start=time.perf_counter()
    try:
     for sector in (0,1):
      draws=uniforms(0x85001002,sector,0,N,len(z['v']))
      with NativeStableClickSampler(z['v'],z['x'][sector],dlls[method],branch_budget=16,runtime_dirs=runtime_dirs()) as model:
       if h is None:h,destroy,setup=cache_fn(model)
       raw,meta=sample_fn(model,h,draws,16)
      parts.append(raw);metas.append(meta)
     combined=parts[0]['clicks']|parts[1]['clicks']
    finally:
     if mode=='cold' and h:destroy(h)
    elapsed=time.perf_counter()-start;row=dict(repetition=rep,method=method,cache=mode,records=N,seconds=elapsed,sectors=metas);rows.append(row);assert all(x['returncode']==0 for x in metas),row;save(out,f'row_{rep}_{method}_{mode}_{N}.json',row);replays[rep,method,mode,N]=parts
    for sector,raw in enumerate(parts):np.savez_compressed(out/f'raw_{rep}_{method}_{mode}_{N}_{sector}.npz',**raw)
    print('timing91',rep,method,mode,N,elapsed,flush=True)
finally:
 for h,d,_ in warm.values():d(h)
medians=[]
for method in dlls:
 for mode in ('cold','warm'):
  for N in (4,64):
   assert all(np.array_equal(replays[0,method,'cold',N][s][k],replays[r,method,mode,N][s][k],equal_nan=True) for r in range(3) for s in (0,1) for k in replays[0,method,'cold',N][s]);medians.append(dict(method=method,cache=mode,records=N,median_seconds=statistics.median(r['seconds'] for r in rows if r['method']==method and r['cache']==mode and r['records']==N)))
save(out,'summary.json',dict(passed=True,rows=rows,medians=medians,scope=__doc__));manifest(out)
