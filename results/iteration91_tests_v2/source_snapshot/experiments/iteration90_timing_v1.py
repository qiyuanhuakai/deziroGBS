"""Three serial4/64-record repeats: canonical89 versus optical90 cold/warm cache.
All include model creation, uniforms, sampling, OR and model destruction. Cold
also includes suffix-cache construction/destruction. Warm excludes its one-time
setup, reported separately. No concurrent heavy job and no new quality samples.
"""
import argparse,statistics
from iteration90_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);old=ROOT/'results/iteration89_pilot_v1/physical.dll';new=ROOT/'results/iteration90_build_v1/optical.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),ROOT/'experiments/iteration90_common_v1.py',old,new,inp];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},experiment=0x85001002,purpose=5,sectors=[0,1],records=[4,64],budget=16,threads=16,repetitions=3,intentional_replay=True));runtime=os.add_dll_directory(str(runtime_dirs()[0]));loaded=[C.CDLL(str(f)) for f in (old,new,ROOT/'results/iteration84_rng_gates_v1/rng.dll')];z=np.load(inp)
with NativeStableClickSampler(z['v'],z['x'][0],new,branch_budget=16,runtime_dirs=runtime_dirs()) as model:warm,destroy,setup=optical_cache(model)
save(out,'warm_setup.json',setup);rows=[];replays={}
try:
 for rep in range(3):
  methods=['canonical','optical_cold','optical_warm'];methods=methods[rep:]+methods[:rep]
  for N in (4,64):
   for method in methods:
    cache=warm if method=='optical_warm' else None;parts=[];metas=[];start=time.perf_counter()
    try:
     for sector in (0,1):
      draws=uniforms(0x85001002,sector,0,N,len(z['v']))
      with NativeStableClickSampler(z['v'],z['x'][sector],old if method=='canonical' else new,branch_budget=16,runtime_dirs=runtime_dirs()) as model:
       if method=='optical_cold' and cache is None:cache,cleanup,setup0=optical_cache(model)
       raw,meta=physical(model,draws,16) if method=='canonical' else optical_sample(model,cache,draws,16)
      parts.append(raw);metas.append(meta)
     combined=parts[0]['clicks']|parts[1]['clicks']
    finally:
     if method=='optical_cold' and cache:cleanup(cache)
    seconds=time.perf_counter()-start;row=dict(repetition=rep,method=method,records=N,seconds=seconds,success=all(m['returncode']==0 for m in metas),sectors=metas);rows.append(row);save(out,f'row_{rep}_{method}_{N}.json',row);replays[rep,method,N]=parts
    for sector,r in enumerate(parts):np.savez_compressed(out/f'raw_{rep}_{method}_{N}_{sector}.npz',**r)
    assert row['success'],row;print('timing90',rep,method,N,seconds,flush=True)
finally:destroy(warm)
checks=[];medians=[]
for N in (4,64):
 for method in ('canonical','optical_cold','optical_warm'):
  same=all(np.array_equal(replays[0,method,N][s][k],replays[r,method,N][s][k],equal_nan=True) for r in (1,2) for s in (0,1) for k in replays[0,method,N][s]);assert same;checks.append(dict(method=method,records=N,identical_repetitions=True));medians.append(dict(method=method,records=N,median_seconds=statistics.median(r['seconds'] for r in rows if r['method']==method and r['records']==N)))
 same=all(np.array_equal(replays[0,'optical_cold',N][s][k],replays[0,'optical_warm',N][s][k],equal_nan=True) for s in (0,1) for k in replays[0,'optical_cold',N][s]);assert same;checks.append(dict(records=N,identical_cold_warm=True))
save(out,'summary.json',dict(passed=True,rows=rows,medians=medians,checks=checks,scope=__doc__));manifest(out)
