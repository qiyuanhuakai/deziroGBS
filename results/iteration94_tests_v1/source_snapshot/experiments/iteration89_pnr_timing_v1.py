"""Three serial paired64-record old/new PNR3/5 timings. Includes model/tree,
output allocation, sampling, spectral sum and destruction; excludes input load,
initial DLL loads and artifact I/O. Common replay addresses are not quality N.
"""
import argparse,statistics
from iteration89_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);paths=dict(old=ROOT/'results/iteration87_blocktree_pilot_v1/blocktree.dll',new=ROOT/'results/iteration89_pilot_v1/puretree.dll');core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';files=[Path(__file__),ROOT/'experiments/iteration89_common_v1.py',*paths.values(),*[core/f'input_{s}.npz' for s in range(2)]];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},records=64,retained=[3,5],experiment=0x87002002,sectors=[0,1],purposes=[1,2,3,4],offset=0,threads=16,block=16,repetitions=3,intentional_replay=True));runtime=os.add_dll_directory(str(runtime_dirs()[0]));loaded=[C.CDLL(str(path)) for path in paths.values()];inputs=[]
for sector in range(2):
 z=np.load(core/f'input_{sector}.npz');t=np.ascontiguousarray(z['t']);inputs.append(dict(v=z['v'],t=t,L=np.ascontiguousarray(z['noise_factor']),mean=np.zeros(2*len(t)),x=np.diag(np.r_[-t/(1+t),t/(1-t)])))
rows=[];replays={}
for repetition in range(3):
 order=[('old',3),('new',3),('old',5),('new',5)];order=order[repetition:]+order[:repetition]
 for method,retained in order:
  parts=[];metas=[];start=time.perf_counter()
  for sector,q in enumerate(inputs):
   with NativeDirectClickSampler(q['v'],q['x'],paths[method],runtime_dirs=runtime_dirs()) as model:
    handle,destroy=tree(model)
    try:raw,meta=(pnr if method=='old' else puretree)(model,handle,q,retained,64,0x87002002,sector=sector)
    finally:destroy(handle)
   parts.append(raw);metas.append(meta)
  combined=parts[0]['counts'].astype(np.uint64)+parts[1]['counts'];seconds=time.perf_counter()-start;row=dict(repetition=repetition,method=method,retained=retained,records=64,seconds=seconds,success=all(m['returncode']==0 for m in metas),sectors=metas);rows.append(row);save(out,f'row_{repetition}_{method}_{retained}.json',row)
  for sector,r in enumerate(parts):np.savez_compressed(out/f'raw_{repetition}_{method}_{retained}_{sector}.npz',**r)
  replays[repetition,method,retained]=parts;assert row['success'],row;print('formal PNR89',repetition,method,retained,seconds,flush=True)
checks=[]
for R in (3,5):
 for method in paths:
  equal=all(np.array_equal(replays[0,method,R][s][k],replays[rep,method,R][s][k],equal_nan=True) for rep in (1,2) for s in range(2) for k in replays[0,method,R][s]);assert equal;checks.append(dict(method=method,retained=R,identical_repetitions=equal))
 same=all(np.array_equal(replays[0,'old',R][s]['counts'],replays[0,'new',R][s]['counts']) and np.array_equal(replays[0,'old',R][s]['statistics'][:,:16],replays[0,'new',R][s]['statistics'][:,:16]) and np.array_equal(replays[0,'old',R][s]['errors'],replays[0,'new',R][s]['errors']) for s in range(2));assert same;checks.append(dict(retained=R,identical_old_new=True))
medians=[dict(method=m,retained=R,median_seconds=statistics.median(r['seconds'] for r in rows if r['method']==m and r['retained']==R)) for m in paths for R in (3,5)];save(out,'summary.json',dict(passed=True,rows=rows,medians=medians,checks=checks,scope=__doc__));manifest(out)
