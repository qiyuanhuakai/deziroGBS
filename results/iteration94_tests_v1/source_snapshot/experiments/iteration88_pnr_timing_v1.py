"""Three serial paired64-record full1152 PNR timings at retained3/4/5.
Includes model/tree creation, output allocation, native RNG/sampling, spectral
sum and destruction; excludes input loading, initial DLL load and artifact I/O.
Fixed common addresses are deliberate timing replays, not new quality records.
"""
import argparse,statistics
from iteration88_pnr_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration87_blocktree_pilot_v1/blocktree.dll';core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';files=[Path(__file__),ROOT/'experiments/iteration88_pnr_common_v1.py',dll,*[core/f'input_{s}.npz' for s in range(2)]];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},records=64,retained=[3,4,5],experiment=0x87002002,sectors=[0,1],purposes=[1,2,3,4],offset=0,threads=16,block=16,repetitions=3,intentional_replay=True));runtime=os.add_dll_directory(str(dirs()[0]));loaded=C.CDLL(str(dll));inputs=[]
for sector in range(2):
 z=np.load(core/f'input_{sector}.npz');t=np.ascontiguousarray(z['t']);inputs.append(dict(v=z['v'],t=t,L=np.ascontiguousarray(z['noise_factor']),mean=np.zeros(2*len(t)),x=np.diag(np.r_[-t/(1+t),t/(1-t)])))
rows=[];replays={}
for repetition in range(3):
 order=[3,4,5];order=order[repetition:]+order[:repetition]
 for retained in order:
  parts=[];metas=[];start=time.perf_counter()
  for sector,q in enumerate(inputs):
   with NativeDirectClickSampler(q['v'],q['x'],dll,runtime_dirs=dirs()) as model:
    handle,destroy=tree(model)
    try:raw,meta=pnr(model,handle,q,retained,64,0x87002002,sector=sector)
    finally:destroy(handle)
   parts.append(raw);metas.append(meta)
  combined=parts[0]['counts'].astype(np.uint64)+parts[1]['counts'];seconds=time.perf_counter()-start;row=dict(repetition=repetition,retained=retained,records=64,seconds=seconds,success=all(m['returncode']==0 for m in metas),sectors=metas);rows.append(row);save(out,f'row_{repetition}_{retained}.json',row)
  for sector,r in enumerate(parts):np.savez_compressed(out/f'raw_{repetition}_{retained}_{sector}.npz',**r)
  replays[repetition,retained]=parts;assert row['success'],row;print('formal PNR',repetition,retained,seconds,flush=True)
checks=[]
for retained in (3,4,5):
 equal=all(np.array_equal(replays[0,retained][sector][key],replays[rep,retained][sector][key],equal_nan=True) for rep in (1,2) for sector in range(2) for key in replays[0,retained][sector]);assert equal;checks.append(dict(retained=retained,identical_repetitions=equal))
medians={str(r):statistics.median(row['seconds'] for row in rows if row['retained']==r) for r in (3,4,5)};save(out,'summary.json',dict(passed=True,rows=rows,medians=medians,checks=checks,scope=__doc__));manifest(out)
