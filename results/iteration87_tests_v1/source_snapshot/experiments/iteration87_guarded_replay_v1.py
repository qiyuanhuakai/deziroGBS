"""Replay all512 frozen86 B64 records, then a fresh-domain B128 oracle pilot.
These are numerical equivalence gates, not new quality samples. Save every row.
"""
import argparse
from iteration87_guarded_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
gates=ROOT/'results/iteration87_guarded_gates_v2';assert read(gates/'summary.json')['passed'];dll=gates/'guarded.dll';old=ROOT/'results/iteration86_shared_gates_v4/shared.dll';base=ROOT/'results/iteration86_shared_quality_v1';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),ROOT/'experiments/iteration87_guarded_common_v1.py',gates/'summary.json',dll,old,base/'manifest.json',inp,ROOT/'research/iteration87_validation_plan.json'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},jobs=[dict(budget=64,records=512,experiment=0x86004001,source='frozen86',intentional_replay=True),dict(budget=128,records=16,experiment=0x87003001,source='matched frozen86 numerical oracle',intentional_replay=False)],threads=16,batch=32,sectors=[0,1],purpose=5))
d=np.load(inp);v,x=d['v'],d['x'];rows=[]
for budget,N,experiment in ((64,512,0x86004001),(128,16,0x87003001)):
 for sector in range(2):
  with NativeStableClickSampler(v,x[sector],dll,branch_budget=budget,runtime_dirs=runtime_dirs()) as model:
   for offset in range(0,N,32):
    count=min(32,N-offset);draws=uniforms(experiment,sector,offset,count,len(v))
    if budget==64:oldraw=dict(np.load(base/f'raw_{sector}_{offset}.npz'));oldmeta=dict(source='frozen86')
    else:
     with NativeStableClickSampler(v,x[sector],old,branch_budget=budget,runtime_dirs=runtime_dirs()) as oracle:oldraw,oldmeta=sample(oracle,draws,budget,False)
     np.savez_compressed(out/f'oracle_{budget}_{sector}_{offset}.npz',**oldraw);save(out,f'oracle_{budget}_{sector}_{offset}.json',oldmeta)
    raw,meta=sample(model,draws,budget);np.savez_compressed(out/f'raw_{budget}_{sector}_{offset}.npz',**raw);row=dict(budget=budget,sector=sector,offset=offset,records=count,**meta,comparison=compare(oldraw,raw));rows.append(row);save(out,f'row_{budget}_{sector}_{offset}.json',row);c=row['comparison'];print('guard replay',budget,sector,offset,'seconds',meta['seconds'],'wide',c['old_wide_calls'],c['new_wide_calls'],flush=True);assert meta['returncode']==0 and c['equal_decisions_errors_state_counters'] and c['oracle_probabilities_enclosed'] and c['counter_identities'] and c['completed']==count,row
    if offset==0:
     # One-worker and irregular chunks use exactly the same record/uniforms.
     serial,sm=sample(model,draws[:7],budget,threads=1);chunks=[]
     for first,end in ((0,3),(3,7)):
      chunk,cm=sample(model,draws[first:end],budget,threads=16);chunks.append(chunk);assert cm['returncode']==0,cm
     joined={key:np.concatenate([c[key] for c in chunks]) for key in raw};np.savez_compressed(out/f'serial_{budget}_{sector}.npz',**serial);np.savez_compressed(out/f'chunks_{budget}_{sector}.npz',**joined);equal=sm['returncode']==0 and all(np.array_equal(serial[k],raw[k][:7],equal_nan=True) and np.array_equal(joined[k],raw[k][:7],equal_nan=True) for k in raw);save(out,f'invariance_{budget}_{sector}.json',dict(passed=equal));assert equal
save(out,'summary.json',dict(passed=True,rows=rows,scope=__doc__));manifest(out)
