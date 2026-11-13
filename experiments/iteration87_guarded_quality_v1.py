"""Execute the preregistered fresh B64/B128 domains after all numerical gates.
Retain failures as separate outcomes; inference is performed by the joint-family
analysis. Explicit midpoint values are diagnostics, never exact probabilities.
"""
import argparse
from iteration87_guarded_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=int,choices=(64,128),required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
gates=ROOT/'results/iteration87_guarded_gates_v2';replay=ROOT/'results/iteration87_guarded_replay_v1';assert read(gates/'summary.json')['passed'] and read(replay/'summary.json')['passed'];planpath=ROOT/'research/iteration87_validation_plan.json';plan=read(planpath);job=next(row for row in plan['fresh_direct'] if row['budget']==a.budget);N=job['records'];experiment=int(job['experiment'],16);dll=gates/'guarded.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),ROOT/'experiments/iteration87_guarded_common_v1.py',dll,inp,gates/'summary.json',replay/'summary.json',planpath,ROOT/'results/iteration84_rng_gates_v1/rng.dll'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},job=job,batch_size=32,threads=16,sectors=[0,1],purpose=5,quality_family=plan))
d=np.load(inp);v,x=d['v'],d['x'];parts=[];valids=[];rows=[]
for sector in range(2):
 output=[];completed=[]
 with NativeStableClickSampler(v,x[sector],dll,branch_budget=a.budget,runtime_dirs=runtime_dirs()) as model:
  for offset in range(0,N,32):
   count=min(32,N-offset);draws=uniforms(experiment,sector,offset,count,len(v));raw,meta=sample(model,draws,a.budget);valid=(raw['errors'][:,0]==0)&(raw['statistics'][:,5]==len(v));row=dict(budget=a.budget,sector=sector,offset=offset,records=count,failed=int((~valid).sum()),**meta);rows.append(row);np.savez_compressed(out/f'raw_{sector}_{offset}.npz',**raw);save(out,f'row_{sector}_{offset}.json',row);output.append(raw['clicks']);completed.append(valid);print('fresh guard',a.budget,sector,offset,'failures',row['failed'],'seconds',meta['seconds'],flush=True)
 parts.append(np.concatenate(output));valids.append(np.concatenate(completed))
bits=np.bitwise_or(*parts);valid=valids[0]&valids[1];np.savez_compressed(out/'combined.npz',clicks=bits,valid=valid);save(out,'summary.json',dict(completed=True,records=N,failed_records=int((~valid).sum()),mean_clicks_successful=float(bits[valid].sum(1).mean()) if valid.any() else None,rows=rows,sampling_seconds=sum(r['seconds'] for r in rows),unique_domains=2*N,scope=__doc__));manifest(out)
