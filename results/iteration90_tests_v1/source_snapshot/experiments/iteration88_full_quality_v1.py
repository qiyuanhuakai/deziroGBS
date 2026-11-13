"""Execute one fixed full1152-output method from the preregistered47-event family.
All failures stay in the denominator; projection rejection is an explicit,
predefined closure choice, separately counted from any sample failure.
"""
import argparse
from iteration88_project_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=('block16','project16','block64','project64'),required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);planpath=ROOT/'research/iteration88_full_quality_plan.json';plan=read(planpath);job=next(row for row in plan['direct'] if row['name']==a.method);gates=ROOT/'results/iteration88_project_gates_v1';assert read(gates/'summary.json')['passed'];dll=ROOT/'results/iteration87_guarded_gates_v2/guarded.dll' if job['method']=='block' else ROOT/'results/iteration88_project_pilot_v2/projected.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),ROOT/'experiments/iteration88_project_common_v1.py',ROOT/'experiments/iteration87_guarded_common_v1.py',dll,inp,planpath,gates/'summary.json',ROOT/'results/iteration84_rng_gates_v1/rng.dll'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},job=job,batch_size=64,threads=16,sectors=[0,1],purpose=5,quality_family=plan));z=np.load(inp);v,x=z['v'],z['x'];N=job['records'];experiment=int(job['experiment'],16);parts=[];valids=[];rows=[]
for sector in range(2):
 output=[];completed=[]
 with NativeStableClickSampler(v,x[sector],dll,branch_budget=job['budget'],runtime_dirs=runtime_dirs()) as model:
  for offset in range(0,N,64):
   count=min(64,N-offset);draws=uniforms(experiment,sector,offset,count,len(v));raw,meta=sample(model,draws,job['budget']) if job['method']=='block' else projected(model,draws,job['budget']);valid=(raw['errors'][:,0]==0)&(raw['statistics'][:,5]==len(v));row=dict(method=job['name'],sector=sector,offset=offset,records=count,failed=int((~valid).sum()),**meta);rows.append(row);np.savez_compressed(out/f'raw_{sector}_{offset}.npz',**raw);save(out,f'row_{sector}_{offset}.json',row);output.append(raw['clicks']);completed.append(valid)
   if offset%256==0:print('full quality',job['name'],sector,offset,'failures',row['failed'],'seconds',meta['seconds'],flush=True)
 parts.append(np.concatenate(output));valids.append(np.concatenate(completed))
bits=np.bitwise_or(*parts);valid=valids[0]&valids[1];np.savez_compressed(out/'combined.npz',clicks=bits,valid=valid);save(out,'summary.json',dict(completed=True,job=job,records=N,failed_records=int((~valid).sum()),mean_clicks_successful=float(bits[valid].sum(1).mean()) if valid.any() else None,rows=rows,sampling_seconds=sum(r['seconds'] for r in rows),unique_domains=2*N,scope=__doc__));manifest(out)
