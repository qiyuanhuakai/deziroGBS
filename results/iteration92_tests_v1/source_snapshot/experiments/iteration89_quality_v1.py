"""One frozen new quality job: PNR5 or block/physical B16. All failures retained."""
import argparse
from iteration89_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--method',choices=('pnr5','block16','physical16'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);planpath=ROOT/'research/iteration89_quality_plan.json';plan=read(planpath);job=next(j for j in plan['jobs'] if j['name']==a.method);gates=ROOT/'results/iteration89_gates_v1';assert read(gates/'summary.json')['passed'];pilot=ROOT/'results/iteration89_pilot_v1';dll=(pilot/'puretree.dll' if job['kind']=='pnr' else pilot/'physical.dll' if job['method']=='physical' else ROOT/'results/iteration87_guarded_gates_v2/guarded.dll');inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),ROOT/'experiments/iteration89_common_v1.py',dll,planpath,inp,gates/'summary.json',*[ROOT/f'results/iteration84_correlated_pnr_pilot_v1/input_{s}.npz' for s in range(2)]];batch=256 if job['kind']=='pnr' else 64;save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},job=job,batch_size=batch,threads=16,quality_family=plan));N=job['records'];experiment=int(job['experiment'],16);parts=[];valids=[];rows=[];input0=np.load(inp)
for sector in range(2):
 if job['kind']=='pnr':
  z=np.load(ROOT/f'results/iteration84_correlated_pnr_pilot_v1/input_{sector}.npz');t=np.ascontiguousarray(z['t']);q=dict(v=z['v'],t=t,L=np.ascontiguousarray(z['noise_factor']),x=np.diag(np.r_[-t/(1+t),t/(1-t)]),mean=np.zeros(2*len(t)));model=NativeDirectClickSampler(q['v'],q['x'],dll,runtime_dirs=runtime_dirs());handle,destroy=tree(model)
 else:model=NativeStableClickSampler(input0['v'],input0['x'][sector],dll,branch_budget=16,runtime_dirs=runtime_dirs());handle=None
 output=[];good=[]
 try:
  for offset in range(0,N,batch):
   count=min(batch,N-offset)
   if job['kind']=='pnr':raw,meta=puretree(model,handle,q,5,count,experiment,offset=offset,sector=sector);valid=raw['valid'];values=raw['counts']
   else:
    draws=uniforms(experiment,sector,offset,count,len(input0['v']));raw,meta=(physical if job['method']=='physical' else sample)(model,draws,16);valid=(raw['errors'][:,0]==0)&(raw['statistics'][:,5]==len(input0['v']));values=raw['clicks']
   row=dict(method=job['name'],sector=sector,offset=offset,records=count,failed=int((~valid).sum()),**{k:v for k,v in meta.items() if k!='failed'});rows.append(row);np.savez_compressed(out/f'raw_{sector}_{offset}.npz',**raw);save(out,f'row_{sector}_{offset}.json',row);output.append(values);good.append(valid)
   if offset%512==0:print('quality89',job['name'],sector,offset,'failed',row['failed'],'seconds',meta['seconds'],flush=True)
 finally:
  if handle:destroy(handle)
  model.close()
 parts.append(np.concatenate(output));valids.append(np.concatenate(good))
valid=valids[0]&valids[1]
if job['kind']=='pnr':counts=parts[0].astype(np.uint64)+parts[1];bits=counts>0;np.savez_compressed(out/'combined.npz',counts=counts,clicks=bits,valid=valid)
else:bits=np.bitwise_or(*parts);np.savez_compressed(out/'combined.npz',clicks=bits,valid=valid)
save(out,'summary.json',dict(completed=True,job=job,records=N,failed_records=int((~valid).sum()),mean_clicks_successful=float(bits[valid].sum(1).mean()) if valid.any() else None,sampling_seconds=sum(r['seconds'] for r in rows),rows=rows,scope=__doc__));manifest(out)
