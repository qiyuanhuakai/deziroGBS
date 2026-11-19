"""Fixed8192 full dual-spectrum optical-space quality records, purpose5.
One immutable suffix cache is shared across the two identical optical maps.
"""
import argparse
from iteration91_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);planpath=ROOT/'research/iteration91_quality_plan.json';plan=read(planpath);job=next(j for j in plan['jobs'] if j['name']=='compact16');dll=ROOT/'results/iteration91_build_v2/optical.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';gates=ROOT/'results/iteration91_optical_gates_v1/summary.json';assert read(gates)['passed'];files=[Path(__file__),ROOT/'experiments/iteration91_common_v1.py',ROOT/'native/gaussian_click_compact91_v2.cpp',dll,planpath,inp,gates];save(out,'plan.json',dict(job=job,scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},threads=16,batch=64));z=np.load(inp);N=job['records'];rows=[];parts=[];good=[];cache=None;destroy=None
try:
 for sector in (0,1):
  values=[];valid=[]
  with NativeStableClickSampler(z['v'],z['x'][sector],dll,branch_budget=16,runtime_dirs=runtime_dirs()) as model:
   if cache is None:cache,destroy,setup=optical_cache(model);save(out,'cache.json',setup)
   for offset in range(0,N,64):
    draws=uniforms(int(job['experiment'],16),sector,offset,min(64,N-offset),len(z['v']));raw,meta=optical_sample(model,cache,draws,16);np.savez_compressed(out/f'raw_{sector}_{offset}.npz',**raw);v=(raw['errors'][:,0]==0)&(raw['statistics'][:,5]==len(z['v']));row=dict(sector=sector,offset=offset,records=len(v),failed=int((~v).sum()),**meta);save(out,f'row_{sector}_{offset}.json',row);rows.append(row);values.append(raw['clicks']);valid.append(v)
    if offset%512==0:print('quality91 compact16',sector,offset,row['failed'],row['seconds'],flush=True)
  parts.append(np.concatenate(values));good.append(np.concatenate(valid))
finally:
 if cache:destroy(cache)
bits=parts[0]|parts[1];valid=good[0]&good[1];np.savez_compressed(out/'combined.npz',clicks=bits,valid=valid);save(out,'summary.json',dict(completed=True,job=job,records=N,failed_records=int((~valid).sum()),mean_clicks_successful=float(bits[valid].sum(1).mean()),sampling_seconds=sum(r['seconds'] for r in rows),rows=rows,scope=__doc__));manifest(out)
