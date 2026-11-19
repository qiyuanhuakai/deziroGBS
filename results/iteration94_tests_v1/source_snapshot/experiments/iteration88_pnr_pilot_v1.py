"""Retained3/4/5 cost/failure pilots on declared two-mode and full1152 inputs.
Fixed common domains are paired diagnostics, not fresh independent quality data.
"""
import argparse
from iteration88_pnr_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
planpath=ROOT/'research/iteration88_small_pnr_plan.json';plan=read(planpath);dll=ROOT/'results/iteration87_blocktree_pilot_v1/blocktree.dll';core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';files=[Path(__file__),ROOT/'experiments/iteration88_pnr_common_v1.py',planpath,dll,*[core/f'input_{s}.npz' for s in range(2)]];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},small_records=128,full_records=[16,64],full_experiments=[0x88002201,0x88002202],retained=[3,4,5],block=16,threads=16));rows=[]
for j,case in enumerate(plan['cases']):
 q=small_input(case);np.savez_compressed(out/f'input_{j}.npz',**q)
 with NativeDirectClickSampler(q['v'],q['x'],dll,runtime_dirs=dirs()) as model:
  handle,destroy=tree(model)
  try:
   for retained in (3,4,5):
    raw,meta=pnr(model,handle,q,retained,128,0x88002100+j);np.savez_compressed(out/f'raw_small_{j}_{retained}.npz',**raw);row=dict(case=case['name'],retained=retained,records=128,**meta);rows.append(row);save(out,f'row_small_{j}_{retained}.json',row);print('PNR small',row,flush=True)
  finally:destroy(handle)
for rep,count in enumerate((16,64)):
 for retained in (3,4,5):
  parts=[];valids=[];metas=[]
  for sector in range(2):
   z=np.load(core/f'input_{sector}.npz');t=np.ascontiguousarray(z['t']);q=dict(v=z['v'],t=t,L=np.ascontiguousarray(z['noise_factor']),x=np.diag(np.r_[-t/(1+t),t/(1-t)]),mean=np.zeros(2*len(t)))
   with NativeDirectClickSampler(q['v'],q['x'],dll,runtime_dirs=dirs()) as model:
    handle,destroy=tree(model)
    try:raw,meta=pnr(model,handle,q,retained,count,0x88002201+rep,sector=sector)
    finally:destroy(handle)
   np.savez_compressed(out/f'raw_full_{count}_{retained}_{sector}.npz',**raw);save(out,f'row_full_{count}_{retained}_{sector}.json',meta);parts.append(raw['counts']);valids.append(raw['valid']);metas.append(meta);print('PNR full',count,retained,sector,meta,flush=True)
  counts=parts[0].astype(np.uint64)+parts[1];valid=valids[0]&valids[1];np.savez_compressed(out/f'combined_{count}_{retained}.npz',counts=counts,valid=valid);row=dict(case='full1152',records=count,retained=retained,failed=int((~valid).sum()),mean_photons_successful=float(counts[valid].sum(1).mean()) if valid.any() else None,mean_clicks_successful=float((counts[valid]>0).sum(1).mean()) if valid.any() else None,seconds=sum(m['seconds'] for m in metas));rows.append(row);save(out,f'combined_{count}_{retained}.json',row)
save(out,'summary.json',dict(completed=True,rows=rows,scope=__doc__));manifest(out)
