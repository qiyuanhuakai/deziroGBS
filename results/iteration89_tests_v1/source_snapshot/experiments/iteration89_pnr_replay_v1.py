"""Replay every786432 small-PNR88 record, including207 failures, plus the first
4096 frozen full1152 dual-spectrum PNR86 records. No independent sample is added.
Counts, errors, validity and original16 evolution/RNG statistics must be exact;
tree accounting intentionally changes because selection now includes pure steps.
"""
import argparse
from iteration89_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration89_pilot_v1/puretree.dll';source=ROOT/'results/iteration88_small_pnr_quality_v1';planpath=ROOT/'research/iteration88_small_pnr_plan.json';files=[Path(__file__),ROOT/'experiments/iteration89_common_v1.py',dll,planpath,source/'manifest.json',ROOT/'results/iteration89_gates_v1/summary.json'];assert read(files[-1])['passed'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},full_records=4096,small_records_per_task=65536,intentional_replay=True,threads=16));rows=[]
def compare_raw(new,old):return bool(all(np.array_equal(new[k],old[k]) for k in ('counts','errors')) and np.array_equal(new['statistics'][:,:16],old['statistics'][:,:16]))
for j in range(4):
 z=np.load(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{j}.npz');q={k:np.ascontiguousarray(z[k]) for k in ('v','t','x','L','mean')}
 with NativeDirectClickSampler(q['v'],q['x'],dll,runtime_dirs=runtime_dirs()) as model:
  handle,destroy=tree(model)
  try:
   for R in (3,4,5):
    for offset in range(0,65536,1024):
     raw,meta=puretree(model,handle,q,R,1024,0x88001000+16*j+R,offset);old=np.load(source/f'raw_{j}_{R}_{offset}.npz');equal=compare_raw(raw,old);np.savez_compressed(out/f'small_{j}_{R}_{offset}.npz',**raw);row=dict(case=j,retained=R,offset=offset,records=1024,identical=equal,**meta);rows.append(row);save(out,f'small_{j}_{R}_{offset}.json',row);assert equal,row
    print('all small frozen records',j,R,'passed',flush=True)
  finally:destroy(handle)
for sector in (0,1):
 z=np.load(ROOT/f'results/iteration84_correlated_pnr_pilot_v1/input_{sector}.npz');t=np.ascontiguousarray(z['t']);q=dict(v=z['v'],t=t,L=np.ascontiguousarray(z['noise_factor']),x=np.diag(np.r_[-t/(1+t),t/(1-t)]),mean=np.zeros(2*len(t)))
 with NativeDirectClickSampler(q['v'],q['x'],dll,runtime_dirs=runtime_dirs()) as model:
  handle,destroy=tree(model)
  try:
   for offset in range(0,4096,256):
    raw,meta=puretree(model,handle,q,3,256,0x86001001,offset,sector=sector);old=np.load(ROOT/f'results/iteration86_pnr_quality_v1/raw_{sector}_{offset}.npz');equal=compare_raw(raw,old);np.savez_compressed(out/f'full_{sector}_{offset}.npz',**raw);row=dict(case='full1152',sector=sector,retained=3,offset=offset,records=256,identical=equal,**meta);rows.append(row);save(out,f'full_{sector}_{offset}.json',row);assert equal,row;print('frozen full PNR replay',sector,offset,meta['seconds'],flush=True)
  finally:destroy(handle)
save(out,'summary.json',dict(passed=True,rows=rows,small_records=786432,full_dual_records=4096,small_failed=sum(r['failed'] for r in rows if r['case']!='full1152'),scope=__doc__));manifest(out)
