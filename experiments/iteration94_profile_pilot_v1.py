"""Inclusive timing diagnostics on frozen paired controls, not formal throughput.
Timers overlap (QR is inside canonicalization, rates, waits and compression).
Projection diagnostics include attempted final jumps that may fail the budget.
"""
import argparse
from iteration94_profile_common_v1 import *

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=ROOT/'results/iteration94_wait_pilot_v1';plan=json.loads((source/'plan.json').read_text())
lib,runtime=load93();save(out,'backend.json',lib._backend)
names=['wait','rates','attenuate','full_jump','centered_rates','compress','canonical','qr','svd']
files=[Path(__file__),ROOT/'native/gaussian_mps_profile94_v1.cpp',ROOT/'experiments/iteration94_profile_common_v1.py',ROOT/'results/iteration94_profile_build_v1/mps.dll',source/'plan.json']
save(out,'plan.json',dict(scope=__doc__,source_plan=plan,timer_names=names,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files}))
rows=[]
for j,case in enumerate(plan['cases']):
    z=np.load(source/f'input_{j}.npz');print('starting',case['name'],flush=True)
    raw,meta=sample93(lib,z['u'],z['t'],z['alpha'],case['records'],plan['experiment_base']+j,d=case['d'],bond=case['bond'],budget=case['budget'],cap=256,noise=z['noise'])
    np.savez_compressed(out/f'raw_{j}.npz',**raw)
    previous=np.load(source/f'raw_0_{j}_new.npz')
    same_counts=bool(np.array_equal(raw['counts'],previous['counts']));same_errors=bool(np.array_equal(raw['errors'],previous['errors']));same_stats=bool(np.array_equal(raw['statistics'][:,:18],previous['statistics']))
    row=dict(name=case['name'],case=j,records=case['records'],**meta,inclusive_seconds={name:float(raw['statistics'][:,18+k].sum()) for k,name in enumerate(names)},calls={name:int(raw['statistics'][:,27+k].sum()) for k,name in enumerate(names)},projection=dict(sum_local_weight=float(raw['statistics'][:,36].sum()),sum_schmidt_weight=float(raw['statistics'][:,37].sum()),max_local_weight=float(raw['statistics'][:,38].max()),max_schmidt_weight=float(raw['statistics'][:,39].max()),sum_local_sqrt=float(raw['statistics'][:,40].sum()),sum_schmidt_sqrt=float(raw['statistics'][:,41].sum())),replay=dict(counts=same_counts,errors=same_errors,first18_statistics=same_stats),max_statistics_difference=float(np.max(abs(raw['statistics'][:,:18]-previous['statistics']))))
    rows.append(row);save(out,f'row_{j}.json',row);print(json.dumps(row),flush=True)
    assert same_counts and same_errors and same_stats,row
save(out,'summary.json',dict(completed=True,scope=__doc__,rows=rows));manifest(out)
