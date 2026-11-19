"""Fixed feasibility grid before quality choices, including the actual50-input task.
No pilot sample will be included in subsequent confidence estimates.
"""
import argparse
from iteration93_common_v2 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);lib,runtime=load93();rng=np.random.default_rng(930030);cases=[]
for n,m,count in ((4,4,64),(4,4,64),(8,16,16),(16,64,8)):
    u=np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0];t=np.linspace(.15,.4,n);alpha=np.zeros(n,complex)
    if len(cases)==1:alpha=(rng.normal(size=n)+1j*rng.normal(size=n))*.15
    cases.append(dict(name=('pure4','displaced4','rank8','rank16')[len(cases)],u=u,t=t,alpha=alpha,noise=np.zeros((2*n,2*n)),count=count,cap=64,configs=[(4,4,.05),(6,16,.01)] if n==4 else [(4,8,.02)]))
for sector in (0,1):
    z=np.load(ROOT/f'results/iteration84_correlated_pnr_pilot_v1/input_{sector}.npz');cases.append(dict(name=f'actual1152_sector{sector}',u=z['v'],t=-z['t'],alpha=np.zeros(50,complex),noise=z['noise_factor']/np.sqrt(2),count=2,cap=256,configs=[(4,8,.02),(6,16,.02)]))
plan=[]
for j,case in enumerate(cases):
    np.savez_compressed(out/f'input_{j}.npz',**{k:case[k] for k in ('u','t','alpha','noise')});plan.append(dict(case=j,name=case['name'],count=case['count'],cap=case['cap'],configs=case['configs'],inputs=case['u'].shape[1],outputs=len(case['u']),mean_photons=float(np.sum(case['t']**2/(1-case['t']**2))+np.sum(abs(case['alpha'])**2)+np.sum(case['noise']**2)),experiment=0x93003000+j))
save(out,'plan.json',dict(scope=__doc__,cases=plan,source_sha256=H(Path(__file__)),native_sha256=H(ROOT/'native/gaussian_mps_core93_v2.cpp'),library_sha256=H(ROOT/'results/iteration93_build_v2/mps.dll'),common_random_numbers_across_configs=True));rows=[]
for j,case in enumerate(cases):
    for d,bond,budget in case['configs']:
        save(out,f'started_{j}_{d}_{bond}.json',dict(case=j,d=d,bond=bond,budget=budget));raw,meta=sample93(lib,case['u'],case['t'],case['alpha'],case['count'],0x93003000+j,d=d,bond=bond,budget=budget,cap=case['cap'],noise=case['noise']);np.savez_compressed(out/f'raw_{j}_{d}_{bond}.npz',**raw);row=dict(case=j,name=case['name'],d=d,bond=bond,budget=budget,records=case['count'],max_trace_budget=float(raw['statistics'][:,5].max()),max_bond=float(raw['statistics'][:,8].max()),max_root_residual=float(raw['statistics'][:,4].max()),max_hazard_residual=float(raw['statistics'][:,13].max()),completed_photons=raw['counts'].sum(1).tolist(),**meta);rows.append(row);save(out,f'row_{j}_{d}_{bond}.json',row);print(row,flush=True)
save(out,'summary.json',dict(completed=True,rows=rows,scope=__doc__));manifest(out)
