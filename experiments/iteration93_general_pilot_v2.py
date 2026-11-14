"""Resource/parameter pilot, never used as independent formal quality data.
Common control records across configurations are deliberate paired probes.
All failures retained. One native worker and one BLAS thread.
"""
import argparse
from iteration93_common_v5 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);lib,runtime=load93();save(out,'backend.json',lib._backend)
rng=np.random.default_rng(930060);cases=[]
for n in (4,4,8):
    j=len(cases);u=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0];t=np.array([.5,.6,.65,.7]) if n==4 else np.linspace(.15,.4,8);alpha=np.zeros(n,complex) if j!=1 else np.array([.2+.1j,-.15+.25j,.1j,-.1]);configs=[(6,16,.02),(8,32,.02),(10,64,.01),(16,128,.01)] if n==4 else [(4,8,.05),(6,16,.05),(8,32,.02)];cases.append(dict(name=['pure4','displaced4','pure8'][j],N=32 if n==4 else 16,configs=configs));np.savez_compressed(out/f'input_{j}.npz',u=u,t=t,alpha=alpha,noise=np.zeros((2*n,2*n)))
z=np.load(ROOT/'results/iteration84_correlated_pnr_pilot_v1/input_0.npz');np.savez_compressed(out/'input_3.npz',u=z['v'],t=-z['t'],alpha=np.zeros(50,complex),noise=z['noise_factor']/np.sqrt(2));cases.append(dict(name='actual_main50',N=1,configs=[(6,32,.1),(6,16,.2)]))
save(out,'plan.json',dict(scope=__doc__,cases=cases,experiment_base=0x93006000,threads=1,cap=256,hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),ROOT/'native/gaussian_mps_core93_v5.cpp',ROOT/'results/iteration93_build_v5/mps.dll')}));rows=[]
for j,case in enumerate(cases):
    z=np.load(out/f'input_{j}.npz')
    for d,bond,budget in case['configs']:
        name=f'{j}_{d}_{bond}';print('starting',name,flush=True);raw,meta=sample93(lib,z['u'],z['t'],z['alpha'],case['N'],0x93006000+j,d=d,bond=bond,budget=budget,cap=256,noise=z['noise']);np.savez_compressed(out/f'raw_{name}.npz',**raw);row=dict(case=case['name'],records=case['N'],d=d,bond=bond,budget=budget,max_charge=float(raw['statistics'][:,5].max()),max_photons=int(raw['counts'].sum(1).max()),detector_draws=int(raw['statistics'][:,15].sum()),auxiliary_norms=int(raw['statistics'][:,11].sum()),**meta);rows.append(row);save(out,f'row_{name}.json',row);print(row,flush=True)
save(out,'summary.json',dict(completed=True,rows=rows));manifest(out)
