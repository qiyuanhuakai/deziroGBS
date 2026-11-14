"""Paired controls for the waiting-root initial guess; not independent quality data.
All implementations and repeats intentionally use identical domain addresses.
Strictly serial calls; one native worker and one BLAS thread. Failures retained.
"""
import argparse
from iteration94_common_v2 import *
from iteration94_common_v1 import load93 as load_old

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
new,runtime=load93();old,oldruntime=load_old();libs={'old':old,'new':new}
save(out,'backend.json',new._backend)
cases=[]
for j,name in enumerate(('pure4','displaced4','pure8','lossy_tmsv')):
    source=ROOT/f'results/iteration93_reference_v2/input_{j}.npz'
    z=np.load(source);np.savez_compressed(out/f'input_{j}.npz',**{k:z[k] for k in z.files})
    cases.append(dict(name=name,records=64,d=8 if j==2 else 16,bond=32 if j==2 else (64 if j==3 else 128),budget=.02 if j==2 else .01,source=str(source.relative_to(ROOT)),sha256=H(source)))
for sector in (0,1):
    source=ROOT/f'results/iteration93_actual_quality_v1/input_{sector}.npz'
    z=np.load(source);j=len(cases);np.savez_compressed(out/f'input_{j}.npz',**{k:z[k] for k in z.files})
    cases.append(dict(name=f'actual_sector{sector}',records=1,d=6,bond=32,budget=.1 if sector==0 else .01,source=str(source.relative_to(ROOT)),sha256=H(source)))
files=[Path(__file__),ROOT/'native/gaussian_mps_core94_v2.cpp',ROOT/'native/gaussian_mps_core94_v1.cpp',ROOT/'results/iteration94_build_v2/mps.dll',ROOT/'results/iteration94_build_v1/mps.dll']
save(out,'plan.json',dict(scope=__doc__,cases=cases,replicates=3,experiment_base=0x94001000,threads=1,cap=256,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files}))
rows=[];raws={}
for rep in range(3):
    for j,case in enumerate(cases):
        z=np.load(out/f'input_{j}.npz')
        for impl in (('old','new') if rep%2==0 else ('new','old')):
            name=f'{rep}_{j}_{impl}';print('starting',name,case['name'],flush=True)
            raw,meta=sample93(libs[impl],z['u'],z['t'],z['alpha'],case['records'],0x94001000+j,d=case['d'],bond=case['bond'],budget=case['budget'],cap=256,noise=z['noise'])
            np.savez_compressed(out/f'raw_{name}.npz',**raw);raws[(rep,j,impl)]=raw
            row=dict(rep=rep,case=j,name=case['name'],impl=impl,records=case['records'],root_iterations=int(raw['statistics'][:,2].sum()),survival_calls=int(raw['statistics'][:,3].sum()),photons=int(raw['counts'].sum()),max_charge=float(raw['statistics'][:,5].max()),max_residual=float(raw['statistics'][:,4].max()),**meta)
            rows.append(row);save(out,f'row_{name}.json',row);print(row,flush=True)
comparisons=[]
for j,case in enumerate(cases):
    x=raws[0,j,'old'];y=raws[0,j,'new'];same={k:bool(np.array_equal(x[k],y[k])) for k in ('counts','errors')};same['rng_words']=bool(np.array_equal(x['statistics'][:,14:17],y['statistics'][:,14:17]))
    for impl in libs:
        for rep in (1,2):
            for k in x:assert np.array_equal(raws[0,j,impl][k],raws[rep,j,impl][k]),(j,impl,rep,k)
    times={impl:float(np.median([r['seconds'] for r in rows if r['case']==j and r['impl']==impl])) for impl in libs}
    iterations={impl:next(r['root_iterations'] for r in rows if r['case']==j and r['impl']==impl) for impl in libs}
    comparisons.append(dict(name=case['name'],replay=same,median_seconds=times,old_over_new=times['old']/times['new'],root_iterations=iterations,max_charge_difference=float(np.max(abs(x['statistics'][:,5]-y['statistics'][:,5])))))
save(out,'summary.json',dict(completed=True,scope=__doc__,comparisons=comparisons,rows=rows));manifest(out);print(json.dumps(comparisons,indent=2),flush=True)
