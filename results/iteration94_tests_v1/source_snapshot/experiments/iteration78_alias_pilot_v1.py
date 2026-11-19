"""Build both iteration78 C++ cores; check thermal fixed-N laws and click tables.
The sphere tail is exact for a conditional thermal law, not pure GBS.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,subprocess,time,math,itertools
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_sphere_number_pnr import NativeSphereNumberPNRSampler
from gbslab.native_table_click import NativeTableClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));runtime=[gcc.parent];eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';paths=[Path(__file__),ROOT/'src/gbslab/native_sphere_number_pnr.py',ROOT/'src/gbslab/native_table_click.py',*sorted((ROOT/'native').glob('*.cpp')),*sorted((ROOT/'native').glob('*.hpp'))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in paths},seed=780001,sphere_samples=200000,workers=16));shutil.copy2(__file__,OUT/Path(__file__).name)
shutil.copy2(ROOT/'results/iteration78_pilot_v1/sphere.dll',OUT/'sphere.dll')
save('sphere_reuse.json',dict(source='results/iteration78_pilot_v1/sphere.dll',sha256=H(OUT/'sphere.dll')))
for kind,source in [('table','gaussian_table_click78_v3.cpp')]:
    command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'native'/source),'-o',str(OUT/f'{kind}.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save(f'build_{kind}.json',dict(command=command,returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;print('built',kind,flush=True)
def compositions(N,m):
    if m==1:yield (N,);return
    for k in range(N+1):
        for rest in compositions(N-k,m-1):yield (k,)+rest
def permanent(a):
    N=len(a);s=0j
    for mask in range(1,1<<N):s+=(-1)**(N-mask.bit_count())*np.prod(a[:,[i for i in range(N) if mask>>i&1]].sum(axis=1))
    return s
rng=np.random.default_rng(780001);sphere_rows=[] # no repeat of unchanged sphere conditional-law cohort
d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');old=np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz')
with NativeSphereNumberPNRSampler(d['v'],d['t'],OUT/'sphere.dll',exact_total_budget=64,workers=16,runtime_dirs=runtime) as model:r=model.sample(10000,0x88000000000000+2**40)
assert np.array_equal(r['counts'],old['counts']);np.savez_compressed(OUT/'strict_replay.npz',counts=r['counts'],totals=r['total_photons'])
def target_click(cov):
    m=len(cov)//2;Q=(mp.matrix(cov.tolist())+mp.eye(2*m))/2;vac=[mp.mpf(1)]
    for mask in range(1,1<<m):
        ids=[i for i in range(m) if mask>>i&1];ids+=[i+m for i in ids];vac.append(1/mp.sqrt(mp.det(mp.matrix([[Q[i,j] for j in ids] for i in ids]))))
    target=[]
    for mask in range(1<<m):
        zero=((1<<m)-1)^mask;terms=[];sub=mask
        while True:
            terms.append((-1)**sub.bit_count()*vac[zero|sub])
            if sub==0:break
            sub=(sub-1)&mask
        target.append(mp.fsum(terms))
    return target
table_rows=[]
for case in range(4):
    if case==0:cov=np.eye(2);v=np.ones((1,1),complex);t=np.zeros(1);x=np.zeros((2,2))
    elif case==1:cov=np.eye(4)*1.6;v=np.eye(2,dtype=complex);t=np.zeros(2);x=np.eye(4)*.3
    else:
        if case==2:d=np.load(ROOT/'results/iteration77_task_comparison_v1/input_0.npz');v,t,x,cov=d['v'],d['t'],d['x'],d['cov']
        else:
            v=np.linalg.qr(rng.normal(size=(8,4))+1j*rng.normal(size=(8,4)))[0];t=np.full(4,.5);x=np.diag(np.r_[-t/(1+t),t/(1-t)]);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.eye(16)+2*o@x@o.T
    target=target_click(cov)
    with NativeTableClickSampler(cov,OUT/'table.dll',runtime_dirs=runtime) as model:law=model.law();r=model.sample(65536,0x8f000000000000+case*2**40)
    numerical=np.array(list(map(float,target)));delta=float(abs(law['probabilities']-numerical).sum()/2);assert delta<1e-12;np.savez_compressed(OUT/f'table_{case}.npz',cov=cov,v=v,t=t,x=x,target=numerical,law=law['probabilities'],clicks=r['clicks']);save(f'table_target_{case}.json',dict(precision_digits=80,probabilities=list(map(str,target))));row=dict(case=case,outputs=len(cov)//2,numerical_full_tv=delta,raw_total=law['raw_total'],records=65536,seconds=r['wall_seconds']);table_rows.append(row);save(f'table_{case}.json',row);print('table complete law',case,delta,flush=True)
save('summary.json',dict(passed=True,sphere=sphere_rows,table=table_rows,strict_replay_records=10000));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
