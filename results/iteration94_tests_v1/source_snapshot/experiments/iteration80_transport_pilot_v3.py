"""Use smooth calibrated native latent-pair PNR and compare complete independent small laws.
LP table creation is separate and explicitly timed; it is not a universal method.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,subprocess,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_transport_alias_pnr import NativeTransportAliasPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));runtime=[gcc.parent];eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';sources=[Path(__file__),ROOT/'src/gbslab/native_transport_alias_pnr.py',*sorted((ROOT/'native').glob('*.cpp')),*sorted((ROOT/'native').glob('*.hpp'))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},workers=16,records_per_fixed_total=100000,totals=[2,4,8,14],seed_base=0xa6000000000000));shutil.copy2(__file__,OUT/Path(__file__).name)
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'native/gaussian_pnr_transport80_v2.cpp'),'-o',str(OUT/'transport.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;print('transport alias built',flush=True)
d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');base=int(d['cutoff'])+1;states=np.array([(d['codes']//base**j)%base for j in range(4)]).T;totals=states.sum(axis=1);table=np.load(ROOT/'results/iteration80_ipf_pilot_v1/strong4.npz');v,t=d['v'],d['t'];couplings=table['couplings'];weights=table['weights'];m,n=v.shape
def complete(N):
    combined={}
    for weight,P in zip(weights,couplings):
        pairs=[]
        for i in range(m):
            for j in range(i,m):
                probability=float(P[i,j]*(1 if i==j else 2))
                if probability>0:
                    count=[0]*m;count[i]+=1;count[j]+=1;pairs.append((tuple(count),probability))
        law={(0,)*m:1.}
        for _ in range(N//2):
            new={}
            for x,p0 in law.items():
                for y,q in pairs:key=tuple(a+b for a,b in zip(x,y));new[key]=new.get(key,0.)+p0*q
            law=new
        for x,p0 in law.items():combined[x]=combined.get(x,0.)+weight*p0
    return combined
rows=[]
for panel,N in enumerate((2,4,8,14)):
    law=complete(N);mask=totals==N;patterns=states[mask];pure=d['target'][:-1][mask];pure=pure/pure.sum();prob=np.array([law.get(tuple(x),0.) for x in patterns]);lookup={tuple(x):i for i,x in enumerate(patterns)};assert abs(prob.sum()-1)<1e-9
    with NativeTransportAliasPNRSampler(v,t,OUT/'transport.dll',couplings=couplings,latent_weights=weights,exact_total_budget=0,workers=16,runtime_dirs=runtime) as model:r=model.sample(100000,0xa6000000000000+panel*2**40,fixed_total=N);alias_bound=model.alias_latent_tv+N/2*model.alias_maximum_pair_tv;np.savez_compressed(OUT/f'alias_{N}.npz',weights=model.actual_latent_weights,pairs=model.actual_pair_masses,bound=alias_bound)
    hist=np.bincount([lookup[tuple(x)] for x in r['counts']],minlength=len(patterns));empirical=float(abs(hist/100000-prob).sum()/2);assert empirical<.04 and np.all(r['counts'].sum(axis=1)==N);x=patterns.astype(float);pair=(x.T@(x*prob[:,None])-np.diag(prob@x))/(N*(N-1));target_pair=(table['Q']+(N-2)*table['R'])/(N-1);pair_error=float(abs(pair-target_pair).max());assert pair_error<1e-9
    # Independent exact polynomial quadratures for complex/real sphere controls.
    from scipy.special import gammaln
    zz,ww=np.polynomial.legendre.leggauss(N+1);phi=2*np.pi*np.arange(2*N+2)/(2*N+2)
    directions=np.array([[np.sqrt((1+z)/2),np.sqrt((1-z)/2)*np.exp(1j*f)] for z in zz for f in phi]);quadrature=np.repeat(ww/2,len(phi))/len(phi)
    intensities=abs(directions@v.T)**2;factor=np.exp(gammaln(N+1)-gammaln(patterns+1).sum(axis=1));complex_law=np.zeros(len(patterns))
    for p0,weight in zip(intensities,quadrature):complex_law+=weight*factor*np.prod(p0[None,:]**patterns,axis=1)
    real_law=np.zeros(len(patterns));real_intensities=abs(table['directions']@v.T)**2
    for p0,weight in zip(real_intensities,weights):real_law+=weight*factor*np.prod(p0[None,:]**patterns,axis=1)
    assert abs(complex_law.sum()-1)<1e-11 and abs(real_law.sum()-1)<1e-11
    np.savez_compressed(OUT/f'controls_{N}.npz',states=patterns,pure=pure,transport=prob,complex_sphere=complex_law,real_sphere=real_law)
    row=dict(total=N,records=100000,empirical_tv_to_model=empirical,numerical_full_tv_to_pure=float(abs(prob-pure).sum()/2),maximum_pair_error=pair_error,ideal_uint64_alias_tv_bound=alias_bound,complex_sphere_tv=float(abs(complex_law-pure).sum()/2),real_sphere_tv=float(abs(real_law-pure).sum()/2),seconds=r['wall_seconds'],statistics=r['statistics']);rows.append(row);np.savez_compressed(OUT/f'law_{N}.npz',states=patterns,law=prob,pure=pure,counts=r['counts'],histogram=hist,pair=pair,target_pair=target_pair);save(f'law_{N}.json',row);print(row,flush=True)
with NativeTransportAliasPNRSampler(v,t,OUT/'transport.dll',couplings=couplings,latent_weights=weights,exact_total_budget=64,workers=16,runtime_dirs=runtime) as model:r=model.sample(10000,0x88000000000000+2**40)
assert np.array_equal(r['counts'],np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz')['counts']);np.savez_compressed(OUT/'strict_replay.npz',counts=r['counts'],totals=r['total_photons'])
extra=[]
for label in ('identity2','balanced_phase2','random_3_5_0','random_4_8_0'):
    d=np.load(ROOT/f'results/iteration80_ipf_pilot_v1/{label}.npz')
    with NativeTransportAliasPNRSampler(d['v'],.4,OUT/'transport.dll',couplings=d['couplings'],latent_weights=d['weights'],exact_total_budget=0,workers=16,runtime_dirs=runtime) as model:r=model.sample(100000,0xa7000000000000+len(extra)*2**40,fixed_total=14)
    x=r['counts'].astype(float);pair=(x.T@x-np.diag(x.sum(axis=0)))/(len(x)*14*13);target=(d['Q']+12*d['R'])/13;row=dict(label=label,maximum_empirical_pair_error=float(abs(pair-target).max()),seconds=r['wall_seconds']);assert row['maximum_empirical_pair_error']<.006;extra.append(row);save(label+'.json',row);np.savez_compressed(OUT/(label+'.npz'),counts=r['counts'],pair=pair,target_pair=target);print(row,flush=True)
save('summary.json',dict(passed=True,full_laws=rows,extra=extra));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
