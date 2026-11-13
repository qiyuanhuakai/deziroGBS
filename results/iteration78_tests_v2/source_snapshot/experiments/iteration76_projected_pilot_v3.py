"""Full thin span without numerical-rank truncation; positive QR norm shared by detectors.
Followed by paired old/new records and expanded-input finite pilot runs.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,subprocess,ctypes as C,functools
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_projected_pnr import NativeProjectedPNRSampler
from gbslab.native_prefix_pnr import NativePrefixPNRSampler
from gbslab.native_deep_pnr import NativeDeepPNRSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int)
def ptr(a):return a.ctypes.data_as(DP)
sources=[Path(__file__),ROOT/'src/gbslab/native_projected_pnr.py']+[ROOT/'native'/name for name in ('gaussian_pnr_projected76_v3.cpp','gaussian_projected76_v3.hpp','gaussian_hermite75_v1.hpp','gaussian_pnr_deep75_v1.cpp','gaussian_complex_moments75_v1.hpp','gaussian_positive_norm74_v1.hpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';library=OUT/'projected.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[2]),'-o',str(library)]
save('plan.json',dict(scope=__doc__,command=command,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},input_seed=760001,record_seed_base=0x82000000000000,seed_stride=2**40));
for p in sources:shutil.copy2(p,OUT/p.name)
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));assert build.returncode==0,build.stderr;print('Projected prefix compiled',flush=True)
runtime=[gcc.parent];rng=np.random.default_rng(760001);checks=[];compressions=[];ratechecks=[]
def wick_target(v,t,mu,md,ids,tau):
    V=v[ids];a=np.c_[V.real,-V.imag];b=np.c_[V.imag,V.real];alpha=(a@mu+1j*b@mu)/np.sqrt(2);ad=(a@md+1j*b@md)/np.sqrt(2);den=1-tau*tau*t*t;M=(V*(-t/den))@V.T;N=(V*(tau*t*t/den))@V.conj().T;Md=(V*(2*tau*t**3/den**2))@V.T;Nd=(V*(-t*t*(1+tau*tau*t*t)/den**2))@V.conj().T;center=np.r_[alpha,alpha.conj()];centerd=np.r_[ad,ad.conj()];matrix=np.block([[M,N],[N.T,M.conj()]]);matrixd=np.block([[Md,Nd],[Nd.T,Md.conj()]])
    @functools.lru_cache(None)
    def f(mask):
        if not mask:return 1+0j,0j
        i=(mask&-mask).bit_length()-1;rest=mask&~(1<<i);h,hd=f(rest);value=center[i]*h;derivative=centerd[i]*h+center[i]*hd
        for j in range(2*len(ids)):
            if rest>>j&1:
                h,hd=f(rest&~(1<<j));value+=matrix[i,j]*h;derivative+=matrixd[i,j]*h+matrix[i,j]*hd
        return value,derivative
    return f((1<<(2*len(ids)))-1)
for n in (2,4,8,32,128):
    m=2*n;v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]);t=rng.uniform(.1,.7,n);mu=rng.normal(size=2*n)*.1;md=rng.normal(size=2*n)*.1;np.savez_compressed(OUT/f'input_{n}.npz',v=v,t=t,mean=mu,meanprime=md)
    with NativeProjectedPNRSampler(v,t,library,retained=3,runtime_dirs=runtime) as model,NativeDeepPNRSampler(v,t,ROOT/'results/iteration75_deep_pilot_v1/deep.dll',runtime_dirs=runtime) as old:
        fn=model.lib.prefix76_probe;fn.argtypes=[C.c_void_p,DP,DP,DP,C.c_int,IP,C.c_double,DP,DP,DP,DP,DP,C.c_char_p,C.c_int]
        for k in (0,1,2,4):
            ids=rng.integers(0,m,k,dtype=np.int32)
            for tau in (0.,.2,.8,1.):
                mass=np.zeros(1);prime=np.zeros(1);cov=np.zeros((2*n,2*n));mean=np.zeros(2*n);diag=np.zeros(3);error=C.create_string_buffer(4096);rc=fn(model.handle,ptr(t),ptr(mu),ptr(md),k,ids.ctypes.data_as(IP),tau,ptr(mass),ptr(prime),ptr(cov),ptr(mean),ptr(diag),error,len(error));assert rc==0,error.value;ref,refd=wick_target(v,t,mu,md,ids,tau);scale=max(1,abs(ref),abs(refd));row=dict(inputs=n,k=k,tau=tau,mass_difference=float(abs(mass[0]-ref)),prime_difference=float(abs(prime[0]-refd)),reference_scale=float(scale),active_rank=int(diag[0]),terms=int(diag[1]),span_residual=float(diag[2]));checks.append(row);np.savez_compressed(OUT/f'probe_{n}_{k}_{tau}.npz',ids=ids,mass=mass,prime=prime,reference=ref,reference_prime=refd,covariance=cov,mean=mean,diagnostics=diag);save(f'probe_{n}_{k}_{tau}.json',row);assert max(row['mass_difference'],row['prime_difference'])<1e-10*scale,row
                if tau==1 and k:
                    oldfn=old.lib.pnr74_probe;oldfn.argtypes=[C.c_void_p,DP,DP,C.c_int,IP,DP,DP,DP,C.c_char_p,C.c_int];x=np.diag(np.r_[-t/(1+t),t/(1-t)]);oldcov=np.zeros_like(x);oldmean=np.zeros(2*n);value=np.zeros(1);rc=oldfn(old.handle,ptr(x),ptr(mu),k,ids.ctypes.data_as(IP),ptr(value),ptr(oldcov),ptr(oldmean),error,len(error));assert rc==0,error.value;delta=max(float(abs(cov-oldcov).max()),float(abs(mean-oldmean).max()));row=dict(inputs=n,k=k,maximum_difference=delta);compressions.append(row);np.savez_compressed(OUT/f'compression_{n}_{k}.npz',projected_cov=cov,reference_cov=oldcov,projected_mean=mean,reference_mean=oldmean);save(f'compression_{n}_{k}.json',row);assert delta<1e-9,row
            ratefn=model.lib.prefix76_rates;ratefn.argtypes=[C.c_void_p,DP,DP,C.c_int,IP,DP,C.c_char_p,C.c_int];rates=np.zeros(m);rc=ratefn(model.handle,ptr(t),ptr(mu),k,ids.ctypes.data_as(IP),ptr(rates),error,len(error));assert rc==0,error.value;selected=sorted(set([0,m//2,m-1]));references=[wick_target(v,t,mu,np.zeros_like(mu),np.r_[ids,i].astype(np.int32),1.)[0] for i in selected];delta=float(abs(rates[selected]-references).max());row=dict(inputs=n,k=k,maximum_difference=delta,minimum_rate=float(rates.min()));ratechecks.append(row);np.savez_compressed(OUT/f'rates_{n}_{k}.npz',rates=rates,selected=selected,references=references);save(f'rates_{n}_{k}.json',row);assert delta<1e-9 and rates.min()>=0,row
    print('independent checks passed',n,flush=True)
save('checks.json',dict(passed=True,moments=checks,compressions=compressions,rates=ratechecks));rows=[]
for case in (0,1):
    data=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{case}.npz')
    for k in (3,7,15,31):
        seed=0x82000000000000+(case*4+k)*2**40;values=[]
        for kind,cls,binary in [('old',NativePrefixPNRSampler,ROOT/'results/iteration75_prefix_pilot_v1/prefix.dll'),('projected',NativeProjectedPNRSampler,library)]:
            start=time.perf_counter()
            with cls(data['v'],data['t'],binary,retained=k,workers=16,runtime_dirs=runtime) as model:r=model.sample(4096,seed)
            row=dict(case=case,kind=kind,retained=k,seed=seed,records=4096,wall_seconds=time.perf_counter()-start,statistics=r['statistics']);values.append(r['counts']);rows.append(row);save(f'row_{case}_{k}_{kind}.json',row);np.savez_compressed(OUT/f'counts_{case}_{k}_{kind}.npz',counts=r['counts']);print(case,k,kind,row['wall_seconds'],flush=True)
        different=int(np.any(values[0]!=values[1],axis=1).sum());save(f'replay_{case}_{k}.json',dict(different_records=different));assert different==0,(case,k,different)
save('summary.json',dict(passed=True,rows=rows,checks_passed=True));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
