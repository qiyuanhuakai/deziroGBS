"""Sequential Hermite prefix: independent moments/tangents, compression and tail-bound pilot."""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,subprocess,ctypes as C,functools,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_prefix_pnr import NativePrefixPNRSampler,DP
from gbslab.native_structured_pnr import NativeStructuredPNRSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(DP)
sources=[Path(__file__),ROOT/'src/gbslab/native_prefix_pnr.py']+[ROOT/'native'/n for n in ('gaussian_pnr_prefix75_v1.cpp','gaussian_hermite75_v1.hpp','gaussian_pnr_deep75_v1.cpp','gaussian_complex_moments75_v1.hpp','gaussian_positive_norm74_v1.hpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
for p in sources:shutil.copy2(p,OUT/p.name)
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';library=OUT/'prefix.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[2]),'-o',str(library)]
save('plan.json',dict(source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},command=command,scope=__doc__,probe_seed=756001,sample_seed_base=0x7b000000000000,seed_stride=2**40,depths=[7,15,23,31],pilot_samples=20000,workers=16,bound_scope='TV <= probability of reaching first compression, in real arithmetic with ideal random draws; no numerical/PRNG certification.'))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));assert build.returncode==0,build.stderr;print('Prefix PNR built',flush=True)
runtime=[gcc.parent];rng=np.random.default_rng(756001);checks=[];high=[];compressions=[]
def configure(model):
    fn=model.lib.prefix75_probe;fn.argtypes=[C.c_void_p,DP,DP,DP,C.c_int,C.POINTER(C.c_int),C.c_double,DP,DP,DP,DP,C.c_char_p,C.c_int];return fn
def probe(model,mu,md,ids,tau):
    ids=np.ascontiguousarray(ids,dtype=np.int32);mass=np.zeros(1);prime=np.zeros(1);cov=np.zeros((len(mu),len(mu)));mean=np.zeros(len(mu));error=C.create_string_buffer(4096);rc=configure(model)(model.handle,ptr(model.squeezing),ptr(mu),ptr(md),len(ids),ids.ctypes.data_as(C.POINTER(C.c_int)),tau,ptr(mass),ptr(prime),ptr(cov),ptr(mean),error,len(error));assert rc==0,error.value;return mass[0],prime[0],cov,mean
for rep in range(5):
    n=2;m=4;v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]);t=rng.uniform(.1,.7,n);mu=rng.normal(size=2*n)*.2;md=rng.normal(size=2*n)*.1;a=np.c_[v.real,-v.imag];b=np.c_[v.imag,v.real]
    with NativePrefixPNRSampler(v,t,library,runtime_dirs=runtime) as model,NativeStructuredPNRSampler(v,t,ROOT/'results/iteration74_structured_pilot_v1/structured.dll',runtime_dirs=runtime) as old:
        for k in range(1,5):
            ids=rng.integers(0,m,k,dtype=np.int32);V=v[ids];alpha=(a[ids]@mu+1j*b[ids]@mu)/np.sqrt(2);ad=(a[ids]@md+1j*b[ids]@md)/np.sqrt(2)
            for tau in (0.,.2,.8,1.):
                mass,prime,cov,mean=probe(model,mu,md,ids,tau);den=1-tau*tau*t*t;M=(V*(-t/den))@V.T;N=(V*(tau*t*t/den))@V.conj().T;Md=(V*(2*tau*t**3/den**2))@V.T;Nd=(V*(-t*t*(1+tau*tau*t*t)/den**2))@V.conj().T;center=np.r_[alpha,alpha.conj()];centerd=np.r_[ad,ad.conj()];matrix=np.block([[M,N],[N.T,M.conj()]]);matrixd=np.block([[Md,Nd],[Nd.T,Md.conj()]])
                @functools.lru_cache(None)
                def wick(mask):
                    if not mask:return 1+0j,0j
                    i=(mask&-mask).bit_length()-1;rest=mask&~(1<<i);h,hd=wick(rest);value=center[i]*h;derivative=centerd[i]*h+center[i]*hd
                    for j in range(2*k):
                        if rest>>j&1:
                            h,hd=wick(rest&~(1<<j));value+=matrix[i,j]*h;derivative+=matrixd[i,j]*h+matrix[i,j]*hd
                    return value,derivative
                ref,refd=wick((1<<(2*k))-1);row=dict(rep=rep,k=k,tau=tau,mass=mass,prime=prime,mass_difference=float(abs(mass-ref)),derivative_difference=float(abs(prime-refd)));checks.append(row);np.savez_compressed(OUT/f'norm_{rep}_{k}_{tau}.npz',v=v,t=t,mu=mu,muprime=md,modes=ids,mass=mass,prime=prime,reference=ref,reference_prime=refd);assert max(row['mass_difference'],row['derivative_difference'])<1e-10*max(1,abs(ref),abs(refd)),row
                if tau==1:
                    oldfn=old.lib.pnr74_probe;oldfn.argtypes=[C.c_void_p,DP,DP,C.c_int,C.POINTER(C.c_int),DP,DP,DP,C.c_char_p,C.c_int];x=np.diag(np.r_[-t/(1+t),t/(1-t)]);out=np.zeros(1);oldcov=np.zeros_like(x);oldmean=np.zeros_like(mu);error=C.create_string_buffer(4096);rc=oldfn(old.handle,ptr(x),ptr(mu),k,ids.ctypes.data_as(C.POINTER(C.c_int)),ptr(out),ptr(oldcov),ptr(oldmean),error,len(error));delta=max(float(abs(cov-oldcov).max()),float(abs(mean-oldmean).max()));compressions.append(dict(rep=rep,k=k,returncode=rc,maximum_difference=delta));np.savez_compressed(OUT/f'compression_{rep}_{k}.npz',new_cov=cov,old_cov=oldcov,new_mean=mean,old_mean=oldmean);assert rc==0 and delta<1e-10,(rc,error.value,delta)
with NativePrefixPNRSampler(np.ones((1,1),complex),[1/3],library,retained=31,runtime_dirs=runtime) as model:
    t=mp.mpf(float(1/3))
    for k in (8,15,16,23,24,31,32):
        for tau in (0.,.25,1.):
            mass,prime,_,_=probe(model,np.zeros(2),np.zeros(2),np.zeros(k,np.int32),tau);z=mp.mpf(tau);den=1-z*z*t*t;M=-t/den;N=z*t*t/den;Md=2*z*t**3/den**2;Nd=-t*t*(1+z*z*t*t)/den**2;ref=mp.mpf(0);refd=mp.mpf(0)
            for p in range(k//2+1):
                q=k-2*p;c=mp.factorial(k)**2/(mp.factorial(q)*(2**p*mp.factorial(p))**2);ref+=c*M**(2*p)*N**q
                if p:refd+=c*2*p*M**(2*p-1)*Md*N**q
                if q:refd+=c*q*M**(2*p)*N**(q-1)*Nd
            row=dict(k=k,tau=tau,mass=mass,prime=prime,mass_reference=str(ref),prime_reference=str(refd),relative_mass_error=float(abs(mp.mpf(mass)-ref)/max(1,abs(ref))),relative_prime_error=float(abs(mp.mpf(prime)-refd)/max(1,abs(refd))));high.append(row);assert max(row['relative_mass_error'],row['relative_prime_error'])<1e-11,row
            if tau==0 and k%2:assert mass==0
save('probe_checks.json',dict(passed=True,moments=checks,compressions=compressions,high_degree=high));print('80 tangent, 20 compression and 21 high-degree checks passed',flush=True)
rows=[]
for ci in (0,1):
    data=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{ci}.npz');v,t,cutoff,codes,target=data['v'],data['t'],int(data['cutoff']),data['codes'],data['target'];m=len(v);base=cutoff+1;powers=np.array([base**j for j in range(m)],np.uint64);lookup=np.full(base**m,-1,np.int32);lookup[codes]=np.arange(len(codes),dtype=np.int32)
    for ki,k in enumerate((7,15,23,31)):
        sector=[mp.mpf(1)]
        for tj in t:
            tj=mp.mpf(float(tj));local=[mp.mpf(0)]*(k+1);local[0]=mp.sqrt(1-tj*tj)
            for j in range(1,k//2+1):local[2*j]=local[2*j-2]*mp.mpf(2*j-1)/(2*j)*tj*tj
            sector=[sum(sector[j]*local[i-j] for j in range(min(i,len(sector)-1)+1)) for i in range(k+1)]
        tail=1-sum(sector);row=dict(case=ci,retained=k,count=20000,seed=0x7b000000000000+(ci*4+ki)*2**40,ideal_prefix_tv_upper=float(tail),tail_decimal=str(tail));save(f'started_{ci}_{k}.json',row);start=time.perf_counter()
        try:
            with NativePrefixPNRSampler(v,t,library,retained=k,workers=16,runtime_dirs=runtime) as model:r=model.sample(row['count'],row['seed'])
            counts=r['counts'];total=counts.sum(axis=1);inside=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];assert labels.min()>=0;hist=np.bincount(labels,minlength=len(target));row.update(success=True,wall_seconds=time.perf_counter()-start,sampling_seconds=r['wall_seconds'],statistics=r['statistics'],empirical_tv=float(abs(hist/len(counts)-target).sum()/2),odd_records=int(np.count_nonzero(total%2)),mean_photons=float(total.mean()));np.savez_compressed(OUT/f'counts_{ci}_{k}.npz',counts=counts,histogram=hist)
        except Exception as e:row.update(success=False,wall_seconds=time.perf_counter()-start,error=str(e))
        rows.append(row);save(f'row_{ci}_{k}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,all_samples_succeeded=all(r['success'] for r in rows),probe_checks_passed=True,rows=rows));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
