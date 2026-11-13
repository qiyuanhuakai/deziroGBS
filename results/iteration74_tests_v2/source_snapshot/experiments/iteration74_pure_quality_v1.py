"""Independent pure-state Fock targets, new samples, and old failure replay."""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,subprocess,ctypes as C,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=90
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_structured_pnr import NativeStructuredPNRSampler,DP
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(DP)
gcc=Path(shutil.which('g++'));runtime=[gcc.parent];binary=ROOT/'results/iteration74_structured_pilot_v1/structured.dll';source=ROOT/'native/gaussian_pnr_fock_reference74_v1.cpp';library=OUT/'reference.dll';N=2000000;radius=math.sqrt(math.log(1200)/(2*N))
save('plan.json',dict(source_sha256=H(source),driver_sha256=H(Path(__file__)),structured_binary_sha256=H(binary),samples_per_law=N,laws=6,one_sided_bounds=12,joint_failure=.01,radius=radius,seed_base=0x76000000000000,stride=2**40,pilot_seed=743001,pilot_count=20000,scope=__doc__))
shutil.copy2(source,OUT/source.name);shutil.copy2(__file__,OUT/Path(__file__).name);command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math',str(source),'-o',str(library)];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
handles=[os.add_dll_directory(str(p)) for p in runtime];ref=C.CDLL(str(library));ref.fock74_reference.argtypes=[C.c_int,C.c_int,DP,C.c_double,C.c_uint64,C.POINTER(C.c_uint64),DP,C.c_char_p,C.c_int];ref.fock74_reference.restype=C.c_int64;error=C.create_string_buffer(4096)
rng=np.random.default_rng(743000);q,_=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)));cases=[(np.ones((1,1),complex),np.array([1/3]),128),(q[:,:2],np.full(2,np.tanh(1.1)),64)];rows=[];failures=[]
for ci,(v,t,cutoff) in enumerate(cases):
    m,n=v.shape;matrix=np.ascontiguousarray(-(v*t)@v.T);p0=float(np.sqrt(np.prod(1-t*t)));capacity=math.comb(cutoff+m,m);codes=np.empty(capacity,np.uint64);target=np.zeros(capacity);count=ref.fock74_reference(m,cutoff,ptr(matrix),p0,capacity,codes.ctypes.data_as(C.POINTER(C.c_uint64)),ptr(target),error,len(error));assert count==capacity,error.value
    # Independent total-number convolution: passive V preserves total count.
    sector=np.array([1.]);
    for value in t:
        local=np.zeros(cutoff+1);local[0]=np.sqrt(1-value*value)
        for k in range(1,cutoff//2+1):local[2*k]=local[2*k-2]*(2*k-1)/(2*k)*value*value
        sector=np.convolve(sector,local)[:cutoff+1]
    base=cutoff+1;powers=np.array([base**j for j in range(m)],np.uint64);totals=sum((codes//powers[j])%base for j in range(m));measured=np.bincount(totals.astype(int),weights=target,minlength=cutoff+1);assert np.max(abs(measured-sector))<1e-12
    exact_sector=[mp.mpf(1)]
    for value in t:
        value=mp.mpf(float(value));local=[mp.mpf(0)]*(cutoff+1);local[0]=mp.sqrt(1-value*value)
        for k in range(1,cutoff//2+1):local[2*k]=local[2*k-2]*mp.mpf(2*k-1)/(2*k)*value*value
        exact_sector=[sum(exact_sector[j]*local[i-j] for j in range(min(i,len(exact_sector)-1)+1)) for i in range(cutoff+1)]
    tail=float(1-sum(exact_sector));assert tail>0;target=np.r_[target,tail];lookup={int(code):i for i,code in enumerate(codes)};np.savez_compressed(OUT/f'input_{ci}.npz',v=v,t=t,cutoff=cutoff,codes=codes,target=target,total_sector=sector)
    def histogram(counts):
        total=counts.sum(axis=1);inside=total<=cutoff;packed=counts[inside].astype(np.uint64)@powers;hist=np.zeros(len(target),np.int64);unique,freq=np.unique(packed,return_counts=True)
        for code,frequency in zip(unique,freq):hist[lookup[int(code)]]=frequency
        hist[-1]=int((~inside).sum());return hist
    for ki,k in enumerate((0,1,3)):
        x=np.diag(np.r_[-t/(1+t),t/(1-t)])
        if ci==0:
            with NativePolynomialPNRSampler(v,x,ROOT/'results/iteration73_polynomial_pilot_v2/polynomial.dll',retained=k,workers=16,runtime_dirs=runtime) as old:
                try:oldr=old.sample(100000,736001);oldrow=dict(retained=k,success=True,odd=int(np.count_nonzero(oldr['counts'].sum(axis=1)%2)))
                except Exception as e:oldrow=dict(retained=k,success=False,error=str(e))
            failures.append(oldrow);save(f'old_{k}.json',oldrow)
        with NativeStructuredPNRSampler(v,t,binary,retained=k,workers=16,runtime_dirs=runtime) as model:
            pilot=model.sample(20000,743001+100*ci+k)['counts'];prior=histogram(pilot);event=prior/len(pilot)>target;np.savez_compressed(OUT/f'pilot_{ci}_{k}.npz',counts=pilot,histogram=prior,fixed_event=event);hist=np.zeros(len(target),np.int64);seconds=0;photons=odd=maximum=0
            for batch in range(20):
                seed=0x76000000000000+(ci*60+ki*20+batch)*2**40;save(f'started_{ci}_{k}_{batch}.json',dict(seed=seed,count=100000));r=model.sample(100000,seed);counts=r['counts'];hist+=histogram(counts);total=counts.sum(axis=1);photons+=int(total.sum());odd+=int(np.count_nonzero(total%2));maximum=max(maximum,int(total.max()));seconds+=r['wall_seconds'];np.savez_compressed(OUT/f'counts_{ci}_{k}_{batch}.npz',counts=counts);save(f'status_{ci}_{k}_{batch}.json',dict(seed=seed,seconds=r['wall_seconds'],statistics=r['statistics']))
        empirical=hist/N;tv=float(abs(empirical-target).sum()/2);lower=max(0.,float((empirical[event]-target[event]).sum())-radius);upper=min(1.,tv+radius+tail);row=dict(case=ci,retained=k,records=N,empirical_tv=tv,nominal_full_tv_interval=[lower,upper],odd_records=odd,maximum_photons=maximum,mean_photons=photons/N,seconds=seconds,target_tail=tail,target_mean=float(np.sum(t*t/(1-t*t))));rows.append(row);save(f'row_{ci}_{k}.json',row);np.savez_compressed(OUT/f'histogram_{ci}_{k}.npz',histogram=hist,fixed_event=event);print(row,flush=True)
save('summary.json',dict(passed=True,rows=rows,old_failure_replays=failures,radius=radius));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
