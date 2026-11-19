"""Independent Williamson/Schur decomposition checks and repeated-resplit PNR pilot."""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,subprocess,ctypes as C
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import schur
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_resplit_pnr import NativeResplitPNRSampler,DP
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(DP)
sources=[Path(__file__),ROOT/'src/gbslab/native_resplit_pnr.py']+[ROOT/'native'/n for n in ('gaussian_pnr_resplit75_v1.cpp','gaussian_resplit75_v1.hpp','gaussian_pnr_deep75_v1.cpp','gaussian_complex_moments75_v1.hpp','gaussian_positive_norm74_v1.hpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
for p in sources:shutil.copy2(p,OUT/p.name)
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';library=OUT/'resplit.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[2]),'-o',str(library)]
save('plan.json',dict(source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},command=command,scope=__doc__,probe_seed=753001,sample_seed_base=0x78000000000000,seed_stride=2**40,depths=[0,3,5,7],pilot_samples=20000,workers=16))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));assert build.returncode==0,build.stderr;print('Resplit PNR built',flush=True)
runtime=[gcc.parent];handles=[os.add_dll_directory(str(p)) for p in runtime];lib=C.CDLL(str(library));lib.split75_probe.argtypes=[C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];error=C.create_string_buffer(4096);checks=[];rng=np.random.default_rng(753001)
for n in (1,2,4,8,50):
    for rep in range(5):
        q=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0];o=np.block([[q.real,-q.imag],[q.imag,q.real]]);r=rng.uniform(.05,1.1,n);nu=rng.uniform(1.2,3,n);v=(o*np.r_[nu*np.exp(-2*r),nu*np.exp(2*r)])@o.T;raw=rng.normal(size=(2*n,2*n))*.05/np.sqrt(n);v+=raw@raw.T;x=np.ascontiguousarray((v-np.eye(2*n))/2);rotation=np.zeros_like(x);t=np.zeros(n);noise=np.zeros_like(x);res=np.zeros(1);rc=lib.split75_probe(n,ptr(x),ptr(rotation),ptr(t),ptr(noise),ptr(res),error,len(error));assert rc==0,(n,rep,rc,error.value)
        omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);values,vectors=np.linalg.eigh(v);root=(vectors*np.sqrt(values))@vectors.T;a=root@omega@root;tri,O=schur((a-a.T)/2,output='real');signed=np.array([tri[2*j,2*j+1] for j in range(n)]);S=root@np.c_[O[:,::2],O[:,1::2]*np.sign(signed)]/np.sqrt(np.r_[abs(signed),abs(signed)]);reference=S@S.T;pure=rotation@np.diag(np.r_[(1-t)/(1+t),(1+t)/(1-t)])@rotation.T;reconstructed=pure+2*noise@noise.T
        row=dict(n=n,rep=rep,returncode=rc,williamson_difference=float(abs(pure-reference).max()),covariance_difference=float(abs(reconstructed-v).max()),purity_residual=float(abs(pure@omega@pure-omega).max()),rotation_residual=float(abs(rotation@omega@rotation.T-omega).max()),native_residual=res[0]);checks.append(row);np.savez_compressed(OUT/f'split_{n}_{rep}.npz',x=x,rotation=rotation,t=t,noise_factor=noise,pure=pure,reference=reference,reconstructed=reconstructed);save(f'split_{n}_{rep}.json',row);assert max(row[k] for k in ('williamson_difference','covariance_difference','purity_residual','rotation_residual'))<1e-9,row
save('decomposition_checks.json',dict(passed=True,rows=checks));print('25 independent Williamson checks passed, through 50 inputs',flush=True)
rows=[]
for ci in (0,1):
    data=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{ci}.npz');v,t,cutoff,codes,target=data['v'],data['t'],int(data['cutoff']),data['codes'],data['target'];m=len(v);base=cutoff+1;powers=np.array([base**j for j in range(m)],np.uint64);lookup=np.full(base**m,-1,np.int32);lookup[codes]=np.arange(len(codes),dtype=np.int32)
    for ki,k in enumerate((0,3,5,7)):
        row=dict(case=ci,retained=k,count=20000,seed=0x78000000000000+(ci*4+ki)*2**40);save(f'started_{ci}_{k}.json',row);start=time.perf_counter()
        try:
            with NativeResplitPNRSampler(v,t,library,retained=k,workers=16,runtime_dirs=runtime) as model:r=model.sample(row['count'],row['seed'])
            counts=r['counts'];total=counts.sum(axis=1);inside=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];assert labels.min()>=0;hist=np.bincount(labels,minlength=len(target));row.update(success=True,wall_seconds=time.perf_counter()-start,sampling_seconds=r['wall_seconds'],statistics=r['statistics'],empirical_tv=float(abs(hist/len(counts)-target).sum()/2),odd_records=int(np.count_nonzero(total%2)),mean_photons=float(total.mean()));np.savez_compressed(OUT/f'counts_{ci}_{k}.npz',counts=counts,histogram=hist)
        except Exception as e:row.update(success=False,wall_seconds=time.perf_counter()-start,error=str(e))
        rows.append(row);save(f'row_{ci}_{k}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,all_samples_succeeded=all(r['success'] for r in rows),decomposition_checks_passed=True,rows=rows));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
