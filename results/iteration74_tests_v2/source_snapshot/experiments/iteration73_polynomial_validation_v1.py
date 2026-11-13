"""Independent complex Gauss-Hermite moments, coherent Poisson and reset replay."""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,argparse,sys,itertools,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
from gbslab.native_gaussian_reset_parallel import NativeParallelGaussianResetSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
library=ROOT/'results/iteration73_polynomial_pilot_v2/polynomial.dll';runtime=[Path(shutil.which('g++')).parent];checks=[]
save('plan.json',dict(source_sha256=H(Path(__file__)),library_sha256=H(library),wrapper_sha256=H(ROOT/'src/gbslab/native_polynomial_pnr.py'),quadrature_order=6,scope='Direct quadrature of degree <=10 polynomials, with complex covariance square root for indefinite Gaussian P moment functional; no sampling from a negative P distribution. Hermite order6 is algebraically exact for these polynomials. Numerical errors are measured, not rigorously enclosed.'))
nodes,weights=np.polynomial.hermite_e.hermegauss(6);weights/=np.sqrt(2*np.pi);ids=np.array(list(itertools.product(range(6),repeat=4)));points=nodes[ids];quadrature_weights=np.prod(weights[ids],axis=1)
for case in range(5):
    data=np.load(ROOT/f'results/iteration73_polynomial_pilot_v2/input_{case}.npz');v=data['v'];x=data['x'];w=np.random.default_rng(734001+case).normal(size=4)*.2;ev,u=np.linalg.eigh(x);root=(u*np.sqrt(ev.astype(complex)))@u.T;z=w+points@root.T;a=np.c_[v.real,-v.imag];b=np.c_[v.imag,v.real]
    with NativePolynomialPNRSampler(v,x,library,mean=w,runtime_dirs=runtime) as model:
        model.lib.pnr74_probe.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.POINTER(C.c_double),C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_double),C.c_char_p,C.c_int]
        for modes in ([0],[1,1],[0,1],[0,1,2],[1,1,2,3],[0,0,0,0]):
            f=np.ones(len(z),complex)
            for mode in modes:f*=((z@a[mode])**2+(z@b[mode])**2)/2
            mass=np.dot(quadrature_weights,f);mu=np.einsum('s,s,si->i',quadrature_weights,f,z)/mass;cov=np.einsum('s,s,si,sj->ij',quadrature_weights,f,z,z)/mass-np.outer(mu,mu)
            out=np.zeros((4,4));outmean=np.zeros(4);value=np.zeros(1);indices=np.array(modes,np.int32);error=C.create_string_buffer(4096);rc=model.lib.pnr74_probe(model.handle,ptr(x),ptr(w),len(modes),indices.ctypes.data_as(C.POINTER(C.c_int)),ptr(value),ptr(out),ptr(outmean),error,len(error))
            row=dict(case=case,modes=modes,returncode=rc,error=error.value.decode() if rc else '',normalization_error=float(abs(value[0]-mass)/max(1,abs(mass))),mean_error=float(abs(outmean-mu).max()),covariance_error=float(abs(out-cov).max()),imaginary_residual=float(max(abs(mu.imag).max(),abs(cov.imag).max(),abs(mass.imag))))
            checks.append(row);save(f'probe_{case}_{len(checks)}.json',row);assert rc==0 and max(row[k] for k in ('normalization_error','mean_error','covariance_error','imaginary_residual'))<1e-9,row
            np.savez_compressed(OUT/f'probe_{case}_{len(checks)}.npz',v=v,x=x,mean=w,modes=indices,native_x=out,native_mean=outmean,native_mass=value,quadrature_x=cov,quadrature_mean=mu,quadrature_mass=mass)
coherent=[]
for retained in (0,1,2,3):
    with NativePolynomialPNRSampler(np.ones((1,1),complex),np.zeros((2,2)),library,mean=[1,0],retained=retained,workers=16,runtime_dirs=runtime) as model:r=model.sample(200000,734101)
    counts=r['counts'][:,0];hist=np.bincount(counts,minlength=15)/len(counts);target=np.array([np.exp(-.5)*.5**k/math.factorial(k) for k in range(len(hist))]);tv=float(abs(hist-target).sum()/2);assert tv<.005;coherent.append(dict(retained=retained,tv=tv,statistics=r['statistics']));np.savez_compressed(OUT/f'coherent_{retained}.npz',counts=counts,histogram=hist,target=target)
replays=[]
for case in (3,4):
    data=np.load(ROOT/f'results/iteration73_polynomial_pilot_v2/input_{case}.npz')
    with NativePolynomialPNRSampler(data['v'],data['x'],library,noise_variance=data['variance'],retained=0,workers=16,runtime_dirs=runtime) as model:r=model.sample(4096,734200+case)
    with NativeParallelGaussianResetSampler(data['v'],data['x'],ROOT/'results/iteration70_build/gaussian_reset_parallel.dll',noise_variance=data['variance'],workers=16,runtime_dirs=runtime) as model:old=model.sample(4096,734200+case,selector='eigen')
    mismatches=int(np.count_nonzero(r['counts']!=old['counts']));replays.append(dict(case=case,mismatched_entries=mismatches));np.savez_compressed(OUT/f'reset_replay_{case}.npz',new=r['counts'],old=old['counts']);assert mismatches==0,replays[-1]
save('summary.json',dict(passed=True,moment_checks=checks,coherent_checks=coherent,reset_replays=replays));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}));print(json.dumps(dict(passed=True,moment_checks=len(checks),maximum_moment_error=max(max(r['normalization_error'],r['mean_error'],r['covariance_error']) for r in checks),coherent=coherent,replays=replays)),flush=True)
