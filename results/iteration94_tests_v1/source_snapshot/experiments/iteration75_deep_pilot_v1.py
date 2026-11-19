"""Build deep PNR, check derivatives by exact Gauss-Hermite quadrature, then sample."""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,subprocess,ctypes as C,itertools
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_deep_pnr import NativeDeepPNRSampler
from gbslab.native_structured_pnr import NativeStructuredPNRSampler,DP
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(DP)
sources=[Path(__file__),ROOT/'src/gbslab/native_deep_pnr.py']+[ROOT/'native'/n for n in ('gaussian_pnr_deep75_v1.cpp','gaussian_complex_moments75_v1.hpp','gaussian_positive_norm74_v1.hpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
for p in sources:shutil.copy2(p,OUT/p.name)
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';library=OUT/'deep.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[2]),'-o',str(library)]
save('plan.json',dict(source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},command=command,scope=__doc__,seed=751001,depths=[3,5,7],pilot_samples=20000,workers=16))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));assert build.returncode==0,build.stderr;print('Deep PNR built',flush=True)
runtime=[gcc.parent];handles=[os.add_dll_directory(str(p)) for p in runtime];lib=C.CDLL(str(library));lib.deep75_moments.argtypes=[C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];error=C.create_string_buffer(4096);checks=[];rng=np.random.default_rng(751001)
nodes,weights=np.polynomial.hermite.hermgauss(9);nodes=nodes*np.sqrt(2);weights=weights/np.sqrt(np.pi)
for k in range(1,9):
    for rep in range(5):
        mean=np.ascontiguousarray(rng.normal(size=2*k)*.3);factor=rng.normal(size=(2*k,2))*.4;cov=np.ascontiguousarray(factor@factor.T);expected=0.;eg=np.zeros(2*k);eh=np.zeros((2*k,2*k))
        for a,b in itertools.product(range(9),repeat=2):
            x=mean+factor@np.array([nodes[a],nodes[b]]);weight=weights[a]*weights[b];local=(x[::2]**2+x[1::2]**2)/2;expected+=weight*np.prod(local)
            for i in range(2*k):
                others=np.prod(np.delete(local,i//2));eg[i]+=weight*x[i]*others;eh[i,i]+=weight*others
                for j in range(2*k):
                    if i//2!=j//2:eh[i,j]+=weight*x[i]*x[j]*np.prod(np.delete(local,[i//2,j//2]))
        out=np.zeros(1);gradient=np.zeros(2*k);hessian=np.zeros((2*k,2*k));rc=lib.deep75_moments(2*k,ptr(mean),ptr(cov),ptr(out),ptr(gradient),ptr(hessian),error,len(error));delta=max(abs(out[0]-expected),float(abs(gradient-eg).max()),float(abs(hessian-eh).max()));scale=max(1,abs(expected),float(abs(eg).max()),float(abs(eh).max()));row=dict(k=k,rep=rep,returncode=rc,error=error.value.decode() if rc else '',maximum_difference=delta,reference_scale=scale);checks.append(row);np.savez_compressed(OUT/f'quadrature_{k}_{rep}.npz',mean=mean,factor=factor,cov=cov,value=out,gradient=gradient,hessian=hessian,reference_value=expected,reference_gradient=eg,reference_hessian=eh);save(f'quadrature_{k}_{rep}.json',row);assert rc==0 and delta<1e-11*scale,row
save('moment_checks.json',dict(passed=True,rows=checks));print('40 value/gradient/Hessian quadrature checks passed',flush=True)
rows=[];replays=[]
for ci in (0,1):
    data=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{ci}.npz');v,t,cutoff,codes,target=data['v'],data['t'],int(data['cutoff']),data['codes'],data['target'];m=len(v);base=cutoff+1;powers=np.array([base**j for j in range(m)],np.uint64);lookup=np.full(base**m,-1,np.int32);lookup[codes]=np.arange(len(codes),dtype=np.int32)
    for k in range(4):
        outputs=[]
        for cls,binary in ((NativeStructuredPNRSampler,ROOT/'results/iteration74_structured_pilot_v1/structured.dll'),(NativeDeepPNRSampler,library)):
            with cls(v,t,binary,retained=k,workers=16,runtime_dirs=runtime) as model:r=model.sample(1000,751101+100*ci+k)
            outputs.append(r['counts'])
        replay=dict(case=ci,retained=k,different_entries=int(np.count_nonzero(outputs[0]!=outputs[1])));replays.append(replay);np.savez_compressed(OUT/f'replay_{ci}_{k}.npz',old=outputs[0],new=outputs[1]);save(f'replay_{ci}_{k}.json',replay)
    for k in (3,5,7):
        row=dict(case=ci,retained=k,count=20000,seed=751301+100*ci+k);save(f'started_{ci}_{k}.json',row);start=time.perf_counter()
        try:
            with NativeDeepPNRSampler(v,t,library,retained=k,workers=16,runtime_dirs=runtime) as model:r=model.sample(row['count'],row['seed'])
            counts=r['counts'];total=counts.sum(axis=1);inside=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];assert labels.min()>=0;hist=np.bincount(labels,minlength=len(target));row.update(success=True,wall_seconds=time.perf_counter()-start,sampling_seconds=r['wall_seconds'],statistics=r['statistics'],empirical_tv=float(abs(hist/len(counts)-target).sum()/2),odd_records=int(np.count_nonzero(total%2)),mean_photons=float(total.mean()));np.savez_compressed(OUT/f'counts_{ci}_{k}.npz',counts=counts,histogram=hist)
        except Exception as e:row.update(success=False,wall_seconds=time.perf_counter()-start,error=str(e))
        rows.append(row);save(f'row_{ci}_{k}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,all_samples_succeeded=all(r['success'] for r in rows),moment_checks_passed=True,replays=replays,rows=rows));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
