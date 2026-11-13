"""Six predeclared input-only detector orders with unchanged old CPU/GPU kernels.
Map all output bits back to physical ports; compare complete small laws.
Construction and exploratory sampling times are separate, not final benchmarks.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,subprocess,time,ctypes as C
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.native_click_branches import NativeBranchClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));runtime=[gcc.parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(p)) for p in runtime];eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';source=ROOT/'native/gaussian_click_order80_v1.cpp';methods=['original','click_ascending','click_descending','spectral','neighbor','degree_ascending'];cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';gpu=ROOT/'results/iteration74_fused_pilot_v2/philox.dll';branch=ROOT/'results/iteration71_branch_pilot_v1/branches.dll';save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),source,cpu,gpu,branch]},methods=methods,budgets=[4,8],seed_base=0xa8000000000000))
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(source),'-o',str(OUT/'order.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
lib=C.CDLL(str(OUT/'order.dll'));DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int32);fn=lib.order80_build;fn.argtypes=[C.c_int,DP,IP,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;ptr=lambda a:a.ctypes.data_as(DP);rows=[]
for case in (0,1,3,4):
    path=ROOT/f'results/iteration79_pilot_v1/{"click_input" if case<2 else "large_input"}_{case}.npz'
    d=np.load(path);v,x=d['v'],d['x'];m,n=v.shape;o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);orders=np.zeros((6,m),np.int32);vacuum=np.zeros(m);correlations=np.zeros((m,m));message=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(m,ptr(cov),orders.ctypes.data_as(IP),ptr(vacuum),ptr(correlations),message,len(message));construction=time.perf_counter()-start;assert rc==0,message.value
    assert all(np.array_equal(np.sort(order),np.arange(m)) for order in orders);np.savez_compressed(OUT/f'input_{case}.npz',v=v,x=x,cov=cov,orders=orders,vacuum=vacuum,correlations=correlations);save(f'ordering_{case}.json',dict(case=case,construction_seconds=construction,methods=methods))
    for method,order in zip(methods,orders):
        for B in ((4,8) if case<2 else (4,)):
            try:
                with NativePhiloxClickSampler(v[order],x,cpu,gpu_library=gpu,branch_budget=B,runtime_dirs=runtime) as model:
                    r=model.sample(8192 if case<2 else 64,0xa8000000000000+case*2**40);bits=np.zeros_like(r['clicks']);bits[:,order]=r['clicks'];row=dict(success=True,case=case,method=method,budget=B,seconds=r['wall_seconds'],statistics=r['statistics'],cuda_timings=r['cuda_timings']);arrays=dict(clicks=bits)
                    if case<2:
                        patterns=np.array([[(s>>j)&1 for j in range(m)] for s in range(1<<m)],np.uint8);forced=np.where(patterns[:,order],0.,np.nextafter(1.,0.));replay=model.replay(np.zeros((len(patterns),2*n)),forced);assert np.array_equal(replay['clicks'],patterns[:,order]);prob=replay['conditional_click_probabilities'];law=np.prod(np.where(patterns[:,order],prob,1-prob),axis=1)
                        with NativeBranchClickSampler(v[order],x,branch,branch_budget=B,runtime_dirs=runtime) as reference:cpu_law=np.array([reference.prefix_probability(bits[order])['probability'] for bits in patterns])
                        assert np.max(abs(law-cpu_law))<1e-9 and abs(law.sum()-1)<1e-9;row.update(numerical_full_tv=float(abs(law-d['target']).sum()/2),maximum_cpu_law_error=float(abs(law-cpu_law).max()));arrays.update(law=law,cpu_law=cpu_law,target=d['target'],conditional_probabilities=prob,patterns=patterns)
                np.savez_compressed(OUT/f'row_{case}_{method}_{B}.npz',**arrays)
            except Exception as e:row=dict(success=False,case=case,method=method,budget=B,error=str(e))
            rows.append(row);save(f'row_{case}_{method}_{B}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
