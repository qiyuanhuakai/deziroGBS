"""C++ cyclic marginal scaling for smooth feasible photon-pair tables.
Explicit nonconvergence remains a failure, never a sampled fallback table.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,subprocess,time,ctypes as C
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));source=ROOT/'native/gaussian_pair_ipf80_v1.cpp';save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),source]},maximum_iterations=10000,tolerance=1e-10));command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math',str(source),'-o',str(OUT/'ipf.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr
handle=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'ipf.dll'));DP=C.POINTER(C.c_double);fn=lib.pair80_ipf;fn.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_int,C.c_double,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;ptr=lambda x:x.ctypes.data_as(DP);rows=[]
summary=json.loads((ROOT/'results/iteration80_pair_transport_preflight_v1/summary.json').read_text())
for original in summary['rows']:
    label=original['label'];d=np.load(ROOT/f'results/iteration80_pair_transport_preflight_v1/{label}.npz');v=d['v'];m,n=v.shape;weights=np.ascontiguousarray(d['weights']);marginal=np.ascontiguousarray(d['marginals']);Q=np.ascontiguousarray(d['Q']);couplings=np.zeros((len(weights),m,m));stats=np.zeros(4);message=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(m,len(weights),ptr(weights),ptr(marginal),ptr(Q),10000,1e-10,ptr(couplings),ptr(stats),message,len(message));row=dict(label=label,success=rc==0,returncode=rc,error=message.value.decode() if rc else '',seconds=time.perf_counter()-start,iterations=int(stats[0]),last_projection_error=stats[1],symmetric_error=stats[2],cell_iterations=stats[3],lp_feasible=original['success'])
    if rc==0:
        error=max(float(abs(couplings.sum(axis=2)-marginal).max()),float(abs(np.einsum('l,lij->ij',weights,couplings)-Q).max()));assert error<1e-10 and couplings.min()>=0 and np.array_equal(couplings,couplings.transpose(0,2,1));row['independent_constraint_error']=error
    rows.append(row);save(label+'.json',row);np.savez_compressed(OUT/(label+'.npz'),**{k:d[k] for k in d.files if k not in ('couplings','dual','slack_solution')},couplings=couplings,stats=stats);print(row,flush=True)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
