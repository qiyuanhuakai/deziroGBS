"""Build official Frugal SIMD dimension 1152 and audit full-input prefixes.
K3+ full-target storage is not attempted; cutoff <=2 is an explicit adapter cap.
"""
from pathlib import Path
import os,json,hashlib,argparse,subprocess,time,ctypes as C,shutil
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
src=ROOT/'native/frugal_full83_adapter_v1.cpp';base=ROOT/'results/iteration81_prefix_build_v1';command=json.loads((base/'build.json').read_text())['command'];command=[str(src) if s==str(ROOT/'native/frugal_prefix81_adapter_v1.cpp') else s for s in command];command[-1]=str(OUT/'frugal.dll');save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),src,ROOT/'native/frugal_prefix81_adapter_v1.cpp',ROOT/'native/frugal_cpu75_adapter_v1.cpp']},parent_manifest=H(base/'manifest.json'),command=command))
start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr;print('Full Frugal built',flush=True)
runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(str(OUT/'frugal.dll'));DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);fn=lib.fc83_tree;fn.argtypes=[C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
rows=[]
for s in range(2):
 cov=np.load(ROOT/f'results/iteration83_full_quality_v1/covariance_{s}.npz')['cov'];q=np.zeros(255);prev=q.copy();prop=q.copy();stats=np.zeros(3);err=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(2,8,16,ptr(cov),ptr(q),ptr(prev),ptr(prop),ptr(stats),err,len(err));row=dict(sector=s,seconds=time.perf_counter()-start,returncode=rc,error=err.value.decode(),stats=stats.tolist());save(f'row_{s}.json',row);assert rc==0,err.value
 prior=json.loads((ROOT/f'results/iteration83_full_quality_v1/frugal_tree_2_{s}.json').read_text())['tree'];old=np.array([float.fromhex(r['threshold_hex']) for r in prior]);row['maximum_subsystem_threshold_difference']=float(np.nanmax(abs(q-old)));row['same_nonfinite']=bool(np.array_equal(np.isnan(q),np.isnan(old)));assert row['same_nonfinite'] and row['maximum_subsystem_threshold_difference']<1e-12
 tree=[dict(prefix=''.join(str((bits>>j)&1) for j in range(length)),threshold_hex=float(q[(1<<length)-1+bits]).hex()) for length in range(8) for bits in range(1<<length)];save(f'tree_{s}.json',dict(tree=tree));np.savez_compressed(OUT/f'prefix_{s}.npz',q=q,previous=prev,proposed=prop);save(f'audit_{s}.json',row);rows.append(row);print(row,flush=True)
save('summary.json',dict(passed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
