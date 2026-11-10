from pathlib import Path
import os,sys,ctypes as C,subprocess,hashlib,json,shutil,time,traceback
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from gbslab.displaced_sampling import DisplacedThresholdSampler,coherent_direction
OUT=ROOT/'.git/iteration68_native_v1_20261110';OUT.mkdir(exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(value,indent=2)+'\n').encode())
source=ROOT/'.git/iteration68_native_pilot_v1.cpp';compiler=Path(shutil.which('g++'));input_path=ROOT/'.git/iteration68_profile_20261110/normalized_input.npz';hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),source,ROOT/'src/gbslab/displaced_sampling.py')}
for n in hashes:p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
command=[str(compiler),'-O3','-march=native','-std=c++20','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math','-Wall','-Wextra',str(source),'-o',str(OUT/'native.dll')];save('plan.json',dict(source_hashes=hashes,input_sha256=H(input_path),compiler_sha256=H(compiler),command=command,threads=[1,2,4,8],tiles=[8,16],samples=128,scope='Exploratory same-stored-tensor native implementation,not new universal GBS algorithm.'))
start=time.perf_counter();built=subprocess.run(command,capture_output=True,text=True);save('build.json',dict(command=command,returncode=built.returncode,stdout=built.stdout,stderr=built.stderr,seconds=time.perf_counter()-start));assert built.returncode==0
runtime=os.add_dll_directory(str(compiler.parent));lib=C.CDLL(str(OUT/'native.dll'));DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);PP=C.POINTER(DP)
lib.nds_create.argtypes=[C.c_int,IP,PP,C.c_char_p,C.c_int];lib.nds_create.restype=C.c_void_p;lib.nds_destroy.argtypes=[C.c_void_p];lib.nds_sample.argtypes=[C.c_void_p,C.c_int,PP,PP,DP,C.c_int,C.c_int,IP,DP,DP,DP,C.c_char_p,C.c_int]
z=np.load(input_path);tensors=[np.ascontiguousarray(z[f'A{i}']) for i in range(144)];beta=z['beta'];u=np.ascontiguousarray(z['uniforms']);dims=np.array([a.shape for a in tensors],np.int32);ptrs=(DP*144)(*(a.ctypes.data_as(DP) for a in tensors));error=C.create_string_buffer(1024);start=time.perf_counter();handle=lib.nds_create(144,dims.ctypes.data_as(IP),ptrs,error,1024);assert handle,error.value;packing=time.perf_counter()-start
reference=DisplacedThresholdSampler.__new__(DisplacedThresholdSampler);reference.tensors=tensors;reference.modes=144;reference.input_log_norm=0.
start=time.perf_counter();expected=reference.sample(beta,u);baseline=time.perf_counter()-start;rows=[]
for threads in (1,2,4,8):
 for tile in (8,16):
  row=dict(threads=threads,tile=tile)
  try:
   start=time.perf_counter();cs=[];ps=[]
   for site,a in enumerate(tensors):
    c,w,t=coherent_direction(beta[:,site],a.shape[1]);cs.append(np.ascontiguousarray(c));ps.append(np.ascontiguousarray(np.column_stack((w,t))))
   prep=time.perf_counter()-start;cp=(DP*144)(*(a.ctypes.data_as(DP) for a in cs));pp=(DP*144)(*(a.ctypes.data_as(DP) for a in ps));labels=np.empty(u.shape,np.int32);selected=np.empty(u.shape);vacuum=np.empty(u.shape);residual=C.c_double();start=time.perf_counter();rc=lib.nds_sample(handle,len(u),cp,pp,u.ctypes.data_as(DP),threads,tile,labels.ctypes.data_as(IP),selected.ctypes.data_as(DP),vacuum.ctypes.data_as(DP),C.byref(residual),error,1024);seconds=time.perf_counter()-start
   row.update(returncode=rc,native_seconds=seconds,coherent_preparation_seconds=prep,total_seconds=seconds+prep,error=error.value.decode(),normalization_residual=residual.value)
   if rc==0:
    row.update(labels_equal=bool(np.array_equal(labels,expected['refined_labels'])),selected_max_error=float(np.max(abs(selected-expected['selected_refined_probabilities']))),vacuum_max_error=float(np.max(abs(vacuum-expected['vacuum_given_hidden_history']))));np.savez_compressed(OUT/f't{threads}_b{tile}.npz',labels=labels,selected=selected,vacuum=vacuum)
   row['status']='completed' if rc==0 and row['labels_equal'] and max(row['selected_max_error'],row['vacuum_max_error'])<1e-11 else 'failed'
  except Exception:row.update(status='failed',traceback=traceback.format_exc())
  rows.append(row);save(f't{threads}_b{tile}.json',row);print(row,flush=True)
lib.nds_destroy(handle);assert all(H(ROOT/n)==h for n,h in hashes.items());save('summary.json',dict(completed=all(r['status']=='completed' for r in rows),baseline_seconds=baseline,packing_seconds=packing,rows=rows,source_hashes=hashes,dll_sha256=H(OUT/'native.dll'),scope='One observed call per configuration; no isolated repeated benchmark conclusion.'))
save('pilot_manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}));print('baseline',baseline,'packing',packing,flush=True)
