from pathlib import Path
import os,sys,ctypes as C,subprocess,hashlib,json,shutil,time,traceback
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from gbslab.displaced_sampling import DisplacedThresholdSampler,coherent_direction
OUT=ROOT/'.git/iteration68_cuda_v1_20261110';OUT.mkdir(exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(value,indent=2)+'\n').encode())
source=ROOT/'.git/iteration68_cuda_pilot_v1.cu';compiler=Path(shutil.which('nvcc'));input_path=ROOT/'.git/iteration68_profile_20261110/normalized_input.npz';hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),source,ROOT/'src/gbslab/displaced_sampling.py')}
for n in hashes:p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
build=Path(os.environ['LOCALAPPDATA'])/'Temp/deziro_gbs_cuda68_20261110_v1';build.mkdir(exist_ok=False);shutil.copy2(source,build/'kernel.cu');vcvars=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat');env=os.environ.copy();values=subprocess.check_output(['cmd','/d','/s','/c',f'"{vcvars}" >nul && set'],text=True,errors='replace')
for line in values.splitlines():
 if '=' in line:key,value=line.split('=',1);env[key]=value
command=[str(compiler),'-O3','-arch=sm_120','--fmad=false','-std=c++17','-shared','-Xcompiler','/MD','kernel.cu','-lcublas','-o','native_cuda.dll'];save('plan.json',dict(source_hashes=hashes,input_sha256=H(input_path),compiler_sha256=H(compiler),compiler_version=subprocess.check_output([str(compiler),'--version'],text=True),command=command,build_directory=str(build),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version,compute_cap','--format=csv,noheader'],text=True),samples=128,repetitions=3,scope='FP64 exploratory C++ CUDA engine on identical finite tensors. Model upload and coherent preparation reported separately.'))
start=time.perf_counter();built=subprocess.run(command,cwd=build,env=env,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=built.returncode,stdout=built.stdout,stderr=built.stderr,seconds=time.perf_counter()-start));print('compile',built.returncode,built.stderr[:1000],flush=True);assert built.returncode==0;shutil.copy2(build/'native_cuda.dll',OUT/'native_cuda.dll')
runtime=[]
for p in (compiler.parent,compiler.parent/'x64'):
 if p.is_dir():runtime.append(os.add_dll_directory(str(p)))
lib=C.CDLL(str(OUT/'native_cuda.dll'));DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);PP=C.POINTER(DP);lib.gds_create.argtypes=[C.c_int,IP,PP,C.c_char_p,C.c_int];lib.gds_create.restype=C.c_void_p;lib.gds_destroy.argtypes=[C.c_void_p];lib.gds_sample.argtypes=[C.c_void_p,C.c_int,PP,PP,DP,IP,DP,DP,DP,C.c_char_p,C.c_int]
z=np.load(input_path);tensors=[np.ascontiguousarray(z[f'A{i}']) for i in range(144)];beta=z['beta'];u=np.ascontiguousarray(z['uniforms']);dims=np.array([a.shape for a in tensors],np.int32);ptrs=(DP*144)(*(a.ctypes.data_as(DP) for a in tensors));error=C.create_string_buffer(1024);start=time.perf_counter();handle=lib.gds_create(144,dims.ctypes.data_as(IP),ptrs,error,1024);assert handle,error.value;upload=time.perf_counter()-start
reference=DisplacedThresholdSampler.__new__(DisplacedThresholdSampler);reference.tensors=tensors;reference.modes=144;reference.input_log_norm=0.;start=time.perf_counter();expected=reference.sample(beta,u);baseline=time.perf_counter()-start;rows=[]
for repetition in range(3):
 row=dict(repetition=repetition)
 try:
  start=time.perf_counter();cs=[];ps=[]
  for site,a in enumerate(tensors):
   c,w,t=coherent_direction(beta[:,site],a.shape[1]);cs.append(np.ascontiguousarray(c));ps.append(np.ascontiguousarray(np.column_stack((w,t))))
  prep=time.perf_counter()-start;cp=(DP*144)(*(a.ctypes.data_as(DP) for a in cs));pp=(DP*144)(*(a.ctypes.data_as(DP) for a in ps));labels=np.empty(u.shape,np.int32);selected=np.empty(u.shape);vacuum=np.empty(u.shape);residual=C.c_double();start=time.perf_counter();rc=lib.gds_sample(handle,len(u),cp,pp,u.ctypes.data_as(DP),labels.ctypes.data_as(IP),selected.ctypes.data_as(DP),vacuum.ctypes.data_as(DP),C.byref(residual),error,1024);seconds=time.perf_counter()-start
  row.update(returncode=rc,native_seconds=seconds,coherent_preparation_seconds=prep,total_seconds=seconds+prep,error=error.value.decode(),normalization_residual=residual.value)
  if rc==0:
   row.update(labels_equal=bool(np.array_equal(labels,expected['refined_labels'])),selected_max_error=float(np.max(abs(selected-expected['selected_refined_probabilities']))),vacuum_max_error=float(np.max(abs(vacuum-expected['vacuum_given_hidden_history']))));np.savez_compressed(OUT/f'run_{repetition}.npz',labels=labels,selected=selected,vacuum=vacuum)
  row['status']='completed' if rc==0 and row['labels_equal'] and max(row['selected_max_error'],row['vacuum_max_error'])<1e-11 else 'failed'
 except Exception:row.update(status='failed',traceback=traceback.format_exc())
 rows.append(row);save(f'run_{repetition}.json',row);print(row,flush=True)
lib.gds_destroy(handle);assert all(H(ROOT/n)==h for n,h in hashes.items());save('summary.json',dict(completed=all(r['status']=='completed' for r in rows),baseline_seconds=baseline,model_upload_seconds=upload,rows=rows,source_hashes=hashes,dll_sha256=H(OUT/'native_cuda.dll'),scope='Three consecutive pilot calls including first call; formal interleaved benchmark and adversarial controls remain.'))
save('pilot_manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}));print('baseline',baseline,'upload',upload,flush=True)
