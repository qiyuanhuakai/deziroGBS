from pathlib import Path
import os, sys, ctypes as C, hashlib, json, time, shutil, inspect
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import gbslab.displaced_sampling as sampling
OUT=ROOT/'.git/iteration68_fullbatch_pilot_20261110';OUT.mkdir(exist_ok=False)
def H(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 p=OUT/n;assert not p.exists();p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
paths=[Path(__file__),ROOT/'src/gbslab/displaced_sampling.py',ROOT/'.git/iteration68_cuda_pilot_v2.cu']
for p in paths:
 dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
dll=ROOT/'.git/iteration68_cuda_v2_20261110/native_cuda.dll'
inp=ROOT/'.git/iteration68_profile_20261110/normalized_input.npz';random=ROOT/'results/iteration30_panel/large_random.npz'
save('plan.json',dict(sources={str(p.relative_to(ROOT)):H(p) for p in paths},dependencies={str(p.relative_to(ROOT)):H(p) for p in [dll,inp,random]},batch=512,samples=4096,threads=1,gpu_repetitions=3,scope='Existing fixed inputs, one CPU call per variant, repeated GPU; exploratory throughput and trajectory agreement, not a formal interleaved benchmark or Gaussian distribution certificate.'))
z=np.load(inp);tensors=[np.ascontiguousarray(z[f'A{i}']) for i in range(144)];r=np.load(random);beta=np.ascontiguousarray(r['beta']);u=np.ascontiguousarray(r['uniforms']);assert beta.shape==(4096,144)
source=inspect.getsource(sampling);old='M=np.stack([env@A[:,n,:] for n in range(A.shape[1])],axis=1)';assert source.count(old)==1
flat_source=source.replace(old,'M=(env @ A.reshape(A.shape[0], -1)).reshape(count, A.shape[1], A.shape[2])')
(OUT/'flattened_reference.py').write_bytes(flat_source.encode());namespace={'__name__':'gbslab.iteration68_flattened_reference','__package__':'gbslab'};exec(compile(flat_source,str(OUT/'flattened_reference.py'),'exec'),namespace)
outputs={};rows=[]
for name,cls in [('original',sampling.DisplacedThresholdSampler),('flattened',namespace['DisplacedThresholdSampler'])]:
 ref=cls.__new__(cls);ref.tensors=tensors;ref.modes=144;ref.input_log_norm=0.;parts=[];elapsed=0.
 for start in range(0,len(u),512):
  t=time.perf_counter();part=ref.sample(beta[start:start+512],u[start:start+512]);elapsed+=time.perf_counter()-t;parts.append(part);print(name,start+512,'seconds',elapsed,flush=True)
 output={key:np.concatenate([p[key] for p in parts]) for key in ('refined_labels','selected_refined_probabilities','vacuum_given_hidden_history')};outputs[name]=output;np.savez_compressed(OUT/(name+'.npz'),**output)
 row=dict(backend=name,seconds=elapsed,normalization_residual=max(p['maximum_normalization_residual'] for p in parts))
 if name=='flattened': row.update(labels_equal=bool(np.array_equal(output['refined_labels'],outputs['original']['refined_labels'])),selected_error=float(np.max(abs(output['selected_refined_probabilities']-outputs['original']['selected_refined_probabilities']))))
 rows.append(row);save(name+'.json',row)
runtime=[]
cuda=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.5/bin')
for p in (cuda,cuda/'x64'):
 if p.is_dir():runtime.append(os.add_dll_directory(str(p)))
lib=C.CDLL(str(dll));DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);PP=C.POINTER(DP)
lib.gds_create.argtypes=[C.c_int,IP,PP,C.c_char_p,C.c_int];lib.gds_create.restype=C.c_void_p;lib.gds_destroy.argtypes=[C.c_void_p]
lib.gds_sample_beta.argtypes=[C.c_void_p,C.c_int,DP,DP,IP,DP,DP,DP,C.c_char_p,C.c_int]
dims=np.array([a.shape for a in tensors],np.int32);ptrs=(DP*144)(*(a.ctypes.data_as(DP) for a in tensors));error=C.create_string_buffer(1024)
t=time.perf_counter();handle=lib.gds_create(144,dims.ctypes.data_as(IP),ptrs,error,1024);upload=time.perf_counter()-t;assert handle,error.value
for repetition in range(3):
 labels=np.empty(u.shape,np.int32);selected=np.empty(u.shape);vacuum=np.empty(u.shape);elapsed=0.;residual_max=0.
 for start in range(0,len(u),512):
  residual=C.c_double();t=time.perf_counter();rc=lib.gds_sample_beta(handle,512,beta[start:start+512].ctypes.data_as(DP),u[start:start+512].ctypes.data_as(DP),labels[start:start+512].ctypes.data_as(IP),selected[start:start+512].ctypes.data_as(DP),vacuum[start:start+512].ctypes.data_as(DP),C.byref(residual),error,1024);elapsed+=time.perf_counter()-t;assert rc==0,error.value;residual_max=max(residual_max,residual.value)
 expected=outputs['original'];row=dict(backend='cuda_v2',repetition=repetition,seconds=elapsed,includes_first_cuda_sampling_call=repetition==0,labels_equal=bool(np.array_equal(labels,expected['refined_labels'])),selected_error=float(np.max(abs(selected-expected['selected_refined_probabilities']))),vacuum_error=float(np.max(abs(vacuum-expected['vacuum_given_hidden_history']))),normalization_residual=residual_max)
 rows.append(row);save(f'cuda_{repetition}.json',row);np.savez_compressed(OUT/f'cuda_{repetition}.npz',labels=labels,selected=selected,vacuum=vacuum);print(row,flush=True)
lib.gds_destroy(handle)
save('summary.json',dict(rows=rows,model_upload_seconds=upload,all_labels_equal=all(row.get('labels_equal',True) for row in rows),scope='Fixed-input full4096 exploratory benchmark; no independent holdout and no interleaved repeated CPU measurements.'))
save('pilot_manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
print('COMPLETE',flush=True)
