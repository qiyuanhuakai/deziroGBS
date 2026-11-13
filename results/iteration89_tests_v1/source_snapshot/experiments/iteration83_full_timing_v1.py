"""Three serial rotations for our complete 1152-bit samplers and one full Frugal panel.
The full Frugal comparison is explicitly a single observation, not a median.
All methods start from the same transmission/input covariance and include native
loss reduction. Frugal also pays full-covariance export. DLL/file I/O and initial
CUDA context are excluded. No concurrent heavy work; no phase timing splicing.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,ctypes as C,time,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(d)) for d in dirs];paths=dict(loss=ROOT/'results/iteration83_full_quality_v1/loss.dll',frugal=ROOT/'results/iteration83_frugal_build_v2/frugal.dll',stable=ROOT/'results/iteration82_stable_pilot_v1/stable.dll',cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu=ROOT/'results/iteration74_fused_pilot_v2/philox.dll');data=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz');T,x0=data['T'],data['input_x'];m,n=T.shape;DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);ptr=lambda x:x.ctypes.data_as(DP)
libs={k:C.CDLL(str(paths[k])) for k in ('loss','frugal')};prepare=libs['loss'].loss83_prepare;prepare.argtypes=[C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];prepare.restype=C.c_int;export=libs['loss'].loss83_covariance;export.argtypes=[C.c_int,C.c_int,DP,DP,C.c_int,DP,C.c_char_p,C.c_int];export.restype=C.c_int;frugal=libs['frugal'].fc83_sample;frugal.argtypes=[C.c_int,C.c_int,C.c_uint,C.c_int,DP,BP,DP,C.c_char_p,C.c_int];frugal.restype=C.c_int
methods=[('frugal',2),('gpu',4),('gpu',8),('stable',8),('stable',32)];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*paths.values(),ROOT/'results/iteration83_loss_pilot_v1/input.npz']},records=4,methods=methods,rotations=3,workers=16,frugal_workers=1,frugal_repetitions=1,frugal_workspace_bytes_per_worker=32*(1152*1151+1152*1151*1150//3),extra_gpu_records=256))
up=subprocess.run(['gh','api','repos/tomdodd4598/FrugalGBS/commits/main'],capture_output=True,text=True,encoding='utf-8');save('upstream.json',dict(returncode=up.returncode,stdout=up.stdout,stderr=up.stderr));assert up.returncode==0 and json.loads(up.stdout)['sha']=='fa8b796f2afebd4845a2e335e457735952617ef8'
with NativePhiloxClickSampler(np.eye(4),np.zeros((8,8)),paths['cpu'],gpu_library=paths['gpu'],branch_budget=4,runtime_dirs=dirs) as model:model.sample(1,830100)
rows=[]
def run(rep,method,budget,count):
 row=dict(rep=rep,method=method,budget=budget,records=count,success=False);outputs=[];details=[];start=time.perf_counter()
 try:
  v=np.zeros_like(T);xs=np.zeros_like(x0);ss=np.zeros(5);err=C.create_string_buffer(4096);assert prepare(m,n,2,ptr(T),ptr(x0),ptr(v),ptr(xs),ptr(ss),err,len(err))==0,err.value;row['prepare_seconds']=time.perf_counter()-start
  for s in range(2):
   seed=830200+2*rep+s
   if method=='frugal':
    cs=time.perf_counter();cov=np.zeros((2*m,2*m));assert export(m,n,ptr(v),ptr(xs[s]),16,ptr(cov),err,len(err))==0,err.value;covtime=time.perf_counter()-cs;bits=np.zeros((count,m),np.uint8);stats=np.zeros(7);assert frugal(budget,count,seed,1,ptr(cov),bits.ctypes.data_as(BP),ptr(stats),err,len(err))==0,err.value;details.append(dict(covariance_seconds=covtime,native_times=stats.tolist()));del cov
   else:
    if method=='stable':model=NativeStableClickSampler(v,xs[s],paths['stable'],branch_budget=budget,workers=16,runtime_dirs=dirs)
    else:model=NativePhiloxClickSampler(v,xs[s],paths['cpu'],gpu_library=paths['gpu'],branch_budget=budget,runtime_dirs=dirs)
    with model:result=model.sample(count,seed)
    bits=result['clicks'];details.append(dict(statistics=result['statistics'],wall_seconds=result['wall_seconds']))
   outputs.append(bits)
  result=np.bitwise_or(*outputs);row.update(success=True,mean_clicks=float(result.sum(1).mean()));del v,xs
 except Exception as e:row['error']=str(e)
 row['seconds']=time.perf_counter()-start;row['sectors']=details;save(f'row_{count}_{rep}_{method}_{budget}.json',row)
 if row['success']:np.savez_compressed(OUT/f'samples_{count}_{rep}_{method}_{budget}.npz',clicks=result)
 rows.append(row);print({k:v for k,v in row.items() if k!='sectors'},flush=True)
for rep in range(3):
 for method,budget in methods[rep:]+methods[:rep]:
  if method!='frugal':run(rep,method,budget,4)
for rep in range(3):
 for budget in (4,8):run(rep,'gpu',budget,256)
run(0,'frugal',2,4)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
