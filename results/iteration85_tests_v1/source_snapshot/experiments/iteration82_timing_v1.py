"""Serial three-rotation speed comparisons, paired with iteration82 accuracy.
Input loading, DLL loading and initial CUDA context excluded. Factor timings
include native finite design, calibration, plan build, sampling, thresholding
and destruction. Reuse throughput is separately labelled. No heavy concurrent job.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,argparse,time,ctypes as C,statistics,subprocess
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_factor_pnr import NativeFactorPNRSampler
from gbslab.native_cached_transport_pnr import NativeCachedTransportPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.native_click_branches import NativeBranchClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(p)) for p in dirs]
bins={k:ROOT/v for k,v in dict(factor='results/iteration82_factor_accuracy_v1/factor.dll',calibrate='results/iteration81_factor_pilot_v2/factor.dll',cached='results/iteration80_cached_pilot_v1/cached.dll',dense='results/iteration80_wide_pilot_v1/ipf.dll',frugal='results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll',stable='results/iteration82_stable_pilot_v1/stable.dll',cpu='results/iteration72_cuda_pilot_v1/cpu.dll',gpu='results/iteration74_fused_pilot_v2/philox.dll').items()}
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);ptr=lambda x:x.ctypes.data_as(DP);libs={k:C.CDLL(str(bins[k])) for k in ('calibrate','dense','frugal')}
design=libs['calibrate'].factor81_design;design.argtypes=[C.c_int,C.c_int,C.c_int,DP,DP,C.c_int,DP,DP,DP,C.c_char_p,C.c_int];design.restype=C.c_int
ipf=libs['calibrate'].factor81_ipf;ipf.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_int,C.c_double,C.c_int,DP,DP,DP,DP,C.c_char_p,C.c_int];ipf.restype=C.c_int
dense=libs['dense'].pair80_ipf;dense.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_int,C.c_double,DP,DP,C.c_char_p,C.c_int];dense.restype=C.c_int
frugal=libs['frugal'].fc73_sample;frugal.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,DP,BP,DP,C.c_char_p,C.c_int];frugal.restype=C.c_int
upstream=subprocess.run(['gh','api','repos/tomdodd4598/FrugalGBS/commits/main'],capture_output=True,text=True,encoding='utf-8');save('upstream.json',dict(returncode=upstream.returncode,stdout=upstream.stdout,stderr=upstream.stderr));assert upstream.returncode==0;assert json.loads(upstream.stdout)['sha']=='fa8b796f2afebd4845a2e335e457735952617ef8'
cases=[('wide2',256,[('factor',8),('cached',8),('gpu',4),('gpu',8),('frugal',2),('frugal',3)]),('large128',256,[('factor',8),('gpu',4),('gpu',8),('frugal',2)])]
save('plan.json',dict(scope=__doc__,cases=cases,jiuzhang_methods=[['frugal',5],['stable',8],['stable',16],['stable',32],['gpu',4]],jiuzhang_records=8192,repetitions=3,workers=16,reuse_records=100000,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*bins.values()]}))
with NativePhiloxClickSampler(np.eye(4),np.zeros((8,8)),bins['cpu'],gpu_library=bins['gpu'],branch_budget=4,runtime_dirs=dirs) as model:warm=model.sample(1,820001)
save('cuda_context_warmup.json',dict(clicks=warm['clicks'].tolist(),seconds=warm['wall_seconds']))
rows=[];reuse=[]
for label,count,methods in cases:
 inp=np.load(ROOT/f'results/iteration82_factor_accuracy_v1/input_{label}.npz');v,x,cov,t=inp['v'],inp['x'],inp['cov'],float(inp['t']);m,n=v.shape;L=n+2**(2*n.bit_length());vr=np.ascontiguousarray(v.real);vi=np.ascontiguousarray(v.imag)
 ref=np.load(ROOT/f'results/iteration81_factor_pilot_v2/input_{label}.npz');refk=ref['K'];refw=ref['weights'];parts=[np.load(ROOT/f'results/iteration81_factor_pilot_v2/factors_{label}_{first}.npz') for first in range(0,L,256)];refa=np.concatenate([p['left'] for p in parts]);refb=np.concatenate([p['right'] for p in parts])
 def prepare(kind):
  start=time.perf_counter();u=np.zeros((L,n));w=np.zeros(L);p=np.zeros((L,m));q=np.ascontiguousarray(abs(v@v.T)**2/n);err=C.create_string_buffer(4096);rc=design(n,m,L,ptr(vr),ptr(vi),16,ptr(u),ptr(w),ptr(p),err,len(err));assert rc==0,err.value;design_time=time.perf_counter()-start
  if kind=='factor':
   k=np.zeros((m,m));aa=np.zeros((L,m));bb=np.zeros_like(aa);ss=np.zeros(6);rc=ipf(m,L,ptr(w),ptr(p),ptr(q),64 if n==16 else 32,1e-10,16,ptr(k),ptr(aa),ptr(bb),ptr(ss),err,len(err));assert rc==0,err.value
   calibrated=time.perf_counter()-start;model=NativeFactorPNRSampler(v,t,bins['factor'],kernel=k,left=aa,right=bb,latent_weights=w,exact_total_budget=8,workers=16,runtime_dirs=dirs);meta=dict(design_seconds=design_time,calibration_and_design_seconds=calibrated,plan_statistics=model.plan_statistics)
   # Record numerical factor changes due to native design arithmetic. No RNG draws.
   drift=[]
   for new,old in ((k,refk),(aa,refa),(bb,refb)):
    assert np.array_equal(new==0,old==0);nonzero=old!=0;drift.append(float(np.max(abs(new[nonzero]/old[nonzero]-1))))
   meta['factor_max_relative_differences']=drift;meta['maximum_weight_difference']=float(abs(w-refw).max());assert max(drift)<1e-8
  else:
   table=np.zeros((L,m,m));ss=np.zeros(4);rc=dense(m,L,ptr(w),ptr(p),ptr(q),min(256,2000000000//table.size),1e-10,ptr(table),ptr(ss),err,len(err));assert rc==0,err.value;calibrated=time.perf_counter()-start;model=NativeCachedTransportPNRSampler(v,t,bins['cached'],couplings=table,latent_weights=w,exact_total_budget=8,workers=16,runtime_dirs=dirs);meta=dict(design_seconds=design_time,calibration_and_design_seconds=calibrated,alias_payload_bytes=model.alias_payload_bytes)
  return model,meta
 for rep in range(3):
  for kind,budget in methods[rep:]+methods[:rep]:
   name=f'{label}_{rep}_{kind}_{budget}';save('started_'+name+'.json',dict(label=label,rep=rep,method=kind,budget=budget));start=time.perf_counter();counts=None;meta={}
   try:
    if kind in ('factor','cached'):
     model,meta=prepare(kind);setup=time.perf_counter()-start
     with model:r=model.sample(count,820200+rep)
     counts=r['counts'];bits=(counts>0).astype(np.uint8);meta.update(setup_seconds=setup,sampling_seconds=r['wall_seconds'],statistics=r['statistics'])
    elif kind=='gpu':
     with NativePhiloxClickSampler(v,x,bins['cpu'],gpu_library=bins['gpu'],branch_budget=budget,runtime_dirs=dirs) as model:r=model.sample(count,820200+rep)
     bits=r['clicks'];meta['cuda_timings']=r['cuda_timings']
    else:
     bits=np.zeros((count,m),np.uint8);ss=np.zeros(7);err=C.create_string_buffer(4096);assert frugal(m,budget,count,820200+rep,16,ptr(cov),bits.ctypes.data_as(BP),ptr(ss),err,len(err))==0,err.value;meta['component_timings']=ss.tolist()
    row=dict(success=True,seconds=time.perf_counter()-start,**meta);np.savez_compressed(OUT/(name+'.npz'),clicks=bits,**({} if counts is None else dict(counts=counts)))
   except (ValueError,ArithmeticError,AssertionError) as e:row=dict(success=False,error=str(e),seconds=time.perf_counter()-start)
   row.update(label=label,rep=rep,method=kind,budget=budget,records=count);rows.append(row);save(name+'.json',row);print(row,flush=True)
 # Reuse batches have one separately measured full preparation per case.
 start=time.perf_counter();model,meta=prepare('factor');setup=time.perf_counter()-start
 with model:
  for rep in range(3):
   start=time.perf_counter();r=model.sample(100000,820300+rep);bits=r['counts']>0;elapsed=time.perf_counter()-start;np.savez_compressed(OUT/f'reuse_{label}_{rep}.npz',counts=r['counts']);row=dict(label=label,rep=rep,records=100000,setup_seconds=setup,seconds=elapsed,statistics=r['statistics']);reuse.append(row);save(f'reuse_{label}_{rep}.json',row);print('Reuse',row,flush=True)
save('synthetic_summary.json',dict(completed=True,rows=rows,reuse=reuse))
covs=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/input_0.npz')['covariances'];methods=[('frugal',5),('stable',8),('stable',16),('stable',32),('gpu',4)];jzrows=[]
for rep in range(3):
 for method,budget in methods[rep:]+methods[:rep]:
  start=time.perf_counter();outputs=[]
  for sector,cov in enumerate(covs):
   seed=820400+rep*2+sector
   if method=='frugal':
    bits=np.zeros((8192,8),np.uint8);ss=np.zeros(7);err=C.create_string_buffer(4096);assert frugal(8,budget,8192,seed,16,ptr(cov),bits.ctypes.data_as(BP),ptr(ss),err,len(err))==0,err.value
   elif method=='stable':
    with NativeBranchClickSampler(np.eye(8),(cov-np.eye(16))/2,bins['stable'],branch_budget=budget,runtime_dirs=dirs) as model:
     fn=model.lib.tc82_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,BP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;bits=np.zeros((8192,8),np.uint8);ss=np.zeros(7);err=C.create_string_buffer(4096);assert fn(model.handle,8192,seed,16,budget,bits.ctypes.data_as(BP),ptr(ss),err,len(err))==0,err.value
   else:
    with NativePhiloxClickSampler(np.eye(8),(cov-np.eye(16))/2,bins['cpu'],gpu_library=bins['gpu'],branch_budget=budget,runtime_dirs=dirs) as model:bits=model.sample(8192,seed)['clicks']
   outputs.append(bits)
  bits=np.bitwise_or(*outputs);elapsed=time.perf_counter()-start;row=dict(rep=rep,method=method,budget=budget,records=8192,seconds=elapsed);jzrows.append(row);save(f'jiuzhang_{rep}_{method}_{budget}.json',row);np.savez_compressed(OUT/f'jiuzhang_{rep}_{method}_{budget}.npz',clicks=bits);print('Jiuzhang',row,flush=True)
save('summary.json',dict(completed=True,rows=rows,reuse=reuse,jiuzhang=jzrows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
