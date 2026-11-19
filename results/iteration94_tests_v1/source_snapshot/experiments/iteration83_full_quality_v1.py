"""All first-eight paths of complete 1152-output lossy low-rank sampling.
Includes actual GPU replay through the full trajectory and preserves failures.
Full Frugal covariance is exported by native code, used unchanged in comparisons.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,ctypes as C,time,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.frugal_law import full_law
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));dirs=[gcc.parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(d)) for d in dirs];src=ROOT/'native/gaussian_loss_map83_v2.cpp';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';data=np.load(inp);v,x=data['v'],data['x'];m,n=v.shape;reference=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/input_0.npz')['two_common_sectors'];patterns=np.array([[(i>>j)&1 for j in range(8)] for i in range(256)],np.uint8)
save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),src,ROOT/'native/gaussian_loss_map83_v1.cpp',inp]},budgets=[4,8,16,32],frugal_cutoffs=[2,3,5],ports=data['order'][:8].tolist()))
command=[str(gcc),'-O3','-std=c++20','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(OUT/'loss.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
lib=C.CDLL(str(OUT/'loss.dll'));DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);fn=lib.loss83_covariance;fn.argtypes=[C.c_int,C.c_int,DP,DP,C.c_int,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
covs=[]
for s in range(2):
 cov=np.zeros((2*m,2*m));err=C.create_string_buffer(4096);start=time.perf_counter();assert fn(m,n,ptr(v),ptr(x[s]),16,ptr(cov),err,len(err))==0,err.value;save(f'covariance_{s}.json',dict(seconds=time.perf_counter()-start));np.savez_compressed(OUT/f'covariance_{s}.npz',cov=cov);covs.append(cov)
def combine(laws):
 out=np.zeros(256,np.longdouble)
 for i in range(256):
  for j in range(256):out[i|j]+=np.longdouble(laws[0][i])*np.longdouble(laws[1][j])
 return out
rows=[]
for budget in (4,8,16,32):
 laws=[];valid=True
 for s in range(2):
  law=np.full(256,np.nan);errors=[]
  with NativeStableClickSampler(v,x[s],ROOT/'results/iteration82_stable_pilot_v1/stable.dll',branch_budget=budget,runtime_dirs=dirs) as model:
   for j,bits in enumerate(patterns):
    try:law[j]=model.prefix_probability(bits)['probability']
    except Exception as e:errors.append(dict(pattern=j,error=str(e)))
  np.savez_compressed(OUT/f'stable_{budget}_{s}.npz',law=law);save(f'stable_errors_{budget}_{s}.json',errors);laws.append(law);valid &= not errors
 row=dict(method='stable',budget=budget,success=bool(valid))
 if valid:law=combine(laws);row.update(numerical_tv=float(abs(law-reference).sum()/2),normalization_residual=float(abs(law.sum()-1)));np.savez_compressed(OUT/f'stable_{budget}.npz',law=law)
 rows.append(row);save(f'row_stable_{budget}.json',row);print(row,flush=True)
 if budget not in (4,8):continue
 laws=[];valid=True
 for s in range(2):
  rowg=dict(method='gpu',budget=budget,sector=s,success=False)
  try:
   with NativePhiloxClickSampler(v,x[s],ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=ROOT/'results/iteration74_fused_pilot_v2/philox.dll',branch_budget=budget,runtime_dirs=dirs) as model:
    draws=np.full((256,m),.5);draws[:,:8]=np.where(patterns,0.,np.nextafter(1.,0.));result=model.replay(np.zeros((256,2*n)),draws)
   assert np.array_equal(result['clicks'][:,:8],patterns);law=np.prod(np.where(patterns,result['conditional_click_probabilities'][:,:8],1-result['conditional_click_probabilities'][:,:8]),axis=1);laws.append(law);np.savez_compressed(OUT/f'gpu_{budget}_{s}.npz',law=law,conditional=result['conditional_click_probabilities'][:,:8],clicks=result['clicks']);rowg['success']=True
  except Exception as e:rowg['error']=str(e);valid=False
  save(f'gpu_status_{budget}_{s}.json',rowg);print(rowg,flush=True)
 row=dict(method='gpu',budget=budget,success=valid)
 if valid:law=combine(laws);row['numerical_tv']=float(abs(law-reference).sum()/2);np.savez_compressed(OUT/f'gpu_{budget}.npz',law=law)
 rows.append(row);save(f'row_gpu_{budget}.json',row)
# Frugal first-eight thresholds depend only on this prefix covariance, verified
# against the exported full covariance. Avoid needlessly large analysis calls.
flib=C.CDLL(str(ROOT/'results/iteration81_prefix_build_v1/prefix.dll'));ff=flib.fc81_tree;ff.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];ff.restype=C.c_int
for budget in (2,3,5):
 laws=[]
 for s in range(2):
  ids=list(range(8))+list(range(m,m+8));cov=np.ascontiguousarray(covs[s][np.ix_(ids,ids)]);q=np.zeros(255);prev=q.copy();prop=q.copy();stats=np.zeros(3);err=C.create_string_buffer(4096);assert ff(8,budget,8,16,ptr(cov),ptr(q),ptr(prev),ptr(prop),ptr(stats),err,len(err))==0,err.value
  tree=[dict(prefix=''.join(str((bits>>j)&1) for j in range(length)),threshold_hex=float(q[(1<<length)-1+bits]).hex()) for length in range(8) for bits in range(1<<length)];law=np.array([float(p) for p in full_law(8,tree,model='uint64')['distribution']]);laws.append(law);save(f'frugal_tree_{budget}_{s}.json',dict(tree=tree));np.savez_compressed(OUT/f'frugal_{budget}_{s}.npz',law=law)
 law=combine(laws);row=dict(method='frugal',budget=budget,success=True,numerical_tv=float(abs(law-reference).sum()/2));rows.append(row);np.savez_compressed(OUT/f'frugal_{budget}.npz',law=law);save(f'row_frugal_{budget}.json',row);print(row,flush=True)
save('summary.json',dict(rows=rows,scope='G0 first eight physical detectors in the specified complete-output order; not full TV'));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
