"""Lightweight localization of complete-output bias using existing samples.
Fixed prefix/block regions, exact Gaussian single/pair moment identities, one
native CPU worker. No new samples or timing claims. DKW is simultaneous over
both methods and all declared regions; clipped thresholds/directions are uniform.
"""
from pathlib import Path
import os,argparse,json,hashlib,math,ctypes as C,shutil,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
base=ROOT/'results/iteration83_global_quality_v1';dll=base/'moments.dll';covfiles=[ROOT/f'results/iteration83_full_quality_v1/covariance_{s}.npz' for s in range(2)];regions=list(dict.fromkeys([(0,j) for j in (8,32,64,128,256,512,768,1024,1152)]+[(j,j+128) for j in range(0,1152,128)]));delta=.01/(2*len(regions));files=[Path(__file__),dll,*covfiles,base/'combined_4.npz',base/'combined_8.npz'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},regions=regions,delta_per_method_region=delta,native_threads=1));covs=[np.load(f)['cov'] for f in covfiles];samples={b:np.load(base/f'combined_{b}.npz')['clicks'] for b in (4,8)};runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(str(dll));DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);fn=lib.click83_moments;fn.argtypes=[C.c_int,C.c_int,DP,C.c_int,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;rows=[];reference=[];native_seconds=0.
def witness(hist,mu,var,eps):
 size=len(hist)-1;best=dict(lower=0.,direction=None,clip_lower=None,clip_upper=None)
 for direction,h,mean in [('increasing',hist,mu),('decreasing',hist[::-1],size-mu)]:
  cdf=np.cumsum(h)/h.sum();integral=np.r_[0,np.cumsum(1-cdf)]
  for U in range(1,size+1):
   L=np.arange(U);width=U-L;emp=(integral[U]-integral[L])/width;stop=(math.hypot(math.sqrt(var),U-mean)-(U-mean))/2;ref=(mean-L-stop)/width;bounds=ref-emp-eps;k=int(np.argmax(bounds))
   if bounds[k]>best['lower']:best=dict(lower=float(bounds[k]),direction=direction,clip_lower=int(L[k]),clip_upper=U,empirical_expectation=float(emp[k]),reference_expectation_lower=float(ref[k]))
 return best
for first,end in regions:
 width=end-first;ix=list(range(first,end))+list(range(1152+first,1152+end));cov=np.ascontiguousarray([c[np.ix_(ix,ix)] for c in covs]);single=np.zeros(width);stats=np.zeros(4);err=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(width,2,ptr(cov),1,ptr(single),ptr(stats),err,len(err));elapsed=time.perf_counter()-start;native_seconds+=elapsed;assert rc==0,err.value;mu,var=stats[:2];reference.append(dict(first=first,end=end,mean=mu,variance=var,native_seconds=elapsed))
 for budget,bits in samples.items():
  totals=bits[:,first:end].sum(1).astype(int);hist=np.bincount(totals,minlength=width+1);eps=math.sqrt(math.log(2/delta)/(2*len(bits)));bound=witness(hist,mu,var,eps);row=dict(budget=budget,first=first,end=end,reference_mean=mu,reference_variance=var,empirical_mean=float(totals.mean()),mean_bias=float(totals.mean()-mu),empirical_mean_standard_error=float(totals.std(ddof=1)/math.sqrt(len(bits))),dkw_radius=eps,nominal_joint99_full_bit_tv_lower_bound=bound.pop('lower'),witness=bound);rows.append(row);np.savez_compressed(OUT/f'hist_{budget}_{first}_{end}.npz',histogram=hist)
best=[max((r for r in rows if r['budget']==b),key=lambda r:r['nominal_joint99_full_bit_tv_lower_bound']) for b in (4,8)];save('summary.json',dict(rows=rows,best_regions=best,reference=reference,native_seconds=native_seconds,scope=__doc__,confidence='joint 99% over two methods and 17 predeclared regions, assuming iid samples and fixed numerical reference moments'));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('native single-worker reference seconds',native_seconds,flush=True);print('best regions',best,flush=True)
