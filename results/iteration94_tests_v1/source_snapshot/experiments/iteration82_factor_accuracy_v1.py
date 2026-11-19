"""Actual integer factor-rejection click laws, including the 128/384 case.
Evaluate actual native alias/rejection probabilities, replace J<=4 by the exact
pure-Gaussian total branch, and compare identical ports with Frugal and direct.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,argparse,time,ctypes as C
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_factor_pnr import NativeFactorPNRSampler
from gbslab.native_click_branches import NativeBranchClickSampler
from gbslab.frugal_law import full_law
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));runtime=[gcc.parent];dirs=[os.add_dll_directory(str(p)) for p in runtime];eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';sources=[Path(__file__),ROOT/'native/gaussian_pnr_factor82_v2.cpp',ROOT/'native/gaussian_pnr_factor82_v1.cpp',ROOT/'src/gbslab/native_factor_pnr.py'];base=ROOT/'results/iteration81_factor_pilot_v2';DP=C.POINTER(C.c_double);ptr=lambda a:a.ctypes.data_as(DP)
save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},labels=['strong4','wide2','large128'],frugal_cutoffs=[2,3],direct_budgets=[4,8],ports='first min(8,m) physical ports',threads=16))
command=[str(gcc),'-O3','-std=c++20','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math','-I',str(eigen),str(ROOT/'native/gaussian_pnr_factor82_v2.cpp'),'-o',str(OUT/'factor.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr;print('Actual factor law evaluator built',flush=True)
flib=C.CDLL(str(ROOT/'results/iteration81_prefix_build_v1/prefix.dll'));ffn=flib.fc81_tree;ffn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];ffn.restype=C.c_int
def mobius(vac):
 law=vac[np.arange(len(vac))^(len(vac)-1)].copy()
 for bit in range((len(vac)-1).bit_length()):
  for mask in range(len(vac)):
   if mask>>bit&1:law[mask]-=law[mask^(1<<bit)]
 assert law.min()>-1e-12 and abs(law.sum()-1)<1e-10
 return law
rows=[]
for label in ('strong4','wide2','large128'):
 d=np.load(base/f'input_{label}.npz');v,w,k=d['v'],d['weights'],d['K'];L=len(w);m,n=v.shape;depth=min(8,m);ports=np.arange(depth,dtype=np.int32);patterns=np.array([[(mask>>j)&1 for j in range(depth)] for mask in range(1<<depth)],np.uint8);chunks=[np.load(base/f'factors_{label}_{first}.npz') for first in range(0,L,256)];aa=np.concatenate([p['left'] for p in chunks]);bb=np.concatenate([p['right'] for p in chunks]);t=float(np.tanh(1.1));z=t*t;x=np.diag(np.r_[np.full(n,-t/(1+t)),np.full(n,t/(1-t))]);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);cov=(cov+cov.T)/2
 # Exact low-total no-click coefficients use traces of H^T H, H=V_R^dagger V_R.
 exactlow=np.ones(1<<depth,np.longdouble);vac=np.ones_like(exactlow)
 for mask in range(1<<depth):
  ids=[j for j in range(depth) if mask>>j&1];quad=ids+[m+j for j in ids]
  if ids:
   sign,ld=np.linalg.slogdet((cov[np.ix_(quad,quad)]+np.eye(len(quad)))/2);assert sign>0;vac[mask]=np.exp(-np.longdouble(ld)/2)
  remain=v[[j for j in range(m) if j not in ids]];h=remain.conj().T@remain;matrix=h.T@h;power=np.eye(n,dtype=complex);traces=[]
  for j in range(1,5):power=power@matrix;tr=np.trace(power);assert abs(tr.imag)<1e-10;traces.append(np.longdouble(tr.real)/2)
  coeff=np.ones(5,np.longdouble)
  for j in range(1,5):coeff[j]=sum(traces[q-1]*coeff[j-q] for q in range(1,j+1))/j
  exactlow[mask]=np.longdouble(1-z)**(np.longdouble(n)/2)*sum(coeff[j]*np.longdouble(z)**j for j in range(5))
 target=mobius(vac);np.savez_compressed(OUT/f'input_{label}.npz',v=v,x=x,cov=cov,t=t,target=target,vacuum=vac,exactlow=exactlow,patterns=patterns)
 with NativeFactorPNRSampler(v,t,OUT/'factor.dll',kernel=k,left=aa,right=bb,latent_weights=w,exact_total_budget=8,workers=16,runtime_dirs=runtime) as model:
  fn=model.lib.factor82_vacuum;fn.argtypes=[C.c_void_p,C.c_int,C.c_double,C.c_int,C.POINTER(C.c_int),C.c_int,DP,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;full=np.zeros(1<<depth);low=full.copy();stats=np.zeros(4);err=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(model._plan,n,z,depth,ports.ctypes.data_as(C.POINTER(C.c_int)),16,ptr(full),ptr(low),ptr(stats),err,len(err));elapsed=time.perf_counter()-start;assert rc==0,err.value
  law=mobius(full.astype(np.longdouble)-low+exactlow);np.savez_compressed(OUT/f'factor_{label}.npz',law=law,full_vacuum=full,low_vacuum=low,statistics=stats);row=dict(label=label,method='factor_pnr_b8',numerical_tv=float(abs(law-target).sum()/2),analysis_seconds=elapsed,plan_statistics=model.plan_statistics,normalization_error=float(abs(law.sum()-1)));rows.append(row);save(f'row_{label}_factor.json',row);print(row,flush=True)
  # Rebuild v1/v2 plans and compare identical integer records, no timing claim.
  replay=model.sample(4096,820182,fixed_total=34)['counts']
 with NativeFactorPNRSampler(v,t,ROOT/'results/iteration82_factor_pilot_v1/factor.dll',kernel=k,left=aa,right=bb,latent_weights=w,exact_total_budget=8,workers=16,runtime_dirs=runtime) as model:old=model.sample(4096,820182,fixed_total=34)['counts']
 assert np.array_equal(replay,old);np.savez_compressed(OUT/f'replay_{label}.npz',counts=replay)
 for cutoff in (2,3):
  q=np.zeros((1<<depth)-1);previous=q.copy();proposed=q.copy();stats=np.zeros(3);err=C.create_string_buffer(4096);start=time.perf_counter();rc=ffn(m,cutoff,depth,16,ptr(cov),ptr(q),ptr(previous),ptr(proposed),ptr(stats),err,len(err));assert rc==0,err.value;elapsed=time.perf_counter()-start
  tree=[dict(prefix=''.join(str((bits>>j)&1) for j in range(length)),threshold_hex=float(q[(1<<length)-1+bits]).hex()) for length in range(depth) for bits in range(1<<length)];result=full_law(depth,tree,model='uint64');law=np.array([float(p) for p in result['distribution']],np.longdouble);row=dict(label=label,method='frugal_cpu',budget=cutoff,numerical_tv=float(abs(law-target).sum()/2),analysis_seconds=elapsed,normalization_error=float(abs(law.sum()-1)));rows.append(row);save(f'row_{label}_frugal_{cutoff}.json',row);save(f'tree_{label}_{cutoff}.json',dict(tree=tree));np.savez_compressed(OUT/f'frugal_{label}_{cutoff}.npz',law=law,thresholds=q);print(row,flush=True)
 for budget in (4,8):
  try:
   with NativeBranchClickSampler(v,x,ROOT/'results/iteration71_branch_pilot_v1/branches.dll',branch_budget=budget,runtime_dirs=runtime) as model:law=np.array([model.prefix_probability(bits)['probability'] for bits in patterns],np.longdouble)
   row=dict(label=label,method='direct_cpu_law',budget=budget,numerical_tv=float(abs(law-target).sum()/2),normalization_error=float(abs(law.sum()-1)));np.savez_compressed(OUT/f'direct_{label}_{budget}.npz',law=law)
  except ArithmeticError as e:row=dict(label=label,method='direct_cpu_law',budget=budget,failed=True,error=str(e))
  rows.append(row);save(f'row_{label}_direct_{budget}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
