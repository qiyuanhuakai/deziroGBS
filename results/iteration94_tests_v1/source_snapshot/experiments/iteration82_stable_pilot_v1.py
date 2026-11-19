"""Diagnose all weak-sector failures and test extended Gaussian/binary128
weights against the same three fixed Jiuzhang eight-bin marginal references.
No probability clipping, threshold relaxation or deletion of failed paths.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,argparse,ctypes as C,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_click_branches import NativeBranchClickSampler
from gbslab.native_direct_click import DP,BP
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';runtime=[gcc.parent];source=ROOT/'native/gaussian_click_stable82_v1.cpp';base=ROOT/'results/iteration81_jiuzhang_accuracy_v3';oldlib=ROOT/'results/iteration71_branch_pilot_v1/branches.dll';sources=[Path(__file__),source,ROOT/'native/gaussian_click_branches_v1.cpp',ROOT/'native/gaussian_direct_click.cpp',ROOT/'native/gaussian_reset_cpu.cpp',oldlib]
save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},groups=[0,1,2],budgets=[1,2,4,8,16,32,128],reference_digits=80,timing='exploratory',fixed_input_hashes={str(i):H(base/f'input_{i}.npz') for i in range(3)}))
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(source),'-lquadmath','-o',str(OUT/'stable.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr;print('Stable branch sampler built',flush=True)
mp.mp.dps=80;patterns=np.array([[(i>>j)&1 for j in range(8)] for i in range(256)],np.uint8)
def reference(c):
 v=[]
 for mask in range(256):
  ids=[j for j in range(8) if mask>>j&1];ids+=[8+j for j in ids]
  v.append(1/mp.sqrt(mp.det(mp.matrix([[(mp.mpf(float(c[i,j]))+int(i==j))/2 for j in ids] for i in ids]))) if ids else mp.mpf(1))
 law=[v[i^255] for i in range(256)]
 for bit in range(8):
  for i in range(256):
   if i>>bit&1:law[i]-=law[i^(1<<bit)]
 return np.array([float(p) for p in law])
def combine(a,b):
 out=np.zeros(256,np.longdouble)
 for i in range(256):
  for j in range(256):out[i|j]+=np.longdouble(a[i])*np.longdouble(b[j])
 return out
rows=[]
for g in range(3):
 inp=np.load(base/f'input_{g}.npz');target=inp['two_common_sectors'];covs=inp['covariances'];refs=[reference(c) for c in covs];np.savez_compressed(OUT/f'references_{g}.npz',sector_laws=refs,combined=combine(*refs));assert abs(combine(*refs)-target).sum()<1e-11
 for budget in (1,2,4,8,16,32,128):
  sectorlaws=[]
  for sector,cov in enumerate(covs):
   for method,dll in [('old',oldlib),('stable',OUT/'stable.dll')]:
    laws=np.full(256,np.nan);stats=np.zeros((256,6));errors={};badstates={}
    with NativeBranchClickSampler(np.eye(8),(cov-np.eye(16))/2,dll,branch_budget=budget,runtime_dirs=runtime) as model:
     if method=='stable':
      fn=model.lib.tc82_prefix;fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
     def probe(bits):
      bits=np.ascontiguousarray(bits,np.uint8)
      if method=='old':
       result=model.prefix_probability(bits);return result['probability'],[result[k] for k in ('component_evaluations','compressions','peak_transient_components','maximum_absolute_weight_sum')]+[0,len(bits)]
      prob=np.zeros(1);ss=np.zeros(6);err=C.create_string_buffer(4096);rc=fn(model.handle,len(bits),bits.ctypes.data_as(BP),budget,prob.ctypes.data_as(DP),ss.ctypes.data_as(DP),err,len(err))
      if rc:raise ArithmeticError(err.value.decode())
      return float(prob[0]),ss
     for i,bits in enumerate(patterns):
      try:laws[i],stats[i]=probe(bits)
      except ArithmeticError as e:
       errors[str(i)]=str(e)
       for length in range(1,9):
        try:previous,previous_stats=probe(bits[:length])
        except ArithmeticError:
         key=''.join(map(str,bits[:length-1]));badstates[key]=dict(reachable_prefix_probability=previous,previous_condition=previous_stats[3],first_failed_probe_length=length);break
     if method=='stable' and not errors and g==0 and budget in (8,128):
      fn2=model.lib.tc82_sample;fn2.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,BP,DP,C.c_char_p,C.c_int];fn2.restype=C.c_int;sample=np.zeros((8192,8),np.uint8);ss=np.zeros(7);err=C.create_string_buffer(4096);start=time.perf_counter();rc=fn2(model.handle,8192,820820+sector,16,budget,sample.ctypes.data_as(BP),ss.ctypes.data_as(DP),err,len(err));elapsed=time.perf_counter()-start;assert rc==0,err.value;np.savez_compressed(OUT/f'samples_{g}_{budget}_{sector}.npz',clicks=sample,statistics=ss);save(f'sample_time_{g}_{budget}_{sector}.json',dict(seconds=elapsed,statistics=ss.tolist()))
    minimal={k:v for k,v in badstates.items() if not any(k.startswith(other) and k!=other for other in badstates)};abort=sum(v['reachable_prefix_probability'] for v in minimal.values());row=dict(group=g,sector=sector,budget=budget,method=method,failed_patterns=len(errors),failure_prefixes=minimal,abort_probability_under_continuous_draws=abort,target_mass_of_failed_patterns=float(refs[sector][[int(i) for i in errors]].sum()),numerical_tv=None if errors else float(abs(laws-refs[sector]).sum()/2),normalization_error=None if errors else float(abs(laws.sum()-1)),maximum_weight_condition=float(stats[:,3].max()),maximum_mass_residual=float(stats[:,4].max()))
    rows.append(row);save(f'row_{g}_{sector}_{budget}_{method}.json',row);np.savez_compressed(OUT/f'law_{g}_{sector}_{budget}_{method}.npz',law=laws,statistics=stats);save(f'errors_{g}_{sector}_{budget}_{method}.json',errors)
    if method=='stable':sectorlaws.append(None if errors else laws)
  if all(p is not None for p in sectorlaws):
   law=combine(*sectorlaws);row=dict(group=g,budget=budget,method='stable_two_sectors_or',numerical_tv=float(abs(law-target).sum()/2),normalization_error=float(abs(law.sum()-1)));rows.append(row);save(f'combined_{g}_{budget}.json',row);np.savez_compressed(OUT/f'combined_{g}_{budget}.npz',law=law);print(row,flush=True)
  else:print('Stable failure',g,budget,flush=True)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
