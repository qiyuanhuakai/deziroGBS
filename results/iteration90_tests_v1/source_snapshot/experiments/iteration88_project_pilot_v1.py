"""Build moment-preserving closure, enumerate small complete laws, then paired
32-record full-model exploratory paths. This is not a fresh quality family.
"""
import argparse,subprocess
from iteration88_project_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=ROOT/'native/gaussian_click_projected88_v1.cpp';dll=out/'projected.dll';old=ROOT/'results/iteration87_guarded_gates_v2/guarded.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),ROOT/'experiments/iteration88_project_common_v1.py',ROOT/'experiments/iteration87_guarded_common_v1.py',old,inp,*[p for p in (ROOT/'native').iterdir() if p.suffix in ('.cpp','.hpp','.h')]];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},small_budgets=[1,2,4,8,16,32,64,128],full_budgets=[16,32,64],full_records=32,experiment=0x88002001,threads=16))
gcc=runtime_dirs()[0]/'g++.exe';cmd=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(dll),'-lquadmath'];start=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');save(out,'build.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr
small=ROOT/'results/iteration86_shared_gates_v4';rows=[]
for name in ('thermal','lossy_squeezed','physical_first8'):
 d=np.load(small/f'input_{name}.npz');v,x,target=d['v'],d['x'],d['target']
 with NativeStableClickSampler(v,x,dll,branch_budget=128,runtime_dirs=runtime_dirs()) as model:
  fn=model.lib.tc88_project_prefix;fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int]
  for budget in (1,2,4,8,16,32,64,128):
   law=[];stats=[];start=time.perf_counter()
   for label in range(256):
    bits=np.array([(label>>j)&1 for j in range(8)],np.uint8);prob=np.zeros(1);s=np.zeros(10);err=C.create_string_buffer(4096);rc=fn(model.handle,8,bits.ctypes.data_as(BP),budget,ptr(prob),ptr(s),err,len(err));assert rc==0,(name,budget,label,err.value);law.append(prob[0]);stats.append(s)
   law=np.array(law);row=dict(case=name,budget=budget,mass=float(law.sum()),minimum=float(law.min()),tv=float(abs(law-target).sum()/2),seconds=time.perf_counter()-start);rows.append(row);np.savez_compressed(out/f'law_{name}_{budget}.npz',law=law,target=target,statistics=stats);save(out,f'law_{name}_{budget}.json',row);assert law.min()>=0 and abs(law.sum()-1)<2e-11,row
   if budget==128:assert row['tv']<2e-11,row
   if budget in (1,8,32,64,128):
    oldlaw=np.load(small/f'law_{name}_shared_{budget}.npz')['law'];row['old_tv']=float(abs(oldlaw-target).sum()/2)
   draws=uniforms(0x88002002,0,0,16,8);a,am=projected(model,draws,budget,0);b,bm=projected(model,draws,budget,1);c=compare(a,b);save(out,f'guard_{name}_{budget}.json',dict(old=am,new=bm,comparison=c));assert am['returncode']==0 and bm['returncode']==0 and c['equal_decisions_errors_state_counters'] and c['oracle_probabilities_enclosed'] and c['counter_identities'],c
 print('window small',name,flush=True)
save(out,'small.json',dict(passed=True,rows=rows));data=np.load(inp);v,x=data['v'],data['x'];full=[]
for budget in (16,32,64):
 for method in ('block','window'):
  parts=[];good=[];metas=[]
  for sector in range(2):
   draws=uniforms(0x88002001,sector,0,32,len(v))
   with NativeStableClickSampler(v,x[sector],old if method=='block' else dll,branch_budget=budget,runtime_dirs=runtime_dirs()) as model:raw,meta=sample(model,draws,budget) if method=='block' else projected(model,draws,budget)
   np.savez_compressed(out/f'raw_{method}_{budget}_{sector}.npz',**raw);save(out,f'raw_{method}_{budget}_{sector}.json',meta);parts.append(raw['clicks']);good.append((raw['errors'][:,0]==0)&(raw['statistics'][:,5]==len(v)));metas.append(meta);print('window full',method,budget,sector,meta,flush=True)
  bits=np.bitwise_or(*parts);valid=good[0]&good[1];np.savez_compressed(out/f'combined_{method}_{budget}.npz',clicks=bits,valid=valid);row=dict(method=method,budget=budget,failed=int((~valid).sum()),mean_successful=float(bits[valid].sum(1).mean()) if valid.any() else None,seconds=sum(m['seconds'] for m in metas));full.append(row);save(out,f'combined_{method}_{budget}.json',row)
save(out,'summary.json',dict(completed=True,small=rows,full=full,scope=__doc__));manifest(out)
