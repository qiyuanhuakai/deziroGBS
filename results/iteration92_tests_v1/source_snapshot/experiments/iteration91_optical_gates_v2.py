"""Complete small laws, actual optical-space numeric/guarded and thread/chunk gates."""
import argparse
from iteration91_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration91_build_v3/optical.dll';files=[Path(__file__),ROOT/'experiments/iteration91_common_v1.py',dll,ROOT/'results/iteration91_cache_audit_v4/summary.json'];assert read(files[-1])['passed'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}));rows=[]
for case in ('thermal','lossy_squeezed','physical_first8'):
 z=np.load(ROOT/f'results/iteration86_shared_gates_v4/input_{case}.npz')
 with NativeStableClickSampler(z['v'],z['x'],dll,branch_budget=128,runtime_dirs=runtime_dirs()) as model:
  cache,destroy,setup=optical_cache(model)
  fn=model.lib.tc91_project_prefix;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int]
  for budget in (1,2,8,16,64,128):
   law=[]
   for label in range(256):
    bits=np.array([(label>>j)&1 for j in range(8)],np.uint8);prob=np.zeros(1);s=np.zeros(10);err=C.create_string_buffer(4096);rc=fn(model.handle,cache,8,bits.ctypes.data_as(BP),budget,ptr(prob),ptr(s),err,len(err));assert rc==0,(case,budget,label,err.value);law.append(prob[0])
   law=np.array(law);row=dict(case=case,budget=budget,mass=float(law.sum()),numerical_tv=float(abs(law-z['target']).sum()/2));assert law.min()>=0 and abs(law.sum()-1)<2e-11,row
   if budget==128:assert row['numerical_tv']<2e-11,row
   rows.append(row);np.savez_compressed(out/f'law_{case}_{budget}.npz',law=law,target=z['target']);draws=uniforms(0x91003001,0,0,16,8);old,om=optical_sample(model,cache,draws,budget,guarded=0);new,nm=optical_sample(model,cache,draws,budget);c=compare(old,new);assert om['returncode']==0 and nm['returncode']==0 and c['equal_decisions_errors_state_counters'] and c['oracle_probabilities_enclosed'] and c['counter_identities'],c
  destroy(cache)
 print('small laws',case,flush=True)
save(out,'small.json',dict(passed=True,rows=rows));z=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz');full=[]
for budget in (16,64):
 for sector in (0,1):
  with NativeStableClickSampler(z['v'],z['x'][sector],dll,branch_budget=budget,runtime_dirs=runtime_dirs()) as model:
   cache,destroy,setup=optical_cache(model)
   draws=uniforms(0x91003002,sector,0,16,len(z['v']));old,om=optical_sample(model,cache,draws,budget,guarded=0);new,nm=optical_sample(model,cache,draws,budget);c=compare(old,new);assert om['returncode']==0 and nm['returncode']==0 and c['equal_decisions_errors_state_counters'] and c['oracle_probabilities_enclosed'] and c['counter_identities'] and np.array_equal(old['statistics'][:,12:],new['statistics'][:,12:]),c;np.savez_compressed(out/f'oracle_{budget}_{sector}.npz',**old);np.savez_compressed(out/f'guarded_{budget}_{sector}.npz',**new);serial,sm=optical_sample(model,cache,draws[:7],budget,threads=1);left,lm=optical_sample(model,cache,draws[:3],budget,threads=2);right,rm=optical_sample(model,cache,draws[3:7],budget,threads=4);joined={k:np.concatenate([left[k],right[k]]) for k in new};assert all(np.array_equal(new[k][:7],serial[k],equal_nan=True) and np.array_equal(new[k][:7],joined[k],equal_nan=True) for k in new);np.savez_compressed(out/f'serial_{budget}_{sector}.npz',**serial);np.savez_compressed(out/f'chunks_{budget}_{sector}.npz',**joined);full.append(dict(budget=budget,sector=sector,comparison=c));print('full optical gates',budget,sector,flush=True);destroy(cache)
save(out,'full.json',dict(passed=True,rows=full));save(out,'summary.json',dict(passed=True,small_laws=len(rows),full_gates=len(full),scope=__doc__));manifest(out)
