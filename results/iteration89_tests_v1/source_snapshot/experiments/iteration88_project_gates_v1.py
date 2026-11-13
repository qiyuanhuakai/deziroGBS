"""Matched full16-record projected numeric/guarded paths and thread/chunk gates.
Also enumerate the actual full model's first8 outputs for both spectra and form
their independent OR law. Numerical finite-budget probabilities are not formal
whole-state certificates; the guard's midpoints are never multiplied as a law.
"""
import argparse
from iteration88_project_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);pilot=ROOT/'results/iteration88_project_pilot_v2';assert read(pilot/'summary.json')['completed'] and read(pilot/'projection.json')['passed'];dll=pilot/'projected.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';reference=ROOT/'results/iteration85_joint_quality_v1/reference_0.npz';planpath=ROOT/'research/iteration88_full_quality_plan.json';files=[Path(__file__),ROOT/'experiments/iteration88_project_common_v1.py',ROOT/'experiments/iteration87_guarded_common_v1.py',dll,inp,reference,planpath];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},records=16,experiment=0x88003001,budgets=[16,64],sectors=[0,1],threads=16));z=np.load(inp);v,x=z['v'],z['x'];rows=[];laws={}
for budget in (16,64):
 for sector in range(2):
  draws=uniforms(0x88003001,sector,0,16,len(v))
  with NativeStableClickSampler(v,x[sector],dll,branch_budget=budget,runtime_dirs=runtime_dirs()) as model:
   old,om=projected(model,draws,budget,0);new,nm=projected(model,draws,budget,1);c=compare(old,new);same_fit=bool(np.array_equal(old['statistics'][:,12:],new['statistics'][:,12:]));row=dict(budget=budget,sector=sector,oracle=om,guarded=nm,comparison=c,identical_projection_stats=same_fit);rows.append(row);np.savez_compressed(out/f'oracle_{budget}_{sector}.npz',**old);np.savez_compressed(out/f'guarded_{budget}_{sector}.npz',**new);save(out,f'row_{budget}_{sector}.json',row);assert om['returncode']==0 and nm['returncode']==0 and same_fit and c['equal_decisions_errors_state_counters'] and c['oracle_probabilities_enclosed'] and c['counter_identities'],row
   serial,sm=projected(model,draws[:7],budget,1,1);left,lm=projected(model,draws[:3],budget,1,2);right,rm=projected(model,draws[3:7],budget,1,4);joined={key:np.concatenate([left[key],right[key]]) for key in new};np.savez_compressed(out/f'serial_{budget}_{sector}.npz',**serial);np.savez_compressed(out/f'chunks_{budget}_{sector}.npz',**joined);same=all(np.array_equal(new[key][:7],serial[key],equal_nan=True) and np.array_equal(new[key][:7],joined[key],equal_nan=True) for key in new);assert same and all(meta['returncode']==0 for meta in (sm,lm,rm));save(out,f'invariance_{budget}_{sector}.json',dict(passed=same))
   for method in ('block','projected'):
    fn=getattr(model.lib,'tc86_shared_prefix' if method=='block' else 'tc88_project_prefix');fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int];law=[];statistics=[]
    for label in range(256):
     bits=np.array([(label>>j)&1 for j in range(8)],np.uint8);prob=np.zeros(1);s=np.zeros(6 if method=='block' else 10);err=C.create_string_buffer(4096);rc=fn(model.handle,8,bits.ctypes.data_as(BP),budget,ptr(prob),ptr(s),err,len(err));assert rc==0,(method,budget,sector,label,err.value);law.append(prob[0]);statistics.append(s)
    law=np.array(law);assert law.min()>=0 and abs(law.sum()-1)<2e-11;np.savez_compressed(out/f'prefix_{method}_{budget}_{sector}.npz',law=law,statistics=statistics);laws[method,budget,sector]=law
  print('project full gates',budget,sector,'wide',c['old_wide_calls'],c['new_wide_calls'],flush=True)
prefix=[];target=np.load(reference)['target']
for budget in (16,64):
 for method in ('block','projected'):
  law=np.zeros(256)
  for i,p0 in enumerate(laws[method,budget,0]):
   for j,p1 in enumerate(laws[method,budget,1]):law[i|j]+=p0*p1
  row=dict(method=method,budget=budget,numerical_tv=float(abs(law-target).sum()/2),mass=float(law.sum()));prefix.append(row);np.savez_compressed(out/f'prefix_combined_{method}_{budget}.npz',law=law,target=target);assert abs(law.sum()-1)<2e-11
save(out,'summary.json',dict(passed=True,rows=rows,prefix=prefix,scope=__doc__));manifest(out)
