"""Complete small laws, full numeric/guarded replay, and pure detector boundaries."""
import argparse,subprocess
from iteration89_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);pilot=ROOT/'results/iteration89_pilot_v1';assert read(pilot/'summary.json')['passed'];dll=pilot/'physical.dll';files=[Path(__file__),ROOT/'experiments/iteration89_common_v1.py',dll,pilot/'puretree.dll',ROOT/'native/gaussian_pnr_puretree_probe89_v1.cpp',ROOT/'research/iteration89_quality_plan.json'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}));rows=[]
for case in ('thermal','lossy_squeezed','physical_first8'):
 z=np.load(ROOT/f'results/iteration86_shared_gates_v4/input_{case}.npz')
 with NativeStableClickSampler(z['v'],z['x'],dll,branch_budget=128,runtime_dirs=runtime_dirs()) as model:
  fn=model.lib.tc89_project_prefix;fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int]
  for budget in (1,2,8,16,64,128):
   law=[]
   for label in range(256):
    bits=np.array([(label>>j)&1 for j in range(8)],np.uint8);prob=np.zeros(1);s=np.zeros(10);err=C.create_string_buffer(4096);rc=fn(model.handle,8,bits.ctypes.data_as(BP),budget,ptr(prob),ptr(s),err,len(err));assert rc==0,(case,budget,label,err.value);law.append(prob[0])
   law=np.array(law);row=dict(case=case,budget=budget,mass=float(law.sum()),numerical_tv=float(abs(law-z['target']).sum()/2));assert law.min()>=0 and abs(law.sum()-1)<2e-11,row
   if budget==128:assert row['numerical_tv']<2e-11,row
   rows.append(row);np.savez_compressed(out/f'law_{case}_{budget}.npz',law=law,target=z['target']);draws=uniforms(0x89003001,0,0,16,8);old,om=physical(model,draws,budget,guarded=0);new,nm=physical(model,draws,budget);c=compare(old,new);assert om['returncode']==0 and nm['returncode']==0 and c['equal_decisions_errors_state_counters'] and c['oracle_probabilities_enclosed'] and c['counter_identities'],c
 print('small laws',case,flush=True)
save(out,'small.json',dict(passed=True,rows=rows));z=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz');full=[]
for budget in (16,64):
 for sector in (0,1):
  with NativeStableClickSampler(z['v'],z['x'][sector],dll,branch_budget=budget,runtime_dirs=runtime_dirs()) as model:
   draws=uniforms(0x89003002,sector,0,16,len(z['v']));old,om=physical(model,draws,budget,guarded=0);new,nm=physical(model,draws,budget);c=compare(old,new);assert om['returncode']==0 and nm['returncode']==0 and c['equal_decisions_errors_state_counters'] and c['oracle_probabilities_enclosed'] and c['counter_identities'] and np.array_equal(old['statistics'][:,12:],new['statistics'][:,12:]),c;np.savez_compressed(out/f'oracle_{budget}_{sector}.npz',**old);np.savez_compressed(out/f'guarded_{budget}_{sector}.npz',**new);serial,sm=physical(model,draws[:7],budget,threads=1);left,lm=physical(model,draws[:3],budget,threads=2);right,rm=physical(model,draws[3:7],budget,threads=4);joined={k:np.concatenate([left[k],right[k]]) for k in new};assert all(np.array_equal(new[k][:7],serial[k],equal_nan=True) and np.array_equal(new[k][:7],joined[k],equal_nan=True) for k in new);np.savez_compressed(out/f'serial_{budget}_{sector}.npz',**serial);np.savez_compressed(out/f'chunks_{budget}_{sector}.npz',**joined);full.append(dict(budget=budget,sector=sector,comparison=c));print('full physical gates',budget,sector,flush=True)
save(out,'full.json',dict(passed=True,rows=full));gcc=runtime_dirs()[0]/'g++.exe';probe=out/'puretree_probe.dll';cmd=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(ROOT/'native/gaussian_pnr_puretree_probe89_v1.cpp'),'-o',str(probe),'-lquadmath'];r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');save(out,'build.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;cases=[];rng=np.random.default_rng(890001)
for n,m in ((2,5),(3,12),(5,17)):
 v=np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0];t=np.linspace(.1,.7,n);x=np.diag(np.r_[-t/(1+t),t/(1-t)])
 with NativeDirectClickSampler(v,x,probe,runtime_dirs=runtime_dirs()) as model:
  handle,destroy=tree(model,4);fn=model.lib.pnr89_pure_probe;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,IP,C.c_int,DP,IP,DP,DP,C.c_char_p,C.c_int]
  try:
   for length in range(6):
    for displaced in (0,1):
     mean=np.ascontiguousarray(rng.normal(size=2*n)*.2*displaced);modes=np.array([(j*3)%m for j in range(length)],np.int32);rates=np.zeros(m);stats=np.zeros(4);err=C.create_string_buffer(4096);draws=np.array([.2]);selected=np.zeros((1,2),np.int32);rc=fn(model.handle,handle,ptr(t),ptr(mean),length,modes.ctypes.data_as(IP),1,ptr(draws),selected.ctypes.data_as(IP),ptr(rates),ptr(stats),err,len(err));assert rc==0,err.value;edges=np.cumsum(rates)[:-1];draws=np.ascontiguousarray(np.r_[0.,np.nextafter(1.,0.),rng.random(12),np.nextafter(edges,0.),edges,np.nextafter(edges,1.)]);draws=draws[(draws>=0)&(draws<1)];selected=np.zeros((len(draws),2),np.int32);rc=fn(model.handle,handle,ptr(t),ptr(mean),length,modes.ctypes.data_as(IP),len(draws),ptr(draws),selected.ctypes.data_as(IP),ptr(rates),ptr(stats),err,len(err));row=dict(n=n,m=m,length=length,displaced=displaced,returncode=rc,draws=len(draws),equal=bool(np.array_equal(selected[:,0],selected[:,1])),statistics=stats.tolist());cases.append(row);np.savez_compressed(out/f'pure_{n}_{length}_{displaced}.npz',v=v,t=t,mean=mean,modes=modes,draws=draws,selected=selected,rates=rates,statistics=stats);assert rc==0 and row['equal'],(row,err.value)
  finally:destroy(handle)
 print('pure detector boundary cases',n,flush=True)
save(out,'pure_detector.json',dict(passed=True,rows=cases));save(out,'summary.json',dict(passed=True,small_laws=len(rows),full_gates=len(full),pure_boundary_cases=len(cases),pure_boundary_draws=sum(r['draws'] for r in cases),scope=__doc__));manifest(out)
