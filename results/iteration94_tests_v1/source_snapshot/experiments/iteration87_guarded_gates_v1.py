"""Build once, independently check scalar ball enclosures and small exact laws.
Numerical experiments are validation, not a proof of whole-state enclosures.
Paired-quad results remain a numerical oracle, not an exact real-arithmetic law.
"""
import argparse,subprocess,mpmath as mp
from iteration87_guarded_common_v1 import *
from iteration86_shared_gates_v4 import reference
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=ROOT/'native/gaussian_click_guarded87_v1.cpp';old=ROOT/'results/iteration86_shared_gates_v4/shared.dll';files=[Path(__file__),ROOT/'experiments/iteration87_guarded_common_v1.py',ROOT/'experiments/iteration86_shared_gates_v4.py',old,*[p for p in (ROOT/'native').iterdir() if p.suffix in ('.cpp','.hpp','.h')]];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},seed=870010,budgets=[1,8,32,64,128],scalar_precision=220))
gcc=runtime_dirs()[0]/'g++.exe';dll=out/'guarded.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(dll),'-lquadmath'];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save(out,'build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(dll));fn=lib.ball87_probe;fn.argtypes=[C.c_int,*([C.c_char_p]*6),C.c_int,C.c_char_p,C.c_int];fn.restype=C.c_int;mp.mp.dps=220;rng=np.random.default_rng(870010);scalar=[]
def probe(op,a,ar,b,br):
 mid=C.create_string_buffer(512);rad=C.create_string_buffer(512);err=C.create_string_buffer(4096);args=[mp.nstr(x,220).encode() for x in (a,ar,b,br)];rc=fn(op,*args,mid,rad,len(mid),err,len(err));row=dict(op=op,a=args[0].decode(),ar=args[1].decode(),b=args[2].decode(),br=args[3].decode(),returncode=rc,error=err.value.decode(),midpoint=mid.value.decode(),radius=rad.value.decode());scalar.append(row)
 if rc:return row,None
 c=mp.mpf(row['midpoint']);r=mp.mpf(row['radius']);return row,(c-r,c+r)
for trial in range(160):
 # Exactly representable dyadic centers/radii; decimal parsing is not an input
 # interval conversion. Binary64 centers ensure exact binary128 representation.
 a=mp.mpf(float(rng.normal()))*mp.power(2,int(rng.integers(-200,201)));b=mp.mpf(float(rng.uniform(.5,2)))*mp.power(2,int(rng.integers(-200,201)));ar=mp.power(2,int(mp.floor(mp.log(abs(a),2)))-int(rng.integers(10,80)));br=mp.power(2,int(mp.floor(mp.log(b,2)))-int(rng.integers(10,80)))
 for op in range(5):
  aa=abs(a)+2*ar if op==4 else a
  if op==4:
   # This center is still a short dyadic; it fits in binary128 exactly.
   exact=[mp.sqrt(aa-ar),mp.sqrt(aa+ar)]
  else:
   pairs=[(u,v) for u in (aa-ar,aa+ar) for v in (b-br,b+br)];exact=[u+v if op==0 else u-v if op==1 else u*v if op==2 else u/v for u,v in pairs]
  row,interval=probe(op,aa,ar,b,br);row['enclosed']=interval is not None and interval[0]<=min(exact) and max(exact)<=interval[1];save(out,'scalar_progress.json',dict(rows=scalar)) if not row['enclosed'] else None;assert row['enclosed'],row
for op,a0,ar0,b0,br0 in [(3,1,0,0,1),(4,0,1,1,0)]:
 row,interval=probe(op,*map(mp.mpf,(a0,ar0,b0,br0)));assert interval is None
save(out,'scalar.json',dict(passed=True,rows=scalar));print('scalar enclosure checks',len(scalar),flush=True)
rows=[];small=ROOT/'results/iteration86_shared_gates_v4';cases=[]
for name in ('thermal','lossy_squeezed','physical_first8'):
 d=np.load(small/f'input_{name}.npz');cases.append((name,d['v'],d['x'],d['target']))
for intensity in (.001,1e-8):
 v=np.eye(8,dtype=complex);x=np.eye(16)*intensity;prob=mp.mpf(float(intensity))/(1+mp.mpf(float(intensity)));target=np.array([float(prob**label.bit_count()*(1-prob)**(8-label.bit_count())) for label in range(256)]);cases.append((f'diagonal_{intensity:g}',v,x,target))
for name,v,x,target in cases:
 independent=reference(v,x);assert abs(independent-target).sum()/2<2e-11;draws=np.array([[0. if label>>j&1 else np.nextafter(1.,0.) for j in range(8)] for label in range(256)])
 for budget in (1,8,32,64,128):
  outputs=[]
  for guarded,path in ((False,old),(True,dll)):
   with NativeStableClickSampler(v,x,path,branch_budget=budget,runtime_dirs=runtime_dirs()) as model:raw,meta=sample(model,draws,budget,guarded)
   np.savez_compressed(out/f'raw_{name}_{budget}_{int(guarded)}.npz',**raw);outputs.append(raw);assert meta['returncode']==0,meta
  comparison=compare(*outputs);row=dict(case=name,budget=budget,**comparison);rows.append(row);save(out,f'comparison_{name}_{budget}.json',row);assert comparison['equal_decisions_errors_state_counters'] and comparison['oracle_probabilities_enclosed'] and comparison['counter_identities'],row
  if budget==128:
   maximum=0.
   for label in range(256):
    for j in range(8):
     prefix=label&((1<<j)-1);indices=[k for k in range(256) if k&((1<<j)-1)==prefix];den=target[indices].sum();num=target[[k for k in indices if k>>j&1]].sum();maximum=max(maximum,abs(outputs[1]['conditional'][label,j]-num/den))
   save(out,f'independent_{name}.json',dict(maximum_conditional_difference=maximum));assert maximum<2e-10,(name,maximum)
 print('small gates',name,flush=True)
with NativeStableClickSampler(np.eye(2,dtype=complex),np.zeros((4,4)),dll,runtime_dirs=runtime_dirs()) as model:
 raw,meta=sample(model,np.array([[0.,0.],[.5,.9]]),64);assert meta['returncode']==0 and not raw['clicks'].any() and not raw['conditional'].any();np.savez_compressed(out/'vacuum.npz',**raw)
save(out,'summary.json',dict(passed=True,scalar_checks=len(scalar),rows=rows,scope=__doc__));manifest(out)
