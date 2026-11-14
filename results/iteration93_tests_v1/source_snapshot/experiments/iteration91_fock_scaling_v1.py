"""Serial finite-core growth through 2/4/6 complex modes and cutoff12/20/28.
All rows use k4,r0.3. Enumerated layer counts and unresolved tail are explicit;
these costs do not imply polynomial scaling for unrestricted many-mode GBS.
"""
import argparse,statistics
from fractions import Fraction as F
from iteration91_fock_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration91_fock_build_v1/fock.dll';files=[Path(__file__),dll,ROOT/'experiments/iteration91_fock_common_v1.py'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},modes=[2,4,6],cutoffs=[12,20,28],k=4,r=.3,repetitions=3));runtime=os.add_dll_directory(str(runtime_dirs()[0]));lib=C.CDLL(str(dll));rng=np.random.default_rng(910061);rows=[]
for m in (2,4,6):
 u=np.linalg.qr(rng.normal(size=(m,m))+1j*rng.normal(size=(m,m)))[0][:,:2];np.savez_compressed(out/f'input_{m}.npz',u=u)
 for cutoff in (12,20,28):
  repeats=[]
  for rep in range(3):
   keys,probs,meta=core_law(lib,u,4,.3,cutoff);assert meta['returncode']==0,meta;repeats.append(meta);count=sum(math.comb(n+m-1,m-1) for n in range(4,cutoff+1,2));assert count==len(keys)
  s=F(.3)**2;J=(cutoff-4)//2;tail=sum((F(math.comb(J+5,h))*(1-s)**h*s**(J+5-h) for h in range(5)),F(0));row=dict(modes=m,cutoff=cutoff,patterns=count,returned_array_bytes=16*count,binding_allocated_array_bytes=16*math.comb(cutoff+m,m),tail=[str(tail.numerator),str(tail.denominator)],tail_float=float(tail),median_native_seconds=statistics.median(r['seconds'] for r in repeats),repetitions=repeats);rows.append(row);print('scaling91',m,cutoff,count,row['median_native_seconds'],row['tail_float'],flush=True)
save(out,'summary.json',dict(passed=True,rows=rows,scope=__doc__));manifest(out)
