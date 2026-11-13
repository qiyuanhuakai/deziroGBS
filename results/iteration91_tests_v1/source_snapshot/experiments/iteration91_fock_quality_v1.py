"""Complete finite-occupation PNR laws after complex passive mixing.
Positive lattice mixtures preserve the input mean exactly. All mass beyond
the frozen total-photon cutoff is retained in analytic error brackets; no
renormalization. These are conditional U S2(r)|k,0> families, not general GBS.
"""
import argparse,math,mpmath as mp
from fractions import Fraction as F
from iteration91_fock_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration91_fock_build_v1/fock.dll';files=[Path(__file__),ROOT/'experiments/iteration91_fock_common_v1.py',dll];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},cases=[dict(modes=3,r=.6,cutoff=64),dict(modes=4,r=.3,cutoff=32)],targets=[4,6],lattices=[1,2,3,4,8],reference_digits=100,sample_records=65536,experiments=[hex(0x91005001),hex(0x91005002)],purpose=5));runtime=os.add_dll_directory(str(runtime_dirs()[0]));lib=C.CDLL(str(dll));rng=np.random.default_rng(910051);mp.mp.dps=100
def reference(u,k,r,key):
 n=[(int(key)>>(8*j))&255 for j in range(len(u))];total=sum(n)
 if total<k or (total-k)%2:return mp.mpf(0)
 l=(total-k)//2;target=k+l;coeff=[mp.mpc(1)]
 for j,count in enumerate(n):
  row=[mp.binomial(count,a)*mp.mpc(complex(u[j,0]))**a*mp.mpc(complex(u[j,1]))**(count-a) for a in range(count+1)];new=[mp.mpc(0)]*(len(coeff)+count)
  for a0,x in enumerate(coeff):
   for b0,y in enumerate(row):new[a0+b0]+=x*y
  coeff=new
 amplitude=coeff[target]*mp.sqrt(mp.factorial(k+l)*mp.factorial(l)/mp.fprod(mp.factorial(t) for t in n));s=mp.mpf(r)**2;weight=mp.binomial(k+l,k)*(1-s)**(k+1)*s**l;return abs(amplitude)**2*weight
def mass(k,r,cut):
 s=F(r)**2;return sum((F(math.comb(k+j,k))*(1-s)**(k+1)*s**j for j in range((cut-k)//2+1)),F(0))
def rational(x):return [str(x.numerator),str(x.denominator)]
rows=[];cases=[];references=[];draw_rows=[]
for case,(m,r,cut) in enumerate(((3,.6,64),(4,.3,32))):
 u=np.linalg.qr(rng.normal(size=(m,m))+1j*rng.normal(size=(m,m)))[0][:,:2];np.savez_compressed(out/f'input_{case}.npz',u=u);bank={};masses={};keys_union=set();native=[]
 for k in range(cut+1):
  keys,probs,meta=core_law(lib,u,k,r,cut);assert meta['returncode']==0,meta;expected=mass(k,r,cut);error=abs(float(expected)-float(probs.sum()));assert error<2e-11 and meta['stats'][2]<2e-10,(case,k,error,meta);bank[k]=(keys,probs);masses[k]=expected;keys_union.update(map(int,keys));native.append(dict(k=k,exact_mass=rational(expected),sum_error=error,**meta));np.savez_compressed(out/f'law_{case}_{k}.npz',keys=keys,probabilities=probs)
  if k in (0,1,4,6):
   eligible=np.flatnonzero(np.array([sum((int(v)>>(8*j))&255 for j in range(m)) for v in keys])<=12);selected=eligible[np.linspace(0,len(eligible)-1,min(20,len(eligible)),dtype=int)]
   for index in selected:
    ref=reference(u,k,r,keys[index]);error=abs(probs[index]-float(ref));assert error<3e-13;references.append(dict(case=case,k=k,key=int(keys[index]),native=float(probs[index]),reference=mp.nstr(ref,90),absolute_error=error))
  if k%8==0:print('fock bank',case,k,meta['seconds'],flush=True)
 keys=np.array(sorted(keys_union),np.uint64);aligned={k:np.zeros(len(keys)) for k in bank}
 for k,(local,p0) in bank.items():aligned[k][np.searchsorted(keys,local)]=p0
 labels=np.zeros(len(keys),np.int32)
 for j in range(m):labels|=(((keys>>(8*j))&255)>0).astype(np.int32)<<j
 for k in (4,6):
  target=aligned[k];target_tail=1-masses[k]
  for lattice in (1,2,3,4,8):
   residue=k%lattice;q=k//lattice;ratio=F(q,q+1);model=np.zeros(len(keys));model_mass=F(0)
   for source in range(residue,cut+1,lattice):
    j=(source-residue)//lattice;weight=(1-ratio)*ratio**j;model+=float(weight)*aligned[source];model_mass+=weight*masses[source]
   tail=1-model_mass;l1=float(abs(target-model).sum()/2);lower=l1+float(abs(target_tail-tail))/2;upper=l1+float(target_tail+tail)/2;exact_input_tv=1-(1-ratio)*ratio**q
   # Contractivity under the specified squeezing and any passive interferometer.
   # Complete output TV can be smaller than the input trace distance after mixing.
   upper=min(upper,float(exact_input_tv));lower=min(lower,upper) if abs(lower-upper)<3e-13 else lower;assert lower<=upper+3e-12
   target_click=np.bincount(labels,weights=target,minlength=1<<m);model_click=np.bincount(labels,weights=model,minlength=1<<m);cl1=float(abs(target_click-model_click).sum()/2);click_lo=max(0.,cl1-float(target_tail+tail)/2);click_hi=min(1.,cl1+float(target_tail+tail)/2,float(exact_input_tv));row=dict(case=case,modes=m,r=r,cutoff=cut,target_photons=k,lattice=lattice,input_trace_distance=rational(exact_input_tv),target_tail=rational(target_tail),model_tail=rational(tail),pnr_tv_lower=lower,pnr_tv_upper=upper,click_tv_lower=click_lo,click_tv_upper=click_hi,retained_pnr_l1_half=l1,scope='Analytic truncation bounds plus independently checked floating-point probabilities, not interval-certified native amplitudes');rows.append(row);np.savez_compressed(out/f'comparison_{case}_{k}_{lattice}.npz',keys=keys,target=target,model=model,target_click=target_click,model_click=model_click)
   if lattice==8:assert np.array_equal(target,model) and float(exact_input_tv)==0
  # Execute the complete finite-core sampling path; unresolved tail is a failure.
  draws=uniforms(0x91005001+case,0,0,65536,1).ravel();result,failed,meta=core_draw(lib,keys,target,draws);assert meta['returncode']==0,meta;cdf=np.cumsum(target.astype(np.longdouble));idx=np.searchsorted(cdf,draws,side='right');assert np.array_equal(failed,idx==len(keys));valid=idx<len(keys);assert np.array_equal(result[valid],keys[idx[valid]]);np.savez_compressed(out/f'samples_{case}_{k}.npz',draws=draws,keys=result,failures=failed);draw_rows.append(dict(case=case,k=k,records=len(draws),failures=int(failed.sum()),intentional_common_random_numbers_between_target_k=True,**meta))
 cases.append(dict(case=case,modes=m,r=r,cutoff=cut,law_count=len(bank),patterns=len(keys),native=native));save(out,f'case_{case}.json',cases[-1])
save(out,'summary.json',dict(passed=True,cases=cases,comparisons=rows,references=references,samples=draw_rows,scope=__doc__));manifest(out)
