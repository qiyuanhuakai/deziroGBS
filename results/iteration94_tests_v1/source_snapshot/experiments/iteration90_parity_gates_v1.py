"""Compile and validate a general parity-Gaussian inverse-moment building block.
No claims about end-to-end sampling; odd one-photon boundary is explicitly rejected.
"""
import argparse,subprocess,math,mpmath as mp
from fractions import Fraction as F
from iteration90_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'native/gaussian_parity_moment90_v1.cpp';files=[Path(__file__),source,ROOT/'experiments/iteration90_common_v1.py'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},independent_digits=100));gcc=runtime_dirs()[0]/'g++.exe';dll=out/'parity.dll';cmd=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(dll),'-lquadmath'];r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');save(out,'build.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(dll));fn=lib.parity90_probe;fn.argtypes=[C.c_int,C.c_int,C.c_int,DP,DP,DP,C.c_char_p,C.c_int];mp.mp.dps=100
def call(x,p,inverse=0):
 x=np.ascontiguousarray(x);result=np.zeros_like(x);stats=np.zeros(5);err=C.create_string_buffer(4096);rc=fn(len(x),p,inverse,ptr(x),ptr(result),ptr(stats),err,len(err));return result,dict(returncode=rc,error=err.value.decode(),statistics=stats.tolist())
def reference(x,p):
 d=len(x);n=d//2;V=mp.matrix(x.tolist())+mp.eye(d)/2;O=mp.matrix(d)
 for j in range(n):O[j,n+j]=1;O[n+j,j]=-1
 parity=1/mp.sqrt(mp.det(2*V));result=(V+p*parity*(O*V**-1*O.T)/4)/(1+p*parity)-mp.eye(d)/2;return np.array(result.tolist(),float),parity
rows=[];rng=np.random.default_rng(900051)
for n in (1,2,4,8):
 for rep in range(3):
  def unitary():return np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
  def real(U):return np.block([[U.real,-U.imag],[U.imag,U.real]])
  nu=np.linspace(.7,1.8,n);r=rng.normal(size=n)*(.15+.2*rep);S=real(unitary())@np.diag(np.r_[np.exp(r),np.exp(-r)])@real(unitary());V=S@np.diag(np.r_[nu,nu])@S.T;x=(V+V.T)/2-np.eye(2*n)/2
  for p0 in (1,-1):
   forward,fm=call(x,p0);ref,parity=reference(x,p0);assert fm['returncode']==0,fm;forward_error=float(abs(forward-ref).max());assert forward_error<3e-11,(n,rep,p0,forward_error);back,bm=call(forward,p0,1);assert bm['returncode']==0,bm;inverse_error=float(np.linalg.norm(back-x)/(1+np.linalg.norm(x)));assert inverse_error<3e-10,(n,rep,p0,inverse_error);np.savez_compressed(out/f'case_{n}_{rep}_{p0}.npz',source=x,target=forward,reference=ref,recovered=back);rows.append(dict(n=n,repetition=rep,parity=p0,forward_error=forward_error,inverse_error=inverse_error,independent_parity=mp.nstr(parity,80),forward=fm,inverse=bm));print('parity inverse',n,rep,p0,inverse_error,flush=True)
thermal=[]
for mean in (.1,1.,3.):
 for p0 in (1,-1):
  result,meta=call(np.eye(2)*mean,p0);expected=2*mean*mean/(1+2*mean)+(p0==-1);assert meta['returncode']==0 and np.max(abs(result-np.eye(2)*expected))<2e-13;thermal.append(dict(mean=mean,parity=p0,conditional_mean=float(result[0,0]),expected=expected))
boundary=[]
for name,x,p0,inverse in (('odd_vacuum_forward',np.zeros((2,2)),-1,0),('odd_one_photon_inverse',np.eye(2),-1,1),('odd_below_one_photon_inverse',np.eye(2)*.9,-1,1),('nonphysical_small_parity',np.diag([-.1,3.,-.1,3.]),1,0)):
 result,meta=call(x,p0,inverse);assert meta['returncode']!=0,(name,meta);boundary.append(dict(case=name,**meta))
fock=[]
for k in range(2,13):
 p0=(-1)**k;result,meta=call(np.eye(2)*k,p0,1);assert meta['returncode']==0,meta;h=k-(k%2);expected=(h+math.sqrt(h*h+2*h))/2;assert abs(result[0,0]-expected)<2e-12;old=1-F(k,k+1)**k/F(k+1);j=k//2;new=1-F(j,j+1)**j/F(j+1);assert new<old;fock.append(dict(photons=k,source_mean=float(result[0,0]),native=meta,gaussian_full_pnr_tv=[str(old.numerator),str(old.denominator)],parity_full_pnr_tv=[str(new.numerator),str(new.denominator)],gaussian_tv=float(old),parity_tv=float(new),exact_error_ratio=[str((old/new).numerator),str((old/new).denominator)],scope='Conditional squeezed-|k,0> family: Gaussian unitaries preserve charge, so the complete two-port PNR TV equals one minus the correct input number weight. This is not a full sampler result.'))
save(out,'summary.json',dict(passed=True,rows=rows,thermal=thermal,boundary=boundary,fock_comparisons=fock,scope=__doc__));manifest(out)
