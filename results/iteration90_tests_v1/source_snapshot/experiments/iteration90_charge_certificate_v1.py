"""Exact rational support-loss certificates for moment Gaussian closure.
The target is a^a b^b|TMSV(r)>: remaining N1-N2=b-a exactly. Its matched
Gaussian has an analytic two-sided geometric number-difference distribution.
These are conditional-state model errors, not end-to-end sampler TV bounds.
"""
import argparse,math
from fractions import Fraction as F
from iteration90_common_v1 import *
def pair_moments(r,a,b):
 s=r*r;u=s/(1-s);Z=F(0);En=F(0)
 for k in range(min(a,b)+1):
  L=a+b-k;coef=math.comb(a,k)*math.comb(b,k)*math.factorial(k);z=math.factorial(L)*u**L;Z+=coef*z;En+=coef*(math.factorial(L+1)*u**(L+1)+L*z)
 En/=Z;return Z,En-a,En-b,r*(En+1)
def encode(x):return [str(x.numerator),str(x.denominator)]
def sqrt_interval(x,bits):
 k=math.isqrt((x.numerator<<(2*bits))//x.denominator);return F(k,1<<bits),F(k+1,1<<bits)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration89_pilot_v1/puretree.dll';files=[Path(__file__),ROOT/'experiments/iteration90_common_v1.py',dll];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},squeezing=[.2,.6,.85],maximum_subtractions=6,sqrt_bits=256,independent_fourier_points=65536));rows=[]
 for r0 in (.2,.6,.85):
  q=small_input(dict(name='paired_strong',squeezing=r0,eta=1.));O=np.block([[q['v'].real,-q['v'].imag],[q['v'].imag,q['v'].real]])
  with NativeDirectClickSampler(q['v'],q['x'],dll,runtime_dirs=runtime_dirs()) as model:
   probe=model.lib.pnr74_probe;probe.argtypes=[C.c_void_p,DP,DP,C.c_int,IP,DP,DP,DP,C.c_char_p,C.c_int]
   for total in range(7):
    for a in range(total+1):
     b=total-a;r=F.from_float(r0);Z,N1,N2,M=pair_moments(r,a,b);A=(1+N1)*(1+N2)-M*M;B=N1*(1+N2)-M*M;C0=N2*(1+N1)-M*M;E=N1*N2-M*M;D=A+E;disc=D*D-4*B*C0;assert A>0 and B>=0 and C0>=0 and D-B-C0==1 and disc>0
     lo,hi=sqrt_interval(disc,256);difference=b-a;side=B if difference>=0 else C0
     def mass(sqrt):return (2*side/(D+sqrt))**abs(difference)/sqrt
     mass_lower,mass_upper=mass(hi),mass(lo);witness_lower=max(F(0),1-mass_upper);witness_upper=min(F(1),1-mass_lower);assert witness_upper-witness_lower<F(1,1<<230)
     modes=np.array([0]*a+[1]*b,np.int32);value=np.ones(1);cov=q['x'].copy();mu=np.zeros(4);err=C.create_string_buffer(4096)
     if total:
      rc=probe(model.handle,ptr(q['x']),ptr(q['mean']),total,modes.ctypes.data_as(IP),ptr(value),ptr(cov),ptr(mu),err,len(err));assert rc==0,err.value
     optical=O@cov@O.T;nativeN1=(optical[0,0]+optical[2,2])/2;nativeN2=(optical[1,1]+optical[3,3])/2;nativeM=complex((optical[0,1]-optical[2,3])/2,(optical[0,3]+optical[2,1])/2);moment_error=max(abs(nativeN1-float(N1)),abs(nativeN2-float(N2)),abs(abs(nativeM)-float(M)));assert moment_error<2e-10,(r0,a,b,moment_error)
     theta=2*np.pi*np.arange(65536)/65536;z=np.exp(1j*theta);fourier=np.mean(np.exp(-1j*difference*theta)/(float(D)-float(B)*z-float(C0)/z));assert abs(fourier.real-float((mass_lower+mass_upper)/2))<3e-12 and abs(fourier.imag)<3e-12
     parity=1 if total%2==0 else -1;parity_probability=(1+F(parity)/(2*D-1))/2;assert parity_probability>0;parity_lower=max(F(0),1-mass_upper/parity_probability)
     row=dict(squeezing=r0,removed=[a,b],remaining_difference=difference,normalization=encode(Z),N1=encode(N1),N2=encode(N2),M=encode(M),A=encode(A),B=encode(B),C=encode(C0),E=encode(E),D=encode(D),discriminant=encode(disc),sqrt_lower=encode(lo),sqrt_upper=encode(hi),charge_mass_lower=encode(mass_lower),charge_mass_upper=encode(mass_upper),support_tv_lower=encode(witness_lower),support_tv_upper=encode(witness_upper),support_tv_lower_float=float(witness_lower),moment_error=moment_error,fourier_real=float(fourier.real),parity_projected_support_tv_lower=float(parity_lower),limits='The two support lower bounds do not by themselves order actual TV. Parity projection of the matched Gaussian changes its moments; no new sampler or universal accuracy claim.');rows.append(row)
     np.savez_compressed(out/f'closure_{str(r0).replace(".","p")}_{a}_{b}.npz',input_x=q['x'],v=q['v'],closed_x=cov,closed_mean=mu,normalization=value)
  print('charge certificates',r0,'passed',flush=True)
 save(out,'summary.json',dict(passed=True,rows=rows,scope=__doc__,certificates=len(rows),limits='Exact certificates concern analytically specified binary-rational squeezing and its exact matched Gaussian. Native covariance agrees numerically; certificates are not interval bounds for the floating native state. Conditional model error is not the full sampler error.'));manifest(out)
 print(json.dumps([{k:r[k] for k in ('squeezing','removed','support_tv_lower_float','parity_projected_support_tv_lower')} for r in rows if r['squeezing']==.85 and sum(r['removed']) in (4,6)],indent=2),flush=True)
if __name__=='__main__':main()
