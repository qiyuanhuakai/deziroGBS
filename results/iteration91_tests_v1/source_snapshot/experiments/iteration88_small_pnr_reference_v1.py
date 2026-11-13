"""Analytic complete two-mode PNR references and explicit geometric tail bounds.
Lossy paired light is a positive geometric mixture of two conditional binomials.
This independent90-digit calculation does not call a sampler or a hafnian kernel.
"""
import argparse,mpmath as mp
from iteration88_pnr_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);planpath=ROOT/'research/iteration88_small_pnr_plan.json';plan=read(planpath);files=[Path(__file__),ROOT/'experiments/iteration88_pnr_common_v1.py',planpath];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},digits=90,mixture_cutoff=256));mp.mp.dps=90;K=plan['maximum_photons_per_port_in_explicit_grid'];rows=[]
for j,case in enumerate(plan['cases']):
 q=small_input(case);G=[[mp.mpf(0) for _ in range(K+1)] for _ in range(K+1)];omitted=mp.mpf(0)
 if case['name']=='thermal_control':
  n1,n2=map(mp.mpf,case['mean_photons']);s1=n1/(1+n1);s2=n2/(1+n2)
  for a0 in range(K+1):
   for b0 in range(K+1):G[a0][b0]=(1-s1)*(1-s2)*s1**a0*s2**b0
  tail_bound=s1**(K+1)+s2**(K+1)-s1**(K+1)*s2**(K+1);p00=(1-s1)*(1-s2);zero1=1-s1;zero2=1-s2;means=[n1,n2];cross=n1*n2
 else:
  r=mp.mpf(float(case['squeezing']));eta=mp.mpf(float(case['eta']));s=r*r;nbar=eta*s/(1-s);means=[nbar,nbar];cross=eta*eta*s*(1+s)/(1-s)**2
  if eta==1:
   for a0 in range(K+1):G[a0][a0]=(1-s)*s**a0
  else:
   for k in range(257):
    B=[mp.binomial(k,n)*eta**n*(1-eta)**(k-n) for n in range(min(k,K)+1)];weight=(1-s)*s**k
    for a0,pa in enumerate(B):
     for b0,pb in enumerate(B):G[a0][b0]+=weight*pa*pb
   omitted=s**257
  s_marginal=nbar/(1+nbar);tail_bound=min(s**(K+1),2*s_marginal**(K+1));p00=(1-s)/(1-s*(1-eta)**2);zero1=zero2=(1-s)/(1-s*(1-eta))
 mass=mp.fsum(value for row in G for value in row);tail=1-mass;target=np.array([[float(x) for x in row] for row in G]);rounding=mp.fsum(abs(mp.mpf(float(value))-value) for row in G for value in row);target_click=np.array(list(map(float,[p00,zero2-p00,zero1-p00,1-zero1-zero2+p00])));O=np.block([[q['v'].real,-q['v'].imag],[q['v'].imag,q['v'].real]]);X=O@q['target_x']@O.T;independent_vac=1/mp.sqrt(mp.det(mp.eye(4)+mp.matrix(X.tolist())));vacdiff=abs(independent_vac-p00);input_residual=float(np.max(abs(q['x']+q['L']@q['L'].T-q['target_x'])));assert 0<=tail<=tail_bound+omitted and vacdiff<mp.mpf('2e-15') and input_residual<2e-15
 row=dict(case=case,name=case['name'],maximum_photons=K,tail_mass=mp.nstr(tail,85),tail_upper=mp.nstr(tail_bound+omitted,85),omitted_mixture_mass=mp.nstr(omitted,85),float_conversion_l1=mp.nstr(rounding,85),vacuum_reference=mp.nstr(p00,85),vacuum_covariance_difference=mp.nstr(vacdiff,40),input_covariance_residual=input_residual,mean_photons=list(map(float,means)),cross_moment=float(cross));rows.append(row);np.savez_compressed(out/f'case_{j}.npz',target=target,target_click=target_click,**q);save(out,f'case_{j}.json',row);print('PNR reference',case['name'],'tail',float(tail_bound+omitted),flush=True)
save(out,'summary.json',dict(passed=True,rows=rows,scope=__doc__));manifest(out)
