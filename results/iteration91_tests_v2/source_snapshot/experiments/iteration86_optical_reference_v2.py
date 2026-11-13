"""Full sixteen-bin Gaussian PGF for two eight-bin optical-output counts.
No rank truncation. Native LAPACK batched eigenvalues and FFT evaluate the PGF
on 128/256 grids, with marginal Chernoff alias/tail bounds. Independent 80-digit
vacuum determinants, marginal recurrences and photon cross moments are checked.
These are numerical references on float64 covariance, not interval certificates.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]

def marginal_reference(covariances,ids,K):
 d=len(covariances[0])//2;ids=np.r_[ids,np.array(ids)+d];lam=[mp.mpf(float(z)) for x in covariances for z in np.linalg.eigvalsh(x[np.ix_(ids,ids)])];c=[z/(1+z) for z in lam];powers=[mp.mpf(1)]*len(c);traces=[];p=[mp.fprod(1/mp.sqrt(1+z) for z in lam)]
 for n in range(1,K+1):
  powers=[a*b for a,b in zip(powers,c)];traces.append(sum(powers)/2);p.append(sum(traces[j-1]*p[n-j] for j in range(1,n+1))/n)
 positive=[z for z in c if z>0]
 if not positive:return p,mp.mpf(0)
 lo=mp.mpf(1);hi=1/max(positive)
 for _ in range(200):
  r=(lo+hi)/2
  if sum(z*r/(1-z*r) for z in c)/2<K+1:lo=r
  else:hi=r
 r=(lo+hi)/2;tail=min(mp.mpf(1),mp.exp(-sum(mp.log(1+z) for z in lam)/2-sum(mp.log(1-z*r) for z in c)/2-(K+1)*mp.log(r)))
 return p,tail

def fourier_general(covariances,group,L):
 d=len(group);group=np.r_[group,group];z=np.exp(2j*np.pi*np.arange(L)/L);values=np.ones((L,L),complex)
 # One row at a time bounds temporary storage; eigensolvers execute in LAPACK.
 for x in covariances:
  A=np.linalg.solve(np.eye(2*d)+x,x);assert np.linalg.norm(A,2)<1;constant=np.linalg.slogdet(np.eye(2*d)+x)[1]/2
  for i in range(L):
   diagonal=np.where(group[None,:]==0,z[i],z[:,None]);eigs=np.linalg.eigvals(np.eye(2*d)-A*diagonal[:,None,:]);assert eigs.real.min()>0;values[i]*=np.exp(-constant-np.log(eigs).sum(-1)/2)
 return np.fft.fft2(values)/L**2

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);mp.mp.dps=80
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';data=np.load(inp);v,x,order=data['v'],data['x'],data['order'];pairs=[[0,143],[142,143]];K=64;rows=[]
 save('plan.json',dict(scope=__doc__,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pairs=pairs,maximum_photons_per_group=K,fourier_sizes=[128,256],decimal_digits=80,replaces='v1 rejected the rank-one assumption before producing any probabilities.'))
 for index,pair in enumerate(pairs):
  start=time.perf_counter();blocks=[np.flatnonzero(order//8==mode) for mode in pair];ids=np.concatenate(blocks);W=v[ids];O=np.block([[W.real,-W.imag],[W.imag,W.real]]);covs=[O@a@O.T for a in x];group=np.r_[np.zeros(8,int),np.ones(8,int)];singular=[np.linalg.svd(v[b],compute_uv=False) for b in blocks];residual=[float(np.linalg.norm(s[1:])/s[0]) for s in singular];assert all(len(b)==8 for b in blocks)
  low=fourier_general(covs,group,128);high=fourier_general(covs,group,256);raw=high[:K+1,:K+1].real;imag=float(abs(high.imag).max());difference=float(abs(low[:K+1,:K+1]-high[:K+1,:K+1]).max());negative=float(-raw[raw<0].sum());target=np.maximum(raw,0);mass=float(target.sum());normalization=max(1.,mass);target/=normalization;tail=max(0.,1-float(target.sum()));assert difference<2e-12 and imag<1e-12 and negative<1e-12 and abs(mass-1)<1e-10
  marginals=[];tails=[];aliases=[]
  for j in range(2):
   marginal,bound=marginal_reference(covs,np.flatnonzero(group==j),K);_,alias=marginal_reference(covs,np.flatnonzero(group==j),127);marginals.append(np.array([float(p) for p in marginal]));tails.append(bound);aliases.append(alias)
  marginal_error=max(float(abs(target.sum(axis=1)-marginals[0]).max()),float(abs(target.sum(axis=0)-marginals[1]).max()));vacuum=mp.fprod(1/mp.sqrt(mp.det(mp.eye(32)+mp.matrix(c.tolist()))) for c in covs);vacuum_error=abs(target[0,0]-float(vacuum));assert marginal_error<2e-12+float(sum(tails)) and vacuum_error<2e-12
  means=np.array([sum(np.trace(c[np.ix_(np.r_[np.flatnonzero(group==j),np.flatnonzero(group==j)+16],np.r_[np.flatnonzero(group==j),np.flatnonzero(group==j)+16])])/2 for c in covs) for j in range(2)]);truncated_means=np.array([(target*np.arange(K+1)[:,None]).sum(),(target*np.arange(K+1)[None,:]).sum()]);cross=0.
  for c in covs:
   normal=(c[:16,:16]+c[16:,16:]+1j*(c[:16,16:]-c[16:,:16]))/2;anomalous=(c[:16,:16]-c[16:,16:]+1j*(c[:16,16:]+c[16:,:16]))/2;cross+=float((abs(normal[:8,8:])**2+abs(anomalous[:8,8:])**2).sum())
  cross_emp=float((target*np.arange(K+1)[:,None]*np.arange(K+1)[None,:]).sum()-np.prod(truncated_means));assert np.max(abs(means-truncated_means))<1e-8 and abs(cross-cross_emp)<1e-8
  np.savez_compressed(out/f'pair_{index}.npz',target=target,covariances=covs,indices=np.array(blocks),fourier128=low,fourier256=high,means=means,marginals=marginals)
  row=dict(pair=pair,processed_indices=[b.tolist() for b in blocks],rank_relative_residual=residual,tail_mass=str(tail),chernoff_tail_upper=str(sum(tails)),fourier_alias_upper=str(sum(aliases)),fourier_maximum_difference=difference,fourier_maximum_imaginary=imag,negative_roundoff_mass=negative,normalization_divisor=normalization,marginal_maximum_difference=marginal_error,vacuum_80digit=str(vacuum),vacuum_difference=vacuum_error,means=means.tolist(),truncated_means=truncated_means.tolist(),reference_count_covariance=cross,computed_count_covariance=cross_emp,seconds=time.perf_counter()-start);save(f'pair_{index}.json',row);rows.append(row);print('pair',pair,'tail upper',float(sum(tails)),'Fourier difference',difference,flush=True)
 save('summary.json',dict(passed=True,rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':main()
