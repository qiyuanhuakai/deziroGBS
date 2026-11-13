"""Two optical-output PNR marginals: 80-digit bivariate determinant PGF.
Rank-one blocks of eight physical bins collapse unitarily to one occupied mode.
Independent complex Fourier inversion checks coefficients; tails are bounded by
the two marginal Chernoff bounds. Float64 input projection is not interval math.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,itertools,json,time
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]

def multiply(a,b):
 out={}
 for (i,j),x in a.items():
  for (k,l),y in b.items():out[i+k,j+l]=out.get((i+k,j+l),mp.mpf(0))+x*y
 return out

def determinant_polynomial(x):
 d=len(x);out={}
 for perm in itertools.permutations(range(d)):
  sign=(-1)**sum(perm[i]>perm[j] for i in range(d) for j in range(i+1,d));term={(0,0):mp.mpf(sign)}
  for i,j in enumerate(perm):
   v=mp.mpf(float(x[i,j]));power=(1,0) if j%2==0 else (0,1)
   term=multiply(term,{(0,0):v+int(i==j),power:-v})
  for ij,value in term.items():out[ij]=out.get(ij,mp.mpf(0))+value
 return out

def coefficients(poly,K):
 p=[[mp.mpf(0) for _ in range(K+1)] for _ in range(K+1)];p[0][0]=1/mp.sqrt(poly[0,0])
 for degree in range(1,2*K+1):
  for a in range(max(0,degree-K),min(K,degree)+1):
   b=degree-a;p[a][b]=-sum((2*degree-i-j)*value*p[a-i][b-j] for (i,j),value in poly.items() if i+j>0 and i<=a and j<=b)/(2*degree*poly[0,0])
 return p

def tail_bound(covariances,mode,K):
 lam=np.concatenate([np.linalg.eigvalsh(x[np.ix_([mode,mode+2],[mode,mode+2])]) for x in covariances]);lam=[mp.mpf(float(z)) for z in lam];c=[z/(1+z) for z in lam];positive=[z for z in c if z>0]
 if not positive:return mp.mpf(0)
 lo=mp.mpf(1);hi=1/max(positive)
 for _ in range(200):
  r=(lo+hi)/2;tilted=sum(z*r/(1-z*r) for z in c)/2
  if tilted<K+1:lo=r
  else:hi=r
 r=(lo+hi)/2
 return min(mp.mpf(1),mp.exp(-sum(mp.log(1+z) for z in lam)/2-sum(mp.log(1-z*r) for z in c)/2-(K+1)*mp.log(r)))

def fourier(covariances,L):
 z=np.exp(2j*np.pi*np.arange(L)/L);z1,z2=np.meshgrid(z,z,indexing='ij');diagonal=np.stack([z1,z2,z1,z2],axis=-1);values=np.ones((L,L),complex)
 for x in covariances:
  A=np.linalg.solve(np.eye(4)+x,x);eigs=np.linalg.eigvals(np.eye(4)-A*diagonal[...,None,:]);values*=np.exp(-np.linalg.slogdet(np.eye(4)+x)[1]/2-np.log(eigs).sum(-1)/2)
 p=np.fft.fft2(values)/(L*L)
 return p

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);mp.mp.dps=80
 def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
 inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';data=np.load(inp);v,x,order=data['v'],data['x'],data['order'];pairs=[[0,143],[142,143]];K=64;L=128
 save('plan.json',dict(scope=__doc__,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pairs=pairs,maximum_photons_per_mode=K,fourier_size=L,decimal_digits=80))
 rows=[]
 for index,pair in enumerate(pairs):
  start=time.perf_counter();blocks=[np.flatnonzero(order//8==mode) for mode in pair];W=[];residual=[]
  for ids in blocks:
   u,s,vh=np.linalg.svd(v[ids],full_matrices=False);residual.append(float(np.linalg.norm(s[1:])/s[0]));assert residual[-1]<1e-12;W.append(s[0]*vh[0])
  W=np.asarray(W);O=np.block([[W.real,-W.imag],[W.imag,W.real]]);covs=[O@a@O.T for a in x];poly={(0,0):mp.mpf(1)}
  for cov in covs:poly=multiply(poly,determinant_polynomial(cov))
  p=coefficients(poly,K);mass=sum(map(sum,p));tail=1-mass;chernoff=sum(tail_bound(covs,j,K) for j in range(2));assert min(map(min,p))>=0 and tail>=-mp.mpf('1e-60') and tail<=chernoff+mp.mpf('1e-60')
  target=np.array([[float(z) for z in row] for row in p]);fft=fourier(covs,L);error=float(abs(fft[:K+1,:K+1].real-target).max());imag=float(abs(fft.imag).max());alias=sum(tail_bound(covs,j,L-1) for j in range(2));assert error<1e-11+float(alias) and imag<1e-12
  means=np.array([sum(np.trace(c[np.ix_([j,j+2],[j,j+2])])/2 for c in covs) for j in range(2)]);truncated_means=np.array([(target*np.arange(K+1)[:,None]).sum(),(target*np.arange(K+1)[None,:]).sum()]);assert np.max(abs(means-truncated_means))<1e-8
  np.savez_compressed(out/f'pair_{index}.npz',target=target,covariances=covs,collapsed_v=W,fourier=fft,indices=np.array(blocks),means=means)
  row=dict(pair=pair,processed_indices=[z.tolist() for z in blocks],rank_relative_residual=residual,tail_mass=str(tail),chernoff_tail_upper=str(chernoff),fourier_alias_upper=str(alias),fourier_maximum_difference=error,fourier_maximum_imaginary=imag,means=means.tolist(),truncated_means=truncated_means.tolist(),seconds=time.perf_counter()-start)
  save(f'pair_{index}.json',dict(**row,probabilities=[[str(z) for z in r] for r in p],determinant_polynomial={f'{i},{j}':str(z) for (i,j),z in poly.items()}));rows.append(row);print('pair',pair,'tail',float(tail),'Fourier difference',error,flush=True)
 save('summary.json',dict(passed=True,rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':main()
