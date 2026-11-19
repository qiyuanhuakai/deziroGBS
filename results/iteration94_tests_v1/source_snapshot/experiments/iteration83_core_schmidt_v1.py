"""Lightweight low-rank Gaussian Schmidt diagnostics for the optimized core.
Only 100-dimensional matrices, no sampling, full Fock tensor or MPS construction.
Exact-arithmetic cut reduction uses the complex Gram V_A^* V_A; vacuum spectators
contribute unit symplectic eigenvalues. Top geometric weights give a necessary
pure-core rank/trace-distance tradeoff, NOT a measured GBS TV lower bound.
"""
from pathlib import Path
import os,argparse,json,hashlib,math,heapq,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
def realify(u):return np.block([[u.real,-u.imag],[u.imag,u.real]])
def spectrum(c):
 n=len(c)//2;omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);L=np.linalg.cholesky(c);return np.linalg.eigvalsh(1j*(L.T@omega@L))[n:]
def reduced(v,pure):
 gram=v.conj().T@v;e,u=np.linalg.eigh(gram);assert e.min()>0 and e.max()<1+1e-12
 root=realify((u*np.sqrt(e))@u.conj().T);cov=np.eye(len(pure))+root@(pure-np.eye(len(pure)))@root;return (cov+cov.T)/2
def topweights(q,rank):
 assert np.all(q>0) and np.all(q<1),'Nonpositive geometric parameter: no clipping'
 costs=-np.log(q);base=float(np.log1p(-q).sum());heap=[(0.,0)];weights=[]
 # Each occupation vector has exactly one nondecreasing-index creation path.
 while len(weights)<rank:
  energy,last=heapq.heappop(heap);weights.append(math.exp(base-energy))
  for j in range(last,len(q)):heapq.heappush(heap,(energy+float(costs[j]),j))
 return np.asarray(weights)
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';base=ROOT/'results/iteration83_core_energy_probe_v2';files=[Path(__file__),inp,base/'split_0.npz',base/'split_1.npz'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},cuts=[64,144,288,576,864,1008],top_ranks=[1,16,64,256,1024,4096],middle_cut=576));v=np.load(inp)['v'];n=v.shape[1];rows=[]
# Analytic 50/50 split of one squeezed mode: nu=cosh(r).
r=.3;single=spectrum(reduced(np.array([[1/math.sqrt(2)]],complex),np.diag([math.exp(-2*r),math.exp(2*r)])))[0];assert abs(single-math.cosh(r))<1e-14
for sector in range(2):
 start=time.perf_counter();pure=np.load(base/f'split_{sector}.npz')['pure'];cuts=[]
 for cut in (64,144,288,576,864,1008):
  c=reduced(v[:cut],pure);nu=spectrum(c);row=dict(cut=cut,minimum_symplectic_eigenvalue=float(nu.min()),maximum_symplectic_eigenvalue=float(nu.max()),renyi2_effective_rank=float(np.exp(np.log(nu).sum())),maximum_schmidt_probability=float(np.exp(np.log(2/(1+nu)).sum())))
  if cut==64:
   b=realify(v[:cut]);full=np.eye(2*cut)+b@(pure-np.eye(2*n))@b.T;direct=spectrum(full);error=float(abs(np.sort(np.r_[np.ones(cut-n),nu])-direct).max());assert error<1e-11;row['direct_128dim_spectrum_error']=error
  if cut==576:
   q=(nu-1)/(nu+1);weights=topweights(q,4096);row['rank_tradeoffs']=[]
   for rank in (1,16,64,256,1024,4096):
    tail=1-math.fsum(weights[:rank]);assert tail>0;row['rank_tradeoffs'].append(dict(rank=rank,discarded_schmidt_mass=tail,necessary_pure_state_trace_distance=math.sqrt(tail)))
   np.savez_compressed(OUT/f'middle_{sector}.npz',covariance=c,nu=nu,q=q,top_weights=weights)
  cuts.append(row)
 row=dict(sector=sector,seconds=time.perf_counter()-start,cuts=cuts);rows.append(row);save(f'row_{sector}.json',row);print(dict(sector=sector,seconds=row['seconds'],middle=next(c for c in cuts if c['cut']==576)),flush=True)
save('summary.json',dict(rows=rows,analytic_single_mode_error=abs(single-math.cosh(r)),scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
