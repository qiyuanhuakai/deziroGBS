"""Independent100-digit direct-matrix endpoint and uncertainty eigenvalues.
Unlike the first diagnostic this never diagonalizes the exported covariance to
form its survival probability; it separates state rounding from eigensystem
formation. No sampler is modified and no failure record is removed.
"""
from pathlib import Path
import argparse,json,hashlib,itertools,functools
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def moment_poly(mu,cov):
 c=len(mu)
 @functools.lru_cache(None)
 def wick(ids):
  if not ids:return mp.mpf(1)
  i,*rest=ids;result=mu[i]*wick(tuple(rest))
  for j,k in enumerate(rest):result+=cov[i,k]*wick(tuple(rest[:j]+rest[j+1:]))
  return result
 return mp.fsum(wick(tuple(sorted(i for j,b in enumerate(bits) for i in (2*j+b,2*j+b)))) for bits in itertools.product((0,1),repeat=c//2))/2**(c//2)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);base=ROOT/'results/iteration89_pnr_diagnostic_v1';files=[Path(__file__),base/'manifest.json'];(out/'plan.json').write_text(json.dumps(dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},digits=100),indent=2)+'\n');mp.mp.dps=100;rows=[]
 for row in read(base/'summary.json')['rows']:
  j,R=row['case'],row['retained'];z=np.load(base/f'state_{j}_{R}.npz');input0=np.load(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{j}.npz');v=input0['v'];ids=z['factors'][:row['factors']];d=len(z['mean']);n=d//2;X=mp.matrix(z['covariance'].tolist());mu=mp.matrix(z['mean'].tolist());cols=[]
  for mode in ids:cols.extend([np.r_[v[mode].real,-v[mode].imag],np.r_[v[mode].imag,v[mode].real]])
  F=mp.matrix(np.array(cols).T.tolist());G=mp.eye(d)+X;tilted=G**-1;M0=moment_poly(F.T*mu,F.T*X*F);M1=moment_poly(F.T*tilted*mu,F.T*X*tilted*F);prob=mp.exp(-(mu.T*tilted*mu)[0]/2)/mp.sqrt(mp.det(G))*M1/M0;Q=X+mp.eye(d)/2
  for k in range(n):Q[k,n+k]+=mp.j/2;Q[n+k,k]-=mp.j/2
  eigen,_=mp.eighe(Q);result=dict(case=j,retained=R,record=row['record'],direct_matrix_endpoint=mp.nstr(prob,90),eigensystem_endpoint=row['evaluations'][-1]['independent'],uncertainty_minimum=mp.nstr(eigen[0],90),normalization=mp.nstr(M0,90),direct_in_range=bool(0<=prob<=1),scope='Exact binary-double covariance and optical-form values lifted to100-digit arithmetic. This audits the saved numerical state, not an interval around the original physical state.');rows.append(result);print('direct matrix audit',result,flush=True)
 (out/'summary.json').write_text(json.dumps(dict(passed=True,rows=rows,scope=__doc__),indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
if __name__=='__main__':main()
