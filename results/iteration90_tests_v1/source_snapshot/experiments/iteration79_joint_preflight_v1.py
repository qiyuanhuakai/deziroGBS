"""Analytic ordered-pair matrices for independent exact photon blocks and mixtures.
Frobenius-optimal mixing is chosen from input matrices, before observations.
Pair-matrix distance is an observable diagnostic, not complete-law TV.
"""
from pathlib import Path
import os,json,argparse,hashlib,shutil
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
rows=[]
for case,(source,N) in enumerate([('results/iteration74_pure_quality_v1/input_1.npz',14),('results/iteration77_budget_pilot_v1/input_4.npz',34)]):
    d=np.load(ROOT/source);v=d['v'];m,n=v.shape;ell=(abs(v)**2).sum(axis=1);A=abs(v@v.T)**2;C=abs(v@v.conj().T)**2;M=np.outer(ell,ell)/(n*n);Q=A/n;R=(np.outer(ell,ell)+C+A)/(n*(n+2));S=(np.outer(ell,ell)+C)/(n*(n+1));T=(Q+(N-2)*R)/(N-1);np.savez_compressed(OUT/f'input_{case}.npz',v=v,total=N,Q=Q,R=R,S=S,M=M,target=T)
    for b in (2,4,8,12):
        sizes=[b]*(N//b)+([N%b] if N%b else []);k=sum(s*(s-2) for s in sizes)/(N*(N-1));rho=(N-2)/(N-1);J=Q/(N-1)+k*R+(rho-k)*M;D=J-S;alpha=float(np.clip(np.sum((T-S)*D)/np.sum(D*D),0,1));mix=alpha*J+(1-alpha)*S
        for kind,P in [('sphere',S),('joint',J),('mixture',mix)]:
            row=dict(case=case,inputs=n,outputs=m,total=N,block=b,kind=kind,joint_probability=alpha,blocks=sizes,pair_l1_half=float(abs(P-T).sum()/2),pair_frobenius=float(np.linalg.norm(P-T)),collision=float(np.trace(P)),target_collision=float(np.trace(T)),minimum_pair_eigenvalue=float(np.linalg.eigvalsh(P)[0]));rows.append(row);save(f'row_{case}_{b}_{kind}.json',row)
        np.savez_compressed(OUT/f'matrices_{case}_{b}.npz',joint=J,mixture=mix,alpha=alpha)
save('summary.json',dict(passed=True,rows=rows));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}));print(json.dumps(rows),flush=True)
