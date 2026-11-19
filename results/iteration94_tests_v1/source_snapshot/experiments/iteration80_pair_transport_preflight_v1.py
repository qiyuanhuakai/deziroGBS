"""Feasibility of shared-real-direction conditional photon-pair couplings.
Each latent coupling is symmetric/nonnegative, has marginals |V u|^2,
and averages to Q=|VV^T|^2/n. A feasible 4-design restores the exact
fixed-total ordered-pair matrix for all even N; higher laws remain unproven.
The LP is a preflight, not a scalable sampling implementation.
"""
from pathlib import Path
import os,json,hashlib,argparse,shutil,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
save('plan.json',dict(scope=__doc__,source_sha256=H(Path(__file__)),tolerance=1e-8,seed=800001));shutil.copy2(__file__,OUT/Path(__file__).name)
def design(n):
    if n==2:
        theta=np.arange(64)*2*np.pi/64;return np.c_[np.cos(theta),np.sin(theta)],np.full(64,1/64)
    signs=np.array([[1]+[1 if mask&(1<<j) else -1 for j in range(n-1)] for mask in range(1<<(n-1))],float)/np.sqrt(n)
    return np.r_[np.eye(n),signs],np.r_[np.full(n,2/(n*(n+2))),np.full(len(signs),n/((n+2)*len(signs)))]
data=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');cases=[('strong4',data['v']),('identity2',np.eye(2,dtype=complex)),('balanced_phase2',np.array([[1,1j],[1j,1]])/np.sqrt(2))]
rng=np.random.default_rng(800001)
for n,m in ((2,3),(2,4),(3,5),(4,8)):
    for rep in range(2):cases.append((f'random_{n}_{m}_{rep}',np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]))
rows=[]
for label,V in cases:
    start=time.perf_counter();m,n=V.shape;u,weights=design(n);latent=len(u);p=abs(u@V.T)**2;ell=(abs(V)**2).sum(axis=1);Q=abs(V@V.T)**2/n;R=(np.outer(ell,ell)+abs(V@V.conj().T)**2+abs(V@V.T)**2)/(n*(n+2));mu=ell/n
    assert np.max(abs(weights@p-mu))<1e-12 and np.max(abs(p.T@(p*weights[:,None])-R))<1e-12
    pairs=[(i,j) for i in range(m) for j in range(i,m)];K=len(pairs);rr=[];cc=[];vv=[]
    # Variables are upper-triangle ordered-pair probabilities, not unordered masses.
    for l in range(latent):
        for k,(i,j) in enumerate(pairs):
            col=l*K+k;rr.append(l*m+i);cc.append(col);vv.append(1.)
            if j!=i:rr.append(l*m+j);cc.append(col);vv.append(1.)
            rr.append(latent*m+k);cc.append(col);vv.append(weights[l])
    matrix=coo_matrix((vv,(rr,cc)),shape=(latent*m+K,latent*K)).tocsr();rhs=np.r_[p.ravel(),[Q[i,j] for i,j in pairs]]
    result=linprog(np.zeros(latent*K),A_eq=matrix,b_eq=rhs,bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    row=dict(label=label,inputs=n,outputs=m,latent_directions=latent,variables=latent*K,status=int(result.status),success=bool(result.success),message=result.message,seconds=time.perf_counter()-start)
    arrays=dict(v=V,directions=u,weights=weights,marginals=p,Q=Q,R=R,pairs=np.array(pairs),rhs=rhs)
    if result.success:
        solution=result.x.reshape(latent,K);residual=float(np.max(abs(matrix@result.x-rhs)));row.update(maximum_constraint_error=residual,minimum_probability=float(result.x.min()));assert residual<1e-8 and result.x.min()>=0
        full=np.zeros((latent,m,m))
        for k,(i,j) in enumerate(pairs):full[:,i,j]=full[:,j,i]=solution[:,k]
        arrays['couplings']=full;arrays['dual']=result.eqlin.marginals
        row['maximum_row_mass_error']=float(abs(full.sum(axis=2)-p).max());row['maximum_averaged_pair_error']=float(abs(np.einsum('l,lij->ij',weights,full)-Q).max())
    else:
        # L1 slack optimum and its LP dual quantify infeasibility of this finite design.
        from scipy.sparse import hstack,eye
        nr=len(rhs);slackmatrix=hstack([matrix,eye(nr),-eye(nr)],format='csr');objective=np.r_[np.zeros(latent*K),np.ones(2*nr)]
        relaxed=linprog(objective,A_eq=slackmatrix,b_eq=rhs,bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});assert relaxed.success
        dual=relaxed.eqlin.marginals;row.update(minimum_l1_constraint_slack=float(relaxed.fun),dual_objective=float(rhs@dual),dual_original_column_max=float((matrix.T@dual).max()),dual_abs_max=float(abs(dual).max()));arrays.update(slack_solution=relaxed.x,dual=dual)
    rows.append(row);save(label+'.json',row);np.savez_compressed(OUT/(label+'.npz'),**arrays);print(row,flush=True)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
