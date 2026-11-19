"""Independent Fock-density and actual-input checks for one Gaussian click update."""
import os
for env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[env]='1'
import argparse, ctypes, hashlib, json, shutil, subprocess
from pathlib import Path
import numpy as np
from scipy.linalg import expm, solve_triangular

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',required=True,type=Path)
out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
def save(name,data):
    (out/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source=root/'native/gaussian_click_proxy95_v1.cpp'
models=[root/f'results/iteration94_actual_quality_v1/input_{i}.npz' for i in range(2)]
ports=[0,4,13,26]
trials=16
save('plan.json',dict(scope=__doc__,ports=ports,trials=trials,latent_seeds=[950031,950032],
    fock_cutoffs=[48,72],quality_records=0,hashes={f.relative_to(root).as_posix():digest(f) for f in [source,Path(__file__),*models]}))
compiler=Path(shutil.which('g++'))
dll=out/'proxy.dll'
build=subprocess.run([str(compiler),'-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-shared',str(source),'-o',str(dll)],capture_output=True,text=True)
save('build.json',dict(returncode=build.returncode,stderr=build.stderr,compiler=str(compiler)))
build.check_returncode()
runtime=os.add_dll_directory(str(compiler.parent))
fn=ctypes.CDLL(str(dll)).coherent95_fidelity
fn.argtypes=[ctypes.c_double]*7+[np.ctypeslib.ndpointer(dtype=np.float64)]
def proxy(g,mu,point):
    values=np.empty(6)
    rc=fn(g[0,0],g[0,1],g[1,1],*mu,*point,values)
    assert rc==0,(rc,g,mu,point)
    return values

def fock_check(size,thermal,squeeze,alpha):
    a=np.diag(np.sqrt(np.arange(1,size)),1).astype(complex)
    ad=a.conj().T
    unitary=expm(alpha*ad-alpha.conjugate()*a)@expm((squeeze.conjugate()*(a@a)-squeeze*(ad@ad))/2)
    x=(a+ad)/np.sqrt(2)
    quadratures=[x,(a-ad)/(1j*np.sqrt(2))]
    weights=(thermal/(1+thermal))**np.arange(size)/(1+thermal)
    weights/=weights.sum()
    rho=(unitary*weights)@unitary.conj().T
    mu=np.array([np.trace(rho@v).real for v in quadratures])
    g=np.array([[2*(np.trace(rho@v@w).real-m*n) for w,n in zip(quadratures,mu)] for v,m in zip(quadratures,mu)])
    chi_zero=unitary[0]*np.sqrt(weights)
    click=np.diag(weights)-np.outer(chi_zero,chi_zero.conj())
    p1=np.trace(click).real
    click/=p1
    answer=proxy(g,mu,[.31,-.28])
    assert abs(p1-answer[0])<1e-8
    rows=[]
    for point,loss in (([.31,-.28],answer[1]),(answer[2:4],answer[4]),([0.,0.],answer[5])):
        beta=complex(*point)/np.sqrt(2)
        coherent=np.ones(size,dtype=complex)
        for k in range(1,size):coherent[k]=coherent[k-1]*beta/np.sqrt(k)
        chi=(coherent.conj()@unitary)*np.sqrt(weights)
        chi/=np.linalg.norm(chi)
        exact=1-(chi.conj()@click@chi).real
        distance=np.abs(np.linalg.eigvalsh(click-np.outer(chi,chi.conj()))).sum()/2
        assert abs(exact-loss)<2e-8 and distance**2<=loss+2e-8
        rows.append(dict(point=list(point),formula_loss=loss,fock_loss=exact,trace_distance=distance))
    return rows

controls=[(.4,0j,0j),(.25,.22+.1j,.4-.3j),(.1,-.28+.16j,.8+.12j),
          (0.,.23j,-.45+.5j),(.65,0j,1.0+.23j)]
oracles=[]
for idx,params in enumerate(controls):
    small=fock_check(48,*params)
    large=fock_check(72,*params)
    # The optimizer is degenerate for a pure product: compare achieved errors.
    for left,right in zip(small,large):
        assert abs(left['fock_loss']-right['fock_loss'])<2e-8
    oracles.append(dict(case=idx,cutoff48=small,cutoff72=large))
save('fock.json',oracles)

def vacuum(q,means):
    inv=np.linalg.inv(q)
    return np.exp(-np.einsum('...a,...ab,...b->...',means,inv,means)/2)/np.sqrt(np.linalg.det(q))

rows=[]
for sector,path in enumerate(models):
    with np.load(path) as data:
        u,t,alpha,noise=(data[k] for k in ('u','t','alpha','noise'))
    m,n=u.shape
    transform=np.block([[u.real,-u.imag],[u.imag,u.real]])
    pure=np.diag(np.r_[t/(1-t),-t/(1+t)])
    q=np.eye(2*m)+transform@pure@transform.T
    normals=np.random.default_rng(950031+sector).standard_normal((2*n,trials))
    means=np.sqrt(2)*transform@(np.r_[alpha.real,alpha.imag][:,None]+noise@normals)
    np.savez_compressed(out/f'latents_{sector}.npz',normals=normals)
    for k in ports:
        included=np.r_[np.arange(k),m+np.arange(k)]
        L=np.linalg.cholesky(q[np.ix_(included,included)])
        white=solve_triangular(L,q[included],lower=True)
        wm=solve_triangular(L,means[included],lower=True)
        mean=means-white.T@wm
        def cov(i,j):
            left,right=np.broadcast_arrays(i,j)
            return q[left,right]-(white[:,left]*white[:,right]).sum(0)
        ids=np.arange(k,m)
        blocks=np.stack([cov(ids,ids),cov(ids,ids+m),cov(ids+m,ids),cov(ids+m,ids+m)],axis=-1).reshape(-1,2,2)
        future=ids[1:]
        cross=np.stack([cov(future,k),cov(future,k+m),cov(future+m,k),cov(future+m,k+m)],axis=-1).reshape(-1,2,2)
        measured=blocks[0]
        gain=cross@np.linalg.inv(measured)
        next_cov=blocks[1:]-gain@cross.transpose(0,2,1)
        for sample in range(trials):
            mu=mean[[k,k+m],sample]
            rest=mean[np.stack([future,future+m],axis=1),sample]
            a=proxy(2*measured-np.eye(2),mu,[0.,0.])
            p1=float(a[0]); p0=1-p1
            zmean=rest-gain@mu
            exact=(vacuum(blocks[1:],rest)-p0*vacuum(next_cov,zmean))/p1
            surrogate=vacuum(next_cov,zmean+gain@a[2:4])
            witness=float(np.max(np.abs(exact-surrogate)))
            assert -1e-9<=exact.min() and exact.max()<=1+1e-9
            assert witness*witness <= a[4]+2e-10 and a[4]<=a[5]+1e-12
            prefix_log=-np.sum(np.log(np.diag(L)))-np.dot(wm[:,sample],wm[:,sample])/2
            rows.append(dict(sector=sector,port=k,latent=sample,p_click=p1,
                history_weight=float(np.exp(prefix_log)*p1),chosen_x=a[2],chosen_p=a[3],
                vacuum_loss=a[5],chosen_loss=a[4],future_bit_witness=witness))
save('actual.json',rows)
summary=[]
for sector in range(2):
    for k in ports:
        group=[r for r in rows if r['sector']==sector and r['port']==k]
        weights=[r['history_weight'] for r in group]
        def avg(key,sqrt=False):
            vals=np.array([r[key] for r in group])
            return float(np.average(np.sqrt(np.maximum(vals,0)) if sqrt else vals,weights=weights))
        summary.append(dict(sector=sector,port=k,weighted_witness=avg('future_bit_witness'),
            weighted_vacuum_proxy_upper=avg('vacuum_loss',True),weighted_chosen_upper=avg('chosen_loss',True)))
save('validation.json',dict(passed=True,latent_histories=len(rows),fock_states=len(controls),cases=summary,
    scope='Finite fixed latent diagnostic. Each conditional pure-core upper bound is in ideal arithmetic. The weighted means are descriptive ratios, without a population confidence claim.'))
save('manifest.json',{f.name:digest(f) for f in out.iterdir() if f.is_file()})
print(json.dumps(summary),flush=True)
