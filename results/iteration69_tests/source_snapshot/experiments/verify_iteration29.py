"""Independent local quadrature and raw-MPS checks for same-channel outputs."""
import argparse
import itertools
import json
import math
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import shutil
import sys
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from gbslab.banded_gaussian_mps import pairing_covariance
from gbslab.mps_overlap import scaled_overlap
from gbslab.exact import ThresholdModel
from iteration23_support import save,digest,load_mps,mps_fingerprint
from iteration22_global import load_family
from iteration22_external import ARCHIVE_SHA


@lru_cache(None)
def grid(order,rank):
    nodes,weights=hermgauss(order); ids=np.array(list(itertools.product(range(order),repeat=rank)),int)
    return np.sqrt(2)*nodes[ids],np.prod(weights[ids]/np.sqrt(np.pi),axis=1)


def quadrature_density(rho,dimensions,F):
    k=len(dimensions); F=np.asarray(F)
    # For large global noise rank, Cholesky of the tiny positive local covariance
    # yields an equivalent low-dimensional Gaussian integral, without clipping.
    if F.shape[1]>4: F=np.linalg.cholesky(F@F.T)
    K=np.eye(F.shape[1])+F.T@F/2; standard,mass=grid(sum(d-1 for d in dimensions)+1,F.shape[1])
    z=np.linalg.solve(np.linalg.cholesky(K).T,standard.T).T; mu=z@F.T; beta=-(mu[:,:k]+1j*mu[:,k:])/2
    vectors=[]
    for occupation in np.ndindex(dimensions):
        value=np.ones(len(beta),complex)
        for i,n in enumerate(occupation): value*=beta[:,i]**n/math.sqrt(math.factorial(n))
        vectors.append(value)
    vector=np.stack(vectors,axis=1); integrand=np.einsum('bi,ij,bj->b',vector.conj(),rho,vector)
    if np.max(abs(integrand.imag))>1e-10: raise ArithmeticError('Nonreal direct quadrature')
    return math.fsum(mass*integrand.real)/math.sqrt(np.linalg.det(K))


def dense_reduced(dense,modes):
    rest=[i for i in range(dense.ndim) if i not in modes]; vector=dense.transpose(modes+rest).reshape(math.prod(dense.shape[i] for i in modes),-1)
    return vector@vector.conj().T


def raw_log_overlap(tensors,beta,log_norm_squared):
    environment=np.ones(1,complex); logs=0.
    for A,b in zip(tensors,beta):
        coefficient=np.array([b.conjugate()**n/math.sqrt(math.factorial(n)) for n in range(A.shape[1])]); local=np.einsum('n,anb->ab',coefficient,A)
        environment=environment@local; scale=float(np.max(abs(environment)))
        if scale==0: raise ArithmeticError('Raw scalar check encountered zero')
        environment/=scale; logs+=math.log(scale)
    return 2*(logs+math.log(abs(environment[0])))-float(np.sum(abs(beta)**2))-log_norm_squared


def raw_pair_density(tensors,start,log_norm_squared):
    E=np.ones((1,1),complex); R=np.ones((1,1),complex); log_scale=0.
    for A in tensors[:start]:
        E=np.einsum('ac,anb,cnd->bd',E,A.conj(),A,optimize=True); scale=float(np.max(abs(E))); E/=scale; log_scale+=math.log(scale)
    for A in tensors[:start+1:-1]:
        R=np.einsum('anb,bd,cnd->ac',A.conj(),R,A,optimize=True); scale=float(np.max(abs(R))); R/=scale; log_scale+=math.log(scale)
    A,B=tensors[start:start+2]; joined=np.einsum('anc,cpb->anpb',A,B,optimize=True).reshape(A.shape[0],A.shape[1]*B.shape[1],B.shape[2])
    density=np.einsum('ac,amb,cnd,bd->nm',E,joined.conj(),joined,R,optimize=True)
    return density*math.exp(log_scale-log_norm_squared)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--archive',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False); source=Path(__file__); sha=digest(source); shutil.copy2(source,out/source.name)
    def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
    checks=[]
    def check(name,value): checks.append(dict(name=name,passed=bool(value)))
    panel=ROOT/'results/iteration29_panel'; plan=read(panel/'plan.json'); rows=read(panel/'rows.json'); summary=read(panel/'summary.json')
    check('all 22 jobs completed',summary['planned']==summary['finished']==summary['completed']==len(rows)==22)
    for name,value in plan['source_hashes'].items(): check('current and frozen source '+name,digest(ROOT/name)==digest(panel/'source_snapshot'/name)==value)
    for name,value in plan['dependencies'].items(): check('historical dependency '+name,digest(ROOT/name)==value)
    check('external archive',digest(args.archive)==ARCHIVE_SHA)
    inputs=np.load(ROOT/'results/iteration28_panel/case_14.npz',allow_pickle=False); largeB=inputs['full_pairing']; largeL=inputs['full_noise_factor']; proposal=np.load(panel/'large_proposal.npz',allow_pickle=False)
    V=pairing_covariance(largeB); K=np.eye(largeL.shape[1])+largeL.T@np.linalg.solve(V+np.eye(288),largeL)
    check('proposal whitened covariance relation',np.max(abs(proposal['latent']@np.linalg.cholesky(K)-proposal['standard']))<2e-12)
    mu=proposal['latent']@largeL.T; check('proposal displacement convention',np.max(abs(-(mu[:,:144]+1j*mu[:,144:])/2-proposal['beta']))<1e-13)
    _,logdet=np.linalg.slogdet((V+np.eye(288))/2); log_reference=-.5*logdet-.5*np.sum(mu*np.linalg.solve(V+np.eye(288),mu.T).T,axis=1)
    check('all Gaussian reference logs independently from covariance',np.max(abs(log_reference-proposal['reference_log_probability']))<2e-11)
    check('fixed common normal sequence',np.array_equal(proposal['standard'],np.random.default_rng(291301).normal(size=proposal['standard'].shape)))
    exactvac=ThresholdModel(V+largeL@largeL.T).vacuum_probability(tuple(range(144))); check('reference mixed vacuum',abs(exactvac/plan['large_reference_mixed_vacuum']-1)<2e-12)
    errors=[]; raw_errors=[]; rows_checked=[]
    for i,row in enumerate(rows):
        print('checking',i,flush=True); data=np.load(panel/f'case_{i:02d}.npz',allow_pickle=False); m=row['modes']; d=row['dimension']; L=largeL
        gaussian_cov=V+largeL@largeL.T
        if row['panel']=='small':
            ref=np.load(panel/f'case_{i:02d}.reference.npz',allow_pickle=False); L=ref['noise_factor']; dense=ref['dense']; gaussian_cov=pairing_covariance(ref['pairing'])+L@L.T
            entire=quadrature_density(np.outer(dense.ravel(),dense.ravel().conj()),dense.shape,L)
            check(f'{i} entire independent integration',abs(entire-row['independent_moment_vacuum'])<2e-12)
        def gaussian_vacuum(modes):
            ids=list(modes)+[j+m for j in modes]; return float(np.linalg.det((gaussian_cov[np.ix_(ids,ids)]+np.eye(len(ids)))/2)**-.5)
        for j in range(m):
            rho=data['single_densities'][j]; check(f'{i} single density {j}',abs(np.trace(rho)-1)<1e-10 and np.linalg.eigvalsh((rho+rho.conj().T)/2)[0]>-1e-11)
            value=quadrature_density(rho,(d,),L[[j,j+m]]); error=abs(value-data['single_vacuum'][j]); errors.append(error); check(f'{i} single direct integral {j}',error<2e-11)
            check(f'{i} Gaussian single marginal {j}',abs(gaussian_vacuum([j])-data['gaussian_single_vacuum'][j])<2e-12)
            if row['panel']=='small': check(f'{i} dense single trace {j}',np.max(abs(rho-dense_reduced(dense,[j])))<2e-12)
        pair_starts=data['pair_starts']; recomputed=[]
        for pos,j in enumerate(pair_starts):
            rho=data['pair_densities'][pos]; check(f'{i} pair density {j}',abs(np.trace(rho)-1)<1e-10 and np.linalg.eigvalsh((rho+rho.conj().T)/2)[0]>-1e-11)
            value=quadrature_density(rho,(d,d),L[[j,j+1,j+m,j+1+m]]); error=abs(value-data['pair_threshold'][pos,0]); errors.append(error); check(f'{i} pair direct integral {j}',error<2e-11)
            law=data['pair_threshold'][pos]; reference=data['gaussian_pair_threshold'][pos]; check(f'{i} threshold law {j}',np.min(law)>=0 and abs(law.sum()-1)<1e-12 and np.min(reference)>=0)
            g0=gaussian_vacuum([j]); g1=gaussian_vacuum([j+1]); g01=gaussian_vacuum([j,j+1]); direct_reference=np.array([g01,g1-g01,g0-g01,1-g0-g1+g01])
            check(f'{i} Gaussian pair marginal {j}',np.max(abs(direct_reference-reference))<2e-12)
            recomputed.append(.5*float(np.sum(abs(law-reference))))
            if row['panel']=='small': check(f'{i} dense pair trace {j}',np.max(abs(rho-dense_reduced(dense,[j,j+1])))<2e-12)
        check(f'{i} deterministic witness maximum',abs(max(recomputed)-row['marginals']['pair_tv_max'])<1e-14)
        weights=data['weights']; check(f'{i} all weights retained',len(weights)==row['samples'] and np.all(np.isfinite(weights)) and np.all(weights>=0))
        check(f'{i} log weights consistent',np.max(abs(np.exp(data['log_weights'])-weights))<1e-12)
        probability=row['reference_mixed_vacuum']*float(np.mean(weights)); se=row['reference_mixed_vacuum']*float(np.std(weights,ddof=1))/math.sqrt(len(weights))
        check(f'{i} importance estimate and standard error',abs(probability/row['importance']['probability_estimate']-1)<2e-13 and abs(se/row['importance']['probability_standard_error']-1)<2e-12)
        if row['panel']=='large':
            check(f'{i} actual likelihood ratio',np.max(abs(data['log_weights']-(data['log_probabilities']-log_reference)))<2e-11)
            family=row['family']
            if family.startswith('new_'):
                sequence={32:16,64:17,128:18,200:19}[int(family[4:])]; tensors=load_mps(ROOT/f'results/iteration25_panel_retry/run_{sequence}.state.npz')
            else:
                tensors,entries=load_family(family,ROOT/'results/iteration22_external',args.archive)
                saved=read(panel/(family+'_entries.json')); check(f'{i} external entry list',entries==saved)
                for a,b in zip(entries,saved): check(f'{i} member '+a['name'],a['sha256']==b['sha256'])
            check(f'{i} raw array fingerprint',mps_fingerprint(tensors)==row['raw_mps_fingerprint'])
            norm=scaled_overlap(tensors,tensors); lognorm=norm['log_abs']; check(f'{i} independent raw norm',not norm['zero'] and abs(lognorm-row['input_log_norm_squared'])<2e-10)
            for index in (0,1023,2047,4095):
                error=abs(raw_log_overlap(tensors,proposal['beta'][index],lognorm)-data['log_probabilities'][index]); raw_errors.append(error); check(f'{i} raw coherent point {index}',error<2e-9)
            witness=row['marginals']['pair_witness'][0]; pos=list(pair_starts).index(witness); direct=raw_pair_density(tensors,witness,lognorm); discrepancy=float(np.max(abs(direct-data['pair_densities'][pos])))
            check(f'{i} raw two-sided witness density',discrepancy<2e-10); rows_checked.append(dict(sequence=i,family=family,raw_witness_density_error=discrepancy)); del tensors
        save(out/'progress.json',dict(finished=i+1,count=len(checks),passed=sum(c['passed'] for c in checks)))
    pilot=read(ROOT/'results/iteration29_pilot/plan.json')
    for name,value in pilot['source_hashes'].items(): check('pilot historical source '+name,digest(ROOT/'results/iteration29_pilot/source_snapshot'/name)==value)
    check('verifier unchanged',digest(source)==sha)
    report=dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,verifier_sha256=sha,dependencies=plan['dependencies'],local_integrals=len(errors),maximum_local_quadrature_error=max(errors),raw_coherent_checks=len(raw_errors),maximum_raw_coherent_log_error=max(raw_errors),raw_witness_checks=rows_checked)
    save(out/'validation.json',report); print(json.dumps({k:v for k,v in report.items() if k not in ('checks','dependencies','raw_witness_checks')})); assert all(c['passed'] for c in checks)


if __name__=='__main__': main()
