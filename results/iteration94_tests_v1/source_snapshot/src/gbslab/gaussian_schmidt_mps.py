"""Gaussian-basis MPS conversion with downward-closed Fock projections.

Inspired by Liu et al., arXiv:2512.10643v2 (GSVD/PCO). Full Williamson
unitaries keep all virtual modes, including vacuum spectators; no mode freeze.
Float64 diagnostics only, not interval or finite-precision sampler guarantees.
"""
import heapq
import math
from time import perf_counter
import numpy as np
from scipy.linalg import schur
from .banded_gaussian_mps import pairing_covariance,pairing_log_normalization
from .gaussian_mps_feasibility import marginal_photon_tails


def williamson(covariance):
    """V=S diag(nu,nu) S.T in x...p ordering; vacuum covariance is I."""
    V=np.asarray(covariance,float); n=len(V)//2
    if V.shape!=(2*n,2*n) or n<1 or not np.all(np.isfinite(V)) or not np.allclose(V,V.T,atol=1e-11,rtol=0): raise ValueError('Invalid covariance')
    values,vectors=np.linalg.eigh(V)
    if values[0]<=0: raise ValueError('Nonpositive covariance')
    root=(vectors*np.sqrt(values))@vectors.T
    omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
    skew=root@omega@root; tri,O=schur((skew-skew.T)/2,output='real')
    raw=np.asarray([tri[2*j,2*j+1] for j in range(n)])
    if np.min(np.abs(raw))<1-1e-9: raise ArithmeticError('Unphysical symplectic spectrum')
    order=np.argsort(-np.abs(raw)); x=O[:,2*order]; p=O[:,2*order+1]*np.sign(raw[order])
    nu=np.abs(raw[order]); S=(root@np.column_stack((x,p)))/np.sqrt(np.r_[nu,nu])
    errors=dict(covariance_residual=float(np.max(np.abs((S*np.r_[nu,nu])@S.T-V))),
        symplectic_residual=float(np.max(np.abs(S@omega@S.T-omega))),minimum_raw_nu=float(nu.min()))
    if max(errors['covariance_residual'],errors['symplectic_residual'])>1e-9: raise ArithmeticError(f'Williamson residuals {errors}')
    return S,np.maximum(nu,1.),errors


def gaussian_unitary_kernel(S):
    """Bargmann kernel c exp((z A z +2z T x+x D x)/2) of a unitary."""
    n=len(S)//2; xx=S[:n,:n]; xp=S[:n,n:]; px=S[n:,:n]; pp=S[n:,n:]
    X=(xx+pp+1j*(px-xp))/2; Y=(xx-pp+1j*(px+xp))/2
    T=np.linalg.solve(X.conj().T,np.eye(n)); A=T@Y.T; D=-Y.conj().T@T
    Q=np.block([[A,T],[T.T,D]])
    if np.max(np.abs(Q-Q.T))>1e-9: raise ArithmeticError('Nonsymmetric unitary kernel')
    _,ld=np.linalg.slogdet(X); scale=math.exp(-.5*float(ld))
    return (Q+Q.T)/2,scale


def apply_gaussian_kernel(B,kernel,left_modes):
    """Contract input half of a Gaussian kernel with the first left_modes.

    Returns pairing and scalar multiplier excluding the supplied kernel scale.
    Kernel ordering is (new output modes, old first left_modes).
    """
    B=np.asarray(B,complex); K=np.asarray(kernel,complex); l=left_modes; new=len(K)-l
    A=K[:new,:new]; T=K[:new,new:]; D=K[new:,new:]
    H=B[:l,:l]; G=B[:l,l:]; J=B[l:,l:]
    inverse=np.linalg.solve(np.eye(l)-D@H,np.eye(l))
    out=np.block([[A+T@H@inverse@T.T,T@inverse.T@G],
                  [G.T@inverse@T.T,J+G.T@inverse@D@G]])
    eig=np.linalg.eigvals(np.eye(l)-D@H)
    if np.min(eig.real)<=0: raise ArithmeticError('Ambiguous Gaussian determinant continuation')
    factor=np.exp(-.5*np.sum(np.log(eig)))
    asym=float(np.max(np.abs(out-out.T)))
    if asym>1e-9: raise ArithmeticError('Transformed pairing is not symmetric')
    return (out+out.T)/2,complex(factor)


def top_occupations(parameters,rank):
    """Product geometric top states, with all spectator coordinates retained."""
    q=np.asarray(parameters,float)
    if q.ndim!=1 or np.any(q<0) or np.any(q>=1) or not np.all(np.isfinite(q)) or isinstance(rank,bool) or not isinstance(rank,int) or rank<1: raise ValueError('Invalid spectrum/rank')
    active=np.flatnonzero(q>0); logs=np.log(q[active]); log0=float(np.sum(np.log1p(-q)))
    heap=[(-log0,(0,)*len(active))]; states=[]; weights=[]
    while heap and len(states)<rank:
        energy,occ=heapq.heappop(heap); full=np.zeros(len(q),dtype=int); full[active]=occ
        states.append(full); weights.append(math.exp(-energy))
        for j in range(len(active)):
            child=list(occ); child[j]+=1; heapq.heappush(heap,(energy-logs[j],tuple(child)))
            if occ[j]: break
    return np.asarray(states,dtype=int).reshape(len(states),len(q)),np.asarray(weights)


def _validate_basis(states):
    states=np.asarray(states,dtype=int)
    if states.ndim!=2 or len(states)<1 or np.any(states<0): raise ValueError('Invalid occupation basis')
    lookup={tuple(row):i for i,row in enumerate(states)}
    if len(lookup)!=len(states) or (0,)*states.shape[1] not in lookup: raise ValueError('Basis must contain distinct states and vacuum')
    for row in states:
        for j in np.flatnonzero(row):
            below=row.copy(); below[j]-=1
            if tuple(below) not in lookup: raise ValueError('Basis is not downward closed')
    return states,lookup


def projected_pair_tensor(Q,left_basis,dimension,right_basis,*,max_elements=4_000_000):
    """Exact paired-state coefficients on left x physical x right kept basis.

    Polynomial projected pair creations, no global Fock box or hafnians.
    Kernel may represent a nonnormalizable isometry; no normalization assumed.
    """
    if isinstance(dimension,bool) or not isinstance(dimension,int) or not 1<=dimension<=24: raise ValueError('Invalid physical dimension')
    groups=[_validate_basis(left_basis),_validate_basis(np.arange(dimension)[:,None]),_validate_basis(right_basis)]
    sizes=[len(s) for s,_ in groups]; modes=[s.shape[1] for s,_ in groups]; total=sum(modes)
    Q=np.asarray(Q,complex)
    if Q.shape!=(total,total) or not np.all(np.isfinite(Q)) or not np.allclose(Q,Q.T,atol=1e-10,rtol=0): raise ValueError('Invalid local pairing kernel')
    if math.prod(sizes)>max_elements: raise MemoryError('Local coefficient allocation cap')
    value=np.zeros(sizes,complex); value[tuple(look[(0,)*s.shape[1]] for s,look in groups)]=1
    active=[]; offset=0
    for g,(states,_) in enumerate(groups):
        active.extend((offset+j,g,j) for j in range(modes[g]) if np.any(states[:,j])); offset+=modes[g]
    def mapping(g,increment):
        states,lookup=groups[g]; sources=[]; targets=[]; factors=[]
        for target,row in enumerate(states):
            sub=row-increment
            if np.any(sub<0): continue
            source=lookup.get(tuple(sub))
            if source is None: raise ArithmeticError('Downward closure was lost')
            factor=1.
            for j in np.flatnonzero(increment):
                for count in range(int(sub[j])+1,int(row[j])+1): factor*=math.sqrt(count)
            sources.append(source); targets.append(target); factors.append(factor)
        return np.asarray(sources,int),np.asarray(targets,int),np.asarray(factors)
    operations=0; pairs=0
    for ai,(a,ga,ja) in enumerate(active):
        for b,gb,jb in active[ai:]:
            coefficient=Q[a,b]*(.5 if a==b else 1.)
            if coefficient==0: continue
            increments=[np.zeros(n,dtype=int) for n in modes]; increments[ga][ja]+=1; increments[gb][jb]+=1
            result=None; power=1.+0j
            for k in range(1,1+sum(int(s.sum(axis=1).max()) for s,_ in groups)//2):
                maps=[mapping(g,inc*k) if np.any(inc) else (np.arange(sizes[g]),np.arange(sizes[g]),np.ones(sizes[g])) for g,inc in enumerate(increments)]
                if any(len(src)==0 for src,_,_ in maps): break
                if result is None: result=value.copy(); pairs+=1
                power*=coefficient/k
                factors=maps[0][2][:,None,None]*maps[1][2][None,:,None]*maps[2][2][None,None,:]
                result[np.ix_(*(v[1] for v in maps))]+=power*factors*value[np.ix_(*(v[0] for v in maps))]
                operations+=math.prod(len(v[0]) for v in maps)
            if result is not None: value=result
    return value,dict(pair_operators=pairs,coefficient_updates=operations,elements=value.size,active_local_modes=len(active))


def decompose_gaussian(B,*,max_seconds=120.):
    """Untruncated Gaussian Schmidt changes of basis; retain every mode.

    Prefix unitary maps are composed without physical or rank projections.
    Thus each cut spectrum refers to the same original state.
    """
    start=perf_counter(); B=np.asarray(B,complex); m=len(B); lognorm=pairing_log_normalization(B)
    residual=B.copy(); scalar=complex(math.exp(lognorm)); kernels=[]; records=[]
    for site in range(m-1):
        if perf_counter()-start>max_seconds: raise TimeoutError('Gaussian decomposition time cap between sites')
        left=site+1; V=pairing_covariance(residual); idx=list(range(left))+list(range(m,m+left))
        S,nu,errors=williamson(V[np.ix_(idx,idx)]); K,c=gaussian_unitary_kernel(S)
        inverse_order=list(range(left,2*left))+list(range(left)); inverse_kernel=K[np.ix_(inverse_order,inverse_order)].conj()
        updated,factor=apply_gaussian_kernel(residual,inverse_kernel,left)
        recovered,backfactor=apply_gaussian_kernel(updated,K,left)
        error=float(np.max(np.abs(recovered-residual)))
        scale_error=abs(c*c*factor*backfactor-1)
        scalar*=c*factor
        norm_error=abs(math.log(abs(scalar))-pairing_log_normalization(updated))
        if max(error,scale_error,norm_error)>1e-8: raise ArithmeticError(f'Gaussian reconstruction error {error,scale_error,norm_error}')
        records.append(dict(cut=left,parameters=(nu-1)/(nu+1),kernel_scale=c,
            reconstruction_residual=error,scale_residual=float(scale_error),normalization_log_residual=float(norm_error),**errors))
        kernels.append(K); residual=updated
    kernels.append(residual)
    return kernels,records,scalar,dict(seconds=perf_counter()-start,final_scalar_abs=abs(scalar),virtual_mode_policy='All prefix modes retained; no small symplectic mode frozen')


def build_schmidt_core(B,dimension,rank,*,max_seconds=240.,max_elements=4_000_000,return_maps=False,coefficient_backend='pco'):
    """Construct the physical-box projection of original Schmidt-projector MPS.

    Ideal trace upper sqrt(sum original cut tails + original marginal tails)
    follows from gentle sequential measurement after identifying the product
    of original prefix projectors. This is not a conditional-loss formula.
    """
    if coefficient_backend not in ('pco','recurrence'): raise ValueError('Unknown Gaussian coefficient backend')
    coefficient_function=projected_pair_tensor
    if coefficient_backend=='recurrence':
        # Lazy import avoids coupling module initialization through basis validation.
        from .gaussian_pair_recurrence import pair_recurrence_tensor
        coefficient_function=pair_recurrence_tensor
    start=perf_counter(); B=np.asarray(B,complex); m=len(B)
    kernels,records,scalar,decomposition=decompose_gaussian(B,max_seconds=max_seconds)
    decomposition_end=perf_counter()
    bases=[np.zeros((1,0),int)]; weights=[]
    for record in records:
        basis,values=top_occupations(record['parameters'],rank); bases.append(basis); weights.append(values)
    bases.append(np.zeros((1,0),int)); tensors=[]; local=[]; basis_end=perf_counter()
    for site,K in enumerate(kernels):
        if perf_counter()-start>max_seconds: raise TimeoutError('Gaussian MPS tensor construction time cap between sites')
        A,info=coefficient_function(K,bases[site],dimension,bases[site+1],max_elements=max_elements)
        A*=records[site]['kernel_scale'] if site<m-1 else scalar
        tensors.append(A); local.append(info)
    tensors_end=perf_counter()
    # Exact right QR only. The saved success mass is not the fidelity.
    for i in range(m-1,0,-1):
        A=tensors[i]; q,r=np.linalg.qr(A.reshape(A.shape[0],-1).T,mode='reduced')
        tensors[i]=q.T.reshape(q.shape[1],dimension,A.shape[2]); tensors[i-1]=np.tensordot(tensors[i-1],r.T,axes=(2,0))
    success=float(np.linalg.norm(tensors[0])**2)
    if not 0<success<=1+1e-8: raise ArithmeticError(f'Invalid projection success {success}')
    tensors[0]/=math.sqrt(success)
    canonicalization_end=perf_counter()
    inverse=np.linalg.solve(np.eye(m)-B@B.conj().T,np.eye(m)); N=inverse-np.eye(m); M=inverse@B
    tails=[float(marginal_photon_tails(max(0.,float(N[i,i].real)),abs(M[i,i]),[dimension])['tail_upper_estimates'][0]) for i in range(m)]
    cut_tails=[max(0.,1-math.fsum(w)) for w in weights]; loss=math.fsum(tails)+math.fsum(cut_tails)
    budget_end=perf_counter()
    phases=dict(gaussian_decomposition=decomposition_end-start,basis_selection=basis_end-decomposition_end,
        local_tensors=tensors_end-basis_end,canonicalization=canonicalization_end-tensors_end,error_budget=budget_end-canonicalization_end)
    info=dict(decomposition=decomposition,cuts=records,local=local,cut_tail_estimates=cut_tails,marginal_tail_estimates=tails,
        original_loss_sum=loss,construction_trace_budget=math.sqrt(min(1.,loss)),projection_success=success,
        stored_complex_elements=sum(a.size for a in tensors),max_bond_actual=max(a.shape[2] for a in tensors),total_seconds=budget_end-start,
        coefficient_backend=coefficient_backend,phase_seconds=phases,
        scope='Ideal original-projector bound evaluated in float64; success mass is not fidelity; no full finite-precision sampler TV certificate')
    return (tensors,info,dict(kernels=kernels,bases=bases,weights=weights,final_scalar=scalar)) if return_maps else (tensors,info)
