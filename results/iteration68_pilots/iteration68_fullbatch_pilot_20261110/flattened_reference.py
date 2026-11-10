"""Direct threshold sampling of a displaced, finite-input MPS.

Rank-one refinement of the binary POVM keeps a pure virtual trajectory.
There is no output Fock cutoff. Hidden labels are not observed photons.
Floating-point diagnostics, not certified random-real arithmetic.
"""
import math
import numpy as np
from scipy.special import gammainc, gammaincc, gammaln
from .mps_overlap import normalized_right_mps


def coherent_direction(beta, dimension):
    """Normalized finite coherent vector and its exact Poisson split formulas."""
    beta=np.asarray(beta,complex)
    if beta.ndim!=1 or not np.all(np.isfinite(beta)) or not isinstance(dimension,(int,np.integer)) or dimension<1:
        raise ValueError('Invalid coherent-vector input')
    t=abs(beta)**2
    if not np.all(np.isfinite(t)) or np.any((beta!=0)&(t==0)):
        raise ArithmeticError('Unrepresentable coherent intensity')
    n=np.arange(dimension); logs=np.zeros((len(beta),dimension)); positive=t>0
    logs[positive]=.5*(np.log(t[positive,None])*n-gammaln(n+1))
    logs[~positive,1:]=-np.inf
    logs-=np.max(logs,axis=1)[:,None]
    magnitudes=np.exp(logs)
    if np.any(np.isfinite(logs)&(magnitudes==0)):
        raise ArithmeticError('Positive coherent direction underflowed')
    direction=magnitudes*np.exp(1j*np.angle(beta)[:,None]*n)
    direction/=np.linalg.norm(direction,axis=1)[:,None]
    retained=gammaincc(dimension,t); tail=gammainc(dimension,t)
    if np.any(positive&((retained==0)|(tail==0))):
        raise ArithmeticError('Positive Poisson branch underflowed')
    if not np.all(np.isfinite(retained+tail)) or np.max(abs(retained+tail-1),initial=0)>2e-14:
        raise ArithmeticError('Invalid Poisson partition')
    return direction,retained,tail


def refined_amplitudes(amplitudes,beta):
    """Householder rows: first coherent direction, then its orthogonal complement."""
    M=np.asarray(amplitudes,complex)
    if M.ndim!=3 or not np.all(np.isfinite(M)): raise ValueError('Invalid physical amplitudes')
    c,w,tail=coherent_direction(beta,M.shape[1])
    if len(c)!=len(M): raise ValueError('Batch mismatch')
    projection=np.einsum('bn,bnr->br',c.conj(),M)
    transformed=M.copy(); transformed[:,0,:]=-projection
    transformed[:,1:,:]-=c[:,1:,None]*(M[:,0,:]+projection)[:,None,:]/(1+c[:,0].real)[:,None,None]
    masses=np.sum(abs(transformed)**2,axis=2)
    if np.any((masses==0)&np.any(transformed!=0,axis=2)):
        raise ArithmeticError('Positive transformed norm underflowed')
    # Do not label unresolved destructive cancellation as structural zero.
    for j in range(M.shape[1]):
        zero=masses[:,j]==0
        if not np.any(zero): continue
        rows=-c[zero].conj() if j==0 else -c[zero,j,None]*c[zero].conj()/(1+c[zero,0].real)[:,None]
        if j:
            rows[:,j]+=1; rows[:,0]=-c[zero,j]
        reachable=np.any((rows!=0)&np.any(M[zero]!=0,axis=2),axis=1)
        if np.any(reachable): raise ArithmeticError('Unresolved refined-amplitude cancellation')
    weights=np.column_stack([w*masses[:,0],tail*masses[:,0],masses[:,1:]])
    if np.any((w>0)&(masses[:,0]>0)&(weights[:,0]==0)) or np.any((tail>0)&(masses[:,0]>0)&(weights[:,1]==0)):
        raise ArithmeticError('Positive refined probability underflowed')
    if not np.all(np.isfinite(weights)): raise ArithmeticError('Nonfinite refined probabilities')
    return transformed,masses,weights


def select_positive(weights,uniforms):
    """Ascending positive masses avoid losing small intervals after large ones."""
    weights=np.asarray(weights,float); u=np.asarray(uniforms,float)
    if weights.ndim!=2 or u.shape!=(len(weights),) or not np.all(np.isfinite(weights)) or np.any(weights<0) or not np.all(np.isfinite(u)) or np.any((u<0)|(u>=1)):
        raise ValueError('Invalid categorical input')
    order=np.argsort(weights,axis=1,kind='stable'); ordered=np.take_along_axis(weights,order,axis=1)
    cumulative=np.cumsum(ordered,axis=1); previous=np.column_stack([np.zeros(len(weights)),cumulative[:,:-1]])
    if np.any((ordered>0)&(cumulative<=previous)): raise ArithmeticError('Positive categorical interval disappeared')
    total=cumulative[:,-1]
    if np.any(total<=0) or not np.all(np.isfinite(total)): raise ArithmeticError('Empty categorical law')
    target=u*total
    if np.any((u>0)&(target==0)): raise ArithmeticError('Categorical coordinate underflowed')
    positions=np.sum(cumulative<=target[:,None],axis=1)
    if np.any(positions>=weights.shape[1]): raise ArithmeticError('Categorical endpoint rounded outside support')
    labels=order[np.arange(len(weights)),positions]
    return labels,total


class DisplacedThresholdSampler:
    """Draw independent threshold strings conditional on supplied displacements."""
    def __init__(self,tensors):
        self.tensors,self.input_log_norm=normalized_right_mps(tensors)
        self.modes=len(self.tensors)

    def sample(self,beta,uniforms):
        beta=np.asarray(beta,complex); uniforms=np.asarray(uniforms,float)
        if beta.ndim!=2 or beta.shape[1]!=self.modes or uniforms.shape!=beta.shape or not np.all(np.isfinite(beta)) or not np.all(np.isfinite(uniforms)) or np.any((uniforms<0)|(uniforms>=1)):
            raise ValueError('Invalid displacement or uniform array')
        count=len(beta); env=np.ones((count,1),complex)
        labels=np.zeros(beta.shape,np.int32); selected=np.zeros(beta.shape); vacuum=np.zeros(beta.shape); residual=0.
        if count==0: return dict(clicks=labels.astype(np.uint8),refined_labels=labels,selected_refined_probabilities=selected,vacuum_given_hidden_history=vacuum,maximum_normalization_residual=0.)
        for site,A in enumerate(self.tensors):
            M=(env @ A.reshape(A.shape[0], -1)).reshape(count, A.shape[1], A.shape[2])
            transformed,masses,weights=refined_amplitudes(M,beta[:,site])
            chosen,total=select_positive(weights,uniforms[:,site]); error=float(np.max(abs(total-1)))
            residual=max(residual,error)
            if error>1e-9: raise ArithmeticError('Conditional MPS norm drift')
            rows=np.arange(count); physical=np.where(chosen<=1,0,chosen-1)
            env=transformed[rows,physical,:]/np.sqrt(masses[rows,physical])[:,None]
            labels[:,site]=chosen; selected[:,site]=weights[rows,chosen]/total; vacuum[:,site]=weights[:,0]/total
        return dict(clicks=(labels!=0).astype(np.uint8),refined_labels=labels,selected_refined_probabilities=selected,vacuum_given_hidden_history=vacuum,maximum_normalization_residual=residual)


def prior_displacements(noise_factor,standard):
    """Prior Gaussian noise (not the vacuum-tilted importance proposal)."""
    F=np.asarray(noise_factor); z=np.asarray(standard)
    if np.iscomplexobj(F) or np.iscomplexobj(z): raise ValueError('Noise inputs must be real')
    F=np.asarray(F,float); z=np.asarray(z,float)
    if F.ndim!=2 or F.shape[0]%2 or F.shape[0]==0 or z.ndim!=2 or z.shape[1]!=F.shape[1] or not np.all(np.isfinite(F)) or not np.all(np.isfinite(z)):
        raise ValueError('Invalid Gaussian noise inputs')
    mu=z@F.T; m=F.shape[0]//2
    if not np.all(np.isfinite(mu)): raise ArithmeticError('Nonfinite displacement')
    return -(mu[:,:m]+1j*mu[:,m:])/2
