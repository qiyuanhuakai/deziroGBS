"""Final-state all-cut MPS existence diagnostics, not a large-mode constructor.

Box projection + TT-SVD yields ideal squared trace distance <= box_tail +
sum(original Schmidt tails). All numbers here are floating evaluations.
"""
import math
import numpy as np
from .mps_probe import geometric_top_weights,dense_to_mps


def marginal_photon_tails(normal_mean,anomalous_abs,dimensions,*,terms=128):
    """One-mode Gaussian PNR recurrence plus Chernoff remainder, no 1-CDF tail."""
    n=float(normal_mean); s=float(anomalous_abs); dims=tuple(dimensions)
    if not math.isfinite(n+s) or n<0 or s<0 or s*s>n*(n+1)+1e-12: raise ValueError('Invalid one-mode Gaussian moments')
    if not isinstance(terms,int) or terms<2 or any(isinstance(d,bool) or not isinstance(d,(int,np.integer)) or not 1<=d<terms for d in dims): raise ValueError('Invalid cutoff or recurrence length')
    probabilities=np.zeros(terms); A=(1+n-s)*(1+n+s); B=max(0.,n*(n+1)-s*s); C=n*n-s*s
    probabilities[0]=1/math.sqrt(A); probabilities[1]=B/A*probabilities[0]
    for k in range(2,terms): probabilities[k]=((2*k-1)*B*probabilities[k-1]-(k-1)*C*probabilities[k-2])/(A*k)
    if np.any(probabilities<0) or not np.all(np.isfinite(probabilities)): raise ArithmeticError('Invalid PNR recurrence')
    if n+s==0: remainder=0.; tilt=None
    else:
        tilt=1+.5/(n+s); t=1-tilt
        log_mgf=-.5*math.log((1+(n-s)*t)*(1+(n+s)*t))
        remainder=math.exp(log_mgf-terms*math.log(tilt))
    tail_sums=[math.fsum(probabilities[d:]) for d in dims]
    return dict(probabilities=probabilities,tail_sums=np.asarray(tail_sums),tail_upper_estimates=np.asarray(tail_sums)+remainder,
        remainder_upper_estimate=remainder,chernoff_tilt=tilt,normalization_residual=1-math.fsum(probabilities))


def equal_core_spectra(U,squeezing,max_rank):
    """All natural-order cut spectra from anomalous cross-block singular values."""
    u=np.asarray(U,complex); s=float(squeezing)
    if u.ndim!=2 or u.shape[0]!=u.shape[1] or len(u)<2 or not np.all(np.isfinite(u)) or not np.allclose(u.conj().T@u,np.eye(len(u)),atol=1e-11,rtol=0): raise ValueError('Require unitary U')
    if not math.isfinite(s) or not 0<=s<=3 or not isinstance(max_rank,int) or max_rank<1: raise ValueError('Invalid squeeze/rank')
    n=math.sinh(s)**2; M=-math.sinh(s)*math.cosh(s)*(u@u.T); cuts=[]
    for cut in range(1,len(u)):
        singular=np.linalg.svd(M[:cut,cut:],compute_uv=False)
        # q=(nu-1)/(nu+1), nu=sqrt(1+4*sigma^2), with no subtractive cancellation.
        nu=np.sqrt(1+4*singular**2); q=4*singular**2/(nu+1)**2
        weights=geometric_top_weights(q,max_rank)
        cuts.append(dict(cut=cut,cross_singular_values=singular,parameters=q,top_weights=weights))
    return dict(normal_mean=n,anomalous=M,cuts=cuts)


def representation_bounds(U,squeezing,dimensions=(6,8,10,12,16),ranks=(8,16,32,64,128,256,512)):
    """Finite-d MPS existence upper estimates plus necessary rank lower estimates."""
    spectra=equal_core_spectra(U,squeezing,max(ranks)); m=len(U)
    marginals=[marginal_photon_tails(spectra['normal_mean'],abs(spectra['anomalous'][i,i]),dimensions) for i in range(m)]
    tails=np.stack([v['tail_upper_estimates'] for v in marginals]); rows=[]
    for rank in ranks:
        schmidt_tails=[max(0.,1-math.fsum(c['top_weights'][:rank])) for c in spectra['cuts']]
        tail_sum=math.fsum(schmidt_tails); lower=math.sqrt(max(schmidt_tails))
        for j,d in enumerate(dimensions):
            box_tail=min(1.,math.fsum(tails[:,j])); squared=box_tail+tail_sum
            rows.append(dict(dimension=d,rank=rank,box_tail_union_estimate=box_tail,schmidt_tail_sum=tail_sum,
                max_cut_rank_distance_lower_estimate=lower,existence_trace_upper_estimate=math.sqrt(min(1.,squared)),
                untruncated_squared_budget=squared,scope='Ideal existence bound evaluated in float64; not a constructed MPS or interval certificate'))
    return dict(rows=rows,spectra=spectra,marginal_tail_upper_estimates=tails,
        marginal_probabilities=np.stack([v['probabilities'] for v in marginals]),
        remainder_upper_estimates=np.asarray([v['remainder_upper_estimate'] for v in marginals]))


def compress_box_once(unnormalized_box,max_bond,*,max_elements=2_000_000):
    """Small explicit Gaussian-box TT-SVD validation; exponential box cost."""
    x=np.asarray(unnormalized_box,complex); mass=float(np.vdot(x.ravel(),x.ravel()).real)
    if not 0<mass<=1+1e-10: raise ValueError('Require a nonzero subnormalized box')
    tensors,info=dense_to_mps(x/math.sqrt(mass),max_bond,max_elements=max_elements)
    retained=mass*(1-info['discarded_sum'])
    return tensors,dict(info,box_mass=mass,unnormalized_tt_retained_mass=retained,
        original_state_trace_from_orthogonal_loss=math.sqrt(max(0.,1-retained)))
