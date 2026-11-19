"""Original-state Schmidt spectra for arbitrary normalizable pure Gaussian B.

Uses the smaller physical side of every cut. No virtual-chain spectrum is
assumed to be the physical Schmidt spectrum. Float64 diagnostic bounds only.
"""
import math
import numpy as np
from .banded_gaussian_mps import pairing_covariance
from .mps_probe import geometric_top_weights
from .gaussian_mps_feasibility import marginal_photon_tails


def pairing_spectra(pairing,max_rank):
    B=np.asarray(pairing,complex); V=pairing_covariance(B); m=len(B)
    if m<2 or isinstance(max_rank,bool) or not isinstance(max_rank,(int,np.integer)) or max_rank<1:
        raise ValueError('Require at least two modes and positive integer rank')
    cuts=[]
    for cut in range(1,m):
        ids=list(range(cut)) if cut<=m-cut else list(range(cut,m)); n=len(ids)
        block=V[np.ix_(ids+[m+i for i in ids],ids+[m+i for i in ids])]
        values,vectors=np.linalg.eigh(block)
        if values[0]<=0: raise ArithmeticError('Nonpositive reduced covariance')
        root=(vectors*np.sqrt(values))@vectors.T
        omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
        hermitian=1j*root@omega@root; hermitian=(hermitian+hermitian.conj().T)/2
        nu=np.linalg.eigvalsh(hermitian)[n:]
        independent=np.linalg.eigvals(1j*omega@block); other=np.sort(independent.real)[n:]
        discrepancy=float(np.max(np.abs(nu-other))); imaginary=float(np.max(np.abs(independent.imag)))
        if nu.min()<1-1e-9 or max(discrepancy,imaginary)>1e-9:
            raise ArithmeticError('Invalid physical symplectic spectrum')
        q=np.maximum(0.,(nu-1)/(nu+1)); weights=geometric_top_weights(q,int(max_rank))
        cuts.append(dict(cut=cut,physical_side='left' if cut<=m-cut else 'right',side_modes=n,
            minimum_raw_nu=float(nu.min()),symplectic_values=nu,parameters=q,top_weights=weights,
            independent_spectrum_error=discrepancy,independent_imaginary_residual=imaginary))
    inverse=np.linalg.solve(np.eye(m)-B@B.conj().T,np.eye(m)); N=inverse-np.eye(m); M=inverse@B
    return dict(cuts=cuts,covariance=V,normal_matrix=N,anomalous_matrix=M)


def pairing_representation_bounds(pairing,dimensions=(4,6,8),ranks=(32,64,128,200,512,1024,2048)):
    if not ranks or any(isinstance(r,bool) or not isinstance(r,(int,np.integer)) or r<1 for r in ranks): raise ValueError('Invalid ranks')
    spectra=pairing_spectra(pairing,int(max(ranks))); m=len(pairing)
    marginal=[marginal_photon_tails(max(0.,float(spectra['normal_matrix'][i,i].real)),abs(spectra['anomalous_matrix'][i,i]),dimensions) for i in range(m)]
    physical=np.stack([r['tail_upper_estimates'] for r in marginal]); rows=[]
    for rank in ranks:
        tails=[max(0.,1-math.fsum(c['top_weights'][:rank])) for c in spectra['cuts']]; total=math.fsum(tails)
        for j,d in enumerate(dimensions):
            box=math.fsum(physical[:,j]); rows.append(dict(rank=int(rank),dimension=int(d),max_cut_tail=max(tails),schmidt_tail_sum=total,physical_tail_union=box,
                pure_rank_distance_lower_estimate=math.sqrt(max(tails)),original_projector_trace_upper_estimate=math.sqrt(min(1.,total+box)),unclipped_squared_upper=total+box,
                scope='Original physical Gaussian state, ideal spectral/projector bounds evaluated in float64; no output-TV lower bound or finite-precision certificate'))
    return dict(rows=rows,spectra=spectra,marginal_tail_upper_estimates=physical,marginal_probabilities=np.stack([r['probabilities'] for r in marginal]))
