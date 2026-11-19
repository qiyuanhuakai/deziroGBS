"""Finite-MPS vacuum events after a common Gaussian displacement channel.

Importance sampling uses the exact Gaussian reference's vacuum-tilted noise
law. This estimates one event, not a complete output sampler or TV certificate.
"""
import math
import numpy as np
from .banded_gaussian_mps import pairing_covariance,pairing_log_normalization
from .mps_overlap import normalized_right_mps


class CoherentMPS:
    """Whole-state normalization once, then scaled coherent overlaps in batches."""
    def __init__(self,tensors):
        self.tensors,self.input_log_norm=normalized_right_mps(tensors)
        self.modes=len(self.tensors)

    def log_probability(self,beta):
        """log |<beta|psi>|^2. Numeric zero is -inf, not an analytic certificate."""
        beta=np.asarray(beta,complex)
        if beta.ndim!=2 or beta.shape[1]!=self.modes or not np.all(np.isfinite(beta)): raise ValueError('Invalid coherent-state batch')
        count=len(beta); environment=np.ones((count,1),complex); logs=np.zeros(count)
        for site,A in enumerate(self.tensors):
            z=beta[:,site].conj(); coefficient=np.ones(count,complex); coefficients=[coefficient]; temporary=environment@A[:,0,:]
            for n in range(1,A.shape[1]):
                old=coefficient; coefficient=coefficient*z/math.sqrt(n)
                if np.any((old!=0)&(z!=0)&(coefficient==0)): raise ArithmeticError('Positive coherent coefficient magnitude underflowed')
                coefficients.append(coefficient)
                temporary+=(environment@A[:,n,:])*coefficient[:,None]
            scale=np.max(np.abs(temporary),axis=1)
            if not np.all(np.isfinite(scale)): raise ArithmeticError('Nonfinite coherent contraction')
            zero=scale==0
            if np.any(zero):
                possible=np.zeros(np.count_nonzero(zero),bool); incoming=(environment[zero]!=0).astype(np.int32)
                for n,c in enumerate(coefficients):
                    reachable=np.any(incoming@(A[:,n,:]!=0).astype(np.int32)>0,axis=1)
                    possible|=reachable&(c[zero]!=0)
                if np.any(possible): raise ArithmeticError('Unresolved coherent cancellation or positive underflow')
            positive=scale>0; logs[positive]+=np.log(scale[positive]); logs[~positive]=-np.inf
            environment=temporary/np.where(positive,scale,1)[:,None]
        final=np.abs(environment[:,0]); positive=final>0; logs[positive]+=np.log(final[positive]); logs[~positive]=-np.inf
        norm=np.sum(np.abs(beta)**2,axis=1)
        if not np.all(np.isfinite(norm)): raise ArithmeticError('Nonfinite coherent Gaussian exponent')
        result=2*logs-norm
        if np.any(np.isnan(result)) or np.any(result>1e-9): raise ArithmeticError('Invalid normalized coherent probability')
        return result


class GaussianVacuumProposal:
    """For displacement mu=L z, draw z conditional on the reference vacuum event.

Prior z~N(0,I); K=I+L.T(V_B+I)^-1 L; proposal covariance=K^-1.
The exact reference mixed-vacuum probability is p0(B)/sqrt(det K).
"""
    def __init__(self,pairing,noise_factor):
        self.pairing=np.asarray(pairing,complex); self.log_pure_vacuum=2*pairing_log_normalization(self.pairing)
        self.covariance=pairing_covariance(self.pairing); self.modes=len(self.pairing)
        raw=np.asarray(noise_factor)
        if np.iscomplexobj(raw) and np.any(raw.imag!=0): raise ValueError('Noise factor must be real')
        L=np.asarray(raw.real,float)
        if L.ndim!=2 or L.shape[0]!=2*self.modes or not np.all(np.isfinite(L)): raise ValueError('Invalid quadrature noise factor')
        self.noise_factor=L.copy(); self.noise_dimensions=L.shape[1]
        inverse_noise=np.linalg.solve(self.covariance+np.eye(2*self.modes),L)
        H=L.T@inverse_noise; self.precision=np.eye(L.shape[1])+(H+H.T)/2
        self.cholesky=np.linalg.cholesky(self.precision)
        self.log_determinant=float(2*np.log(np.diag(self.cholesky)).sum())
        self.log_mixed_vacuum=self.log_pure_vacuum-.5*self.log_determinant
        self.mixed_vacuum=math.exp(self.log_mixed_vacuum)
        if self.mixed_vacuum==0: raise ArithmeticError('Positive reference mixed-vacuum mass underflowed')

    def beta_from_latent(self,z):
        z=np.asarray(z,float)
        if z.ndim!=2 or z.shape[1]!=self.noise_dimensions or not np.all(np.isfinite(z)): raise ValueError('Invalid latent batch')
        mu=z@self.noise_factor.T
        return -(mu[:,:self.modes]+1j*mu[:,self.modes:])/2

    def from_standard(self,standard):
        standard=np.asarray(standard,float)
        if standard.ndim!=2 or standard.shape[1]!=self.noise_dimensions or not np.all(np.isfinite(standard)): raise ValueError('Invalid normal batch')
        z=np.linalg.solve(self.cholesky.T,standard.T).T
        return z,self.beta_from_latent(z)

    def gaussian_log_probability(self,beta):
        beta=np.asarray(beta,complex)
        if beta.ndim!=2 or beta.shape[1]!=self.modes or not np.all(np.isfinite(beta)): raise ValueError('Invalid coherent-state batch')
        conjugate=beta.conj(); exponent=np.einsum('bi,ij,bj->b',conjugate,self.pairing,conjugate).real-np.sum(abs(beta)**2,axis=1)
        result=self.log_pure_vacuum+exponent
        if not np.all(np.isfinite(result)): raise ArithmeticError('Nonfinite Gaussian reference event')
        return result

    def likelihood_ratios(self,log_probabilities,beta):
        logp=np.asarray(log_probabilities,float); reference=self.gaussian_log_probability(beta)
        if logp.shape!=reference.shape or np.any(np.isnan(logp)) or np.any(logp==np.inf): raise ValueError('Invalid event log probabilities')
        logs=logp-reference
        with np.errstate(over='ignore',under='ignore'):
            weights=np.exp(logs)
        if np.any(~np.isfinite(weights)) or np.any((weights==0)&np.isfinite(logs)):
            raise ArithmeticError('Positive importance weight not representable; no mass was dropped')
        return weights,logs


def importance_summary(weights,reference_probability):
    w=np.asarray(weights,float)
    if w.ndim!=1 or len(w)<2 or not np.all(np.isfinite(w)) or np.any(w<0) or not 0<reference_probability<=1+1e-12: raise ValueError('Invalid importance data')
    scale=float(w.max()); scaled=w/scale if scale else w; mean_scaled=math.fsum(scaled)/len(w)
    variance_scaled=math.fsum((float(x)-mean_scaled)**2 for x in scaled)/(len(w)-1)
    mean=scale*mean_scaled; se=scale*math.sqrt(variance_scaled/len(w)); square_sum=math.fsum(float(x*x) for x in scaled)
    ess=math.fsum(scaled)**2/square_sum if square_sum else 0.
    estimate=reference_probability*mean; uncertainty=reference_probability*se
    if mean>0 and estimate==0: raise ArithmeticError('Positive mixed-vacuum estimate underflowed')
    return dict(samples=len(w),mean_ratio=mean,ratio_standard_error=se,probability_estimate=estimate,
        probability_standard_error=uncertainty,relative_standard_error=se/mean if mean else None,
        empirical_effective_sample_size=ess,maximum_ratio=scale,numeric_zero_weights=int(np.count_nonzero(w==0)),
        normal_approximation_interval=[estimate-1.96*uncertainty,estimate+1.96*uncertainty],
        estimate_outside_unit_interval=not 0<=estimate<=1,
        scope='Unbiased importance identity in exact arithmetic; finite Monte Carlo and floating diagnostics. Standard error and normal interval are not rigorous confidence guarantees, output samples, or full-TV certificates.')
