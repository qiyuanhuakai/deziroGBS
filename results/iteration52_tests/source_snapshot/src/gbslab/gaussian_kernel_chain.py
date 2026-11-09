"""Untruncated Gaussian-kernel contraction under sufficient convergence guards.

Virtual Fock indices contract bilinearly, without complex conjugation.
Float64 diagnostics do not certify rounded matrices by interval arithmetic.
"""
import math
import numpy as np
from .banded_gaussian_mps import pairing_log_normalization


class GaussianContractionError(ArithmeticError):
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


def _symmetric(matrix):
    a=np.asarray(matrix,complex)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError('Require finite square Gaussian kernel')
    if a.size and np.max(np.abs(a-a.T))>1e-10:
        raise ValueError('Require symmetric Gaussian kernel')
    return (a+a.T)/2


def contract_gaussians(pairing,kernel,shared_modes,*,margin=1e-12):
    """B(v,x) and K(y,v) -> C(y,x), log determinant scalar and diagnostics.

    Each internal self block must have singular norm < 1-margin. This is a
    sufficient absolute-convergence condition, not a necessary one. On this
    domain I-t*D*H stays in the right half-plane for 0<=t<=1; summing principal
    eigenvalue logs fixes the continuation, unlike logging the determinant.
    """
    B=_symmetric(pairing); K=_symmetric(kernel); l=shared_modes
    if isinstance(l,bool) or not isinstance(l,(int,np.integer)) or not 0<=l<=min(len(B),len(K)):
        raise ValueError('Invalid shared mode count')
    if not math.isfinite(margin) or not 0<margin<1: raise ValueError('Invalid convergence margin')
    new=len(K)-l; A=K[:new,:new]; T=K[:new,new:]; D=K[new:,new:]
    H=B[:l,:l]; G=B[:l,l:]; J=B[l:,l:]
    if l==0:
        out=np.block([[A,np.zeros((new,len(J)),complex)],[np.zeros((len(J),new),complex),J]])
        return out,0j,dict(shared_modes=0,left_internal_norm=0.,right_internal_norm=0.,contraction_norm_bound=0.,condition=1.,solve_residual=0.,symmetry_residual=0.,log_factor_real=0.,log_factor_imag=0.)
    hn=float(np.linalg.norm(H,2)); dn=float(np.linalg.norm(D,2))
    info=dict(shared_modes=int(l),left_internal_norm=hn,right_internal_norm=dn,contraction_norm_bound=hn*dn)
    if max(hn,dn)>=1-margin: raise GaussianContractionError('Sufficient internal Gaussian convergence guard failed',info)
    S=np.eye(l)-D@H; condition=float(np.linalg.cond(S)); R=np.linalg.solve(S,np.eye(l))
    residual=float(np.linalg.norm(S@R-np.eye(l),np.inf)); eig=np.linalg.eigvals(S)
    info.update(condition=condition,solve_residual=residual,minimum_real_eigenvalue=float(np.min(eig.real)))
    if np.min(eig.real)<=0 or not np.isfinite(condition) or residual>1e-9:
        raise GaussianContractionError('Gaussian solve or logarithm branch guard failed',info)
    logs=np.log(eig); logfactor=-.5*complex(math.fsum(logs.real),math.fsum(logs.imag))
    out=np.block([[A+T@H@R@T.T,T@R.T@G],[G.T@R@T.T,J+G.T@R@D@G]])
    asym=float(np.max(np.abs(out-out.T))) if out.size else 0.
    info.update(symmetry_residual=asym,log_factor_real=logfactor.real,log_factor_imag=logfactor.imag)
    if not np.all(np.isfinite(out)) or asym>1e-9:
        raise GaussianContractionError('Contracted Gaussian symmetry/finite guard failed',info)
    return (out+out.T)/2,logfactor,info


def fold_kernel_chain(kernels,bond_modes,*,log_scales=None,margin=1e-12):
    """Each site has axes (left virtual modes, one physical, right virtual).

    No finite virtual occupations or physical photon cutoffs are applied.
    log_scales holds known local scalar logarithms. Unknown global scale
    cannot be inferred from kernels, although the normalized state can.
    """
    kernels=list(kernels); bonds=list(bond_modes); m=len(kernels)
    if m<1 or len(bonds)!=m+1 or bonds[0]!=0 or bonds[-1]!=0 or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<0 for v in bonds):
        raise ValueError('Invalid chain bond mode counts')
    scales=np.zeros(m,complex) if log_scales is None else np.asarray(log_scales,complex)
    if scales.shape!=(m,) or not np.all(np.isfinite(scales)): raise ValueError('Invalid local scalar logs')
    prefix=np.zeros((0,0),complex); scalar=0j; steps=[]
    for site,raw in enumerate(kernels):
        l,r=bonds[site:site+2]; K=_symmetric(raw)
        if K.shape!=(l+1+r,l+1+r): raise ValueError('Kernel dimensions do not match chain bonds')
        porder=list(range(site,site+l))+list(range(site))
        korder=list(range(l,l+1+r))+list(range(l))
        try:
            result,factor,info=contract_gaussians(prefix[np.ix_(porder,porder)],K[np.ix_(korder,korder)],l,margin=margin)
        except GaussianContractionError as e:
            e.diagnostics=dict(site=site,steps=steps,failed_step=e.diagnostics); raise
        order=list(range(1+r,1+r+site))+list(range(1+r))
        prefix=result[np.ix_(order,order)]; scalar+=scales[site]+factor
        steps.append(dict(site=site,**info))
    norm=float(np.linalg.norm(prefix,2))
    if norm>=1-margin: raise GaussianContractionError('Final Gaussian normalizability guard failed',dict(final_pairing_norm=norm,steps=steps))
    logc=pairing_log_normalization(prefix)
    return prefix,scalar,dict(steps=steps,final_pairing_norm=norm,normalized_log_vacuum_amplitude=logc,
        raw_log_norm_squared=2*(scalar.real-logc),log_scalar_real=scalar.real,log_scalar_imag=scalar.imag,
        scope='Untruncated state defined by supplied kernels; sufficient float64 convergence checks, no original experimental-state identification')


def reverse_kernel_chain(kernels,bond_modes,log_scales=None):
    """Reverse sites and swap virtual groups, keeping physical mode within each."""
    ks=list(kernels); bonds=list(bond_modes); m=len(ks)
    if len(bonds)!=m+1: raise ValueError('Invalid chain length')
    reversed_kernels=[]
    for site in range(m-1,-1,-1):
        l,r=bonds[site:site+2]; order=list(range(l+1,l+1+r))+[l]+list(range(l))
        K=_symmetric(ks[site]); reversed_kernels.append(K[np.ix_(order,order)])
    scales=None if log_scales is None else np.asarray(log_scales)[::-1].copy()
    return reversed_kernels,bonds[::-1],scales
