"""Dyadic Hermitian witnesses and an exact B,F Gaussian model.

Fixed-point Cholesky only proposes a witness. Acceptance uses exact integer
matrix products, never a floating tolerance. PSD boundary cases can remain
unresolved; a failed positive-definite witness is not a negative certificate.
"""
from fractions import Fraction as F
from math import isqrt,prod
import numpy as np
from .rational_gaussian import integer_matrix,outward


class CertificationFailure(ArithmeticError):pass


def nearest(n,d):
    if d<=0:raise ValueError('Positive divisor required')
    return (1 if n>=0 else -1)*((abs(int(n))+d//2)//d)


def sqrt_interval(lo,hi,bits=256):
    lo,hi=F(lo),F(hi)
    if not 0<=lo<=hi:raise ValueError('Nonnegative ordered interval required')
    outward(0,1,bits);scale=1<<bits
    def root(x):
        t=isqrt((x.numerator*scale*scale)//x.denominator)
        return F(t,scale),F(t if t*t*x.denominator==x.numerator*scale*scale else t+1,scale)
    return root(lo)[0],root(hi)[1]


def fixed_cholesky_inverse(ar,ai,den,bits):
    n=len(ar);scale=1<<bits;real_only=not any(int(x) for x in ai.flat)
    a=[[nearest(int(ar[i,j])*scale*scale,den) for j in range(n)] for i in range(n)];b=[[nearest(int(ai[i,j])*scale*scale,den) for j in range(n)] for i in range(n)]
    lr=[[0]*n for _ in range(n)];li=[[0]*n for _ in range(n)]
    for j in range(n):
        pivot=a[j][j]-sum(lr[j][k]**2+li[j][k]**2 for k in range(j))
        if pivot<=0:raise CertificationFailure(f'Fixed-point Cholesky candidate stopped at pivot {j}: {pivot}')
        lr[j][j]=isqrt(pivot)
        if not lr[j][j]:raise CertificationFailure('Zero proposed diagonal')
        for i in range(j+1,n):
            re=a[i][j]-sum(lr[i][k]*lr[j][k]+li[i][k]*li[j][k] for k in range(j));lr[i][j]=nearest(re,lr[j][j])
            if not real_only:
                im=b[i][j]-sum(li[i][k]*lr[j][k]-lr[i][k]*li[j][k] for k in range(j));li[i][j]=nearest(im,lr[j][j])
    rr=[[0]*n for _ in range(n)];ri=[[0]*n for _ in range(n)]
    for i in range(n):
        rr[i][i]=nearest(scale*scale,lr[i][i])
        for j in range(i):
            re=-sum(lr[i][k]*rr[k][j]-li[i][k]*ri[k][j] for k in range(j,i));rr[i][j]=nearest(re,lr[i][i])
            if not real_only:
                im=-sum(lr[i][k]*ri[k][j]+li[i][k]*rr[k][j] for k in range(j,i));ri[i][j]=nearest(im,lr[i][i])
    return np.array(rr,dtype=object),np.array(ri,dtype=object)


class HermitianCertificate:
    def __init__(self,real,imag=None,*,bits=128):
        a=np.asarray(real,dtype=object)
        if a.ndim!=2 or a.shape[0]!=a.shape[1] or not 1<=len(a)<=512 or not isinstance(bits,int) or not 32<=bits<=256:raise ValueError('Invalid Hermitian certificate dimensions or bits')
        b=np.zeros(a.shape,dtype=object) if imag is None else np.asarray(imag,dtype=object)
        if b.shape!=a.shape:raise ValueError('Mismatched imaginary matrix')
        if any(F(x).denominator&(F(x).denominator-1) for matrix in (a,b) for x in matrix.flat):raise ValueError('Every entry must be dyadic')
        ar,dr=integer_matrix(a);ai,di=integer_matrix(b);den=max(dr,di);ar=ar*(den//dr);ai=ai*(den//di)
        if not np.array_equal(ar,ar.T) or not np.array_equal(ai,-ai.T):raise ValueError('Exact Hermitian symmetry required')
        rr,ri=fixed_cholesky_inverse(ar,ai,den,bits);scale=1<<bits;real_only=not any(int(x) for x in ai.flat)
        if real_only:er=(rr@ar)@rr.T-np.eye(len(a),dtype=object)*(den*scale*scale);ei=np.zeros(a.shape,dtype=object)
        else:
            ur=rr@ar-ri@ai;ui=rr@ai+ri@ar;er=ur@rr.T+ui@ri.T-np.eye(len(a),dtype=object)*(den*scale*scale);ei=ui@rr.T-ur@ri.T
        assert np.array_equal(er,er.T) and np.array_equal(ei,-ei.T)
        delta=F(max(sum(abs(int(er[i,j]))+abs(int(ei[i,j])) for j in range(len(a))) for i in range(len(a))),den*scale*scale)
        if delta>=1:raise CertificationFailure('Exact residual cannot prove positive definiteness: '+str(delta))
        self.real=ar;self.imag=ai;self.denominator=den;self.R_real=rr;self.R_imag=ri;self.bits=bits;self.scale=scale;self.delta=delta;self.dimension=len(a);self.real_only=real_only
        self.R_frobenius_squared=F(sum(int(x)**2 for x in rr.flat)+sum(int(x)**2 for x in ri.flat),scale*scale);self.minimum_eigenvalue_lower=(1-delta)/self.R_frobenius_squared

    def determinant_interval(self,bits=256):
        determinant_R=prod((F(int(self.R_real[i,i]),self.scale) for i in range(self.dimension)),start=F(1));n=self.dimension
        return outward((1-self.delta)**n/determinant_R**2,(1+self.delta)**n/determinant_R**2,bits)

    def diagnostics(self):
        return dict(dimension=self.dimension,witness_bits=self.bits,exact_residual_upper=self.delta,minimum_eigenvalue_lower=self.minimum_eigenvalue_lower,positive_definite_proved=True,real_only=self.real_only)


def exact_uncertainty_rayleigh(V,w):
    n=len(V);m=n//2;Z,dv=integer_matrix(V);U,du=integer_matrix(np.stack([np.asarray(w).real,np.asarray(w).imag]));a,b=U;omega_b=np.r_[b[m:],-b[:m]];form=F(int(a@(Z@a)+b@(Z@b)-2*dv*(a@omega_b)),dv*du*du);norm=F(int(a@a+b@b),du*du)
    if not norm:raise ValueError('Nonzero witness required')
    return form/norm


def certify_uncertainty(covariance,*,bits=128):
    raw=np.asarray(covariance)
    if np.iscomplexobj(raw) or raw.ndim!=2 or raw.shape[0]!=raw.shape[1] or not len(raw) or len(raw)%2:raise ValueError('Even real covariance required')
    V=np.asarray(raw,float)
    if not np.all(np.isfinite(V)) or not np.array_equal(V,V.T):raise ValueError('Finite exact covariance symmetry required')
    m=len(V)//2;O=np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]])
    try:
        cert=HermitianCertificate(V,O,bits=bits);return dict(status='strictly_physical',certificate=cert)
    except CertificationFailure as error:
        vals,vectors=np.linalg.eigh(V+1j*O);w=vectors[:,0];rayleigh=exact_uncertainty_rayleigh(V,w)
        return dict(status='strict_violation' if rayleigh<0 else 'unresolved',candidate_failure=str(error),floating_minimum_eigenvalue=float(vals[0]),witness=w,exact_rayleigh=rayleigh)


class ExactPairingModel:
    """The mathematical V*=2(I-S)^-1-I+FF^T, without rounding it to a new target."""
    def __init__(self,pairing,noise,*,bits=128,output_bits=256):
        B=np.asarray(pairing,complex);raw=np.asarray(noise)
        if B.ndim!=2 or B.shape[0]!=B.shape[1] or not len(B) or not np.all(np.isfinite(B)) or not np.array_equal(B,B.T):raise ValueError('Finite exactly symmetric pairing required')
        if np.iscomplexobj(raw) or raw.ndim!=2 or raw.shape[0]!=2*len(B) or not np.all(np.isfinite(raw)):raise ValueError('Finite real noise factor required')
        self.m=len(B);n=2*self.m;bre=np.array([[F(float(x)) for x in row] for row in B.real],dtype=object);bim=np.array([[F(float(x)) for x in row] for row in B.imag],dtype=object);S=np.block([[bre,bim],[bim,-bre]]);K=np.eye(n,dtype=object)-S;core=HermitianCertificate(K,bits=bits);self.core=core;self.output_bits=output_bits
        Z,df=integer_matrix(raw) if raw.shape[1] else (np.zeros(raw.shape,dtype=object),1);noise_gram=Z@Z.T;self.noise_numerators=Z;self.noise_denominator=df
        if raw.shape[1]:
            denominator=2*df*df*core.denominator;Tnum=Z.T@(core.real@Z)+np.eye(raw.shape[1],dtype=object)*denominator;T=np.array([[F(int(x),denominator) for x in row] for row in Tnum],dtype=object);self.noise_certificate=HermitianCertificate(T,bits=bits);tlo,thi=self.noise_certificate.determinant_interval(output_bits)
        else:self.noise_certificate=None;tlo=thi=F(1)
        klo,khi=core.determinant_interval(output_bits)
        if klo<=0 or tlo<=0:raise CertificationFailure('Output determinant interval too coarse')
        self.vacuum_interval=sqrt_interval(klo/thi,khi/tlo,output_bits);self.core_determinant_interval=(klo,khi);self.noise_determinant_interval=(tlo,thi)
        Y=core.R_real.T@core.R_real;dy=core.scale**2;dc=max(dy,df*df);self.center_numerators=(2*Y-np.eye(n,dtype=object)*dy)*(dc//dy)+noise_gram*(dc//(df*df));self.center_denominator=dc
        traceY=F(sum(int(Y[i,i]) for i in range(n)),dy);self.operator_radius=2*core.delta/(1-core.delta)*traceY;self.core_trace_upper=2*traceY/(1-core.delta)-n
        self.core_trace_lower=2*traceY/(1+core.delta)-n;self.covariance_center_float=np.array([[float(F(int(x),dc)) for x in row] for row in self.center_numerators])

    def compare_covariance(self,covariance):
        V=np.asarray(covariance,float)
        if V.shape!=self.center_numerators.shape or not np.array_equal(V,V.T):raise ValueError('Matching symmetric covariance required')
        Z,dv=integer_matrix(V);den=max(dv,self.center_denominator);D=self.center_numerators*(den//self.center_denominator)-Z*(den//dv);max_entry=F(max(abs(int(x)) for x in D.flat),den);row_norm=F(max(sum(abs(int(x)) for x in row) for row in D),den);radius=self.operator_radius
        return dict(max_entry_difference_lower=max(F(0),max_entry-radius),max_entry_difference_upper=max_entry+radius,operator_difference_lower=max(F(0),max_entry-radius),operator_difference_upper=row_norm+radius,center_row_norm=row_norm,model_center_operator_radius=radius)

    def noise_transfer_bound(self,original,shifted):
        old=np.asarray(original,float);new=np.asarray(shifted,float)
        if old.shape!=new.shape or not np.array_equal(old-np.diag(np.diag(old)),new-np.diag(np.diag(new))):raise ValueError('Require explicit diagonal-only shift')
        shifts=[F(float(new[i,i]))-F(float(old[i,i])) for i in range(len(old))];old_error=self.compare_covariance(old)['operator_difference_upper'];margin=min(shifts)-old_error;new_error=self.compare_covariance(new)['operator_difference_upper'];bound=None
        if margin>0:bound=min(F(1),sqrt_interval(0,new_error*self.core_trace_upper/4,self.output_bits)[1])
        return dict(minimum_actual_diagonal_shift=min(shifts),maximum_actual_diagonal_shift=max(shifts),added_noise_PSD_margin_lower=margin,added_noise_operator_upper=new_error,noise_order_proved=margin>0,all_measurement_TV_transfer_upper=bound,scope='Between the exact B,F physical model and the explicitly shifted covariance, not a sampler accuracy certificate')
