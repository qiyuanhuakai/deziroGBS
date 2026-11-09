"""Complete small MPS sampling law, conditional on a supplied displacement.

The floating kernel is the same callable used by the sampler. Branch weights
are accumulated exactly as rational numbers. The 53-bit grid assumes fresh
independent uniform integers: it is NOT the law of a fixed-seed PRNG, nor an
average over the Gaussian displacement generator. High-precision references
are convergence checks, not outward-rounded certificates.
"""
from fractions import Fraction
import math
import numpy as np
from .local_mps import threshold_step


def choice_intervals(probabilities, bits=53):
    """NumPy 2.4.6 weighted replacement choice: cumulative, divide, right search.

    Return exact rational interval lengths and integer counts on k/2**bits.
    Lower endpoints belong to the next interval (searchsorted side='right').
    """
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=1 or not len(p) or not np.all(np.isfinite(p)) or np.any(p<0):
        raise ValueError('Invalid categorical probabilities')
    total=float(p[0]); correction=0.
    for value in p[1:]:
        y=float(value)-correction; updated=total+y
        correction=(updated-total)-y; total=updated
    if abs(total-1)>math.sqrt(np.finfo(float).eps):
        raise ValueError('Invalid categorical probability sum')
    if isinstance(bits,bool) or not isinstance(bits,int) or not 1<=bits<=53:
        raise ValueError('Grid bits must be in [1,53]')
    cdf=p.cumsum(); cdf/=cdf[-1]
    grid=1<<bits
    previous=Fraction(0); last_count=0; intervals=[]; counts=[]
    for value in cdf:
        edge=Fraction.from_float(float(value))
        upper=(edge.numerator*grid+edge.denominator-1)//edge.denominator
        upper=min(grid,max(0,upper))
        intervals.append(edge-previous); counts.append(upper-last_count)
        previous=edge; last_count=upper
    if sum(intervals)!=1 or sum(counts)!=grid or any(x<0 for x in counts):
        raise ArithmeticError('CDF does not partition the uniform grid')
    return intervals,counts,cdf


def enumerate_floating_law(tensors, alpha, *, bits=53, max_nodes=200_000):
    """Full binary output laws plus failure symbol, no renormalization on success."""
    m=len(tensors); alpha=np.asarray(alpha,complex)
    if isinstance(bits,bool) or not isinstance(bits,int) or not 1<=bits<=53:
        raise ValueError('Grid bits must be in [1,53]')
    if not 1<=m<=8 or alpha.shape!=(m,) or not np.all(np.isfinite(alpha)):
        raise ValueError('Invalid displacement or mode cap')
    for a in tensors[1:]:
        flat=a.reshape(a.shape[0],-1)
        if not np.allclose(flat@flat.conj().T,np.eye(len(flat)),atol=1e-10,rtol=0):
            raise ValueError('Require right-canonical MPS')
    grid=1<<bits; denominator=grid**m
    real=[Fraction(0) for _ in range(1<<m)]; integers=[0]*(1<<m)
    real_failure=Fraction(0); integer_failure=0; nodes=0; failures=[]; max_residual=0.
    stack=[(0,0,np.ones(1,complex),Fraction(1),1)]
    while stack:
        depth,mask,left,rweight,iweight=stack.pop()
        nodes+=1
        if nodes>max_nodes: raise MemoryError('Hidden-path enumeration node cap exceeded')
        if depth==m:
            real[mask]+=rweight; integers[mask]+=iweight
            continue
        try:
            branches,weights,total=threshold_step(left,tensors[depth],alpha[depth])
            intervals,counts,_=choice_intervals(weights/total,bits)
            max_residual=max(max_residual,abs(total-1))
        except (ValueError,ArithmeticError) as e:
            real_failure+=rweight; integer_failure+=iweight*grid**(m-depth)
            failures.append({'depth':depth,'mask':mask,'error':str(e),
                'real_weight':str(rweight),'grid_prefix_count':str(iweight)})
            continue
        for j,(width,count) in enumerate(zip(intervals,counts)):
            if (not rweight or not width) and (not iweight or not count): continue
            rw=rweight*width; iw=iweight*count
            if weights[j]<=0:
                real_failure+=rw; integer_failure+=iw*grid**(m-depth-1)
                failures.append({'depth':depth,'mask':mask,'error':'Selected zero branch'})
                continue
            child=branches[j]/math.sqrt(weights[j])
            stack.append((depth+1,mask|((j!=0)<<depth),child,rw,iw))
    assert sum(real)+real_failure==1
    assert sum(integers)+integer_failure==denominator
    quantization_tv=(sum(abs(Fraction(n,denominator)-p) for n,p in zip(integers,real))
        +abs(Fraction(integer_failure,denominator)-real_failure))/2
    return {'real_cdf_law':real,'grid_numerators':integers,'grid_denominator':denominator,
        'real_failure':real_failure,'grid_failure_numerator':integer_failure,
        'quantization_tv':quantization_tv,'nodes':nodes,'failure_records':failures,
        'max_conditional_norm_residual':max_residual,'bits':bits}


def high_precision_law(tensors, alpha, *, digits=80, max_elements=5000):
    """Independent exact-Fock vacuum contraction using mpmath, no hidden paths.

    Treat stored binary64 tensor entries and displacement coordinates as exact
    inputs. Normalize the resulting represented pure state in high precision.
    """
    import mpmath as mp
    m=len(tensors); dims=[a.shape[1] for a in tensors]
    if math.prod(dims)>max_elements or not 40<=digits<=500:
        raise MemoryError('High-precision reference resource cap')
    alpha=np.asarray(alpha,complex)
    if alpha.shape!=(m,) or not np.all(np.isfinite(alpha)): raise ValueError('Invalid alpha')
    with mp.workdps(digits):
        cast=lambda z:mp.mpc(float(np.real(z)),float(np.imag(z)))
        mats=[[[[cast(a[l,n,r]) for r in range(a.shape[2])] for n in range(a.shape[1])]
               for l in range(a.shape[0])] for a in tensors]
        amplitudes={}
        for occ in np.ndindex(tuple(dims)):
            vector=[mp.mpc(1)]
            for j,n in enumerate(occ):
                vector=[sum(vector[l]*mats[j][l][n][r] for l in range(len(vector)))
                        for r in range(tensors[j].shape[2])]
            amplitudes[occ]=vector[0]
        norm=sum(abs(a)**2 for a in amplitudes.values())
        if norm<=0: raise ArithmeticError('Zero state norm')
        local=[]
        for z,d in zip(alpha,dims):
            a=cast(z); values=[mp.exp(-abs(a)**2/2)]
            for n in range(1,d): values.append(values[-1]*(-a.conjugate())/mp.sqrt(n))
            local.append(values)
        marginals=[]
        for zeros in range(1<<m):
            groups={}
            for occ,a in amplitudes.items():
                outside=tuple(occ[j] for j in range(m) if not (zeros>>j)&1)
                value=a
                for j,n in enumerate(occ):
                    if (zeros>>j)&1: value*=local[j][n]
                groups[outside]=groups.get(outside,mp.mpc(0))+value
            marginals.append(sum(abs(v)**2 for v in groups.values())/norm)
        law=[]
        for mask in range(1<<m):
            zeros=((1<<m)-1)^mask; value=mp.mpf(0); subset=mask
            while True:
                value+=(-1)**subset.bit_count()*marginals[zeros|subset]
                if subset==0: break
                subset=(subset-1)&mask
            law.append(value)
        if min(law)<-mp.mpf(10)**(-(digits//2)):
            raise ArithmeticError('High-precision reference negative probability')
        return {'law_decimal':[mp.nstr(v,digits) for v in law],
            'state_norm_decimal':mp.nstr(norm,digits),'digits':digits,
            'mass_residual_decimal':mp.nstr(abs(sum(law)-1),digits)}
