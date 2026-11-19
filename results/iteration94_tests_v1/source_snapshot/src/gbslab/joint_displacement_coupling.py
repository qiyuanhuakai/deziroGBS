"""Certified quantization moments and independent-drive Gaussian coupling."""
from fractions import Fraction as F
from functools import lru_cache
from .growing_normal_grid import growing_grid,growing_cdf,scaled_exp_minus
from .normal_grid_certificate import pi_interval
from .gaussian_fock_bridge import root_interval


def add(*pairs):return sum(x for x,y in pairs),sum(y for x,y in pairs)


def scale(c,pair):
    c=F(c);a,b=pair
    return (c*a,c*b) if c>=0 else (c*b,c*a)


@lru_cache(None)
def density(x):
    x=F(x);a,b=scaled_exp_minus(x*x/2);pl,ph=pi_interval()
    return a*root_interval(1/(2*ph))[0],b*root_interval(1/(2*pl))[1]


@lru_cache(None)
def quantization_moments(sigma,cells,radius):
    sigma=F(sigma);grid=growing_grid(sigma,cells,radius=radius);edges=grid['standard_edges'];R=grid['standard_radius'];rows=[]
    for a,b in zip(edges,edges[1:]):
        c=(a+b)/2;ca,cb,cc=growing_cdf(a),growing_cdf(b),growing_cdf(c);da,db,dc=density(a),density(b),density(c)
        mass=add(cb,scale(-1,ca))
        absolute=add(scale(c,add(scale(2,cc),scale(-1,ca),scale(-1,cb))),scale(2,dc),scale(-1,da),scale(-1,db))
        squared=add(scale(1+c*c,mass),scale(-b,da),scale(a,db))
        rows.append(dict(a=a,b=b,center=c,mass_interval=mass,absolute_interval=scale(sigma,absolute),squared_interval=scale(sigma*sigma,squared)))
    cr=growing_cdf(R);tail=(1-cr[1],1-cr[0]);dr=density(R)
    tail1=scale(2*sigma,add(dr,scale(-R,tail)));tail2=scale(2*sigma*sigma,add(scale(1+R*R,tail),scale(-R,dr)))
    m1=add(tail1,*(r['absolute_interval'] for r in rows));m2=add(tail2,*(r['squared_interval'] for r in rows))
    assert 0<=m1[0]<=m1[1] and 0<=m2[0]<=m2[1]
    return dict(sigma=sigma,cells=cells,radius=R,grid=grid,rows=rows,tail_absolute_interval=tail1,tail_squared_interval=tail2,
                mean_residual=F(0),absolute_interval=m1,squared_interval=m2,
                assumptions='Independent centered normals; each boundary-clipped midpoint quantizer is odd almost everywhere.')


def checked_gram(matrix):
    a=[[F(x) for x in row] for row in matrix];n=len(a)
    if not 1<=n<=4 or any(len(row)!=n for row in a) or any(a[i][j]!=a[j][i] for i in range(n) for j in range(n)):
        raise ValueError('Symmetric Gram matrix of rank dimension1..4 required')
    work=[row[:] for row in a];pivots=[]
    for k in range(n):
        pivot=work[k][k];pivots.append(pivot)
        if pivot<0 or (pivot==0 and any(work[k][j]!=0 for j in range(k+1,n))):raise ValueError('Positive semidefinite Gram matrix required')
        if pivot:
            for i in range(k+1,n):
                for j in range(k+1,n):work[i][j]-=work[i][k]*work[k][j]/pivot
    return a,pivots


def joint_coupling(matrix,axes):
    gram,pivots=checked_gram(matrix)
    if len(axes)!=len(gram):raise ValueError('One independent symmetric quantizer per drive required')
    moments=[quantization_moments(F(a['sigma']),a['cells'],F(a['radius'])) for a in axes]
    mean=add(*(scale(gram[i][i],m['squared_interval']) for i,m in enumerate(moments)))
    l1=sum(root_interval(gram[i][i])[1]*m['absolute_interval'][1] for i,m in enumerate(moments))
    rms=root_interval(mean[1])[1]
    if mean[1]<=64:
        exponent=scaled_exp_minus(mean[1]);curved=root_interval(max(F(0),1-exponent[0]))[1]
    else:exponent=None;curved=F(1)
    old=sum(root_interval(gram[i][i])[1]*((1-m['grid']['tail_interval'][0])*m['grid']['interior_half_width']+m['grid']['tail_mean_excess_interval'][1]) for i,m in enumerate(moments))
    bounds=dict(old_axis_upper=min(F(1),old),exact_l1_upper=min(F(1),l1),rms_upper=min(F(1),rms),joint_curved_upper=min(F(1),curved))
    return dict(gram=gram,psd_pivots=pivots,axes=[dict(sigma=m['sigma'],cells=m['cells'],radius=m['radius']) for m in moments],
                mean_quadratic_interval=mean,exponent_at_mean_upper_interval=exponent,**bounds,
                best_upper=min(bounds.values()),scope='Ideal quantized versus continuous displacement channels on a pure Gaussian input; independent symmetric residuals required.')
