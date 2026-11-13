"""Growing-radius real Gaussian displacement grids with rational bounds.

Ideal discretizations converge for N=4**k, R=k+1. Implemented domain is
R<=8, N<=1024; finite-word selector and state errors remain separate.
"""
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from .normal_grid_certificate import pi_interval
from .gaussian_fock_bridge import root_interval
from .rational_gaussian import outward
from .displaced_fock_certificate import exp_minus_interval


@lru_cache(None)
def scaled_exp_minus(x):
    x=F(x)
    if not 0<=x<=64:raise ValueError('Certified exponent range is [0,64]')
    if not x:return F(1),F(1)
    power=0
    while x/(1<<power)>1:power+=1
    a,b=exp_minus_interval(x/(1<<power));a,b=outward(a,b,192)
    for _ in range(power):a,b=outward(a*a,b*b,192)
    return max(F(0),a),min(F(1),b)


@lru_cache(None)
def growing_cdf(x):
    x=F(x)
    if abs(x)>8:raise ValueError('Certified standardized range is [-8,8]')
    if x<0:
        a,b=growing_cdf(-x);return 1-b,1-a
    if not x:return F(1,2),F(1,2)
    term=total=x
    for k in range(1,321):term*=-x*x*F(2*k-1,2*k*(2*k+1));total+=term
    low=total-term*x*x*F(641,642*643);pl,ph=pi_interval();nl=root_interval(1/(2*ph))[0];nh=root_interval(1/(2*pl))[1]
    a,b=outward(F(1,2)+low*nl,F(1,2)+total*nh,192);return max(F(0),a),min(F(1),b)


def scheduled_radius(cells):
    if type(cells) is not int or cells<16 or cells>1024:raise ValueError('Require a power of four in 16..1024')
    k=0;n=cells
    while n%4==0:n//=4;k+=1
    if n!=1:raise ValueError('Require a power of four')
    return k+1


def growing_grid(sigma,cells,*,radius=None):
    sigma=F(sigma);R=F(scheduled_radius(cells) if radius is None else radius)
    if not 0<sigma<=F(1,2) or type(cells) is not int or not 4<=cells<=1024 or cells%2 or not 1<=R<=8:raise ValueError('Require sigma in (0,1/2], even4..1024 cells and radius in [1,8]')
    edges=[-R+2*R*k/cells for k in range(cells+1)];cdf=[growing_cdf(x) for x in edges];weights=[(max(F(0),b[0]-a[1]),b[1]-a[0]) for a,b in zip(cdf,cdf[1:])];tail=(2*cdf[0][0],2*cdf[0][1]);weights.extend([(tail[0]/2,tail[1]/2)]*2);nodes=[sigma*(a+b)/2 for a,b in zip(edges,edges[1:])]+[-R*sigma,R*sigma]
    prob=np.array([float((a+b)/2) for a,b in weights]);prob/=prob.sum();fcdf=prob.cumsum();fcdf/=fcdf[-1];e=scaled_exp_minus(R*R/2);pl,ph=pi_interval();density=(e[0]*root_interval(1/(2*ph))[0],e[1]*root_interval(1/(2*pl))[1]);excess=(max(F(0),2*sigma*density[0]-R*sigma*tail[1]),max(F(0),2*sigma*density[1]-R*sigma*tail[0]))
    assert all(0<=a<=b<=1 for a,b in weights) and sum(a for a,b in weights)<=1<=sum(b for a,b in weights)
    return dict(sigma=sigma,cells=cells,standard_radius=R,standard_edges=edges,cdf_intervals=cdf,nodes=nodes,weight_intervals=weights,tail_interval=tail,probabilities_hex=[float(x).hex() for x in prob],cdf_hex=[float(x).hex() for x in fcdf],pi_interval=pi_interval(),interior_half_width=R*sigma/cells,standard_density_at_radius_interval=density,tail_mean_excess_interval=excess,scope='Boundary-clipped Gaussian grid; default radius k+1 for N=4**k; fixed finite implementation domain')


def growing_displacement_bound(lam,grid):
    lam=F(lam)
    if not 0<=lam<1:raise ValueError('Require real pairing in [0,1)')
    c=root_interval((1-lam)/(1+lam))[1]
    return min(F(1),c*((1-grid['tail_interval'][0])*grid['interior_half_width']+grid['tail_mean_excess_interval'][1]))
