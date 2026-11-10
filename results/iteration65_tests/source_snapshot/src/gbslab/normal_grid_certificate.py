"""Rational certificates for a bounded normal grid and one-mode noisy sampler.

Interior bins cover [-4 sigma,4 sigma]; both tails map to zero displacement.
This defines a discrete emulator of continuous Gaussian displacement noise.
"""
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from .gaussian_fock_bridge import root_interval
from .rational_gaussian import outward
from .local_mps import sample_local_threshold


def arctan_interval(x,terms=160):
    x=F(x)
    if not 0<x<1 or type(terms) is not int or terms<32 or terms%2:raise ValueError('0<x<1 and even terms>=32 required')
    total=sum((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(terms))
    return total,total+x**(2*terms+1)/F(2*terms+1)


@lru_cache(None)
def pi_interval():
    a,b=arctan_interval(F(1,5));c,d=arctan_interval(F(1,239));return 16*a-4*d,16*b-4*c


@lru_cache(None)
def normal_cdf(x):
    x=F(x)
    if abs(x)>4:raise ValueError('Certified standardized range is[-4,4]')
    if x<0:
        a,b=normal_cdf(-x);return 1-b,1-a
    if x==0:return F(1,2),F(1,2)
    # Integral of exp(-t^2/2) from0 tox: even last index gives upper bound.
    term=x;total=x
    for k in range(1,161):term*=-x*x*F(2*k-1,2*k*(2*k+1));total+=term
    lower=total+term*(-x*x)*F(321,322*323);plo,phi=pi_interval();norm=(root_interval(1/(2*phi))[0],root_interval(1/(2*plo))[1]);return outward(F(1,2)+lower*norm[0],F(1,2)+total*norm[1],192)


def normal_grid(sigma,cells):
    sigma=F(sigma)
    if not 0<sigma<=F(1,2) or type(cells) is not int or cells<4 or cells>512 or cells%2:raise ValueError('0<sigma<=1/2 and even4..512 cells required')
    edges=[F(-4)+F(8*k,cells) for k in range(cells+1)];cdf=[normal_cdf(x) for x in edges];weights=[(max(F(0),b[0]-a[1]),b[1]-a[0]) for a,b in zip(cdf,cdf[1:])];tail=(2*cdf[0][0],2*cdf[0][1]);weights.append(tail);nodes=[sigma*(a+b)/2 for a,b in zip(edges,edges[1:])]+[F(0)];midpoints=np.array([float((a+b)/2) for a,b in weights]);probabilities=midpoints/midpoints.sum();floating_cdf=probabilities.cumsum();floating_cdf/=floating_cdf[-1]
    assert all(a<=b for a,b in weights) and sum(a for a,b in weights)<=1<=sum(b for a,b in weights)
    return dict(sigma=sigma,cells=cells,standard_edges=edges,cdf_intervals=cdf,nodes=nodes,weight_intervals=weights,tail_interval=tail,probabilities_hex=[float(x).hex() for x in probabilities],cdf_hex=[float(x).hex() for x in floating_cdf],pi_interval=pi_interval(),interior_half_width=4*sigma/cells,scope='Exact normal bins; both outer tails mapped to zero; stored float probabilities define selector')


def displacement_grid_bound(lam,grid):
    lam=F(lam)
    if not 0<=lam<1:raise ValueError('Real single-mode pairing0<=lambda<1 required')
    vp=(1-lam)/(1+lam);c=root_interval(vp)[1];a,b=grid['tail_interval'];return min(F(1),b+(1-a)*c*grid['interior_half_width'])


def sample_grid_threshold(tensors,grid,samples,rng):
    if len(tensors)!=1 or type(samples) is not int or samples<0:raise ValueError('One-mode input and nonnegative sample count required')
    p=np.array([float.fromhex(x) for x in grid['probabilities_hex']]);nodes=np.array([float(x) for x in grid['nodes']]);outputs=[];selected=[]
    for _ in range(samples):
        index=int(rng.choice(len(p),p=p));value,_=sample_local_threshold(tensors,np.array([[nodes[index]]],complex),rng);outputs.append(int(value[0,0]));selected.append(index)
    return dict(outputs=outputs,selected_nodes=selected)
