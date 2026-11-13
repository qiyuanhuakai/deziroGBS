"""Boundary tail mapping with a certified normal tail-distance moment."""
from fractions import Fraction as F
import numpy as np
from .normal_grid_certificate import normal_grid,pi_interval
from .gaussian_fock_bridge import root_interval
from .displaced_fock_certificate import exp_minus_interval


def boundary_grid(sigma,cells):
    grid=normal_grid(sigma,cells);sigma=grid['sigma'];tau=grid['tail_interval'];grid['nodes']=grid['nodes'][:-1]+[-4*sigma,4*sigma];grid['weight_intervals']=grid['weight_intervals'][:-1]+[(tau[0]/2,tau[1]/2)]*2;mid=np.array([float((a+b)/2) for a,b in grid['weight_intervals']]);p=mid/mid.sum();cdf=p.cumsum();cdf/=cdf[-1];grid['probabilities_hex']=[float(x).hex() for x in p];grid['cdf_hex']=[float(x).hex() for x in cdf]
    # phi(4)=exp(-8)/sqrt(2*pi). Squaring the certified exp(-4)
    # stays within the existing exponential primitive's validated domain.
    e=exp_minus_interval(F(4));pi=pi_interval();density=(e[0]**2*root_interval(1/(2*pi[1]))[0],e[1]**2*root_interval(1/(2*pi[0]))[1]);excess=(max(F(0),2*sigma*density[0]-4*sigma*tau[1]),max(F(0),2*sigma*density[1]-4*sigma*tau[0]));grid.update(standard_density_at_radius_interval=density,tail_mean_excess_interval=excess,scope='Normal bins; each outer tail mapped to its signed boundary; no target-dependent choices');return grid


def boundary_displacement_bound(lam,grid):
    lam=F(lam)
    if not 0<=lam<1:raise ValueError('Real single-mode pairing0<=lambda<1 required')
    constant=root_interval((1-lam)/(1+lam))[1];return min(F(1),constant*((1-grid['tail_interval'][0])*grid['interior_half_width']+grid['tail_mean_excess_interval'][1]))
