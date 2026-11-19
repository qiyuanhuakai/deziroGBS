"""Exact finite-box MPS for banded pure Gaussian pairing matrices.

Frontier labels count photons already assigned to future modes. No global
Fock tensor; exponential in bandwidth remains. Floating error diagnostics.
"""
import itertools
import math
from time import perf_counter
import numpy as np
from .local_mps import LocalMPS


def pairing_log_normalization(B):
    b=np.asarray(B,complex)
    if b.ndim!=2 or b.shape[0]!=b.shape[1] or len(b)<1 or not np.all(np.isfinite(b)) or not np.allclose(b,b.T,atol=1e-12,rtol=0): raise ValueError('Require finite symmetric pairing matrix')
    values=np.linalg.eigvalsh(np.eye(len(b))-b@b.conj().T)
    if values[0]<=0: raise ValueError('Pairing matrix is not a normalizable pure Gaussian state')
    return .25*math.fsum(np.log(values))


def pairing_covariance(B):
    b=np.asarray(B,complex); pairing_log_normalization(b); m=len(b)
    inverse=np.linalg.solve(np.eye(m)-b@b.conj().T,np.eye(m)); N=inverse-np.eye(m); M=inverse@b
    V=np.block([[np.eye(m)+2*(N.real+M.real),2*(M.imag-N.imag)],
                [2*(M.imag+N.imag),np.eye(m)+2*(N.real-M.real)]])
    return (V+V.T)/2


def pairing_distance(B,C):
    """Pure-state trace distance from Bargmann Gaussian overlap determinants."""
    b=np.asarray(B,complex); c=np.asarray(C,complex)
    if b.shape!=c.shape: raise ValueError('Pairing shapes differ')
    lb=pairing_log_normalization(b); lc=pairing_log_normalization(c)
    if np.array_equal(b,c): return 0.
    _,cross=np.linalg.slogdet(np.eye(len(b))-b.conj().T@c)
    log_fidelity=2*lb+2*lc-float(cross)
    if log_fidelity>1e-10: raise ArithmeticError('Gaussian fidelity exceeds one')
    return math.sqrt(max(0.,-math.expm1(min(0.,log_fidelity))))


def truncate_band(B,width):
    b=np.asarray(B,complex)
    if isinstance(width,bool) or not isinstance(width,(int,np.integer)) or not 0<=width<len(b): raise ValueError('Invalid bandwidth')
    return np.where(np.abs(np.arange(len(b))[:,None]-np.arange(len(b)))<=width,b,0.)


def frontier_element_bound(modes,dimension,width):
    """Conservative dense allocation bound, before reachable-state pruning."""
    return sum((1 if i==0 else dimension**min(width,modes-i))*dimension*(1 if i==modes-1 else dimension**min(width,modes-i-1)) for i in range(modes))


def _allocations(length,budget):
    if length==0:
        yield (); return
    for first in range(budget+1):
        for rest in _allocations(length-1,budget-first): yield (first,)+rest


def build_banded_box(B,dimension,width,*,max_elements=40_000_000,max_seconds=120.):
    """Construct normalized exact box projection; no bond truncation inside."""
    start=perf_counter(); b=np.asarray(B,complex); lognorm=pairing_log_normalization(b); m=len(b); d=dimension
    if isinstance(d,bool) or not isinstance(d,int) or not 1<=d<=16: raise ValueError('Dimension must be in [1,16]')
    if isinstance(width,bool) or not isinstance(width,int) or not 0<=width<m: raise ValueError('Invalid width')
    if np.any(b!=truncate_band(b,width)): raise ValueError('Nonzero pairing entries outside declared band')
    bound=frontier_element_bound(m,d,width)
    if bound>max_elements: raise MemoryError(f'Conservative frontier allocation cap: {bound}>{max_elements}')
    states=[(0,)*min(width,m)]; tensors=[]; transitions=[]; factorial=[math.factorial(k) for k in range(d)]
    for i in range(m):
        if perf_counter()-start>max_seconds: raise TimeoutError('Frontier construction time cap between sites')
        future=min(width,m-i-1); outgoing={}; entries=[]
        patterns={budget:list(_allocations(future,budget)) for budget in range(d)}
        edge_factors={k:math.prod(b[i,i+j+1]**x/factorial[x] for j,x in enumerate(k)) for k in patterns[d-1]}
        for left,pending in enumerate(states):
            existing=pending[0] if pending else 0
            for k in patterns[d-1-existing]:
                dest=tuple((pending[j+1] if j+1<len(pending) else 0)+k[j] for j in range(future))
                if any(x>=d for x in dest): continue
                edge=edge_factors[k]
                if edge==0: continue
                right=outgoing.setdefault(dest,len(outgoing)); used=existing+sum(k)
                for local_pairs in range((d-1-used)//2+1):
                    n=used+2*local_pairs
                    value=math.sqrt(factorial[n])*edge*(b[i,i]/2)**local_pairs/factorial[local_pairs]
                    if value!=0: entries.append((left,n,right,value))
        tensor=np.zeros((len(states),d,len(outgoing)),complex)
        for left,n,right,value in entries: tensor[left,n,right]+=value
        tensors.append(tensor); transitions.append(len(entries)); states=list(outgoing)
    if states!=[()]: raise ArithmeticError('Frontier did not close')
    tensors[0]*=math.exp(lognorm)
    exact_elements=sum(a.size for a in tensors); frontier_bonds=[a.shape[2] for a in tensors[:-1]]
    # Right QR preserves the exact box vector; its norm is then at site zero.
    for i in range(m-1,0,-1):
        if perf_counter()-start>max_seconds: raise TimeoutError('Canonicalization time cap between sites')
        a=tensors[i]; q,r=np.linalg.qr(a.reshape(a.shape[0],-1).T,mode='reduced')
        tensors[i]=q.T.reshape(q.shape[1],d,a.shape[2]); tensors[i-1]=np.tensordot(tensors[i-1],r.T,axes=(2,0))
    mass=float(np.linalg.norm(tensors[0])**2)
    if not 0<mass<=1+1e-9: raise ArithmeticError(f'Invalid finite-box mass {mass}')
    tensors[0]/=math.sqrt(mass)
    return tensors,dict(box_mass=mass,frontier_bonds=frontier_bonds,transition_counts=transitions,
        conservative_element_bound=bound,frontier_elements=exact_elements,seconds=perf_counter()-start)


def build_banded_core(original,dimension,width,rank,*,max_elements=40_000_000,max_seconds=120.):
    start=perf_counter(); band=truncate_band(original,width); band_distance=pairing_distance(original,band)
    tensors,info=build_banded_box(band,dimension,width,max_elements=max_elements,max_seconds=max_seconds)
    mps=LocalMPS(tensors,rank); losses,joint=mps.round_bonds(); mps.move_center(0)
    band_state_distance=math.sqrt(max(0.,1-info['box_mass']*(1-joint)))
    total=band_distance+band_state_distance
    info.update(band_distance=band_distance,rounding_losses=losses,rounding_joint_tail=joint,
        distance_to_band_gaussian=band_state_distance,construction_trace_budget=min(1.,total),unclipped_construction_trace_budget=total,
        stored_complex_elements=sum(a.size for a in mps.tensors),max_bond_actual=max(a.shape[2] for a in mps.tensors),
        total_seconds=perf_counter()-start,scope='Band Gaussian approximation plus exact-box/nested compression distance in ideal arithmetic; float64, no complete sampler TV certificate')
    return mps.tensors,info
