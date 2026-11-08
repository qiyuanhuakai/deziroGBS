"""A posteriori budget-driven Fock dimension and per-cut Schmidt rank selection.

The accepted bound is an ideal-arithmetic construction diagnostic evaluated in
float64, not a finite-precision output-TV certificate. Failed attempts are kept.
"""
import math
from time import perf_counter
import numpy as np
from .local_mps import LocalMPS, _integer, squeezed_product


class ConstructionBudgetUnmet(ArithmeticError):
    def __init__(self, attempts, elapsed):
        super().__init__('No accepted construction within the declared search/resources')
        self.attempts=attempts; self.elapsed=elapsed


def schmidt_rank(singular_values, tail, cap):
    """Smallest allowed rank whose directly summed suffix weight meets tail."""
    s=np.asarray(singular_values,float); tail=float(tail)
    cap=_integer(cap,'cap',1,512)
    if s.ndim!=1 or not len(s) or not np.all(np.isfinite(s)) or np.any(s<0) or np.any(np.diff(s)>0):
        raise ValueError('Require finite descending nonnegative singular values')
    if not math.isfinite(tail) or not 0<=tail<1 or s[0]<=0: raise ValueError('Invalid tail or zero spectrum')
    weights=(s/s[0])**2; suffix=np.cumsum(weights[::-1])[::-1]
    tails=np.r_[suffix[1:],0.]/suffix[0]
    limit=min(cap,len(s)); acceptable=np.flatnonzero(tails[:limit]<=tail)
    rank=int(acceptable[0]+1) if len(acceptable) else limit
    return rank,float(tails[rank-1]),bool(tails[rank-1]>tail)


def adaptive_round_bonds(mps, joint_tail):
    """Nested sweep with an equal conditional tail allowance at each cut."""
    if not math.isfinite(joint_tail) or not 0<=joint_tail<1: raise ValueError('Invalid joint tail')
    per_cut=-math.expm1(math.log1p(-joint_tail)/(len(mps.tensors)-1))
    mps.move_center(0); losses=[]; ranks=[]; misses=[]
    for i in range(len(mps.tensors)-1):
        a=mps.tensors[i]
        u,s,vh=np.linalg.svd(a.reshape(-1,a.shape[2]),full_matrices=False)
        rank,_,miss=schmidt_rank(s,per_cut,mps.max_bond)
        keep=float(np.sum(s[:rank]**2)); drop=float(np.sum(s[rank:]**2))
        if abs(keep+drop-1)>1e-9 or keep<=0: raise ArithmeticError('Adaptive rounding norm check failed')
        losses.append(drop/(keep+drop)); ranks.append(rank); misses.append(miss)
        mps.tensors[i]=u[:,:rank].reshape(a.shape[0],a.shape[1],rank)
        transfer=(s[:rank]/math.sqrt(keep))[:,None]*vh[:rank]
        mps.tensors[i+1]=np.tensordot(transfer,mps.tensors[i+1],axes=(1,0)); mps.center=i+1
    return {'conditional_losses':losses,'joint_tail':-math.expm1(sum(math.log1p(-p) for p in losses)),
        'ranks':ranks,'rank_cap_misses':sum(misses),'per_cut_allowance':per_cut,'joint_allowance':joint_tail}


def budgeted_attempt(modes,squeezing,dimension,cap,gates,target,*,rounding_fraction=.5,
                     max_pair_elements=1_000_000,max_seconds=30.):
    """One dimension/cap attempt; check measured losses, not predicted success."""
    start=perf_counter()
    if not 0<target<1 or not 0<rounding_fraction<1 or not math.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError('Invalid target, split or time cap')
    tensors,initial=squeezed_product(modes,squeezing,dimension)
    if initial['input_trace_budget']>target:
        return None,dict(initial,status='input_budget_rejected',dimension=dimension,cap=cap,seconds=perf_counter()-start)
    layer_count=sum(g[0]=='phases' for g in gates)
    if any(g[0]=='pair' for g in gates) and (not gates or gates[0][0]!='phases'):
        raise ValueError('Require phase-delimited layers')
    allowance=(target-initial['input_trace_budget'])*rounding_fraction/max(1,layer_count)
    mps=LocalMPS(tensors,cap,max_pair_elements=max_pair_elements)
    layers=[]; pending=[]; occupied=set(); error=None
    def finish():
        nonlocal pending,occupied
        if not pending: return
        projection=-math.expm1(sum(math.log1p(-p['local_projection_tail']) for p in pending))
        rounding=adaptive_round_bonds(mps,allowance**2)
        layers.append(dict(rounding,projection_tail=projection,
            projection_trace=math.sqrt(projection),rounding_trace=math.sqrt(rounding['joint_tail'])))
        pending=[]; occupied=set()
    try:
        for g in gates:
            if perf_counter()-start>max_seconds: raise TimeoutError('Attempt time cap reached between gates')
            if g[0]=='phases': finish(); mps.phases(g[1])
            elif g[0]=='pair':
                i=g[1]
                if i in occupied or i+1 in occupied: raise ValueError('Require disjoint gates within each layer')
                occupied.update((i,i+1)); pending.append(mps.pair(i,g[2],truncate_bond=False))
            else: raise ValueError('Unknown gate kind')
        finish(); mps.move_center(0)
    except (MemoryError,TimeoutError,ArithmeticError) as e: error=e
    projection=sum(x['projection_trace'] for x in layers); rounding=sum(x['rounding_trace'] for x in layers)
    raw=initial['input_trace_budget']+projection+rounding
    info=dict(initial,dimension=dimension,cap=cap,layers=layers,gates=mps.ledger,
        projection_trace_sum=projection,rounding_trace_sum=rounding,
        completed_layers=len(layers),pending_gate_count=len(pending),
        rank_cap_misses=sum(x['rank_cap_misses'] for x in layers),
        stored_complex_elements=sum(a.size for a in mps.tensors),max_bond_actual=max(a.shape[2] for a in mps.tensors),
        max_pair_branch_elements=mps.max_live_pair_elements,seconds=perf_counter()-start)
    if error:
        info.update(status='resource_failed' if isinstance(error,(MemoryError,TimeoutError)) else 'numerical_failed',error=repr(error))
        return None,info
    info.update(status='accepted' if raw<=target else 'budget_missed',
        construction_trace_budget=min(1.,raw),unclipped_construction_trace_budget=raw)
    return mps.tensors,info


def build_adaptive_core(modes,squeezing,gates,target,*,dimensions=(4,6,8,10,12),caps=(8,16,32),
                        max_pair_elements=1_000_000,max_attempt_seconds=30.,on_attempt=None):
    """Increase d for input/projection loss; increase cap when rounding dominates.

    First observed accepted state is returned. Search costs include discarded
    attempts. A resource/numerical failure stops the search and is not evidence
    that every larger representation is impossible.
    """
    dims=tuple(_integer(d,'dimension',1,24) for d in dimensions)
    bonds=tuple(_integer(c,'cap',1,512) for c in caps)
    if not dims or not bonds or tuple(sorted(set(dims)))!=dims or tuple(sorted(set(bonds)))!=bonds:
        raise ValueError('Require nonempty increasing dimension and cap ladders')
    start=perf_counter(); attempts=[]; ci=0
    for d in dims:
        while True:
            tensors,info=budgeted_attempt(modes,squeezing,d,bonds[ci],gates,target,
                max_pair_elements=max_pair_elements,max_seconds=max_attempt_seconds)
            attempts.append(info)
            if on_attempt is not None: on_attempt(info)
            if info['status']=='accepted':
                return tensors,dict(info,attempts=attempts,search_seconds=perf_counter()-start,
                    target=target,scope='A posteriori ideal construction bound evaluated in float64; excludes numerical and RNG output error')
            if info['status'] in ('resource_failed','numerical_failed'):
                raise ConstructionBudgetUnmet(attempts,perf_counter()-start)
            if (info['status']=='budget_missed' and info['input_trace_budget']+info['projection_trace_sum']<target
                    and ci+1<len(bonds)):
                ci+=1; del tensors; continue
            del tensors; break
    raise ConstructionBudgetUnmet(attempts,perf_counter()-start)
