"""Layer checkpoint retries and rolling conditional Schmidt-tail allowances.

All construction budgets are ideal-arithmetic diagnostics evaluated in float64.
No finite-precision sampler certificate, optimal-rank or monotonicity claim.
"""
import math
from time import perf_counter
import numpy as np
from .local_mps import LocalMPS, _integer, squeezed_product
from .adaptive_mps import schmidt_rank, adaptive_round_bonds, ConstructionBudgetUnmet


def rolling_round_bonds(mps,joint_tail):
    """Redistribute unused log-survival allowance to remaining cuts."""
    if not math.isfinite(joint_tail) or not 0<=joint_tail<1: raise ValueError('Invalid joint tail')
    target_log=math.log1p(-joint_tail); spent_log=0.
    mps.move_center(0); losses=[]; ranks=[]; allowances=[]; misses=0
    for i in range(len(mps.tensors)-1):
        remaining=len(mps.tensors)-1-i
        allowance=-math.expm1(min(0.,target_log-spent_log)/remaining)
        a=mps.tensors[i]; u,s,vh=np.linalg.svd(a.reshape(-1,a.shape[2]),full_matrices=False)
        rank,_,miss=schmidt_rank(s,allowance,mps.max_bond)
        keep=float(np.sum(s[:rank]**2)); drop=float(np.sum(s[rank:]**2))
        if abs(keep+drop-1)>1e-9 or keep<=0: raise ArithmeticError('Rolling sweep norm check failed')
        loss=drop/(keep+drop); spent_log+=math.log1p(-loss)
        losses.append(loss); ranks.append(rank); allowances.append(allowance); misses+=miss
        mps.tensors[i]=u[:,:rank].reshape(a.shape[0],a.shape[1],rank)
        transfer=(s[:rank]/math.sqrt(keep))[:,None]*vh[:rank]
        mps.tensors[i+1]=np.tensordot(transfer,mps.tensors[i+1],axes=(1,0)); mps.center=i+1
    return {'conditional_losses':losses,'joint_tail':-math.expm1(spent_log),'ranks':ranks,
        'per_cut_allowances':allowances,'rank_cap_misses':int(misses),'joint_allowance':joint_tail}


def layer_schedule(modes,gates):
    """Validate phase-delimited, disjoint layers without changing gate order."""
    layers=[]; phase=None; pairs=[]; occupied=set()
    for gate in gates:
        if gate[0]=='phases':
            if phase is not None: layers.append((phase,pairs))
            phase=np.asarray(gate[1],float)
            if phase.shape!=(modes,) or not np.all(np.isfinite(phase)): raise ValueError('Invalid phases')
            pairs=[]; occupied=set()
        elif gate[0]=='pair':
            if phase is None: raise ValueError('Require phase-delimited layers')
            i=_integer(gate[1],'first',0,modes-2)
            if i in occupied or i+1 in occupied: raise ValueError('Require disjoint pairs')
            occupied.update((i,i+1)); pairs.append(gate)
        else: raise ValueError('Unknown gate kind')
    if phase is not None: layers.append((phase,pairs))
    return layers


def checkpoint_attempt(modes,squeezing,dimension,caps,gates,target,*,rounding_fraction=.5,
                       max_pair_elements=8_000_000,max_checkpoint_elements=150_000_000,
                       max_seconds=60.,allocation='rolling'):
    start=perf_counter(); modes=_integer(modes,'modes',2,1024)
    caps=tuple(_integer(c,'cap',1,512) for c in caps)
    if not caps or tuple(sorted(set(caps)))!=caps: raise ValueError('Invalid cap ladder')
    if allocation not in ('rolling','equal'): raise ValueError('Invalid allocation')
    if not 0<target<1 or not 0<rounding_fraction<1 or not math.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError('Invalid budget parameters')
    _integer(max_checkpoint_elements,'max_checkpoint_elements',1,300_000_000)
    schedule=layer_schedule(modes,gates)
    tensors,initial=squeezed_product(modes,squeezing,dimension)
    if initial['input_trace_budget']>target:
        return None,dict(initial,status='input_budget_rejected',dimension=dimension,seconds=perf_counter()-start)
    allowance=(target-initial['input_trace_budget'])*rounding_fraction/max(1,len(schedule))
    mps=LocalMPS(tensors,caps[-1],max_pair_elements=max_pair_elements)
    layers=[]; records=[]; projection_sum=0.; rounding_sum=0.; max_checkpoint=0; max_branch=0
    status='building'; failure=None; completed_gates=0
    def guard():
        if perf_counter()-start>max_seconds: raise TimeoutError('Attempt time cap reached between operations')
    try:
        for number,(phase,pairs) in enumerate(schedule):
            guard(); mps.phases(phase); local=[]
            for gate in pairs:
                guard(); record=mps.pair(gate[1],gate[2],truncate_bond=False)
                local.append(record); records.append(record); completed_gates+=1
                max_branch=max(max_branch,mps.max_live_pair_elements)
            if not pairs: continue
            projection=-math.expm1(sum(math.log1p(-r['local_projection_tail']) for r in local))
            projection_sum+=math.sqrt(projection)
            layer={'layer_index':number,'projection_tail':projection,'projection_trace':math.sqrt(projection),'rounding_attempts':[]}
            layers.append(layer)
            # This is a monotone ledger lower bound, NOT an actual-error lower bound.
            if initial['input_trace_budget']+projection_sum+rounding_sum>target:
                status='projection_budget_exhausted'; break
            mps.move_center(0)
            size=sum(a.size for a in mps.tensors); max_checkpoint=max(max_checkpoint,size)
            if size>max_checkpoint_elements: raise MemoryError(f'Checkpoint copy cap: {size}>{max_checkpoint_elements}')
            base=mps; selected=None
            for cap in caps:
                guard(); attempt_start=perf_counter()
                candidate=LocalMPS(base.tensors,cap,max_pair_elements=max_pair_elements)
                trial=(rolling_round_bonds if allocation=='rolling' else adaptive_round_bonds)(candidate,allowance**2)
                trial.update(cap=cap,seconds=perf_counter()-attempt_start,
                    accepted=trial['joint_tail']<=allowance**2,
                    stored_complex_elements=sum(a.size for a in candidate.tensors))
                layer['rounding_attempts'].append(trial)
                if trial['accepted']: selected=candidate; break
                del candidate
            if selected is None:
                status='rank_budget_unmet'; break
            mps=selected; del base,candidate
            mps.move_center(0)
            layer.update(rounding_tail=trial['joint_tail'],rounding_trace=math.sqrt(trial['joint_tail']),selected_cap=cap)
            rounding_sum+=layer['rounding_trace']
        else: status='complete'
    except (MemoryError,TimeoutError,ArithmeticError) as e:
        status='resource_failed' if isinstance(e,(MemoryError,TimeoutError)) else 'numerical_failed'; failure=repr(e)
    info=dict(initial,dimension=dimension,caps=list(caps),allocation=allocation,layers=layers,gates=records,
        projection_trace_sum=projection_sum,rounding_trace_sum=rounding_sum,
        completed_gates=completed_gates,max_checkpoint_elements=max_checkpoint,
        max_pair_branch_elements=max_branch,seconds=perf_counter()-start,status=status,
        partial_ledger_sum=initial['input_trace_budget']+projection_sum+rounding_sum)
    if failure: info['error']=failure
    if status!='complete': return None,info
    mps.move_center(0); budget=info['partial_ledger_sum']
    info.update(status='accepted' if budget<=target else 'budget_missed',construction_trace_budget=min(1.,budget),
        unclipped_construction_trace_budget=budget,stored_complex_elements=sum(a.size for a in mps.tensors),
        max_bond_actual=max(a.shape[2] for a in mps.tensors),
        selected_caps=[layer['selected_cap'] for layer in layers])
    return mps.tensors,info


def build_checkpoint_core(modes,squeezing,gates,target,*,dimensions=(10,12),caps=(8,16,32,64,128),
                          max_pair_elements=8_000_000,max_checkpoint_elements=150_000_000,
                          max_attempt_seconds=60.,on_attempt=None,allocation='rolling'):
    dimensions=tuple(_integer(d,'dimension',1,24) for d in dimensions)
    if not dimensions or tuple(sorted(set(dimensions)))!=dimensions: raise ValueError('Invalid dimension ladder')
    attempts=[]; start=perf_counter()
    for d in dimensions:
        tensors,info=checkpoint_attempt(modes,squeezing,d,caps,gates,target,max_pair_elements=max_pair_elements,
            max_checkpoint_elements=max_checkpoint_elements,max_seconds=max_attempt_seconds,allocation=allocation)
        attempts.append(info)
        if on_attempt is not None: on_attempt(info)
        if info['status']=='accepted':
            return tensors,dict(info,attempts=attempts,search_seconds=perf_counter()-start,target=target,
                scope='Ideal construction budget evaluated in float64; no full output-TV certificate')
        if info['status'] in ('resource_failed','numerical_failed'): break
        del tensors
    raise ConstructionBudgetUnmet(attempts,perf_counter()-start)
