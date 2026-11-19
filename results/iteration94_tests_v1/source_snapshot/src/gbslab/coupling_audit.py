"""Exact rational audit of a finite sequential law, including failure.

The target table is evaluator-only. Rational arithmetic certifies relationships
for that supplied discrete table, not its accuracy as a Gaussian probability.
"""
from fractions import Fraction
import heapq


def normalized_rationals(values):
    p=tuple(Fraction(v) for v in values)
    if not p or any(x<0 for x in p):raise ValueError('Invalid target masses')
    total=sum(p,Fraction(0))
    if total<=0:raise ValueError('Empty target support')
    return tuple(x/total for x in p),total


def uniform_grid_probability(probability,bits):
    q=Fraction(probability)
    if not 0<=q<=1 or not isinstance(bits,int) or not 1<=bits<=128:raise ValueError('Invalid uniform-grid law')
    size=1<<bits;n=q.numerator*size;return Fraction((n+q.denominator-1)//q.denominator,size)


def audit(target,oracle,*,order=None,maximum_queries=None,uniform_bits=None):
    """Best-first coupling audit; oracle(prefix) returns q(1) or None=failure.

Unqueried matched probability is charged in full. Complete audits also return
the exact extended-law TV and two hybrid bounds. Exceptions from the oracle
propagate: only an explicit None represents the intended failure outcome.
"""
    p=tuple(Fraction(v) for v in target);count=len(p);m=count.bit_length()-1
    if count<2 or count!=(1<<m) or m>14 or any(v<0 for v in p) or sum(p,Fraction(0))!=1:raise ValueError('Target must be an exactly normalized finite binary law')
    order=tuple(range(m)) if order is None else tuple(order)
    if sorted(order)!=list(range(m)):raise ValueError('Invalid order')
    if maximum_queries is not None and (not isinstance(maximum_queries,int) or maximum_queries<0):raise ValueError('Invalid query budget')
    if uniform_bits is not None:uniform_grid_probability(0,uniform_bits)
    target_prefix={():Fraction(1)}
    for length in range(1,m+1):
        for mask,mass in enumerate(p):
            prefix=tuple((mask>>j)&1 for j in order[:length]);target_prefix[prefix]=target_prefix.get(prefix,Fraction(0))+mass
    # Heap prioritizes still-matched mass; prefix is a deterministic tie-breaker.
    queue=[(-Fraction(1),(),Fraction(1))];distribution=[Fraction(0)]*count;records=[];failure_mass=Fraction(0);known_loss=Fraction(0);confirmed_match=Fraction(0);hybrid_p=Fraction(0);hybrid_q=Fraction(0);level_loss=[Fraction(0)]*m
    while queue and (maximum_queries is None or len(records)<maximum_queries):
        negative_c,prefix,q_mass=heapq.heappop(queue);matched=-negative_c;length=len(prefix);p_mass=target_prefix[prefix];raw=oracle(prefix)
        if raw is None:
            known_loss+=matched;level_loss[length]+=matched;failure_mass+=q_mass;hybrid_p+=p_mass;hybrid_q+=q_mass
            records.append(dict(prefix=prefix,status='failure',target_mass=p_mass,proxy_mass=q_mass,matched_mass=matched,local_disagreement=Fraction(1),charged_loss=matched));continue
        raw=Fraction(raw)
        if not 0<=raw<=1:raise ValueError('Oracle returned invalid conditional')
        q=raw if uniform_bits is None else uniform_grid_probability(raw,uniform_bits)
        # Target-zero histories may use p=q; this does not change target law.
        true=target_prefix[prefix+(1,)]/p_mass if p_mass else q
        delta=abs(true-q);loss=matched*delta;known_loss+=loss;level_loss[length]+=loss;hybrid_p+=p_mass*delta;hybrid_q+=q_mass*delta
        records.append(dict(prefix=prefix,status='conditional',target_mass=p_mass,proxy_mass=q_mass,matched_mass=matched,target_conditional=true,raw_proxy_conditional=raw,proxy_conditional=q,local_disagreement=delta,charged_loss=loss))
        for bit,pp,qq in ((0,1-true,1-q),(1,true,q)):
            if qq==0:continue
            child=prefix+(bit,);weight=q_mass*qq;common=matched*min(pp,qq)
            if length+1==m:
                physical=sum(value<<mode for mode,value in zip(order,child));distribution[physical]=weight;confirmed_match+=common
            else:heapq.heappush(queue,(-common,child,weight))
    unresolved=[dict(prefix=prefix,matched_mass=-negative,proxy_mass=q_mass,target_mass=target_prefix[prefix]) for negative,prefix,q_mass in queue];unresolved_match=sum((r['matched_mass'] for r in unresolved),Fraction(0));unresolved_q=sum((r['proxy_mass'] for r in unresolved),Fraction(0));upper=known_loss+unresolved_match
    assert known_loss+confirmed_match+unresolved_match==1
    assert sum(distribution,Fraction(0))+failure_mass+unresolved_q==1
    complete=not queue;tv=None
    if complete:
        tv=(sum((abs(a-b) for a,b in zip(p,distribution)),Fraction(0))+failure_mass)/2
        assert failure_mass<=tv<=upper<=min(Fraction(1),hybrid_p,hybrid_q)
    return dict(complete=complete,queries=len(records),target=tuple(p),distribution=tuple(distribution),failure_mass_seen=failure_mass,unresolved_proxy_mass=unresolved_q,matched_failure_or_disagreement=known_loss,unresolved_matched_mass=unresolved_match,confirmed_matched_mass=confirmed_match,coupling_upper=upper,extended_tv=tv,hybrid_target_upper=min(Fraction(1),hybrid_p) if complete else None,hybrid_proxy_upper=min(Fraction(1),hybrid_q) if complete else None,level_losses=tuple(level_loss),records=records,unresolved=unresolved,uniform_bits=uniform_bits)
