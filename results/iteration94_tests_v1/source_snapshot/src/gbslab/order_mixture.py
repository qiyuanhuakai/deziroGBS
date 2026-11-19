"""Fixed fair mixture of two opposite mode orders, with explicit relabeling."""
from fractions import Fraction as F

def reverse_active(law,active_modes):
    values=tuple(map(F,law));length=len(values)
    if type(active_modes) is not int or active_modes<1 or not length or length&(length-1) or length<1<<active_modes or min(values)<0 or sum(values)!=1:raise ValueError('Normalized power-of-two law and valid active prefix required')
    result=[F(0)]*length;low=(1<<active_modes)-1
    for mask,p in enumerate(values):
        index=(mask&~low)|sum(((mask>>j)&1)<<(active_modes-1-j) for j in range(active_modes));result[index]=p
    return tuple(result)

def reversal_mixture(forward,reverse,active_modes):
    p=tuple(map(F,forward));q=tuple(map(F,reverse));aligned=reverse_active(q,active_modes);back=reverse_active(p,active_modes)
    if len(p)!=len(q):raise ValueError('Equal output spaces required')
    law=tuple((a+b)/2 for a,b in zip(p,aligned));other=tuple((a+b)/2 for a,b in zip(q,back));assert reverse_active(other,active_modes)==law
    return dict(law=law,aligned_reverse=aligned,reverse_oriented_law=other,weights=(F(1,2),F(1,2)),order_TV=sum(abs(a-b) for a,b in zip(p,aligned))/2,scope='Independent fair-order mixture of given program laws; no target-dependent weight selection')
