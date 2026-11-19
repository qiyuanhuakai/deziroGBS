"""Exact laws of exported Frugal thresholds under explicitly stated RNG models.

The uint64 model assumes independent uniform 64-bit words, nearest-even integer
conversion to binary64, exact division by 2**64, and the inspected GCC upper
endpoint clamp. It is not a theorem about a fixed mt19937 seed or other C++ libs.
"""
import math
from fractions import Fraction as F
from functools import lru_cache
WORDS=1<<64
MAX_UNIFORM=F((1<<53)-1,1<<53)


def word_uniform(word):
    """Exact nearest-even integer conversion and GCC's endpoint clamp."""
    if type(word) is not int or not 0<=word<WORDS:raise ValueError('A uint64 word is required')
    shift=max(0,word.bit_length()-53)
    if shift:
        significand,remainder=divmod(word,1<<shift);half=1<<(shift-1)
        if remainder>half or (remainder==half and significand%2):significand+=1
        rounded=significand<<shift
    else:rounded=word
    return min(F(rounded,WORDS),MAX_UNIFORM)


@lru_cache(maxsize=131072)
def zero_probability(threshold,model='uint64'):
    """Pr(not(U > threshold)); NaN follows the upstream false comparison."""
    q=float(threshold)
    if model not in ('uint64','continuous'):raise ValueError('Unknown uniform model')
    if math.isnan(q):return F(1)
    if q<0:return F(0)
    if q>=1:return F(1)
    if model=='continuous':return F(q)
    if F(q)>=MAX_UNIFORM:return F(1)
    target=F(q);lo=0;hi=WORDS
    while lo<hi:
        mid=(lo+hi)//2
        if word_uniform(mid)<=target:lo=mid+1
        else:hi=mid
    return F(lo,WORDS)


def full_law(modes,rows,*,model='uint64'):
    """Rebuild all leaves, including rare zero-threshold and NaN continuations."""
    if type(modes) is not int or not 1<=modes<=16:raise ValueError('Full trees require 1..16 modes')
    table={}
    for row in rows:
        p=row['prefix']
        if not isinstance(p,str) or len(p)>=modes or set(p)-{'0','1'} or p in table:raise ValueError('Invalid/duplicate prefix')
        table[p]=float.fromhex(row['threshold_hex'])
    if len(table)!=(1<<modes)-1:raise ValueError('Incomplete prefix tree')
    masses={'':F(1)};law=[F(0)]*(1<<modes);nonfinite=[];saturated=[]
    for length in range(modes):
        for p in tuple(k for k in masses if len(k)==length):
            mass=masses[p];q=table[p];z=zero_probability(q,model)
            if mass and not math.isfinite(q):nonfinite.append(p)
            if mass and (q<0 or q>1):saturated.append(p)
            masses[p+'0']=mass*z;masses[p+'1']=mass*(1-z)
    for p,mass in masses.items():
        if len(p)==modes:law[sum((c=='1')<<j for j,c in enumerate(p))]=mass
    assert sum(law)==1
    return dict(distribution=tuple(law),reachable_nonfinite_prefixes=tuple(nonfinite),reachable_saturated_prefixes=tuple(saturated),model=model)


def distance_interval(law,target):
    law=tuple(map(F,law));target=tuple((F(a),F(b)) for a,b in target)
    if len(law)!=len(target) or any(v<0 for v in law) or sum(law)!=1 or any(not 0<=a<=b<=1 for a,b in target):raise ValueError('Normalized law and probability intervals required')
    if not sum(a for a,b in target)<=1<=sum(b for a,b in target):raise ValueError('Target intervals cannot contain a normalized law')
    return (sum(max(F(0),a-q,q-b) for q,(a,b) in zip(law,target))/2,sum(max(abs(q-a),abs(q-b)) for q,(a,b) in zip(law,target))/2)


def uniform_model_transfer_bound(modes):
    """History-wise coupling bound; thresholds must be the same full tree."""
    if type(modes) is not int or modes<1:raise ValueError('Positive integer mode count required')
    return min(F(1),modes*(F(1,1<<53)+F(1,1<<64)))
