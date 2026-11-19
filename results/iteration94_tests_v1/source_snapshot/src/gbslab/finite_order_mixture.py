"""Finite-word order selection and batched dispatch to labelled samplers.

Law claims assume independent uniform selector words and independent component
sampler randomness. They are not claims about a finite set of PRNG seeds.
"""
from fractions import Fraction as F


def _size(count,bits):
    if type(bits) is not int or not 1<=bits<=64:raise ValueError('word bits must be in 1..64')
    if type(count) is not int or not 1<=count<=min(64,1<<bits):raise ValueError('component count must be in 1..min(64,2**bits)')
    return 1<<bits


def select_order(word,count=3,*,bits=64):
    size=_size(count,bits)
    if type(word) is not int or not 0<=word<size:raise ValueError('an unsigned word is required')
    return (count*word)//size


def selector_law(count=3,*,bits=64):
    size=_size(count,bits)
    boundaries=[(j*size+count-1)//count for j in range(count+1)]
    masses=tuple(boundaries[j+1]-boundaries[j] for j in range(count))
    weights=tuple(F(x,size) for x in masses)
    return dict(counts=masses,weights=weights,boundaries=tuple(boundaries),uniform_TV=sum(abs(w-F(1,count)) for w in weights)/2,bits=bits)


def mixture_law(distributions,*,bits=64):
    laws=tuple(tuple(map(F,p)) for p in distributions);weights=selector_law(len(laws),bits=bits)
    if not laws[0] or any(len(p)!=len(laws[0]) or min(p)<0 or sum(p)!=1 for p in laws):raise ValueError('equal-length normalized component laws required')
    law=tuple(sum(w*p[j] for w,p in zip(weights['weights'],laws)) for j in range(len(laws[0])))
    ideal=tuple(sum(p[j] for p in laws)/len(laws) for j in range(len(laws[0])))
    distance=sum(abs(a-b) for a,b in zip(law,ideal))/2
    assert sum(law)==1 and distance<=weights['uniform_TV']
    return dict(distribution=law,ideal_distribution=ideal,selector=weights,ideal_distance=distance)


def sample_order_mixture(words,component_samplers,permutations,*,bits=64):
    """Call each selected sampler once, then restore sample and physical order.

    A sampler receives its requested count and returns that many bit strings in
    its own mode order. permutation[j] is the physical label of position j.
    Selectors are consumed before any callback, and unused samplers are skipped.
    Conditional on selectors, callbacks must provide independent samples of
    their stated component law. Preprocessing and batching costs belong to the
    callbacks; this dispatcher does not claim that they are shared or free.
    """
    samplers=tuple(component_samplers);perms=tuple(tuple(p) for p in permutations);_size(len(samplers),bits)
    if len(perms)!=len(samplers) or not all(callable(s) for s in samplers):raise ValueError('one callable and permutation per component required')
    modes=len(perms[0])
    if not modes or any(len(p)!=modes or any(type(x) is not int for x in p) or sorted(p)!=list(range(modes)) for p in perms):raise ValueError('permutations must label the same nonempty mode set')
    words=tuple(words);choices=tuple(select_order(w,len(samplers),bits=bits) for w in words);positions=[[] for _ in samplers]
    for i,c in enumerate(choices):positions[c].append(i)
    outputs=[None]*len(words);ranks=[None]*len(words)
    for c,where in enumerate(positions):
        if not where:continue
        batch=tuple(samplers[c](len(where)))
        if len(batch)!=len(where) or any(not isinstance(s,str) or len(s)!=modes or set(s)-{'0','1'} for s in batch):raise ValueError('component returned malformed samples')
        for rank,(index,s) in enumerate(zip(where,batch)):
            mapped=['']*modes
            for j,p in enumerate(perms[c]):mapped[p]=s[j]
            outputs[index]=''.join(mapped);ranks[index]=rank
    return dict(outputs=tuple(outputs),choices=choices,component_indices=tuple(ranks),component_counts=tuple(map(len,positions)),selector_words=words,selector=selector_law(len(samplers),bits=bits))
