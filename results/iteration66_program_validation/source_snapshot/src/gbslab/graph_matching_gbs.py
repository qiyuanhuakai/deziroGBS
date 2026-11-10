"""Exact small-graph audits, not a scalable Markov-chain sampler.

Graph law: c**(2*|S|) PM(G[S])**2. Lift horizontal edges have weight c**2
and vertical edges weight 1. Rational arithmetic includes finite-word CDFs.
"""
from bisect import bisect_right
from collections import Counter
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt


def graph(n, edges):
    if type(n) is not int or not 1 <= n <= 8:
        raise ValueError('small-graph audit requires 1..8 vertices')
    result = []
    for e in edges:
        if len(e) != 2 or any(type(x) is not int or not 0 <= x < n for x in e) or e[0] == e[1]:
            raise ValueError('simple undirected edges required')
        result.append(tuple(sorted(e)))
    if len(set(result)) != len(result):
        raise ValueError('duplicate edges')
    return n, tuple(sorted(result))


def subset_counts(n, edges):
    n, edges = graph(n, edges)
    neighbors = [0]*n
    for a,b in edges:
        neighbors[a] |= 1 << b
        neighbors[b] |= 1 << a
    @lru_cache(None)
    def count(mask):
        if not mask: return 1
        if mask.bit_count() % 2: return 0
        first = mask & -mask
        i = first.bit_length()-1
        rest = mask ^ first
        return sum(count(rest ^ (1 << j)) for j in range(n) if rest & neighbors[i] & (1 << j))
    return tuple(count(s) for s in range(1 << n))


def lifted_structure(n, edges):
    n, edges = graph(n, edges)
    lifted = tuple(sorted((*edges, *((a+n,b+n) for a,b in edges), *((i,i+n) for i in range(n)))))
    neighbors = [[] for _ in range(2*n)]
    for a,b in lifted:
        h = int((a < n) == (b < n))
        neighbors[a].append((b,h)); neighbors[b].append((a,h))
    @lru_cache(None)
    def polynomial(mask):
        if not mask: return {(0,0):1}
        first = mask & -mask; i = first.bit_length()-1; rest = mask ^ first
        out = Counter(polynomial(rest))
        for j,h in neighbors[i]:
            if rest & (1 << j):
                for (k,t),v in polynomial(rest ^ (1 << j)).items(): out[k+1,t+h] += v
        return dict(out)
    witnesses = []
    def visit(mask, selected, subset):
        if not mask:
            witnesses.append(dict(edges=tuple(selected), subset=subset)); return
        first = mask & -mask; i = first.bit_length()-1; rest = mask ^ first
        for j,h in neighbors[i]:
            if rest & (1 << j):
                visit(rest ^ (1 << j), selected+[(i,j)], subset | (((1 << i)|(1 << j)) if h and j < n else 0))
    visit((1 << (2*n))-1, [], 0)
    coefficients = tuple((k,h,v) for (k,h),v in sorted(polynomial((1 << (2*n))-1).items()))
    return dict(lifted_edges=lifted, coefficients=coefficients, perfect_matchings=witnesses,
                dynamic_program_states=polynomial.cache_info().currsize)


def partition_by_size(n, coefficients, c):
    c = F(c)
    if c <= 0: raise ValueError('positive c required')
    z = [F(0)]*(n+1)
    for k,h,v in coefficients: z[k] += v*c**(2*h)
    return tuple(z)


def perfect_mass(z, alpha):
    alpha = F(alpha)
    if alpha <= 0: raise ValueError('positive edge multiplier required')
    return alpha**(len(z)-1)*z[-1]/sum(alpha**k*v for k,v in enumerate(z))


def least_half_multiplier(z):
    hi = 1
    while perfect_mass(z,hi) < F(1,2): hi *= 2
    lo = 1
    while lo < hi:
        mid = (lo+hi)//2
        if perfect_mass(z,mid) >= F(1,2): hi = mid
        else: lo = mid+1
    return lo


def categorical(probabilities, *, bits=64):
    p = tuple(map(F,probabilities))
    if type(bits) is not int or not 1 <= bits <= 64: raise ValueError('bits must be 1..64')
    if not p or min(p) < 0 or sum(p) != 1: raise ValueError('normalized nonnegative law required')
    size = 1 << bits; cumulative = []; boundaries = [0]; total = F(0)
    for value in p:
        total += value; cumulative.append(total)
        boundaries.append(-((-total.numerator*size)//total.denominator))
    return dict(cumulative=tuple(cumulative), boundaries=tuple(boundaries),
                counts=tuple(b-a for a,b in zip(boundaries,boundaries[1:])), bits=bits)


def draw_word(table, word):
    if type(word) is not int or not 0 <= word < (1 << table['bits']): raise ValueError('unsigned word required')
    return bisect_right(table['cumulative'], F(word,1 << table['bits']))


def determinant(matrix):
    a = [list(map(F,row)) for row in matrix]; value = F(1); n = len(a)
    for i in range(n):
        pivot = next((j for j in range(i,n) if a[j][i]),None)
        if pivot is None: return F(0)
        if pivot != i: a[i],a[pivot] = a[pivot],a[i]; value = -value
        x = a[i][i]; value *= x
        for j in range(i+1,n):
            q = a[j][i]/x
            for k in range(i+1,n): a[j][k] -= q*a[i][k]
    return value


def gaussian_bridge(n, edges, c, graph_partition, *, bits=192):
    n, edges = graph(n,edges); c = F(c)
    if c <= 0: raise ValueError('positive c required')
    adjacency = [[0]*n for _ in range(n)]
    for i,j in edges: adjacency[i][j] = adjacency[j][i] = 1
    degree = max(map(sum,adjacency)); norm_upper = c*c*degree
    if norm_upper >= 1:
        # For a nonempty unweighted graph ||A|| >= 1. Other parameters can be
        # inconclusive under the degree bound and are explicitly labelled so.
        return dict(status='invalid' if edges and c*c >= 1 else 'uncertified', pairing_norm_upper=norm_upper)
    matrix = [[F(i==j)-c**4*sum(adjacency[i][k]*adjacency[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    d = determinant(matrix); scale = 1 << bits; root = isqrt((d.numerator*scale*scale)//d.denominator)
    lo = F(root,scale); hi = lo if lo*lo == d else F(root+1,scale)
    return dict(status='valid', pairing_norm_upper=norm_upper, determinant=d,
                root_interval=(lo,hi), collision_free_probability=(graph_partition*lo,graph_partition*hi), bits=bits)
