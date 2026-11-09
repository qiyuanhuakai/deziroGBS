"""Selected complete-output overlap bounds; unknown output mass stays unknown.

Target intervals are supplied by the caller. No complete target table is built.
Only an explicit None from the conditional oracle denotes failure.
"""
from fractions import Fraction as F
from .coupling_audit import uniform_grid_probability


def selected_atoms(modes,patterns,oracle,target_interval,*,uniform_bits=None):
    if not isinstance(modes,int) or not 1<=modes<=256:raise ValueError('Invalid mode count')
    if uniform_bits is not None:uniform_grid_probability(0,uniform_bits)
    patterns=tuple(tuple(p) for p in patterns)
    if not patterns or len(set(patterns))!=len(patterns):raise ValueError('Empty or duplicate patterns')
    for p in patterns:
        if any(not isinstance(j,int) for j in p) or tuple(sorted(set(p)))!=p or any(j<0 or j>=modes for j in p):raise ValueError('Invalid click pattern')
    trie={};intervals=[]
    for i,pattern in enumerate(patterns):
        lo,hi=map(F,target_interval(pattern))
        if not 0<=lo<=hi<=1:raise ValueError('Invalid target probability interval')
        intervals.append((lo,hi));node=trie;clicks=set(pattern)
        for j in range(modes):node=node.setdefault(int(j in clicks),{})
        node['atom']=i
    if sum(lo for lo,hi in intervals)>1:raise ValueError('Inconsistent target lower masses')
    masses=[F(0)]*len(patterns);records=[];failures=[];unselected=F(0)
    def visit(prefix,weight,node):
        nonlocal unselected
        if len(prefix)==modes:masses[node['atom']]=weight;return
        raw=oracle(prefix)
        if raw is None:failures.append(dict(prefix=prefix,mass=weight));records.append(dict(prefix=prefix,reach=weight,q=None));return
        raw=F(raw)
        if not 0<=raw<=1:raise ValueError('Invalid conditional probability')
        q=raw if uniform_bits is None else uniform_grid_probability(raw,uniform_bits);records.append(dict(prefix=prefix,reach=weight,raw_q=raw,q=q))
        for bit,prob in ((0,1-q),(1,q)):
            if not prob:continue
            branch=weight*prob
            if bit in node:visit(prefix+(bit,),branch,node[bit])
            else:unselected+=branch
    visit((),F(1),trie);failure=sum((r['mass'] for r in failures),F(0));selected=sum(masses);overlap=sum((min(lo,q) for (lo,hi),q in zip(intervals,masses)),F(0));positive=sum((max(F(0),lo-q) for (lo,hi),q in zip(intervals,masses)),F(0));negative=failure+sum((max(F(0),q-hi) for (lo,hi),q in zip(intervals,masses)),F(0));lower=max(positive,negative);upper=1-overlap
    assert selected+failure+unselected==1 and 0<=lower<=upper<=1
    return dict(modes=modes,patterns=patterns,target_intervals=intervals,proxy_masses=tuple(masses),selected_proxy_mass=selected,target_selected_lower=sum(lo for lo,hi in intervals),target_selected_upper=min(F(1),sum(hi for lo,hi in intervals)),confirmed_overlap=overlap,tv_lower=lower,tv_upper=upper,seen_failure_mass=failure,unselected_proxy_mass=unselected,queries=len(records),records=records,failures=failures,uniform_bits=uniform_bits)
