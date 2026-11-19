"""Weighted lazy add/delete/swap matching chain with audited word semantics.

Exact rejection uses independent uniform words and almost-sure termination;
neither exact stationarity nor local steps assert finite-time mixing accuracy.
"""
from fractions import Fraction as F


def uniform_below(bound, getword, *, bits=64):
    if type(bits) is not int or not 1<=bits<=64: raise ValueError('bits must be 1..64')
    size=1<<bits
    if type(bound) is not int or not 1<=bound<=size: raise ValueError('bound must fit word range')
    if bound==1:return 0
    limit=size-size%bound
    while True:
        word=getword()
        if type(word) is not int or not 0<=word<size:raise ValueError('unsigned word required')
        if word<limit:return word%bound


def rational_bernoulli(probability,getword,*,bits=64):
    p=F(probability)
    if not 0<=p<=1:raise ValueError('probability outside [0,1]')
    if p in (0,1):return bool(p)
    return uniform_below(p.denominator,getword,bits=bits)<p.numerator


class MatchingChain:
    def __init__(self,vertices,edges,weights):
        self.vertices=vertices;self.edges=tuple(tuple(e) for e in edges);self.weights=tuple(map(F,weights))
        if type(vertices) is not int or vertices<1 or not self.edges or len(self.edges)!=len(self.weights):raise ValueError('nonempty weighted simple graph required')
        if any(len(e)!=2 or any(type(v) is not int or not 0<=v<vertices for v in e) or e[0]>=e[1] for e in self.edges):raise ValueError('ordered simple edges required')
        if len(set(self.edges))!=len(self.edges) or min(self.weights)<=0:raise ValueError('distinct edges and positive weights required')
        self.incident=[0]*vertices
        for i,e in enumerate(self.edges):
            for v in e:self.incident[v]|=1<<i
        self.conflicts=tuple(self.incident[a]|self.incident[b] for a,b in self.edges)

    def validate(self,state):
        if type(state) is not int or not 0<=state<(1<<len(self.edges)):raise ValueError('invalid state bits')
        if any((state&mask).bit_count()>1 for mask in self.incident):raise ValueError('state is not a matching')

    def proposal(self,state,edge):
        selected=1<<edge
        if state&selected:return state^selected,min(F(1),1/self.weights[edge]),'delete'
        conflicts=state&self.conflicts[edge]
        if not conflicts:return state|selected,min(F(1),self.weights[edge]),'add'
        if conflicts.bit_count()==1:
            old=conflicts.bit_length()-1
            return (state^conflicts)|selected,min(F(1),self.weights[edge]/self.weights[old]),'swap'
        return state,F(1),'blocked'

    def step(self,state,getword,*,mode='exact',bits=64):
        self.validate(state)
        if mode not in ('exact','naive') or type(bits) is not int or not 1<=bits<=64:raise ValueError('invalid word mode')
        size=1<<bits
        if len(self.edges)>size:raise ValueError('edge count exceeds word range')
        def word():
            w=getword()
            if type(w) is not int or not 0<=w<size:raise ValueError('unsigned word required')
            return w
        lazy=word()
        if mode=='naive':
            edge_word=word();accept_word=word()
            if lazy&1:return state
            edge=len(self.edges)*edge_word//size;new,p,kind=self.proposal(state,edge)
            return new if F(accept_word,size)<p else state
        if lazy&1:return state
        edge=uniform_below(len(self.edges),word,bits=bits);new,p,kind=self.proposal(state,edge)
        if new==state:return state
        return new if rational_bernoulli(p,word,bits=bits) else state

    def states(self):
        """Exponential audit helper; not used by the local sampler."""
        if self.vertices>8:raise ValueError('state enumeration restricted to eight vertices')
        def visit(remaining,state):
            if not remaining:yield state;return
            first=remaining&-remaining;v=first.bit_length()-1;rest=remaining^first
            yield from visit(rest,state)
            for i,(a,b) in enumerate(self.edges):
                other=b if a==v else a if b==v else None
                if other is not None and rest&(1<<other):yield from visit(rest^(1<<other),state|(1<<i))
        return tuple(sorted(visit((1<<self.vertices)-1,0)))

    def weight(self,state):
        value=F(1)
        for i,w in enumerate(self.weights):
            if state&(1<<i):value*=w
        return value

    def transition(self,states,*,bits=None):
        lookup={s:i for i,s in enumerate(states)};m=len(self.edges);rows=[]
        if bits is None:edge_prob=[F(1,m)]*m
        else:
            size=1<<bits;ends=[(j*size+m-1)//m for j in range(m+1)];edge_prob=[F(b-a,size) for a,b in zip(ends,ends[1:])]
        for i,state in enumerate(states):
            row=[F(0)]*len(states);row[i]=F(1)
            for edge,pe in enumerate(edge_prob):
                new,p,kind=self.proposal(state,edge)
                if bits is not None:p=F(-((-p.numerator*(1<<bits))//p.denominator),1<<bits)
                change=pe*p/2;row[i]-=change;row[lookup[new]]+=change
            rows.append(row)
        return rows
