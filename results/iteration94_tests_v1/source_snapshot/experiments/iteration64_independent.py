"""Independent set-based chain, edge combinations and equation certificates."""
from fractions import Fraction as F
from itertools import combinations
from math import prod,lcm

def enumerate_states(vertices,edges):
    out=[]
    for k in range(vertices//2+1):
        for indices in combinations(range(len(edges)),k):
            if len({v for i in indices for v in edges[i]})==2*k:out.append(sum(1<<i for i in indices))
    return sorted(out)

def move(state,edge,edges,weights):
    occupied={i for i in range(len(edges)) if state&(1<<i)}
    if edge in occupied:new=occupied-{edge}
    else:
        touching={i for i in occupied if set(edges[i])&set(edges[edge])}
        if len(touching)>1:new=occupied
        else:new=(occupied-touching)|{edge}
    ratio=prod(weights[i] for i in new-occupied)/prod(weights[i] for i in occupied-new)
    return sum(1<<i for i in new),min(F(1),F(ratio))

def matrix(states,edges,weights,bits):
    n=len(states);m=len(edges);lookup={s:i for i,s in enumerate(states)};p=[[F(0)]*n for _ in states]
    def ceiling(q):return q.numerator//q.denominator+int(q.denominator!=1)
    if bits is None:edgep=[F(1,m)]*m
    else:edgep=[F(ceiling(F((i+1)*(1<<bits),m))-ceiling(F(i*(1<<bits),m)),1<<bits) for i in range(m)]
    for i,s in enumerate(states):
        p[i][i]=F(1,2)
        for e,pe in enumerate(edgep):
            new,a=move(s,e,edges,weights)
            if bits is not None:a=F(ceiling(a*(1<<bits)),1<<bits)
            p[i][lookup[new]]+=pe*a/2;p[i][i]+=pe*(1-a)/2
    return p

def walk_checkpoints(p,checkpoints):
    size=len(p);den=lcm(*(x.denominator for row in p for x in row));columns=[[(i,int(p[i][j]*den)) for i in range(size) if p[i][j]] for j in range(size)];v=[1]+[0]*(size-1);scale=1
    for t in range(1,max(checkpoints)+1):
        v=[sum(v[i]*a for i,a in col) for col in columns];scale*=den
        if t in checkpoints:yield t,v,scale

def project(law,states,edges,n):
    histogram=[F(0)]*(1<<n)
    for s,p in zip(states,law):
        if s.bit_count()==n:
            subset=sum(1<<v for i,(a,b) in enumerate(edges) if b<n and s&(1<<i) for v in (a,b))
            histogram[subset]+=p
    total=sum(histogram)
    return total,None if total==0 else [v/total for v in histogram]

def replay(path,words,mode,edges,weights,bits=64):
    cursor=0;size=1<<bits
    def word():
        nonlocal cursor
        x=words[cursor];cursor+=1;assert type(x) is int and 0<=x<size;return x
    def below(bound):
        if bound==1:return 0
        while True:
            x=word();q=x//bound
            if q<size//bound:return x-q*bound
    generated=[0]
    for step in range(64):
        state=generated[-1];lazy=word()
        if mode=='naive':
            e=len(edges)*word()//size;u=word();new,p=move(state,e,edges,weights)
            generated.append(new if not lazy%2 and u*p.denominator<size*p.numerator else state)
        elif lazy%2:generated.append(state)
        else:
            e=below(len(edges));new,p=move(state,e,edges,weights)
            accepted=True if new==state or p==1 else False if p==0 else below(p.denominator)<p.numerator
            generated.append(new if accepted else state)
    return generated==path and cursor==len(words)
