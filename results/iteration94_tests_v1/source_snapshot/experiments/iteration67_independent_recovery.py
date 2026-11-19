"""Independent backward reward equation and variable-time set-based replay."""
from fractions import Fraction as F
from math import lcm
from math import prod

def move(state,edge,edges,weights):
    occupied={i for i in range(len(edges)) if state&(1<<i)}
    if edge in occupied:new=occupied-{edge}
    else:
        touching={i for i in occupied if set(edges[i])&set(edges[edge])}
        new=occupied if len(touching)>1 else (occupied-touching)|{edge}
    numerator=prod((F(weights[i]) for i in new-occupied),start=F(1))
    denominator=prod((F(weights[i]) for i in occupied-new),start=F(1))
    return sum(1<<i for i in new),min(F(1),numerator/denominator)

def rewards(states,edges,weights):
    M=1<<64;m=len(edges)
    def count(b):return F(0) if b==1 else F(M,b*(M//b))
    values=[]
    for s in states:
        costs=[]
        for e in range(m):
            new,p=move(s,e,edges,weights);costs.append(count(m)+(count(p.denominator) if new!=s and 0<p<1 else 0))
        values.append(F(1)+sum(costs)/(2*m))
    return values

def backward_and_forward(p,reward,times):
    n=len(p);D=lcm(*(x.denominator for row in p for x in row));W=lcm(*(x.denominator for x in reward));a=[int(x*W) for x in reward];rows=[[(j,int(x*D)) for j,x in enumerate(row) if x] for row in p];columns=[[(i,int(p[i][j]*D)) for i in range(n) if p[i][j]] for j in range(n)];v=[1]+[0]*(n-1);u=[0]*n;den=1
    for t in range(1,max(times)+1):
        den*=D;v=[sum(v[i]*x for i,x in col) for col in columns];u=[a[i]*den+sum(x*u[j] for j,x in row) for i,row in enumerate(rows)]
        if t in times:yield t,v,den,F(u[0],W*den)

def replay(path,words,steps,edges,weights,*,start=0):
    cursor=0;M=1<<64
    def word():
        nonlocal cursor
        value=words[cursor];cursor+=1;assert type(value) is int and 0<=value<M;return value
    def below(b):
        if b==1:return 0
        while True:
            value=word();quotient=value//b
            if quotient<M//b:return value-quotient*b
    generated=[start]
    for _ in range(steps):
        state=generated[-1]
        if word()%2:generated.append(state);continue
        e=below(len(edges));new,p=move(state,e,edges,weights);accepted=True if new==state or p==1 else False if p==0 else below(p.denominator)<p.numerator;generated.append(new if accepted else state)
    return generated==path and cursor==len(words)
