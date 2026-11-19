"""Immutable iteration64 artifacts, frozen panel, rational matrix certificates."""
import sys,shutil
sys.set_int_max_str_digits(0)
from fractions import Fraction as F
from math import lcm
from iteration63_common import ROOT,digest,save,read,tv

def panel():
    return [dict(name=name,n=n,edges=e) for name,n,e in [('isolated_1',1,[]),('empty_2',2,[]),('edge_2',2,[(0,1)]),('path_3',3,[(0,1),(1,2)]),('triangle_3',3,[(0,1),(0,2),(1,2)])]]

def snapshot(out):
    out.mkdir(exist_ok=False)
    files=sorted({*ROOT.glob('experiments/iteration64*.py'),*ROOT.glob('experiments/verify_iteration64*.py'),ROOT/'experiments/iteration63_common.py',ROOT/'src/gbslab/matching_chain.py',ROOT/'tests/test_matching_chain.py',ROOT/'research/iteration64_chain_plan.md'})
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    for p in files:
        target=out/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    return hashes

def solve(a,b):
    """Production rational Gauss-Jordan solver; verifier checks equations."""
    x=[list(map(F,row))+[F(y)] for row,y in zip(a,b)];n=len(x)
    for k in range(n):
        pivot=next(i for i in range(k,n) if x[i][k]);x[k],x[pivot]=x[pivot],x[k];v=x[k][k];x[k]=[z/v for z in x[k]]
        for i in range(n):
            if i!=k:
                v=x[i][k]
                if v:x[i]=[u-v*w for u,w in zip(x[i],x[k])]
    return [row[-1] for row in x]

def stationary(p):
    n=len(p);a=[[p[i][j]-F(i==j) for i in range(n)] for j in range(n-1)]+[[F(1)]*n]
    return solve(a,[F(0)]*(n-1)+[F(1)])

def first_hit(p,perfect,start=0):
    indices=[i for i,x in enumerate(perfect) if not x];a=[[F(i==j)-p[i][j] for i in indices] for j in indices]
    g=solve(a,[F(j==start) for j in indices]);hit=[sum(v*p[i][j] for i,v in zip(indices,g)) if perfect[j] else F(0) for j in range(len(p))]
    return dict(transient_indices=indices,occupation=g,law=hit,expected_steps=sum(g))

def propagate(p,checkpoints):
    d=lcm(*(v.denominator for row in p for v in row));rows=[[(j,int(v*d)) for j,v in enumerate(row) if v] for row in p];vector=[0]*len(p);vector[0]=1;den=1;result=[]
    for t in range(1,max(checkpoints)+1):
        new=[0]*len(p)
        for i,v in enumerate(vector):
            if v:
                for j,a in rows[i]:new[j]+=v*a
        vector=new;den*=d
        if t in checkpoints:result.append(dict(step=t,law=[F(v,den) for v in vector]))
    return result

def project(law,states,edges,n):
    out=[F(0)]*(1<<n);mass=F(0)
    for state,p in zip(states,law):
        if state.bit_count()!=n:continue
        subset=0
        for i,(a,b) in enumerate(edges):
            if b<n and state&(1<<i):subset|=(1<<a)|(1<<b)
        out[subset]+=p;mass+=p
    return mass,None if not mass else [x/mass for x in out]
