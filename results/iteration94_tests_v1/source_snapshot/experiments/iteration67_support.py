"""Frozen matching-chain budget panel,exact word rewards,and provenance."""
import sys,shutil,math
from fractions import Fraction as F
from pathlib import Path
from iteration63_common import save,read,digest,tv
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
TIMES=[16,32,64,128,256,512,1024]
EPSILONS=[F(1,100),F(1,1000)]

def graphs():return [dict(name=name,n=n,edges=edges,new_holdout=new) for name,n,edges,new in [('triangle3',3,[(0,1),(0,2),(1,2)],False),('path4',4,[(0,1),(1,2),(2,3)],False),('cycle4',4,[(0,1),(0,3),(1,2),(2,3)],False),('star4',4,[(0,1),(0,2),(0,3)],True),('diamond4',4,[(0,1),(0,2),(0,3),(1,2),(2,3)],True)]]
def jobs():return [dict(graph=g,graph_index=i,c=F(1,4),alpha=a) for i,g in enumerate(graphs()) for a in (1,4,16,4*g['n']*g['n'])]

def snapshot(out):
    out.mkdir(parents=True,exist_ok=False);files=[*ROOT.glob('experiments/iteration67*.py'),*ROOT.glob('experiments/verify_iteration67*.py'),*[ROOT/'experiments'/n for n in ('iteration63_common.py','iteration64_common.py','iteration64_independent.py','iteration22_external.py')],*ROOT.glob('src/**/*.py'),ROOT/'research/iteration67_cost_plan.md'];hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    for n in hashes:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    return hashes

def uniform_cost(b):
    size=1<<64
    if not 1<=b<=size:raise ValueError('word range exceeded')
    return F(0) if b==1 else F(size,size-size%b)

def word_rewards(chain,states):
    out=[];m=len(chain.edges)
    for s in states:
        reward=F(1)+uniform_cost(m)/2
        for e in range(m):
            new,p,_=chain.proposal(s,e)
            if new!=s and p not in (0,1):reward+=uniform_cost(p.denominator)/(2*m)
        out.append(reward)
    return out

def forward_rewards(p,rewards,times):
    d=math.lcm(*(x.denominator for row in p for x in row));w=math.lcm(*(x.denominator for x in rewards));a=[int(x*w) for x in rewards];rows=[[(j,int(x*d)) for j,x in enumerate(row) if x] for row in p];v=[1]+[0]*(len(p)-1);den=1;cost=0;out=[]
    for t in range(1,max(times)+1):
        cost=d*(cost+sum(x*y for x,y in zip(v,a)));new=[0]*len(v)
        for i,x in enumerate(v):
            if x:
                for j,y in rows[i]:new[j]+=x*y
        v=new;den*=d
        if t in times:out.append(dict(steps=t,law=[F(x,den) for x in v],attempt_words=F(cost,w*den)))
    return out

def project_state(state,edges,n):return sum(1<<v for i,(a,b) in enumerate(edges) if b<n and state&(1<<i) for v in (a,b))
