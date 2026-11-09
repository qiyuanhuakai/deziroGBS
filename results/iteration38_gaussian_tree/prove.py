"""Exploratory search, then exact interval certificate on a physical marginal."""
import json,sys,itertools,math,hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'));sys.set_int_max_str_digits(0)
from verify_iteration34 import decode
from iteration33_run import encode
BITS=120;S=1<<BITS;LOGS=[]

def rounded(a,b):return (F(a.numerator*S//a.denominator,S),F(-((-b.numerator*S)//b.denominator),S))
def add(a,b):return rounded(a[0]+b[0],a[1]+b[1])
def sub(a,b):return rounded(a[0]-b[1],a[1]-b[0])
def mul(a,b):
    v=[x*y for x in a for y in b];return rounded(min(v),max(v))
def div(a,b):
    assert b[0]>0;return mul(a,(1/b[1],1/b[0]))
def total(items):
    v=(F(0),F(0))
    for item in items:v=add(v,item)
    return v
def series(z,n=48):
    v=sum((2*z**(2*j+1)/F(2*j+1) for j in range(n)),F(0));return v,v+2*z**(2*n+1)/(F(2*n+1)*(1-z*z))
LOG2=series(F(1,3))
def point_log(x):
    assert x>0;k=x.numerator.bit_length()-x.denominator.bit_length();t=x/(F(2)**k)
    if t<1:k-=1;t*=2
    assert 1<=t<2;v=series((t-1)/(t+1));a,b=(k*LOG2[0],k*LOG2[1]) if k>=0 else (k*LOG2[1],k*LOG2[0]);result=rounded(v[0]+a,v[1]+b);LOGS.append(dict(x=x,k=k,interval=result));return result
def log(a):return point_log(a[0])[0],point_log(a[1])[1]
def absolute(a):return (F(0) if a[0]<=0<=a[1] else min(abs(a[0]),abs(a[1])),max(abs(a[0]),abs(a[1])))
def save(p,x):p.write_text(json.dumps(encode(x),indent=2)+'\n',encoding='utf-8')

def main():
    out=Path(__file__).parent
    if (out/'proof.json').exists() or (out/'search.json').exists():raise FileExistsError('Do not overwrite the evidence')
    inputs={};search=[]
    for start in (0,54,134):
        path=ROOT/f'results/iteration37_local/target_{start:03d}.json';a=decode(json.loads(path.read_text()))['law']['intervals'];inputs[start]=a;p=np.array([float((l+h)/2) for l,h in a])
        for modes in itertools.combinations(range(10),4):
            for target in modes:
                ids=(target,)+tuple(i for i in modes if i!=target);v=np.zeros((2,2,2,2))
                for mask,w in enumerate(p):v[tuple((mask>>i)&1 for i in ids)]+=w
                pa=v.sum(axis=(1,2,3));one={i:v.sum(axis=tuple(k for k in (1,2,3) if k!=i)) for i in (1,2,3)};pair={ij:v.sum(axis=tuple(k for k in (1,2,3) if k not in ij)) for ij in itertools.combinations((1,2,3),2)};mi={ij:sum(pair[ij][aa,x,y]*math.log(pair[ij][aa,x,y]*pa[aa]/(one[ij[0]][aa,x]*one[ij[1]][aa,y])) for aa in range(2) for x in range(2) for y in range(2)) for ij in pair};true=v[1,1,1,1]/v[:,1,1,1].sum();trees=[]
                for c in (1,2,3):
                    edges=[tuple(sorted((c,l))) for l in (1,2,3) if l!=c];weights=[np.prod([pair[e][aa,1,1] for e in edges])/one[c][aa,1] for aa in (0,1)];q=weights[1]/sum(weights);trees.append(dict(center=c,score=sum(mi[e] for e in edges),posterior=q,error=abs(q-true)))
                joint=max(trees,key=lambda t:t['score']);local=min(trees,key=lambda t:t['error']);search.append(dict(start=start,local_modes=ids,physical_modes=[start+i for i in ids],true=true,trees=trees,gap=joint['error']-local['error']))
    winner=max(search,key=lambda r:r['gap']);save(out/'search.json',dict(exploratory_only=True,candidates=len(search),criterion='largest floating diagnostic error gap among searched four-mode marginals; no global optimality claim',rows=search,winner=winner));ids=winner['local_modes'];start=winner['start'];p=[]
    for mask in range(16):p.append(rounded(sum((a for i,(a,b) in enumerate(inputs[start]) if all((i>>mode)&1==(mask>>j)&1 for j,mode in enumerate(ids))),F(0)),sum((b for i,(a,b) in enumerate(inputs[start]) if all((i>>mode)&1==(mask>>j)&1 for j,mode in enumerate(ids))),F(0))))
    assert all(a>0 for a,b in p)
    def marginal(assignment):return total(p[mask] for mask in range(16) if all((mask>>j)&1==v for j,v in assignment.items()))
    true=div(p[15],add(p[14],p[15]));mi={}
    for edge in itertools.combinations((1,2,3),2):
        terms=[]
        for aa,x,y in itertools.product((0,1),repeat=3):
            joint=marginal({0:aa,edge[0]:x,edge[1]:y});ratio=div(mul(joint,marginal({0:aa})),mul(marginal({0:aa,edge[0]:x}),marginal({0:aa,edge[1]:y})));terms.append(mul(joint,log(ratio)))
        mi[edge]=total(terms)
    trees=[]
    for center in (1,2,3):
        leaves=[j for j in (1,2,3) if j!=center];edges=[tuple(sorted((center,j))) for j in leaves];q=[]
        for mask in range(16):
            bits=[(mask>>j)&1 for j in range(4)];factors=[marginal({0:bits[0],center:bits[center],leaf:bits[leaf]}) for leaf in leaves];q.append(div(mul(*factors),marginal({0:bits[0],center:bits[center]})))
        posterior=div(q[15],add(q[14],q[15]));error=absolute(sub(posterior,true));kl=total(mul(a,log(div(a,b))) for a,b in zip(p,q));trees.append(dict(center=center,edges=edges,law=q,score=total(mi[e] for e in edges),posterior=posterior,error=error,KL=kl))
    best=max(trees,key=lambda t:t['score'][0]);other=min(trees,key=lambda t:t['error'][1]);assert all(best['score'][0]>t['score'][1] and best['KL'][1]<t['KL'][0] for t in trees if t!=best);gap=sub(best['error'],other['error']);assert gap[0]>0
    deps=[f'results/iteration37_local/target_{s:03d}.json' for s in (0,54,134)]+['results/iteration35_panel/case_07.audit.json','results/iteration37_validation/validation.json'];result=dict(physical_model='Exact B,F Gaussian target certified in iteration35; genuine four-mode threshold marginal',bits=BITS,log_terms=48,local_modes=ids,physical_modes=winner['physical_modes'],start=start,law=p,true_posterior=true,trees=trees,joint_KL_best_center=best['center'],better_posterior_center=other['center'],error_gap=gap,log_certificates=LOGS,dependencies={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in deps},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Deterministic existence certificate after exploratory search; no statistical selection claim, no global GBS-TV ranking')
    save(out/'proof.json',result);print(json.dumps(dict(candidates=len(search),physical_modes=result['physical_modes'],true_posterior=list(map(float,true)),trees=[{k:list(map(float,t[k])) if k in ('score','posterior','error','KL') else t[k] for k in ('center','score','posterior','error','KL')} for t in trees],gap=list(map(float,gap)),log_certificates=len(LOGS))),flush=True)
if __name__=='__main__':main()
