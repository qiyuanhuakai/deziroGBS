"""Directed-factor reconstruction of the separate exact toy witness."""
import json,hashlib,sys,math
from pathlib import Path
from fractions import Fraction as F
sys.set_int_max_str_digits(0);root=Path(__file__).parent;raw=json.loads((root/'result.json').read_text())
def decode(x):
    if isinstance(x,dict):
        if set(x)=={'numerator','denominator'}:return F(int(x['numerator']),int(x['denominator']))
        return {k:decode(v) for k,v in x.items()}
    if isinstance(x,list):return [decode(v) for v in x]
    return x
r=decode(raw);w=r['integer_weights'];n=sum(w);p=list(map(lambda x:F(x,n),w));checks=0
def check(ok):
    global checks
    assert ok;checks+=1
def event(c):return sum((p[s] for s in range(16) if all((s>>j)&1==x for j,x in c.items())),F(0))
check(n==r['total'] and min(w)>0);check(hashlib.sha256((root/'proof.py').read_bytes()).hexdigest()==r['source_sha256']);products=[];rows=[]
for t in r['trees']:
    reached={1};oriented=[]
    while len(reached)<3:
        for i,j in t['tree']:
            if (i in reached)!=(j in reached):
                a,b=(i,j) if i in reached else (j,i);oriented.append((a,b));reached.add(b);break
        else:raise AssertionError('Disconnected tree')
    q=[]
    for s in range(16):
        a=s&1;value=event({0:a,1:(s>>1)&1})
        for parent,child in oriented:
            condition={0:a,parent:(s>>parent)&1};value*=event(dict(condition,**{} )|{child:(s>>child)&1})/event(condition)
        q.append(value);check(value==t['joint'][s])
    check(sum(q)==1);post=q[15]/(q[14]+q[15]);check(post==t['posterior']);check(abs(post-r['true_posterior_A1_given_all_features1'])==t['absolute_posterior_error']);kl=math.prod((a/b)**weight for a,b,weight in zip(p,q,w));check(kl==t['exp_total_KL']);products.append(kl*t['exp_total_edge_CMI_sum']);rows.append(dict(tree=t['tree'],joint_TV=str(sum(abs(a-b) for a,b in zip(p,q))/2),posterior_error=str(t['absolute_posterior_error'])))
check(len(set(products))==1);check(r['true_posterior_A1_given_all_features1']==F(w[15],w[14]+w[15]));best=r['joint_KL_optimal_tree_index'];other=r['minimum_pointwise_posterior_error_tree_index'];check(all(r['trees'][best]['exp_total_KL']<t['exp_total_KL'] for j,t in enumerate(r['trees']) if j!=best));check(r['strict_error_gap']==r['trees'][best]['absolute_posterior_error']-r['trees'][other]['absolute_posterior_error']>F(1,100))
out=Path(sys.argv[1]) if len(sys.argv)>1 else root/'verification.json'
with out.open('x',encoding='utf-8') as f:json.dump(dict(passed=checks,checks=checks,directed_reconstruction=True,exact_constant_KL_plus_CMI_product=True,rows=rows,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest()),f,indent=2);f.write('\n')
print('exact independent toy checks',checks)
