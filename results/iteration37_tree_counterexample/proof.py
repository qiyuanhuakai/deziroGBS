"""Exact rational toy witness: joint-KL-optimal TAN need not optimize one posterior.

This is an exploratory arbitrary binary distribution, NOT a Gaussian GBS example.
It does not modify the predeclared iteration 37 scientific panels.
"""
from fractions import Fraction as F
from pathlib import Path
import itertools,json,hashlib,sys,math
import numpy as np
import mpmath as mp
sys.set_int_max_str_digits(0)
source=Path(__file__);sha=hashlib.sha256(source.read_bytes()).hexdigest();rng=np.random.default_rng(370004)
edges=((1,2),(1,3),(2,3));trees=list(itertools.combinations(edges,2));mp.mp.dps=90


def make(weights):
    total=sum(weights)
    def count(constraints):return sum(w for mask,w in enumerate(weights) if all((mask>>j)&1==v for j,v in constraints.items()))
    def distribution(tree):
        degrees={j:sum(j in e for e in tree) for j in (1,2,3)};q=[]
        for mask in range(16):
            a=mask&1;na=count({0:a});p=F(na,total)
            for j in (1,2,3):p*=F(count({0:a,j:(mask>>j)&1}),na)**(1-degrees[j])
            for i,j in tree:p*=F(count({0:a,i:(mask>>i)&1,j:(mask>>j)&1}),na)
            q.append(p)
        assert sum(q)==1 and min(q)>0
        return q
    edge_products={}
    for i,j in edges:
        product=F(1)
        for a,x,y in itertools.product((0,1),repeat=3):
            n=count({0:a,i:x,j:y});ratio=F(n*count({0:a}),count({0:a,i:x})*count({0:a,j:y}));product*=ratio**n
        edge_products[(i,j)]=product
    truth=F(weights[15],weights[14]+weights[15]);records=[]
    for tree in trees:
        q=distribution(tree);post=q[15]/(q[14]+q[15]);kl_product=math.prod((F(w,total)/v)**w for w,v in zip(weights,q));score=math.prod(edge_products[e] for e in tree)
        records.append(dict(tree=tree,joint=q,posterior=post,absolute_posterior_error=abs(post-truth),exp_total_KL=kl_product,exp_total_edge_CMI_sum=score))
    best=min(range(3),key=lambda j:records[j]['exp_total_KL']);assert best==max(range(3),key=lambda j:records[j]['exp_total_edge_CMI_sum']);assert len({r['exp_total_KL'] for r in records})==3
    return truth,records,best


for trial in range(1000):
    weights=[int(x) for x in rng.integers(1,41,size=16)];truth,records,best=make(weights);posterior_best=min(range(3),key=lambda j:records[j]['absolute_posterior_error'])
    if records[best]['absolute_posterior_error']>records[posterior_best]['absolute_posterior_error']+F(1,100):break
else:raise RuntimeError('No witness in the fixed exploratory search')
total=sum(weights);mf=lambda r:mp.mpf(r.numerator)/r.denominator
for r in records:
    kl=mp.fsum(mp.mpf(w)/total*mp.log(mf(F(w,total)/q)) for w,q in zip(weights,r['joint']));encoded=mp.log(mf(r['exp_total_KL']))/total;assert abs(kl-encoded)<mp.mpf('1e-85');r['KL_90_digit']=mp.nstr(kl,85)
assert all(records[best]['exp_total_KL']<r['exp_total_KL'] for j,r in enumerate(records) if j!=best)
result=dict(scope='Exact arbitrary-binary toy counterexample only; no claim it is Gaussian realizable; exploratory and separate from fixed GBS confirmation',seed=370004,trial=trial,bit_layout='bit0 next outcome A; bits1,2,3 omitted clicked features',integer_weights=weights,total=total,true_posterior_A1_given_all_features1=truth,trees=records,joint_KL_optimal_tree_index=best,minimum_pointwise_posterior_error_tree_index=posterior_best,strict_error_gap=records[best]['absolute_posterior_error']-records[posterior_best]['absolute_posterior_error'],source_sha256=sha,proof='Compare exp(total*KL) and exp(total*sum edge CMI) as exact rationals, with opposite strict rankings; enumerate normalized tree laws and exact posteriors',high_precision_check_digits=90)
def encode(x):
    if isinstance(x,F):return dict(numerator=str(x.numerator),denominator=str(x.denominator))
    if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [encode(v) for v in x]
    return x
assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
out=Path(sys.argv[1]) if len(sys.argv)>1 else source.with_name('result.json')
with out.open('x',encoding='utf-8') as f:json.dump(encode(result),f,indent=2);f.write('\n')
print('trial',trial,'weights',weights,'total',total,'truth',truth,'best KL',best,'best posterior',posterior_best,'gap',float(result['strict_error_gap']))
for r in records:print(r['tree'],'KL',r['KL_90_digit'][:22],'q',float(r['posterior']),'error',float(r['absolute_posterior_error']))
