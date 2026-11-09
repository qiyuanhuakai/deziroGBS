"""Independent unrounded interval algebra and 64-term logarithm checks."""
import json,sys,itertools,hashlib
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'));sys.set_int_max_str_digits(0)
from verify_iteration34 import decode
OUT=Path(__file__).parent
def read(p):return decode(json.loads(p.read_text()))
def contains(outer,inner):return outer[0]<=inner[0]<=inner[1]<=outer[1]
def plus(x,y):return x[0]+y[0],x[1]+y[1]
def times(x,y):
    a=[i*j for i in x for j in y];return min(a),max(a)
def over(x,y):
    assert y[0]>0;return times(x,(1/y[1],1/y[0]))
def summation(seq):
    v=(F(0),F(0))
    for x in seq:v=plus(v,x)
    return v
def series(z):
    # Independent recurrence with 64 terms, versus the producer's 48 powers.
    v=F(0);power=z
    for j in range(64):v+=2*power/F(2*j+1);power*=z*z
    return v,v+2*power/(129*(1-z*z))
def ln(x):
    k=0;t=x
    while t<1:t*=2;k-=1
    while t>=2:t/=2;k+=1
    a,b=series(F(1,3));c,d=series((t-1)/(t+1));return (c+k*a,d+k*b) if k>=0 else (c+k*b,d+k*a)
def main():
    if (OUT/'verification.json').exists():raise FileExistsError('Keep original verification')
    r=read(OUT/'proof.json');checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    for name,sha in r['dependencies'].items():check('dependency '+name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha)
    check('producer source',hashlib.sha256((OUT/'prove.py').read_bytes()).hexdigest()==r['source_sha256'])
    source=read(ROOT/f'results/iteration37_local/target_{r["start"]:03d}.json')['law']['intervals'];ids=r['local_modes'];p=[]
    for bits in range(16):
        group=[source[mask] for mask in range(1024) if sum(((mask>>v)&1)<<j for j,v in enumerate(ids))==bits];actual=summation(group);check('physical atom '+str(bits),contains(r['law'][bits],actual) and r['law'][bits][0]>0);p.append(tuple(r['law'][bits]))
    check('physical mode map',r['physical_modes']==[r['start']+v for v in ids]);check('normalization enclosure',sum(a for a,b in p)<=1<=sum(b for a,b in p));lookup={}
    for i,c in enumerate(r['log_certificates']):check('rational logarithm '+str(i),contains(c['interval'],ln(c['x'])));lookup[c['x']]=tuple(c['interval'])
    cursor=iter(r['log_certificates']);used=0
    def logarithm(x):
        nonlocal used
        low=next(cursor);high=next(cursor);check('logarithm argument '+str(used),low['x']<=x[0]<=x[1]<=high['x']);used+=1;return low['interval'][0],high['interval'][1]
    def marginal(fixed):
        mask=sum(1<<j for j in fixed);value=sum(v<<j for j,v in fixed.items());return summation(p[i] for i in range(16) if i&mask==value)
    mi={}
    for i,j in itertools.combinations((1,2,3),2):
        terms=[]
        for a,b,c in itertools.product((0,1),repeat=3):
            v=marginal({0:a,i:b,j:c});ratio=over(times(v,marginal({0:a})),times(marginal({0:a,i:b}),marginal({0:a,j:c})));terms.append(times(v,logarithm(ratio)))
        mi[i,j]=summation(terms)
    truth=over(p[15],plus(p[14],p[15]));check('true posterior',contains(r['true_posterior'],truth));trees=[]
    for rec in r['trees']:
        center=rec['center'];leaves=[j for j in (1,2,3) if j!=center];q=[]
        for mask in range(16):
            bits=[(mask>>j)&1 for j in range(4)];factors=[marginal({0:bits[0],center:bits[center],j:bits[j]}) for j in leaves];v=over(times(*factors),marginal({0:bits[0],center:bits[center]}));check(f'tree {center} atom {mask}',contains(rec['law'][mask],v));q.append(v)
        score=summation(mi[tuple(sorted((center,j)))] for j in leaves);kl=summation(times(a,logarithm(over(a,b))) for a,b in zip(p,q));post=over(q[15],plus(q[14],q[15]));diff=(post[0]-truth[1],post[1]-truth[0]);error=(F(0) if diff[0]<=0<=diff[1] else min(abs(diff[0]),abs(diff[1])),max(abs(diff[0]),abs(diff[1])))
        for name,value in (('score',score),('KL',kl),('posterior',post),('error',error)):check(f'tree {center} '+name,contains(rec[name],value))
        trees.append(rec)
    check('used all logarithms',used==72 and next(cursor,None) is None);best=next(t for t in trees if t['center']==r['joint_KL_best_center']);other=next(t for t in trees if t['center']==r['better_posterior_center']);check('strict unique joint KL optimizer',all(best['KL'][1]<t['KL'][0] and best['score'][0]>t['score'][1] for t in trees if t!=best));gap=(best['error'][0]-other['error'][1],best['error'][1]-other['error'][0]);check('strict posterior counterexample',contains(r['error_gap'],gap) and r['error_gap'][0]>0);result=dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,logarithm_certificates=144,physical_atoms=16,tree_atoms=48,proof_sha256=hashlib.sha256((OUT/'proof.json').read_bytes()).hexdigest(),verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Exact rational interval certificate for a physical four-mode Gaussian threshold marginal; exploratory existence, not global sampler error')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
if __name__=='__main__':main()
