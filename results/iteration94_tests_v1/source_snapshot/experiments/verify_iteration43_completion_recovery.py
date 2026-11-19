"""Independent rational-power log enclosure and direct moment/target audit."""
import json,sys,shutil,math
from functools import lru_cache
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration41_independent import target_from_vacua
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

@lru_cache(None)
def direct_unit(y):
    # Independent exact powers, rounding each rational term only once at192bits.
    z=(y-1)/(y+1);a,b=z.numerator,z.denominator;u,v=a,b;S=1<<192;lo=hi=0;T=96
    for k in range(T):
        numerator=2*S*u;denominator=v*(2*k+1);lo+=numerator//denominator;hi+=(numerator+denominator-1)//denominator;u*=a*a;v*=b*b
    tail=F(9,4*(2*T+1)*3**(2*T+1))*S;hi+=(tail.numerator+tail.denominator-1)//tail.denominator
    return F(lo,S),F(hi,S)

def direct_log(x):
    x=F(x);assert x>0;k=0
    while x<1:x*=2;k-=1
    while x>=2:x/=2;k+=1
    lo,hi=direct_unit(x);a,b=direct_unit(F(2))
    return (lo+k*a,hi+k*b) if k>=0 else (lo+k*b,hi+k*a)

def main():
    out=ROOT/'results/iteration43_completion_recovery';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration43_completion_recovery','iteration22_external','iteration33_run','verify_iteration34','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    panel='results/iteration43_completion';plan=read(panel+'/plan.json');summary=read(panel+'/summary.json');check('25completed',summary['planned']==summary['finished']==summary['completed']==25 and summary['unchanged_sources'] and summary['unchanged_dependencies'])
    for n,s in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/panel/'source_snapshot'/n))
    for n,s in plan['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    for i,job in enumerate(plan['jobs']):
        r=read(panel+f'/case_{i:02d}.json');old=read(job['path']);n=r['modes'];size=1<<n;w=old['job']['weights'];a=old['job']['intensity'];vac=[1/(1+a*sum(w[j]**2 for j in range(n) if mask>>j&1)) for mask in range(size)];target=target_from_vacua(vac,n);law=r['projection']['law'];tag=f'case{i} '
        check(tag+'physical family and target',r['job']==job and r['physical_job']==old['job'] and n==len(w) and target==r['target']==old['target'][:size] and min(target)>=0 and sum(target)==1)
        subsets=[(j,) for j in range(n)]+[(k,j) for j in range(n) for k in range(j)];features=[sum(1<<j for j in S) for S in subsets];X=[[math.prod(1-2*((atom>>j)&1) for j in S) for S in subsets] for atom in range(size)]
        # Independent low-order target moments directly from vacuum probabilities.
        mu=[]
        for S in subsets:
            if len(S)==1:mu.append(2*vac[1<<S[0]]-1)
            else:mu.append(4*vac[sum(1<<j for j in S)]-2*sum(vac[1<<j] for j in S)+1)
        check(tag+'true low moments',mu==r['moments']==r['projection']['moments'] and features==r['projection']['features'])
        check(tag+'positive normalized',len(law)==size and min(law)>0 and sum(law)==1 and min(law)==r['projection']['minimum_probability'])
        for j,S in enumerate(subsets):check(tag+'exact moment '+str(S),sum(p*X[k][j] for k,p in enumerate(law))==mu[j])
        weights=r['fit']['probabilities'];q=[p/sum(weights) for p in weights];delta=[m-sum(p*X[k][j] for k,p in enumerate(q)) for j,m in enumerate(mu)]
        check(tag+'projection',q==r['projection']['normalized_candidate'] and delta==r['projection']['corrections'] and all(law[k]==q[k]+sum(d*x for d,x in zip(delta,X[k]))/size for k in range(size)))
        cert=r['certificate'];theta=r['fit']['theta'];check(tag+'parameter identity',theta==cert['theta']);logs=cert['log_intervals'];res=[]
        for k,p in enumerate(law):
            lo,hi=direct_log(p);l,h=logs[k];check(tag+'independent log '+str(k),l<=lo<=hi<=h);e=sum(t*x for t,x in zip(theta,X[k]));res.append((l-e,h-e))
        ent=(-sum(p*h for p,(l,h) in zip(law,logs)),-sum(p*l for p,(l,h) in zip(law,logs)));gap=sum(p*h for p,(l,h) in zip(law,res))-min(l for l,h in res);tvmax=cert['TV_to_exact_maximum_entropy_upper'];physical=sum(abs(p-t) for p,t in zip(law,target))/2;baseline=old['real_sampler_TV'] if job['round']==41 else old['official_formula_real_TV']
        check(tag+'entropy and gap',tuple(cert['entropy_interval'])==ent and cert['entropy_gap_upper']==gap>=0)
        check(tag+'Pinsker outward bound',tvmax==1 if gap>=2 else tvmax**2>=gap/2 and (tvmax-F(1,1<<80))**2<gap/2)
        check(tag+'separate physical error',physical==r['physical_TV'] and baseline==r['reference_TV'])
        rows.append(dict(sequence=i,modes=n,physical_TV=physical,reference_TV=baseline,improvement=baseline-physical,relative_reduction=(baseline-physical)/baseline,maxent_TV_upper=tvmax,entropy_gap_upper=gap));print('verified completion',i,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'comparisons.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=25,improved=sum(r['improvement']>0 for r in rows),source_hashes=hashes,dependencies=deps));print('completion verification',len(checks),flush=True)
if __name__=='__main__':main()
