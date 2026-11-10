"""Reconstruct every structural coefficient, probability, CDF cell and sample."""
from collections import Counter
from fractions import Fraction as F
from itertools import accumulate
import random,time
from iteration63_common import ROOT,read,save,digest,snapshot,tv
from iteration63_independent import reconstruct,integer_determinant,sqrt_interval,word_cells,locate

def main():
    out=ROOT/'results/iteration63_graph_validation';hashes=snapshot(out);deps={};checks=[];shards=[];total=0;started=time.perf_counter();comparisons=[]
    def load(p):
        deps[p]=digest(ROOT/p);return read(ROOT/p)
    def flush():
        nonlocal checks
        name=f'checks_{len(shards):03d}.json';save(out/name,checks);shards.append(name);checks=[]
    def check(name,passed):
        nonlocal total
        checks.append(dict(name=name,passed=bool(passed)));total+=1
        if len(checks)>=25000: flush()
    plan=load('results/iteration63_graph/plan.json');summary=load('results/iteration63_graph/summary.json')
    for name,s in plan['source_hashes'].items():check('frozen:'+name,digest(ROOT/name)==s==digest(ROOT/'results/iteration63_graph/source_snapshot'/name))
    all_witnesses=0;atoms=0;public=0
    for gi,g in enumerate(plan['graphs']):
        n=g['n'];name=g['name'];st=load(f'results/iteration63_graph/{name}_structure.json.gz');pm,hist,witnesses,base_count=reconstruct(n,g['edges'])
        check(name+':subset-counts',pm==st['subset_counts']);check(name+':histogram',hist=={(k,h):v for k,h,v in st['coefficients']})
        got=[(w['subset'],tuple(map(tuple,w['edges']))) for w in st['perfect_matchings']]
        check(name+':witness-bijection',len(got)==len(set(got))==len(witnesses) and set(got)==witnesses);all_witnesses+=len(got)
        subset_frequency=Counter(s for s,w in witnesses)
        for s in range(1<<n):check(f'{name}:multiplicity:{s}',subset_frequency[s]==pm[s]**2)
        for ci,c in enumerate(plan['parameters']):
            index=3*gi+ci;r=load(f'results/iteration63_graph/case_{index:02d}.json.gz');prefix=f'case{index}:';weights=[c**(2*s.bit_count())*v*v for s,v in enumerate(pm)];normalizer=sum(weights);target=[w/normalizer for w in weights];atoms+=len(target)
            z=[sum(v*c**(2*h) for (k,h),v in hist.items() if k==j) for j in range(n+1)]
            check(prefix+'identity',r['graph']==name and r['c']==c and r['index']==index)
            for s,p in enumerate(target):check(prefix+f'target:{s}',r['target'][s]==p)
            check(prefix+'partitions',r['partitions']==z and r['graph_partition']==normalizer==z[-1])
            check(prefix+'ratio',r['near_perfect_ratio']==z[-2]/z[-1]<2*n*n)
            for k in range(1,n):check(prefix+f'logconcavity:{k}',z[k-1]*z[k+1]<=z[k]*z[k])
            mass=lambda a:F(a**n*z[-1],sum(a**k*v for k,v in enumerate(z)))
            alpha=r['least_alpha'];check(prefix+'universal',r['universal_alpha']==4*n*n and r['universal_mass']==mass(4*n*n)>F(1,2))
            check(prefix+'least-alpha',r['least_mass']==mass(alpha)>=F(1,2) and (alpha==1 or r['previous_mass']==mass(alpha-1)<F(1,2)))
            check(prefix+'unweighted',r['unweighted_mass']==mass(1))
            rng=random.Random(630000+index);words=[rng.getrandbits(64) for _ in range(17)];projected=[];distance=[]
            for route in r['routes']:
                tag=route['name'];order=route['order'];labels=list(range(1<<n)) if tag=='direct' else [got[j][0] for j in order]
                if tag=='natural':check(prefix+'natural-order',order==list(range(len(got))))
                if tag=='grouped':check(prefix+'grouped-order',order==sorted(range(len(got)),key=lambda j:(got[j][0],j)))
                p=target if tag=='direct' else [c**(2*s.bit_count())/normalizer for s in labels];counts=word_cells(p);totals=[0]*(1<<n)
                check(prefix+tag+':normalization',sum(p)==1 and sum(counts)==1<<64 and min(counts)>=0)
                for j,(a,b) in enumerate(zip(counts,route['word_counts'])):check(prefix+tag+f':cell:{j}',a==b)
                check(prefix+tag+':cell-length',len(counts)==len(route['word_counts']))
                for s,v in zip(labels,counts):totals[s]+=v
                check(prefix+tag+':projected',totals==route['projected_counts']);projected.append(totals);law=[F(x,1<<64) for x in totals];d=tv(law,target);distance.append(d)
                check(prefix+tag+':TV-bound',d==route['TV']<=route['support_bound']==F(sum(x>0 for x in p)-1,1<<64))
                for j,(sample,w) in enumerate(zip(route['samples'],words)):
                    selected=locate(p,w);check(prefix+tag+f':sample:{j}',sample==dict(word=w,index=selected,subset=labels[selected],witness=None if order is None else order[selected]));public+=1
                check(prefix+tag+':sample-count',len(route['samples'])==17)
            check(prefix+'grouping-commutes',projected[0]==projected[2]);check(prefix+'route-TV',r['natural_direct_TV']==tv([F(x,1<<64) for x in projected[0]],[F(x,1<<64) for x in projected[1]]))
            bridge=r['bridge'];valid=not g['edges'] or c==F(1,4)
            check(prefix+'physical-scope',bridge['status']==('valid' if valid else 'invalid'))
            if valid:
                a=[[F(0)]*n for _ in range(n)]
                for i,j in g['edges']:a[i][j]=a[j][i]=c*c
                d=integer_determinant([[F(i==j)-sum(a[i][k]*a[k][j] for k in range(n)) for j in range(n)] for i in range(n)]);lo,hi=sqrt_interval(d)
                check(prefix+'physical-determinant',d==bridge['determinant']>0)
                check(prefix+'physical-root',bridge['root_interval']==[lo,hi] and lo*lo<=d<=hi*hi)
                check(prefix+'physical-event',bridge['collision_free_probability']==[lo*normalizer,hi*normalizer] and 0<=lo*normalizer<=hi*normalizer<=1)
            comparisons.append(dict(index=index,graph=name,n=n,c=c,base_matchings=base_count,lift_perfect_matchings=len(got),near_perfect_ratio=r['near_perfect_ratio'],unweighted_mass=r['unweighted_mass'],universal_alpha=4*n*n,universal_mass=r['universal_mass'],least_alpha=alpha,alpha_ratio=F(4*n*n,alpha),direct_TV=distance[0],natural_TV=distance[1],natural_direct_TV=r['natural_direct_TV'],bridge=bridge))
            print(prefix+' verified',flush=True)
    check('finished-panel',summary['planned']==summary['finished']==30 and atoms==2070 and public==1530)
    check('sources-unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies-unchanged',all(digest(ROOT/n)==s for n,s in deps.items()))
    if checks:flush()
    passed=sum(sum(c['passed'] for c in read(out/name)) for name in shards)
    save(out/'comparisons.json',comparisons);save(out/'validation.json',dict(count=total,passed=passed,check_shards=shards,graphs=10,cases=30,atoms=atoms,perfect_witnesses=all_witnesses,public_outputs=public,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True,elapsed_seconds=time.perf_counter()-started));assert total==passed

if __name__=='__main__':main()
