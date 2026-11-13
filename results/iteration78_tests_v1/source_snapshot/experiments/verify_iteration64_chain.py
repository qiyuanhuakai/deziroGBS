"""Rebuild finite kernels; certify stationary and hitting laws by exact equations."""
import random,time
from fractions import Fraction as F
from itertools import combinations
from math import prod
from iteration64_common import ROOT,read,save,digest,snapshot,tv
from iteration64_independent import enumerate_states,matrix,walk_checkpoints,project,replay

def main():
    out=ROOT/'results/iteration64_chain_validation';hashes=snapshot(out);deps={};checks=[];shards=[];count=0;started=time.perf_counter();comparisons=[];total_steps=total_words=successes=failures=attempts_count=0
    def load(name):deps[name]=digest(ROOT/name);return read(ROOT/name)
    def flush():
        nonlocal checks
        name=f'checks_{len(shards):03d}.json';save(out/name,checks);shards.append(name);checks=[]
    def check(name,value):
        nonlocal count
        checks.append(dict(name=name,passed=bool(value)));count+=1
        if len(checks)>=25000:flush()
    plan=load('results/iteration64_chain/plan.json');summary=load('results/iteration64_chain/summary.json')
    for name,s in plan['source_hashes'].items():check('frozen:'+name,digest(ROOT/name)==s==digest(ROOT/'results/iteration64_chain/source_snapshot'/name))
    for index in range(30):
        r=load(f'results/iteration64_chain/case_{index:02d}.json.gz');n=r['n'];c=r['c'];alpha=r['alpha'];edges=r['edges'];weights=list(map(F,r['weights']));states=enumerate_states(2*n,edges);prefix=f'case{index}:';size=len(states)
        expected_graph=plan['graphs'][index//6];check(prefix+'panel',r['index']==index and r['graph']==expected_graph and c==plan['parameters'][(index%6)//2] and alpha==(1 if index%2==0 else 4*n*n))
        expected_edges=sorted([*expected_graph['edges'],*[[a+n,b+n] for a,b in expected_graph['edges']],*[[i,i+n] for i in range(n)]]);check(prefix+'lift',edges==expected_edges and weights==[alpha*(c*c if (a<n)==(b<n) else 1) for a,b in edges]);check(prefix+'states',states==r['states'])
        w=[prod(weights[i] for i in range(len(edges)) if s&(1<<i)) for s in states];pi=[F(v,sum(w)) for v in w];check(prefix+'target-stationary',pi==r['target_stationary'])
        pm=[0]*(1<<n)
        for k in range(n//2+1):
            for chosen in combinations(expected_graph['edges'],k):
                vertices=[v for e in chosen for v in e]
                if len(set(vertices))==2*k:pm[sum(1<<v for v in vertices)]+=1
        target_weights=[v*v*c**(2*s.bit_count()) for s,v in enumerate(pm)];target=[v/sum(target_weights) for v in target_weights];pmass,projected=project(pi,states,edges,n);check(prefix+'graph-target',target==projected==r['target'] and pmass==r['target_perfect_mass'])
        variant_summaries=[];ideal_series=None;ideal_p=None
        for variant,bits in zip(r['variants'],(None,8,64)):
            tag=variant['name'];p=matrix(states,edges,weights,bits);check(prefix+tag+':kernel',p==variant['transition'])
            for i,row in enumerate(p):check(prefix+tag+f':row:{i}',sum(row)==1 and min(row)>=0 and row[i]>=F(1,2))
            def reachable(transpose=False):
                seen={0};todo=[0]
                while todo:
                    i=todo.pop()
                    for j in range(size):
                        if (p[j][i] if transpose else p[i][j])>0 and j not in seen:seen.add(j);todo.append(j)
                return len(seen)==size
            check(prefix+tag+':irreducible',reachable() and reachable(True))
            rho=variant['stationary'];check(prefix+tag+':stationary-normalization',min(rho)>0 and sum(rho)==1)
            for j in range(size):check(prefix+tag+f':stationary-equation:{j}',sum(rho[i]*p[i][j] for i in range(size))==rho[j])
            if bits is None:
                check(prefix+'ideal-stationarity',rho==pi)
                for i in range(size):
                    for j in range(size):check(prefix+f'detailed-balance:{i}:{j}',pi[i]*p[i][j]==pi[j]*p[j][i])
            resid=tv([sum(pi[i]*p[i][j] for i in range(size)) for j in range(size)],pi);check(prefix+tag+':residual',resid==variant['target_residual_TV'])
            smass,sproj=project(rho,states,edges,n);check(prefix+tag+':stationary-distances',tv(rho,pi)==variant['stationary_TV'] and smass==variant['stationary_perfect_mass'] and tv(sproj,target)==variant['stationary_projected_TV'])
            hit=variant['first_hit'];transient=[i for i,s in enumerate(states) if s.bit_count()!=n];occupation=hit['occupation'];check(prefix+tag+':hitting-transient',hit['transient_indices']==transient and len(occupation)==len(transient) and min(occupation)>=0)
            for j in transient:check(prefix+tag+f':hitting-equation:{j}',sum(occupation[k]*(F(i==j)-p[i][j]) for k,i in enumerate(transient))==F(j==0))
            first=[sum(occupation[k]*p[i][j] for k,i in enumerate(transient)) if states[j].bit_count()==n else F(0) for j in range(size)];hm,hproj=project(first,states,edges,n);check(prefix+tag+':first-hit',first==hit['law'] and hm==1 and min(first)>=0 and sum(occupation)==hit['expected_steps'] and hproj==variant['first_hit_projected'] and tv(hproj,target)==variant['first_hit_TV'])
            check(prefix+tag+':time-grid',[x['step'] for x in variant['series']]==plan['checkpoints']);previous=F(1)
            for item,(t,nums,den) in zip(variant['series'],walk_checkpoints(p,plan['checkpoints'])):
                law=item['law'];check(prefix+tag+f':law-length:{t}',len(law)==size)
                for j,(a,q) in enumerate(zip(nums,law)):check(prefix+tag+f':finite-law:{t}:{j}',a*q.denominator==q.numerator*den)
                mass,q=project(law,states,edges,n);delta=tv(law,pi);distance=None if q is None else tv(q,target)
                check(prefix+tag+f':finite-summary:{t}',sum(law)==1 and min(law)>=0 and mass==item['perfect_mass'] and q==item['projected'] and delta==item['full_TV'] and distance==item['conditional_TV'] and item['conditioning_bound']==min(F(1),delta/pmass) and (distance is None or distance<=item['conditioning_bound']))
                if bits is None:check(prefix+f'contraction:{t}',delta<=previous);previous=delta
            if bits is None:ideal_series=variant['series'];ideal_p=p;perturbation=F(0)
            else:
                perturbation=max(tv(a,b) for a,b in zip(p,ideal_p))
                for actual,ideal in zip(variant['series'],ideal_series):check(prefix+tag+f':telescoping-bound:{actual["step"]}',tv(actual['law'],ideal['law'])<=actual['step']*perturbation)
            variant_summaries.append(dict(name=tag,stationary_TV=variant['stationary_TV'],stationary_projected_TV=variant['stationary_projected_TV'],target_residual_TV=resid,first_hit_TV=variant['first_hit_TV'],expected_first_hit_steps=hit['expected_steps'],row_perturbation=perturbation,series=[{k:v for k,v in x.items() if k not in ('law','projected')} for x in variant['series']]))
        sample_summaries=[]
        for route_index,sample in enumerate(r['samples']):
            mode=sample['mode'];check(prefix+mode+':sample-plan',mode==('exact','naive')[route_index] and sample['seed']==640000+10*index+route_index and len(sample['runs'])==7);rng=random.Random(sample['seed']);attempts=0;ok=0
            for j,run in enumerate(sample['runs']):
                check(prefix+mode+f':attempt-count:{j}',1<=len(run['attempts'])<=64)
                for k,attempt in enumerate(run['attempts']):
                    path=attempt['path'];words=attempt['words'];check(prefix+mode+f':trace:{j}:{k}',len(path)==65 and replay(path,words,mode,edges,weights));check(prefix+mode+f':random-stream:{j}:{k}',words==[rng.getrandbits(64) for _ in words]);total_steps+=64;total_words+=len(words);attempts+=1;attempts_count+=1
                    check(prefix+mode+f':restart:{j}:{k}',k==len(run['attempts'])-1 or path[-1].bit_count()!=n)
                last=run['attempts'][-1]['path'][-1];success=last.bit_count()==n;check(prefix+mode+f':output:{j}',run['output']==(last if success else None) and (success or len(run['attempts'])==64));ok+=int(success);successes+=int(success);failures+=int(not success)
            p64=r['variants'][0 if mode=='exact' else 2]['series'][6]['perfect_mass'];check(prefix+mode+':stopping-law',sample['attempt_success_mass']==p64 and sample['failure_probability']==(1-p64)**64)
            sample_summaries.append(dict(mode=mode,attempts=attempts,successes=ok,failures=7-ok,failure_probability=sample['failure_probability'],attempt_success_mass=p64))
        comparisons.append(dict(index=index,graph=r['graph']['name'],n=n,c=c,alpha=alpha,states=size,target_perfect_mass=pmass,variants=variant_summaries,samples=sample_summaries));print(prefix+'verified',flush=True)
    check('summary-panel',summary['planned']==summary['finished']==len(comparisons)==30);check('sources-unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies-unchanged',all(digest(ROOT/n)==s for n,s in deps.items()))
    if checks:flush()
    passed=sum(sum(x['passed'] for x in read(out/name)) for name in shards);save(out/'comparisons.json',comparisons);save(out/'validation.json',dict(count=count,passed=passed,check_shards=shards,cases=30,kernels=90,public_outputs=successes,public_failures=failures,public_attempts=attempts_count,replayed_steps=total_steps,replayed_words=total_words,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True,elapsed_seconds=time.perf_counter()-started));assert count==passed

if __name__=='__main__':main()
