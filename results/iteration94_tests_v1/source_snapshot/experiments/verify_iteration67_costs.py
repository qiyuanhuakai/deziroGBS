"""Rebuild all finite laws and restart resources with independent rewards."""
from fractions import Fraction as F
from itertools import combinations,permutations
from math import prod
from iteration67_support import ROOT,save,read,digest,snapshot,jobs,TIMES,tv
from iteration63_common import encode
from iteration64_independent import enumerate_states,matrix,project
from iteration67_independent import rewards,backward_and_forward

def main():
    out=ROOT/'results/iteration67_costs_validation';hashes=snapshot(out);deps={};checks=[];rows=[]
    def load(n):deps[n]=digest(ROOT/n);return read(ROOT/n)
    def ck(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration67_costs';plan=load(base+'/plan.json');summary=load(base+'/summary.json');ck('20 completed',summary['planned']==summary['finished']==summary['completed']==20);ck('fixed panel',encode(plan['jobs'])==encode(jobs()) and plan['checkpoints']==TIMES)
    for n,h in plan['source_hashes'].items():ck('frozen '+n,digest(ROOT/n)==h==digest(ROOT/base/'source_snapshot'/n))
    for index,job in enumerate(plan['jobs']):
        r=load(base+f'/case_{index:02d}.json.gz');tag=f'case{index} ';g=job['graph'];n=g['n'];c=job['c'];alpha=job['alpha'];edges=sorted([*g['edges'],*[[a+n,b+n] for a,b in g['edges']],*[[i,i+n] for i in range(n)]]);weights=[alpha*(c*c if (a<n)==(b<n) else 1) for a,b in edges];states=enumerate_states(2*n,edges);ck(tag+'input and lift',r['job']==job and r['edges']==edges and r['weights']==weights and r['states']==states);p=matrix(states,edges,weights,None);ck(tag+'full independent kernel',p==r['transition']);raw=[prod(weights[e] for e in range(len(edges)) if s&(1<<e)) for s in states];pi=[F(x,sum(raw)) for x in raw];ck(tag+'stationary law',pi==r['stationary'] and sum(pi)==1 and min(pi)>0)
        for i,row in enumerate(p):
            ck(tag+f'kernel row {i}',min(row)>=0 and sum(row)==1 and row[i]>=F(1,2))
            for j,x in enumerate(row):
                if x:ck(tag+f'balance {i}:{j}',pi[i]*x==pi[j]*p[j][i])
        seen={0};todo=[0]
        while todo:
            i=todo.pop()
            for j,x in enumerate(p[i]):
                if x and j not in seen:seen.add(j);todo.append(j)
        ck(tag+'irreducible by balance and reachability',len(seen)==len(states))
        pm=[0]*(1<<n)
        for k in range(n//2+1):
            for chosen in combinations(g['edges'],k):
                vertices=[v for e in chosen for v in e]
                if len(set(vertices))==2*k:pm[sum(1<<v for v in vertices)]+=1
        gw=[F(v*v)*c**(2*s.bit_count()) for s,v in enumerate(pm)];target=[x/sum(gw) for x in gw];mass,projected=project(pi,states,edges,n);ck(tag+'independent graph target',target==r['target']==projected and mass==r['stationary_success_mass'] and sum(gw)==r['graph_partition'])
        adjacency=[[int([min(i,j),max(i,j)] in g['edges']) if i!=j else 0 for j in range(n)] for i in range(n)];A=[[F(i==j)-c**4*sum(adjacency[i][k]*adjacency[k][j] for k in range(n)) for j in range(n)] for i in range(n)];det=sum((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))*prod(A[i][p[i]] for i in range(n)) for p in permutations(range(n)));phy=r['physical'];lo,hi=phy['root_interval'];norm=c*c*max(map(sum,adjacency));ck(tag+'physical Gaussian conditional target',phy['status']=='valid' and norm==phy['pairing_norm_upper']<1 and det==phy['determinant'] and 0<lo<=hi and lo*lo<=det<=hi*hi and hi-lo<=F(1,1<<192) and phy['collision_free_probability']==[sum(gw)*lo,sum(gw)*hi] and sum(gw)*hi<=1)
        reward=rewards(states,edges,weights);ck(tag+'exact expected word rewards',reward==r['word_rewards']);ck(tag+'time grid',[q['steps'] for q in r['series']]==TIMES)
        for item,(t,v,den,attempt_words) in zip(r['series'],backward_and_forward(p,reward,TIMES)):
            law=item['law'];ck(tag+f'finite law shape {t}',len(law)==len(states) and sum(law)==1 and min(law)>=0)
            for j,(num,x) in enumerate(zip(v,law)):ck(tag+f'finite atom {t}:{j}',num*x.denominator==x.numerator*den)
            success,q=project(law,states,edges,n);delta=tv(q,target);ck(tag+f'backward reward {t}',attempt_words==item['attempt_words'] and attempt_words>=t);ck(tag+f'restart resources {t}',success>0 and success==item['success_mass'] and q==item['conditional_law'] and delta==item['conditional_TV'] and item['expected_steps']==F(t)/success and item['expected_words']==attempt_words/success);rows.append(dict(case=index,graph_index=job['graph_index'],graph=g['name'],new_holdout=g['new_holdout'],alpha=alpha,steps=t,states=len(states),conditional_TV=delta,success_mass=success,attempt_words=attempt_words,expected_steps=F(t)/success,expected_words=attempt_words/success))
        print('verified cost case',index,flush=True)
    ck('sources unchanged',all(digest(ROOT/n)==h for n,h in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==h for n,h in deps.items()));save(out/'candidates.json',rows);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=20,checkpoints=140,source_hashes=hashes,dependencies=deps,scope='Exact conditional graph laws and expected restart steps/words; independent set kernels and backward reward recursion.'))
if __name__=='__main__':main()
