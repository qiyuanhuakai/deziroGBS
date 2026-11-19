"""Execute fixed matching chains, exact finite-time laws and stopping audits."""
import random,time,sys
from pathlib import Path
from fractions import Fraction as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gbslab.matching_chain import MatchingChain
from iteration64_common import ROOT,save,digest,snapshot,panel,stationary,first_hit,propagate,project,tv

def main():
    out=ROOT/'results/iteration64_chain';hashes=snapshot(out);started=time.perf_counter();checkpoints=[1,2,4,8,16,32,64,128,256]
    save(out/'plan.json',dict(graphs=panel(),parameters=[F(1,4),F(1),F(2)],alpha_rules=['one','4n2'],checkpoints=checkpoints,public_steps=64,attempt_cap=64,outputs_per_case_route=7,seed_base=640000,source_hashes=hashes));summary=[]
    for g in panel():
        n=g['n'];edges=sorted([*g['edges'],*((a+n,b+n) for a,b in g['edges']),*((i,i+n) for i in range(n))])
        for c in (F(1,4),F(1),F(2)):
            for alpha in (1,4*n*n):
                index=len(summary);weights=[alpha*(c*c if (a<n)==(b<n) else 1) for a,b in edges];chain=MatchingChain(2*n,edges,weights);states=chain.states();w=[chain.weight(s) for s in states];pi=[v/sum(w) for v in w];perfect=[s.bit_count()==n for s in states];pmass,target=project(pi,states,edges,n);variants=[]
                for tag,bits in [('ideal',None),('naive8',8),('naive64',64)]:
                    p=chain.transition(states,bits=bits);rho=stationary(p);hit=first_hit(p,perfect);hm,hp=project(hit['law'],states,edges,n);series=propagate(p,checkpoints)
                    for item in series:
                        mass,q=project(item['law'],states,edges,n);delta=tv(item['law'],pi);item.update(perfect_mass=mass,projected=q,full_TV=delta,conditional_TV=None if q is None else tv(q,target),conditioning_bound=min(F(1),delta/pmass))
                    stationary_mass,stationary_projected=project(rho,states,edges,n)
                    variants.append(dict(name=tag,transition=p,stationary=rho,stationary_TV=tv(rho,pi),stationary_perfect_mass=stationary_mass,stationary_projected_TV=tv(stationary_projected,target),target_residual_TV=tv([sum(pi[i]*p[i][j] for i in range(len(pi))) for j in range(len(pi))],pi),first_hit=hit,first_hit_projected=hp,first_hit_TV=tv(hp,target),series=series))
                samples=[]
                for ri,mode in enumerate(('exact','naive')):
                    rng=random.Random(640000+10*index+ri);runs=[]
                    for sample in range(7):
                        attempts=[];output=None
                        for attempt in range(64):
                            words=[];path=[0]
                            def getword():
                                word=rng.getrandbits(64);words.append(word);return word
                            for t in range(64):path.append(chain.step(path[-1],getword,mode=mode))
                            attempts.append(dict(words=words,path=path))
                            if path[-1].bit_count()==n:output=path[-1];break
                        runs.append(dict(attempts=attempts,output=output))
                    sample_mass=variants[0 if mode=='exact' else 2]['series'][6]['perfect_mass']
                    samples.append(dict(mode=mode,seed=640000+10*index+ri,runs=runs,attempt_success_mass=sample_mass,failure_probability=(1-sample_mass)**64))
                record=dict(index=index,graph=g,n=n,c=c,alpha=alpha,edges=edges,weights=weights,states=states,target_stationary=pi,target=target,target_perfect_mass=pmass,variants=variants,samples=samples)
                save(out/f'case_{index:02d}.json.gz',record);summary.append(dict(index=index,graph=g['name'],c=c,alpha=alpha,states=len(states),public_successes=sum(r['output'] is not None for s in samples for r in s['runs']),public_attempts=sum(len(r['attempts']) for s in samples for r in s['runs'])))
                print('completed',index,g['name'],c,alpha,flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items())
    save(out/'summary.json',dict(planned=30,finished=len(summary),source_hashes=hashes,unchanged_sources=True,results=summary,elapsed_seconds=time.perf_counter()-started))

if __name__=='__main__':main()
