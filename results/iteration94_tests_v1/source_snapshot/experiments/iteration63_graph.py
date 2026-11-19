"""Execute the frozen exact graph construction panel and public word samples."""
import random, sys, time
from fractions import Fraction as F
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gbslab.graph_matching_gbs import subset_counts,lifted_structure,partition_by_size,perfect_mass,least_half_multiplier,categorical,draw_word,gaussian_bridge
from iteration63_common import ROOT,digest,save,snapshot,panel,tv

def main():
    out=ROOT/'results/iteration63_graph'; hashes=snapshot(out); started=time.perf_counter()
    plan=dict(graphs=panel(),parameters=[F(1,4),F(1),F(2)],bits=64,samples_per_case=17,seed_base=630000,source_hashes=hashes)
    save(out/'plan.json',plan); results=[]
    for g in plan['graphs']:
        n=g['n']; pm=subset_counts(n,g['edges']); structure=lifted_structure(n,g['edges']); witnesses=structure['perfect_matchings']
        save(out/(g['name']+'_structure.json.gz'),dict(**g,subset_counts=pm,**structure))
        for c in plan['parameters']:
            index=len(results); weights=[c**(2*s.bit_count())*p*p for s,p in enumerate(pm)]; zgraph=sum(weights); target=[w/zgraph for w in weights]
            z=partition_by_size(n,structure['coefficients'],c); alpha=least_half_multiplier(z); routes=[]
            rng=random.Random(plan['seed_base']+index); words=[rng.getrandbits(64) for _ in range(17)]
            for name,order in [('direct',None),('natural',list(range(len(witnesses)))),('grouped',sorted(range(len(witnesses)),key=lambda j:(witnesses[j]['subset'],j)))]:
                labels=list(range(1<<n)) if order is None else [witnesses[j]['subset'] for j in order]
                p=target if order is None else [c**(2*s.bit_count())/zgraph for s in labels]
                table=categorical(p); counts=[0]*(1<<n)
                for s,count in zip(labels,table['counts']): counts[s]+=count
                law=[F(k,1<<64) for k in counts]; drawn=[draw_word(table,w) for w in words]
                samples=[dict(word=w,index=j,subset=labels[j],witness=None if order is None else order[j]) for w,j in zip(words,drawn)]
                routes.append(dict(name=name,order=order,word_counts=table['counts'],projected_counts=counts,TV=tv(law,target),support_bound=F(sum(x>0 for x in p)-1,1<<64),samples=samples))
            record=dict(index=index,graph=g['name'],n=n,c=c,target=target,graph_partition=zgraph,partitions=z,near_perfect_ratio=z[-2]/z[-1],unweighted_mass=perfect_mass(z,1),universal_alpha=4*n*n,universal_mass=perfect_mass(z,4*n*n),least_alpha=alpha,least_mass=perfect_mass(z,alpha),previous_mass=None if alpha==1 else perfect_mass(z,alpha-1),bridge=gaussian_bridge(n,g['edges'],c,zgraph),seed=plan['seed_base']+index,routes=routes,natural_direct_TV=tv([F(x,1<<64) for x in routes[0]['projected_counts']],[F(x,1<<64) for x in routes[1]['projected_counts']]))
            save(out/f'case_{index:02d}.json.gz',record); results.append(dict(index=index,graph=g['name'],c=c,perfect_matchings=len(witnesses),least_alpha=alpha,universal_mass=record['universal_mass'],natural_direct_TV=record['natural_direct_TV'],physical_status=record['bridge']['status']))
            print(f'case {index:02d} {g["name"]} c={c} completed',flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items())
    save(out/'summary.json',dict(planned=30,finished=len(results),source_hashes=hashes,unchanged_sources=True,results=results,elapsed_seconds=time.perf_counter()-started,public_outputs=30*3*17))

if __name__=='__main__': main()
