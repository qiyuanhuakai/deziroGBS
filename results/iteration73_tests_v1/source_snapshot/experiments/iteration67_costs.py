"""Exact resource laws for fixed-time independent-restart matching samplers."""
import time,traceback
from fractions import Fraction as F
from iteration67_support import ROOT,save,read,digest,snapshot,jobs,TIMES,EPSILONS,word_rewards,forward_rewards,tv
from iteration64_common import project
from gbslab.matching_chain import MatchingChain
from gbslab.graph_matching_gbs import subset_counts,gaussian_bridge

def main():
    out=ROOT/'results/iteration67_costs';hashes=snapshot(out);panel=jobs();save(out/'plan.json',dict(jobs=panel,checkpoints=TIMES,epsilons=EPSILONS,bits=64,source_hashes=hashes));summary=[]
    for index,job in enumerate(panel):
        start=time.perf_counter()
        try:
            g=job['graph'];n=g['n'];c=job['c'];alpha=job['alpha'];edges=sorted([*g['edges'],*((a+n,b+n) for a,b in g['edges']),*((i,i+n) for i in range(n))]);weights=[alpha*(c*c if (a<n)==(b<n) else 1) for a,b in edges];chain=MatchingChain(2*n,edges,weights);states=chain.states();raw=[chain.weight(s) for s in states];pi=[x/sum(raw) for x in raw];pm,target=project(pi,states,edges,n);counts=subset_counts(n,g['edges']);gw=[F(x*x)*c**(2*s.bit_count()) for s,x in enumerate(counts)];assert target==[x/sum(gw) for x in gw];physical=gaussian_bridge(n,g['edges'],c,sum(gw));assert physical['status']=='valid';p=chain.transition(states);rewards=word_rewards(chain,states);series=forward_rewards(p,rewards,TIMES)
            for r in series:
                mass,law=project(r['law'],states,edges,n);assert mass>0;r.update(success_mass=mass,conditional_law=law,conditional_TV=tv(law,target),expected_steps=F(r['steps'])/mass,expected_words=r['attempt_words']/mass)
            save(out/f'case_{index:02d}.json.gz',dict(index=index,job=job,edges=edges,weights=weights,states=states,stationary=pi,stationary_success_mass=pm,target=target,physical=physical,graph_partition=sum(gw),transition=p,word_rewards=rewards,series=series));row=dict(case=index,graph=g['name'],alpha=alpha,states=len(states),status='completed',seconds=time.perf_counter()-start)
        except Exception:row=dict(case=index,job=job,status='failed',traceback=traceback.format_exc());save(out/f'case_{index:02d}.failed.json',row)
        summary.append(row);print('costs',index,row['status'],flush=True)
    assert all(digest(ROOT/n)==h for n,h in hashes.items());save(out/'summary.json',dict(planned=20,finished=len(summary),completed=sum(r['status']=='completed' for r in summary),rows=summary,source_hashes=hashes,unchanged_sources=True))
if __name__=='__main__':main()
