"""Uncapped independent restarts, public word streams, forced rejection controls."""
import random,time
from iteration67_support import ROOT,save,read,digest,snapshot,project_state
from gbslab.matching_chain import MatchingChain

def main():
    out=ROOT/'results/iteration67_samples';hashes=snapshot(out);deps={};rows=[];controls=[]
    def load(n):deps[n]=digest(ROOT/n);return read(ROOT/n)
    valid=load('results/iteration67_selection_validation/validation.json');assert valid['count']==valid['passed'];plans=load('results/iteration67_selection/plans.json');cv=load('results/iteration67_costs_recovery_validation/validation.json');assert cv['count']==cv['passed']
    cases=[load(f'results/iteration67_costs/case_{i:02d}.json.gz') for i in range(20)];M=1<<64
    for i,case in enumerate(cases):
        g=case['job']['graph'];chain=MatchingChain(2*g['n'],case['edges'],case['weights']);p=chain.proposal(0,0)[1];words=[0,M-1,0]+([0] if p<1 else []);cursor=0
        def supplied():
            nonlocal cursor
            x=words[cursor];cursor+=1;return x
        end=chain.step(0,supplied);assert cursor==len(words) and end==1;controls.append(dict(case=i,kind='edge_tail',start=0,path=[0,end],words=words))
        if g['name']=='triangle3' and case['job']['alpha']==36:
            words=[0,M-1,0,M-1,0];cursor=0;end=chain.step(1,supplied);assert cursor==len(words) and end==0;controls.append(dict(case=i,kind='edge_and_acceptance_tail',start=1,path=[1,end],words=words))
    save(out/'controls.json',controls)
    for plan in plans:
        for ri,route in enumerate(('joint','fixed_alpha')):
            chosen=plan[route]
            if chosen is None:continue
            started=time.perf_counter();case=cases[chosen['case']];g=plan['graph'];chain=MatchingChain(2*g['n'],case['edges'],case['weights']);seed=670000+2*plan['index']+ri;rng=random.Random(seed);samples=[];attempts_count=steps_count=words_count=0
            for sample in range(7):
                attempts=[]
                while True:
                    words=[];path=[0]
                    def word():
                        value=rng.getrandbits(64);words.append(value);return value
                    for _ in range(chosen['steps']):path.append(chain.step(path[-1],word))
                    success=path[-1].bit_count()==g['n'];attempts.append(dict(path=path,words=words,success=success));attempts_count+=1;steps_count+=chosen['steps'];words_count+=len(words)
                    if success:break
                samples.append(dict(index=sample,attempts=attempts,output=project_state(path[-1],case['edges'],g['n'])))
            filename=f'route_{plan["index"]:02d}_{route}.json.gz';save(out/filename,dict(plan=plan['index'],route=route,chosen=chosen,seed=seed,samples=samples));rows.append(dict(plan=plan['index'],route=route,file=filename,outputs=7,attempts=attempts_count,steps=steps_count,words=words_count,seconds=time.perf_counter()-started));print('sampled',plan['index'],route,attempts_count,flush=True)
    assert len(rows)==valid['routes'];assert all(digest(ROOT/n)==h for n,h in hashes.items()) and all(digest(ROOT/n)==h for n,h in deps.items());save(out/'summary.json',dict(completed=True,routes=len(rows),public_samples=7*len(rows),controls=len(controls),attempts=sum(r['attempts'] for r in rows),steps=sum(r['steps'] for r in rows),words=sum(r['words'] for r in rows),rows=rows,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
