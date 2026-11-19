"""Independent complete trajectory replay and binding to certified chosen laws."""
import random
from fractions import Fraction as F
from iteration67_support import ROOT,save,read,digest,snapshot
from iteration67_independent_recovery import replay
from iteration67_independent_recovery import move

def main():
    out=ROOT/'results/iteration67_samples_validation';hashes=snapshot(out);deps={};checks=[]
    def load(n):deps[n]=digest(ROOT/n);return read(ROOT/n)
    def ck(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name
    v=load('results/iteration67_selection_validation/validation.json');plans=load('results/iteration67_selection/plans.json');cv=load('results/iteration67_costs_recovery_validation/validation.json');s=load('results/iteration67_samples/summary.json');controls=load('results/iteration67_samples/controls.json');ck('prior certificates',v['count']==v['passed'] and cv['count']==cv['passed']);ck('sample input binding',all(deps[n]==s['dependencies'][n] for n in ('results/iteration67_selection_validation/validation.json','results/iteration67_selection/plans.json','results/iteration67_costs_recovery_validation/validation.json')));cases=[load(f'results/iteration67_costs/case_{i:02d}.json.gz') for i in range(20)];ck('case input binding',all(deps[n]==s['dependencies'][n] for n in deps if '/case_' in n));ck('21 controls',len(controls)==s['controls']==21);ck('edge control coverage',sorted(c['case'] for c in controls if c['kind']=='edge_tail')==list(range(20)));ck('acceptance control count',sum(c['kind']=='edge_and_acceptance_tail' for c in controls)==1);M=1<<64
    for i,c in enumerate(controls):
        case=cases[c['case']];edges=case['edges'];weights=case['weights'];m=len(edges);ck(f'control{i} edge tail exists',M%m>0)
        if c['kind']=='edge_tail':
            new,p=move(0,0,edges,weights);ck(f'control{i} exact forced words',c['start']==0 and c['path']==[0,1] and c['words']==[0,M-1,0]+([0] if p<1 else []))
        else:
            new,p=move(1,0,edges,weights);ck(f'control{i} rational acceptance tail',case['job']['graph']['name']=='triangle3' and case['job']['alpha']==36 and p==F(4,9) and M%p.denominator>0 and c['start']==1 and c['path']==[1,0] and c['words']==[0,M-1,0,M-1,0])
        ck(f'control{i} independent replay',replay(c['path'],c['words'],1,edges,weights,start=c['start']))
    requested={(p['index'],route) for p in plans for route in ('joint','fixed_alpha') if p[route] is not None};ck('route coverage',len(s['rows'])==len(requested)==v['routes']==s['routes'] and {(r['plan'],r['route']) for r in s['rows']}==requested);total_attempts=total_steps=total_words=total_outputs=0
    for row in s['rows']:
        item=load('results/iteration67_samples/'+row['file']);plan=plans[row['plan']];route=row['route'];chosen=plan[route];case=cases[chosen['case']];n=plan['graph']['n'];edges=case['edges'];weights=case['weights'];tag=f'plan{plan["index"]} {route} ';seed=670000+2*plan['index']+('joint','fixed_alpha').index(route);rng=random.Random(seed);ck(tag+'selection binding',item['plan']==plan['index'] and item['route']==route and item['chosen']==chosen and item['seed']==seed and chosen['conditional_TV']<=plan['epsilon']);point=next(q for q in case['series'] if q['steps']==chosen['steps']);ck(tag+'certified restart law and resources',all(chosen[k]==point[k] for k in ('conditional_TV','success_mass','attempt_words','expected_steps','expected_words')) and point['success_mass']>0);ck(tag+'seven public outputs',len(item['samples'])==row['outputs']==7);attempts=steps=words=0
        for si,sample in enumerate(item['samples']):
            ck(tag+f'sample{si} nonempty',sample['index']==si and len(sample['attempts'])>=1)
            for ai,a in enumerate(sample['attempts']):
                end=a['path'][-1];success=end.bit_count()==n;ck(tag+f'{si}:{ai} complete fixed attempt',len(a['path'])==chosen['steps']+1 and a['path'][0]==0 and a['success']==success==(ai==len(sample['attempts'])-1));ck(tag+f'{si}:{ai} independent replay',replay(a['path'],a['words'],chosen['steps'],edges,weights));ck(tag+f'{si}:{ai} seed provenance',a['words']==[rng.getrandbits(64) for _ in a['words']]);attempts+=1;steps+=chosen['steps'];words+=len(a['words'])
            occupied={v for e,(a,b) in enumerate(edges) if a<n and b<n and end&(1<<e) for v in (a,b)};output=sum(2**v for v in occupied);ck(tag+f'sample{si} projection',output==sample['output'] and case['target'][output]>0)
        ck(tag+'observed resource totals',attempts==row['attempts'] and steps==row['steps'] and words==row['words']);total_attempts+=attempts;total_steps+=steps;total_words+=words;total_outputs+=7;print('replayed',plan['index'],route,flush=True)
    ck('all totals',total_attempts==s['attempts'] and total_steps==s['steps'] and total_words==s['words'] and total_outputs==s['public_samples']);ck('sources unchanged',all(digest(ROOT/n)==h for n,h in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==h for n,h in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),routes=len(requested),public_samples=total_outputs,attempts=total_attempts,steps=total_steps,words=total_words,controls=21,source_hashes=hashes,dependencies=deps,scope='Exact independent-word restart law; public seeded traces replayed without a finite-PRNG distribution claim.'))
if __name__=='__main__':main()
