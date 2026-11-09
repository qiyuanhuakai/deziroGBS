"""Independent depth-first rational reconstruction; does not import coupling_audit."""
import json,shutil
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from iteration22_external import digest,save
ROOT=Path(__file__).resolve().parents[1]

def decode(value):
    if isinstance(value,dict):
        if set(value)=={'rational'}:return F(value['rational'])
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(v) for v in value]
    return value

def reconstruct(target,table,bits=None,allowed=None):
    m=len(target).bit_length()-1;law=[F(0)]*len(target);failure=F(0);confirmed=F(0);unresolved=[];nodes={};levels=[F(0)]*m;hp=F(0);hq=F(0)
    def visit(prefix,qmass,common):
        nonlocal failure,confirmed,hp,hq
        length=len(prefix)
        if length==m:law[sum(b<<j for j,b in enumerate(prefix))]=qmass;confirmed+=common;return
        matching=[mass for mask,mass in enumerate(target) if all((mask>>j)&1==b for j,b in enumerate(prefix))];pmass=sum(matching,F(0))
        if allowed is not None and prefix not in allowed:unresolved.append((prefix,pmass,qmass,common));return
        raw=table[prefix];node=dict(target_mass=pmass,proxy_mass=qmass,matched_mass=common);nodes[prefix]=node
        if raw is None:
            failure+=qmass;levels[length]+=common;hp+=pmass;hq+=qmass;node.update(status='failure',local_disagreement=F(1),charged_loss=common);return
        q=raw if bits is None else F(-((-raw.numerator*(1<<bits))//raw.denominator),1<<bits)
        p1=sum((mass for mask,mass in enumerate(target) if (mask>>length)&1 and all((mask>>j)&1==b for j,b in enumerate(prefix))),F(0))/pmass if pmass else q
        disagreement=abs(p1-q);hp+=pmass*disagreement;hq+=qmass*disagreement;levels[length]+=common*disagreement
        node.update(status='conditional',target_conditional=p1,raw_proxy_conditional=raw,proxy_conditional=q,local_disagreement=disagreement,charged_loss=common*disagreement)
        for bit,pp,qq in ((0,1-p1,1-q),(1,p1,q)):
            if qq:visit(prefix+(bit,),qmass*qq,common*min(pp,qq))
    visit((),F(1),F(1));unresolved_q=sum((x[2] for x in unresolved),F(0));unresolved_c=sum((x[3] for x in unresolved),F(0));complete=not unresolved
    tv=(sum(abs(p-q) for p,q in zip(target,law))+failure)/2 if complete else None
    return dict(complete=complete,queries=len(nodes),distribution=law,failure_mass_seen=failure,unresolved_proxy_mass=unresolved_q,confirmed_matched_mass=confirmed,matched_failure_or_disagreement=sum(levels),unresolved_matched_mass=unresolved_c,coupling_upper=1-confirmed,extended_tv=tv,hybrid_target_upper=min(F(1),hp) if complete else None,hybrid_proxy_upper=min(F(1),hq) if complete else None,level_losses=levels,nodes=nodes,frontier=unresolved)

def main():
    out=ROOT/'results/iteration33_validation';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);checks=[]
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
    def check(name,passed,**values):
        checks.append(dict(name=name,passed=bool(passed),**values))
        if not passed:save(out/'checks.failed.json',checks);raise AssertionError(name)
    panel=ROOT/'results/iteration33_panel';plan=read(panel/'plan.json');summary=read(panel/'summary.json');rows=read(panel/'rows.json');dependencies=dict(plan['dependencies'])
    check('formal jobs',summary['planned']==summary['finished']==summary['completed']==38 and len(rows)==38 and summary['unchanged_sources'])
    for name,value in dependencies.items():check('dependency '+name,digest(ROOT/name)==value)
    for name,value in plan['source_hashes'].items():check('source '+name,digest(ROOT/name)==value);check('snapshot '+name,digest(panel/'source_snapshot'/name)==value)
    build=read(ROOT/'results/iteration31_build/build.json')
    for name,value in build['source_hashes'].items():check('native binding '+name,digest(ROOT/name)==value);dependencies[name]=value
    max_old_tv_change=0.;replay_differences=[];native_failures=[];max_grid=F(0);maximum_normalization=F(0);all_records=0
    for i,row in enumerate(rows):
        data=decode(read(panel/f'case_{i:02d}.audit.json'));target=data['target'];m=row['modes'];table={tuple(r['prefix']):r['q'] for r in data['oracle']};check(f'{i} normalized target',sum(target)==1 and min(target)>=0)
        for r in data['oracle']:
            if r['q'] is not None:check(f'{i} binary64 input '+str(r['prefix']),F(float.fromhex(r['hex']))==r['q'])
            else:check(f'{i} failure message '+str(r['prefix']),bool(r['error']))
        if i<24:
            old=np.load(ROOT/f"results/iteration32_panel/case_{row['source_case']:02d}.npz");raw=[F(float(x)) for x in old['target']];total=sum(raw);check(f'{i} explicit target normalization',data['raw_target']==raw and data['original_target_mass']==total and target==[p/total for p in raw]);maximum_normalization=max(maximum_normalization,abs(total-1))
            for prefix,q in table.items():check(f'{i} saved condition '+str(prefix),q==F(float(old['conditionals'][(1<<len(prefix))-1+sum(b<<j for j,b in enumerate(prefix))])))
        else:
            V=np.load(panel/f'case_{i:02d}.npz')['covariance'];v=F(float(V[0,0]));q=(v-1)/(v+1);check(f'{i} exact analytic target',np.array_equal(V,float(v)*np.eye(2*m)) and target==[q**mask.bit_count()*(1-q)**(m-mask.bit_count()) for mask in range(1<<m)])
        independent=[]
        for key,bits in (('ideal',None),('grid53',53)):
            stored=data[key];expected=reconstruct(target,table,bits);independent.append(expected);all_records+=len(expected['nodes'])
            for name,value in expected.items():
                if name not in ('nodes','frontier'):check(f'{i} {key} '+name,stored[name]==value)
            check(f'{i} {key} exact bounds',expected['failure_mass_seen']<=expected['extended_tv']<=expected['coupling_upper']<=min(expected['hybrid_target_upper'],expected['hybrid_proxy_upper']))
            check(f'{i} {key} node coverage',{tuple(r['prefix']) for r in stored['records']}==set(expected['nodes']))
            frontier={():F(1)}
            for r in stored['records']:
                prefix=tuple(r['prefix']);node=expected['nodes'][prefix];chosen=min(frontier,key=lambda h:(-frontier[h],h));check(f'{i} {key} best first '+str(prefix),chosen==prefix);frontier.pop(prefix)
                check(f'{i} {key} node '+str(prefix),all(r[name]==value for name,value in node.items()))
                if node['status']=='conditional':
                    p=node['target_conditional'];q=node['proxy_conditional'];c=node['matched_mass']
                    if len(prefix)+1<m:
                        if q<1:frontier[prefix+(0,)]=c*min(1-p,1-q)
                        if q>0:frontier[prefix+(1,)]=c*min(p,q)
            previous=F(1)
            for partial in (r for r in data['partials'] if r['uniform_bits']==bits):
                allowed={tuple(r['prefix']) for r in stored['records'][:partial['budget']]};computed=reconstruct(target,table,bits,allowed)
                for name,value in computed.items():
                    if name not in ('nodes','frontier','distribution'):check(f'{i} {key} partial {partial["budget"]} '+name,partial[name]==value)
                check(f'{i} {key} partial monotonic {partial["budget"]}',expected['extended_tv']<=partial['coupling_upper']<=previous);previous=partial['coupling_upper']
                check(f'{i} {key} partial conservation {partial["budget"]}',computed['matched_failure_or_disagreement']+computed['confirmed_matched_mass']+computed['unresolved_matched_mass']==1 and sum(computed['distribution'])+computed['failure_mass_seen']+computed['unresolved_proxy_mass']==1)
        ideal,grid=independent;shift=(sum(abs(p-q) for p,q in zip(ideal['distribution'],grid['distribution']))+abs(ideal['failure_mass_seen']-grid['failure_mass_seen']))/2;max_grid=max(max_grid,shift)
        check(f'{i} grid-law TV',shift==data['grid_law_tv'] and shift<=F(m,1<<53))
        check(f'{i} summary metrics',all(float(ideal[k])==v for k,v in row['metrics'].items()))
        if i<24:
            oldtv=.5*sum(abs(old['target']-old['law']));change=abs(float(ideal['extended_tv'])-oldtv);max_old_tv_change=max(max_old_tv_change,change);check(f'{i} old floating law agreement',max(abs(float(a)-b) for a,b in zip(ideal['distribution'],old['law']))<2e-15 and change<2e-15)
        else:
            replay_table={tuple(r['prefix']):r['q'] for r in data['replay_oracle']};replay=reconstruct(target,replay_table);replay_tv=(sum(abs(p-q) for p,q in zip(ideal['distribution'],replay['distribution']))+abs(ideal['failure_mass_seen']-replay['failure_mass_seen']))/2;identical=table==replay_table
            check(f'{i} alternate query-order law',float(replay_tv)==row['replay_extended_law_tv'] and identical==row['replay_table_identical']);replay_differences.append(dict(case=i,identical=identical,tv=float(replay_tv)))
            check(f'{i} native call accounting',row['native_diagnostics']['probability_calls']==len(table) and row['replay_native_diagnostics']['probability_calls']==len(replay_table))
            native_failures.append(dict(case=i,**data['job'],failure_nodes=row['failure_nodes'],failure_mass=float(ideal['failure_mass_seen']),extended_tv=float(ideal['extended_tv']),coupling_upper=float(ideal['coupling_upper'])))
    check('verifier unchanged',digest(source)==sha);save(out/'checks.json',checks);result=dict(count=len(checks),passed=sum(c['passed'] for c in checks),dependencies=dependencies,verifier_sha256=sha,records_verified=all_records,maximum_old_tv_change=max_old_tv_change,maximum_absolute_original_target_mass_error=float(maximum_normalization),maximum_grid_law_tv=float(max_grid),native_replay=replay_differences,native_failures=native_failures);save(out/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('dependencies','native_failures')},indent=2))
if __name__=='__main__':main()
