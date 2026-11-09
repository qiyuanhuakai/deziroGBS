"""Independent replay and exact endpoints for the fixed five-configuration panel."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from verify_iteration34 import decode
from iteration39_independent import replay
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def binomial_tail(n,k,p):
    if k==0:return F(1)
    a,b=p.numerator,p.denominator;c=b-a;aa=[1];cc=[1]
    for _ in range(n):aa.append(aa[-1]*a);cc.append(cc[-1]*c)
    if k<=n//2:return 1-F(sum(math.comb(n,j)*aa[j]*cc[n-j] for j in range(k)),b**n)
    return F(sum(math.comb(n,j)*aa[j]*cc[n-j] for j in range(k,n+1)),b**n)


def mixture(values,m):
    stakes=[F(1,1<<j) for j in range(6,-1,-1)]
    return sum((math.prod((1-t+t*x/m for x in values),start=F(1)) for t in stakes),start=F(0))/7


def main():
    out=ROOT/'results/iteration40_validation';out.mkdir(exist_ok=False);names=['experiments/verify_iteration40.py','experiments/iteration39_independent.py','experiments/verify_iteration34.py','experiments/iteration22_external.py'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:shutil.copy2(ROOT/n,out/Path(n).name)
    read=lambda p:decode(json.loads(Path(p).read_text()));checks=[];deps={};steps=product_roots=mixture_roots=quantiles=0
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(name)
    primary=np.load(ROOT/'results/iteration35_panel/inputs.npz')['shifted'];mom=read(ROOT/'results/iteration36_recovery/case_11.audit.json')['moments'];mu=mom['mean_interval'];e2=mom['second_moment_interval'];var=mom['variance_interval'];targets={'X':tuple(x/144 for x in mu),'Y':((e2[0]-144*mu[1]+5184)/5184,(e2[1]-144*mu[0]+5184)/5184)};transfer=read(ROOT/'results/iteration35_panel/case_07.audit.json')['transfers']['-40']['all_measurement_TV_transfer_upper'];alpha=F(1,5700);delta=F(1,1<<40);tails={}
    for k in (40,50,60,70,80,90,100):
        candidates=[F(1),mu[1]/k,var[1]/(var[1]+(k-mu[1])**2) if k>mu[1] else F(1)]
        for a in range(-1,k-1):
            den=(k-a)*(k-a-1);check(f'quadratic majorant {k}:{a}',all((n-a)*(n-a-1)>=int(n>=k)*den for n in range(145)));coefficient=-(2*a+1);candidates.append((e2[1]+coefficient*(mu[1] if coefficient>=0 else mu[0])+a*(a+1))/den)
        tails[k]=min(candidates)
    for label,expected in (('frugal',3),('recent',1),('conditional_mi',1)):
        folder=ROOT/f'results/iteration40_{label}';plan=read(folder/'plan.json');summary=read(folder/'summary.json');rows=read(folder/'rows.json');bounds=read(folder/'target_bounds.json');check(label+' completion',summary['planned']==summary['finished']==summary['completed']==len(rows)==expected and summary['unchanged_sources'] and summary['unchanged_dependencies']);check(label+' target identity',bounds['moments']==mom and all(tuple(bounds['targets'][k])==v for k,v in targets.items()) and bounds['transfer_upper']==transfer and all(bounds['tail_bounds'][str(k)]==v for k,v in tails.items()));check(label+' alpha and stakes',plan['alpha_per_tail']==alpha and plan['total_tails']==57 and plan['familywise_alpha']==57*alpha==F(1,100) and plan['stakes']==[F(1,1<<j) for j in range(6,-1,-1)])
        for n,sha in plan['source_hashes'].items():check(label+' source '+n,digest(ROOT/n)==sha);check(label+' snapshot '+n,digest(folder/'source_snapshot'/n)==sha)
        for n,sha in plan['dependencies'].items():check(label+' dependency '+n,digest(ROOT/n)==sha);deps[n]=sha
        if label!='frugal':
            U=np.load(folder/'uniforms.npz')['uniforms'];check(label+' common fixed stream',np.array_equal(U,np.random.default_rng(401440).random((256,144))))
        for i,job in enumerate(plan['jobs']):
            row=rows[i];a=read(folder/f'case_{i:02d}.audit.json');check(f'{label}:{i} fixed job',job==dict(method='frugal',modes=144,cutoff=(1,2,4)[i],attempts=256,seed=401440) if label=='frugal' else job==dict(method='forest',modes=144,selection=label,forest='static_mi_tree',budget=4,attempts=256,seed=401440));counts=[];case_steps=0
            if label=='frugal':
                raw=read(folder/f'case_{i:02d}.stdout.json');check(f'frugal {i} input',np.array_equal(primary,np.loadtxt(folder/f'case_{i:02d}.csv',delimiter=',')));check(f'frugal {i} metadata',raw['modes']==144 and raw['samples']==256 and raw['seed']==401440 and raw['cutoff']==job['cutoff'] and raw['threads']==1);case_steps=replay(raw);check(f'frugal {i} replay count',case_steps==a['replayed_steps']==256*144);counts=[s.count('1') for s in raw['outputs']]
            else:
                outputs=np.load(folder/f'case_{i:02d}.npz')['outputs'];check(label+' attempts',len(a['attempts'])==256)
                for j,attempt in enumerate(a['attempts']):
                    r=read(folder/f'attempts/case_{i:02d}/attempt_{j:03d}.json');check(f'{label} attempt {j} summary',all(r[k]==v for k,v in attempt.items()));prefix='';failed=False;consumed=0
                    for q in r['conditionals']:
                        check(f'{label} prefix {j}:{consumed}',q['prefix']==prefix)
                        if q['q_hex'] is None:failed=True;break
                        probability=float.fromhex(q['q_hex']);check(f'{label} range {j}:{consumed}',0<=probability<=1);prefix+=str(int(U[j,consumed]<probability));consumed+=1
                    case_steps+=len(r['conditionals']);check(f'{label} outcome {j}',consumed==r['uniforms_consumed'] and len(r['conditionals'])==r['conditional_attempts'] and ((failed and r['outcome']=='numerical_failure' and r['failed_prefix']==prefix and np.all(outputs[j]==-1)) or (not failed and consumed==144 and r['outcome']=='completed' and r['clicks']==prefix.count('1') and r['output']==list(map(int,prefix)) and np.array_equal(outputs[j],r['output']))));counts.append(None if failed else prefix.count('1'))
                check(label+' native accounting',a['native_diagnostics']['probability_calls']<=case_steps==a['conditional_attempts'])
            steps+=case_steps;check(f'{label}:{i} all outcomes retained',len(counts)==256 and row['completed_attempts']==sum(n is not None for n in counts) and row['failed_attempts']==sum(n is None for n in counts));lowers=[]
            for name in ('X','Y'):
                values=[F(0) if n is None else F(n,144) if name=='X' else F((n-72)**2,5184) for n in counts];r=a['intervals'][name];forward=math.prod(values,start=F(1));reverse=math.prod((1-x for x in values),start=F(1));check(f'{label}:{i}:{name} products',r['forward_product']==forward and r['reverse_product']==reverse and r['attempts']==256 and r['alpha_per_tail']==alpha and r['bits']==40);pairs=[('upper',1-r['upper'],reverse)]
                if name=='X':pairs.append(('lower',r['lower'],forward))
                else:
                    check(f'{label}:{i} fixed mixture',r['lower_method']=='fixed_uniform_betting_mixture' and r['stakes']==plan['stakes'] and r['weights']==[F(1,7)]*7);q=r['lower'];check(f'{label}:{i} mixture inverse',0<=q<1 and (q*(1<<40)).denominator==1 and (q==0 or mixture(values,q)>=1/alpha) and mixture(values,q+delta)<1/alpha);mixture_roots+=1;diagnostic=r['product_lower_diagnostic'];check(f'{label}:{i} retained product diagnostic',diagnostic**256<=alpha*forward<(diagnostic+delta)**256)
                for tag,q,value in pairs:check(f'{label}:{i}:{name}:{tag} root',0<=q<=1 and (q*(1<<40)).denominator==1 and q**256<=alpha*value<(q+delta)**256);product_roots+=1
                lower=max(F(0),r['lower']-targets[name][1],targets[name][0]-r['upper']);check(f'{label}:{i}:{name} TV',a['observable_TV_lower'][name]==lower);lowers.append(lower)
            event_family=[*tails,*(['failure'] if label!='frugal' else [])];check(f'{label}:{i} event family',[e['event'] for e in a['events']]==event_family)
            for event in a['events']:
                k=event['event'];count=sum(n is None for n in counts) if k=='failure' else sum(n is not None and n>=k for n in counts);q=event['probability_lower'];check(f'{label}:{i}:{k} counts',event['attempts']==256 and event['successes']==count);check(f'{label}:{i}:{k} inverse',q==0 if count==0 else 0<=q<1 and (q*(1<<40)).denominator==1 and binomial_tail(256,count,q)<=alpha<binomial_tail(256,count,q+delta));quantiles+=1;upper=F(0) if k=='failure' else tails[k];lower=max(F(0),q-upper);check(f'{label}:{i}:{k} TV',event['target_upper']==upper and event['TV_lower']==lower);lowers.append(lower)
            lower=max(lowers);check(f'{label}:{i} total and transfer',a['full_TV_statistical_lower']==lower and a['shifted_target_TV_statistical_lower']==max(F(0),lower-transfer));print('verified confirmation',label,i,flush=True)
    pilot=ROOT/'results/iteration40_betting_pilot';p=read(pilot/'summary.json');check('exploratory pilot scope',p['exploratory_only'] and p['rows']==16 and p['unchanged_sources'])
    for n,sha in p['source_hashes'].items():check('pilot source '+n,digest(ROOT/n)==sha);check('pilot snapshot '+n,digest(pilot/'source_snapshot'/n)==sha)
    for n,sha in p['dependencies'].items():check('pilot dependency '+n,digest(ROOT/n)==sha);deps[n]=sha
    check('all formal confidence tails',product_roots+mixture_roots+quantiles==57);check('verifier unchanged',hashes=={n:digest(ROOT/n) for n in names});save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),native_steps_replayed=steps,formal_product_root_checks=product_roots,formal_mixture_root_checks=mixture_roots,formal_binomial_quantiles=quantiles,source_hashes=hashes,dependencies=deps,scope='99% simultaneous confidence under iid attempts; internal Frugal saturation is not a failure event; exploratory old-cohort pilot is not new confirmation'));print('verified',len(checks),flush=True)
if __name__=='__main__':main()
