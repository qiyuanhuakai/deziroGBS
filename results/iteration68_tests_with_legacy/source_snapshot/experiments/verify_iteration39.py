"""Independent rounding-cell laws, native RNG replay and exact witness endpoints."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,math,subprocess
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from verify_iteration34 import decode
from iteration39_independent import reconstruct,replay,cdf_count
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def direct_tail(n,k,p):
    if not k:return F(1)
    a,b=p.numerator,p.denominator;c=b-a;aa=[1];cc=[1]
    for _ in range(n):aa.append(aa[-1]*a);cc.append(cc[-1]*c)
    if k<=n//2:return 1-F(sum(math.comb(n,j)*aa[j]*cc[n-j] for j in range(k)),b**n)
    return F(sum(math.comb(n,j)*aa[j]*cc[n-j] for j in range(k,n+1)),b**n)


def main():
    out=ROOT/'results/iteration39_validation';out.mkdir(exist_ok=False);names=['experiments/verify_iteration39.py','experiments/iteration39_independent.py','experiments/verify_iteration34.py','experiments/iteration22_external.py'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:shutil.copy2(ROOT/n,out/Path(n).name)
    read=lambda p:decode(json.loads(Path(p).read_text(encoding='utf-8-sig')));checks=[];deps={};steps=roots=quantiles=prefixes=0
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(name)
    def bind(folder,record,snapshot='source_snapshot'):
        for n,sha in record.get('source_hashes',{}).items():check(folder.name+' source '+n,digest(ROOT/n)==sha);check(folder.name+' snapshot '+n,digest(folder/snapshot/n)==sha)
        for n,sha in record.get('dependencies',{}).items():check(folder.name+' dependency '+n,digest(ROOT/n)==sha);deps[n]=sha
    for label,count in (('global',4),('local',15)):
        folder=ROOT/f'results/iteration39_{label}';s=read(folder/'summary.json');p=read(folder/'plan.json');r=read(folder/'rows.json');check(label+' completed',s['planned']==s['finished']==s['completed']==len(r)==count and s['unchanged_sources'] and s['unchanged_dependencies'] and all(x['status']=='completed' for x in r));bind(folder,p)
    builddir=ROOT/'results/iteration39_build';build=read(builddir/'build.json');bind(builddir,build);check('build unchanged and pinned',build['returncode']==0 and build['unchanged_sources'] and build['upstream_commit']=='fa8b796f2afebd4845a2e335e457735952617ef8');check('binary bound',digest(builddir/'frugal_extended.exe')==build['executable_sha256'])
    for n,sha in build['runtime_files'].items():check('runtime '+n,digest(builddir/n)==sha)
    for label in ('pilot','calibration'):
        folder=ROOT/f'results/iteration39_{label}';p=read(folder/('plan.json' if label=='pilot' else 'validation.json'));bind(folder,p)
    cal=read(ROOT/'results/iteration39_calibration/validation.json');check('calibration passed',cal['count']==cal['passed']==7516 and cal['pilot_cases']==19 and cal['native_steps_replayed']==7146)
    primary=np.load(ROOT/'results/iteration35_panel/inputs.npz')['shifted'];full=ROOT/'results/iteration39_global';plan=read(full/'plan.json');bounds=read(full/'target_bounds.json');mom=read(ROOT/'results/iteration36_recovery/case_11.audit.json')['moments'];mu=mom['mean_interval'];e2=mom['second_moment_interval'];var=mom['variance_interval'];target={'X':tuple(x/144 for x in mu),'Y':((e2[0]-134*mu[1]+4489)/5929,(e2[1]-134*mu[0]+4489)/5929)};transfer=read(ROOT/'results/iteration35_panel/case_07.audit.json')['transfers']['-40']['all_measurement_TV_transfer_upper'];check('physical moment binding',bounds['moments']==mom and all(tuple(bounds[k])==v for k,v in target.items()) and bounds['transfer_upper']==transfer);check('preallocated simultaneous confidence',plan['mean_alpha']==F(1,2000) and plan['event_alpha']==F(1,14000) and plan['familywise_alpha']==16*F(1,2000)+28*F(1,14000)==F(1,100));tail_bounds={}
    for k in (40,50,60,70,80,90,100):
        candidates=[F(1),mu[1]/k,var[1]/(var[1]+(k-mu[1])**2) if k>mu[1] else F(1)]
        for a in range(-1,k-1):
            denominator=(k-a)*(k-a-1);check(f'integer majorant {k}:{a}',all((n-a)*(n-a-1)>=int(n>=k)*denominator for n in range(145)));coefficient=-(2*a+1);candidates.append((e2[1]+coefficient*(mu[1] if coefficient>=0 else mu[0])+a*(a+1))/denominator)
        tail_bounds[k]=min(candidates);check(f'target tail {k}',bounds['tail_bounds'][str(k)]==tail_bounds[k])
    delta=F(1,1<<40)
    for i,job in enumerate(plan['jobs']):
        check(f'global {i} fixed job',job==dict(modes=144,cutoff=i+1,samples=1024,seed=391440));rec=read(full/f'case_{i:02d}.stdout.json');audit=read(full/f'case_{i:02d}.audit.json');V=np.load(full/f'case_{i:02d}.npz')['covariance'];check(f'global {i} input identity',np.array_equal(V,primary) and np.array_equal(np.loadtxt(full/f'case_{i:02d}.csv',delimiter=','),primary));check(f'global {i} metadata',all(rec[k]==v for k,v in job.items()) and rec['threads']==1 and rec['official_sampler_unchanged']);n=replay(rec);steps+=n;check(f'global {i} replay count',n==audit['replayed_steps']==1024*144);counts=[s.count('1') for s in rec['outputs']];lower=[]
        for name in ('X','Y'):
            values=[F(n,144) if name=='X' else F((n-67)**2,5929) for n in counts];m=audit['intervals'][name];forward=math.prod(values,start=F(1));reverse=math.prod((1-x for x in values),start=F(1));check(f'global {i}:{name} products',m['forward_product']==forward and m['reverse_product']==reverse and m['attempts']==1024 and m['alpha_per_tail']==F(1,2000) and m['bits']==40)
            for tag,q,power in (('lower',m['lower'],forward/2000),('upper-complement',1-m['upper'],reverse/2000)):
                check(f'global {i}:{name}:{tag} root',0<=q<=1 and (q*(1<<40)).denominator==1 and q**1024<=power<(q+delta)**1024);roots+=1
            v=max(F(0),m['lower']-target[name][1],target[name][0]-m['upper']);check(f'global {i}:{name} TV lower',audit['observable_TV_lower'][name]==v);lower.append(v)
        check(f'global {i} event family',[e['threshold'] for e in audit['events']]==list(tail_bounds))
        for e in audit['events']:
            k=e['threshold'];count=sum(n>=k for n in counts);q=e['sampler_probability_lower'];check(f'global {i}:{k} count',count==e['successes']);check(f'global {i}:{k} inverse tail',q==0 if count==0 else 0<=q<1 and (q*(1<<40)).denominator==1 and direct_tail(1024,count,q)<=F(1,14000)<direct_tail(1024,count,q+delta));quantiles+=1;v=max(F(0),q-tail_bounds[k]);check(f'global {i}:{k} TV lower',e['target_probability_upper']==tail_bounds[k] and e['TV_lower']==v);lower.append(v)
        v=max(lower);check(f'global {i} full and shifted bounds',audit['full_TV_statistical_lower']==v and audit['shifted_target_TV_statistical_lower']==max(F(0),v-transfer));print('verified global',i,flush=True)
    local=ROOT/'results/iteration39_local';lp=read(local/'plan.json');targets={s:read(ROOT/f'results/iteration37_local/target_{s:03d}.json')['law']['intervals'] for s in (0,54,134)}
    for i,job in enumerate(lp['jobs']):
        check(f'local {i} fixed job',job==dict(modes=10,start=(0,54,134)[i//5],cutoff=i%5+1,samples=257,seed=391001+i));rec=read(local/f'case_{i:02d}.stdout.json');audit=read(local/f'case_{i:02d}.audit.json');old=read(local/f'case_{i:02d}.legacy.json');rev=read(local/f'case_{i:02d}.reverse.json');s=job['start'];ix=list(range(s,s+10))+list(range(s+144,s+154));V=np.load(local/f'case_{i:02d}.npz')['covariance'];check(f'local {i} input identity',np.array_equal(V,primary[np.ix_(ix,ix)]) and np.array_equal(V,np.loadtxt(local/f'case_{i:02d}.csv',delimiter=',')));check(f'local {i} metadata',all(rec[k]==v for k,v in job.items() if k!='start') and old['seed']==job['seed'] and old['sample_count']==257 and old['cutoff']==job['cutoff'] and old['omp_threads']==1 and old['modes']==10);n=replay(rec);steps+=n;check(f'local {i} replay count',n==audit['replayed_steps']==2570);table={r['prefix']:r['threshold_hex'] for r in rec['prefix_tree']};check(f'local {i} reverse complete tree',{r['prefix']:r for r in rec['prefix_tree']}=={r['prefix']:r for r in rev['prefix_tree']} and len(table)==1023);hist=[0]*1024
        for bits in rec['outputs']:hist[sum((b=='1')<<j for j,b in enumerate(bits))]+=1
        check(f'local {i} legacy outputs',hist==old['histogram_counts'] and audit['legacy_histogram_equal']);laws={}
        for model in ('continuous','uint64'):
            law=reconstruct(10,table,model);laws[model]=law;check(f'local {i}:{model} independent law',law==audit['laws'][model]['distribution'] and sum(law)==1 and all(q>=0 for q in law));lo=sum(max(F(0),a-q,q-b) for q,(a,b) in zip(law,targets[s]))/2;hi=sum(max(abs(q-a),abs(q-b)) for q,(a,b) in zip(law,targets[s]))/2;check(f'local {i}:{model} physical TV interval',audit['TV_intervals'][model]==[lo,hi]);reachable_bad=[];reachable_sat=[]
            for prefix,qhex in table.items():
                mass=F(1)
                for depth,bit in enumerate(prefix):
                    q=float.fromhex(table[prefix[:depth]]);z=F(cdf_count(q),1<<64) if model=='uint64' else F(1) if math.isnan(q) or q>=1 else F(0) if q<0 else F(q);mass*=z if bit=='0' else 1-z
                q=float.fromhex(qhex)
                if mass and not math.isfinite(q):reachable_bad.append(prefix)
                if mass and (q<0 or q>1):reachable_sat.append(prefix)
            check(f'local {i}:{model} reachable diagnostics',set(reachable_bad)==set(audit['laws'][model]['reachable_nonfinite_prefixes']) and set(reachable_sat)==set(audit['laws'][model]['reachable_saturated_prefixes']))
        distance=sum(abs(a-b) for a,b in zip(laws['continuous'],laws['uint64']))/2;bound=10*(F(1,1<<53)+F(1,1<<64));check(f'local {i} uniform coupling',distance==audit['uniform_model_distance'] and distance<=audit['uniform_model_transfer_bound']==bound);legacy=list(map(F,old['distribution']));check(f'local {i} legacy floating diagnostic',audit['legacy_half_L1']==sum(abs(a-b) for a,b in zip(legacy,laws['continuous']))/2 and audit['legacy_normalization_defect']==sum(legacy)-1);prefixes+=1023;print('verified local',i,flush=True)
    check('verifier sources unchanged',hashes=={n:digest(ROOT/n) for n in names});save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),native_steps_replayed=steps,exact_mean_root_checks=roots,exact_binomial_quantiles=quantiles,complete_local_prefixes=prefixes,source_hashes=hashes,dependencies=deps,scope='Conditional independent-random-word laws and iid confidence; fixed PRNG replay is deterministic verification, not an independence theorem'));print('verified',len(checks),flush=True)
if __name__=='__main__':main()
