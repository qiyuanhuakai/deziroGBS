"""Independent exact confidence endpoints, full paths and external-vector checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,math,subprocess
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from verify_iteration34 import decode,reconstruct
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration38_validation';out.mkdir(exist_ok=False);names=['experiments/verify_iteration38.py','experiments/verify_iteration34.py','experiments/iteration22_external.py'];hashes={p:digest(ROOT/p) for p in names}
    for p in names:shutil.copy2(ROOT/p,out/Path(p).name)
    read=lambda p:decode(json.loads(Path(p).read_text()));checks=[];deps={};paths=0;root_checks=0;quantiles=0;context_checks=0;external_replays=0
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(name)
    full=ROOT/'results/iteration38_confirmation';external=ROOT/'results/iteration38_external'
    for folder,n in ((full,4),(external,40)):
        p=read(folder/'plan.json');s=read(folder/'summary.json');rows=read(folder/'rows.json');check(folder.name+' complete',s['planned']==s['finished']==s['completed']==n and s['unchanged_sources'] and len(rows)==n and all(r['status']=='completed' for r in rows));deps.update(p['dependencies'])
        for name,sha in p['source_hashes'].items():check(folder.name+' source '+name,digest(ROOT/name)==sha);check(folder.name+' snapshot '+name,digest(folder/'source_snapshot'/name)==sha)
    for name,sha in deps.items():check('dependency '+name,digest(ROOT/name)==sha)
    build=ROOT/'results/iteration38_frugal_build_attempt';bp=read(build/'plan.json');check('official build succeeded',read(build/'attempt.json')['returncode']==0 and read(build/'attempt.json')['unchanged_sources'])
    for name,sha in bp['source_hashes'].items():check('build source '+name,digest(ROOT/name)==sha);check('build snapshot '+name,digest(build/'source_snapshot'/name)==sha)
    meta=read(build/'build/build.json');runtime=read(ROOT/'results/iteration38_frugal_runtime/runtime.json');check('unchanged pinned upstream',meta['upstream_commit']=='fa8b796f2afebd4845a2e335e457735952617ef8' and meta['sampler_body_sha256']=='9c14a870f2875e808025d9e3535a71ba86144e5639ab14f0b7be5b594dabb3fb');check('same compiled executable',meta['executable_sha256']==runtime['original_sha256']==digest(build/'build/frugal_adapter')==digest(ROOT/'results/iteration38_frugal_runtime/frugal_adapter.exe'));check('official Eigen archive hash',bp['acquisition']['archive_sha256']=='8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72')
    for n,r in runtime['files'].items():check('runtime '+n,digest(ROOT/'results/iteration38_frugal_runtime'/n)==r['sha256'])
    plan=read(full/'plan.json');bounds=read(full/'target_bounds.json');mom=read(ROOT/'results/iteration36_recovery/case_11.audit.json')['moments'];check('same certified target moments',bounds['moments']==mom);mu=mom['mean_interval'];e2=mom['second_moment_interval'];var=mom['variance_interval'];target=tuple(v/144 for v in mu);check('normalized target interval',tuple(bounds['target_X_interval'])==target);transfer=read(ROOT/'results/iteration35_panel/case_07.audit.json')['transfers']['-40']['all_measurement_TV_transfer_upper'];check('same physical transfer',bounds['transfer_upper']==transfer);check('fixed joint alpha',plan['mean_alpha_per_tail']==F(1,1000) and plan['mean_tails']==8 and plan['event_alpha']==F(1,16000) and plan['event_tails']==32 and 8*F(1,1000)+32*F(1,16000)==plan['familywise_alpha']==F(1,100));tail_bounds={}
    for k in (40,50,60,70,80,90,100):
        r=bounds['tail_bounds'][str(k)];candidates=[]
        for a in range(-1,k-1):
            den=(k-a)*(k-a-1);check(f'integer majorant {k}:{a}',all((n-a)*(n-a-1)>=int(n>=k)*den for n in range(145)));coefficient=-(2*a+1);bound=(e2[1]+coefficient*(mu[1] if coefficient>=0 else mu[0])+a*(a+1))/den;candidates.append(dict(root=a,upper=bound))
        best=min(candidates,key=lambda x:x['upper']);upper=min(F(1),best['upper']);check(f'integer target bound {k}',r['discrete']['candidates']==candidates and r['discrete']['upper']==upper and r['discrete']['root']==(best['root'] if upper<1 else None));old=min(F(1),mu[1]/k,var[1]/(var[1]+(k-mu[1])**2) if k>mu[1] else F(1));check(f'combined target bound {k}',r['previous']==old and r['upper']==min(old,upper));tail_bounds[k]=r['upper']
    U=np.load(full/'uniforms.npz')['uniforms'];check('new fixed uniform stream',np.array_equal(U,np.random.default_rng(381440).random((96,144))));delta=F(1,1<<40)
    for i,job in enumerate(plan['jobs']):
        rec=read(full/f'case_{i:02d}.audit.json');outputs=np.load(full/f'case_{i:02d}.npz')['outputs'];check(f'full {i} fixed attempts',len(rec['attempts'])==96);values=[];calls=0
        for j,a in enumerate(rec['attempts']):
            r=read(full/f'attempts/case_{i:02d}/attempt_{j:03d}.json');check(f'attempt {i}:{j} summary',all(r[k]==v for k,v in a.items()));h='';n=0;failed=False
            for q in r['conditionals']:
                check(f'path {i}:{j}:{n}',q['prefix']==h)
                if q['q_hex'] is None:failed=True;break
                p=float.fromhex(q['q_hex']);check(f'probability {i}:{j}:{n}',0<=p<=1);h+=str(int(U[j,n]<p));n+=1
            calls+=len(r['conditionals']);check(f'outcome {i}:{j}',n==r['uniforms_consumed'] and len(r['conditionals'])==r['conditional_attempts'] and ((failed and r['outcome']=='numerical_failure' and r['failed_prefix']==h and np.all(outputs[j]==-1)) or (not failed and n==144 and r['outcome']=='completed' and r['clicks']==h.count('1') and r['output']==list(map(int,h)) and np.array_equal(outputs[j],r['output']))));values.append(F(0) if failed else F(h.count('1'),144))
        check(f'full {i} native accounting',rec['native_diagnostics']['probability_calls']<=calls==rec['conditional_attempts']);paths+=calls;m=rec['mean_interval'];forward=math.prod(values,start=F(1));reverse=math.prod((1-v for v in values),start=F(1));check(f'mean {i} exact products',m['forward_product']==forward and m['reverse_product']==reverse and m['attempts']==96 and m['alpha_per_tail']==F(1,1000) and m['bits']==40)
        for tag,q,value in (('lower',m['lower'],forward),('upper-complement',1-m['upper'],reverse)):
            power=F(1,1000)*value;check(f'mean {i} root {tag}',0<=q<=1 and (q*(1<<40)).denominator==1 and q**96<=power<(q+delta)**96);root_checks+=1
        mean_tv=max(F(0),m['lower']-target[1],target[0]-m['upper']);check(f'mean {i} TV',rec['mean_TV_statistical_lower']==mean_tv);event_lowers=[];check(f'full {i} event family',[e['threshold'] for e in rec['events']]==[40,50,60,70,80,90,100,None])
        for e in rec['events']:
            k=e['threshold'];count=sum(a['outcome']=='numerical_failure' if k is None else a['outcome']=='completed' and a['clicks']>=k for a in rec['attempts']);q=e['sampler_probability_lower'];check(f'event {i}:{k} counts',count==e['successes'] and e['attempts']==96)
            def tail(p):return sum(F(math.comb(96,j))*p**j*(1-p)**(96-j) for j in range(count,97))
            check(f'event {i}:{k} exact inverse',q==0 if count==0 else 0<=q<=1 and tail(q)<=F(1,16000)<tail(q+delta));upper=F(0) if k is None else tail_bounds[k];v=max(F(0),q-upper);check(f'event {i}:{k} target and TV',e['target_probability_upper']==upper and e['full_TV_statistical_lower']==v);event_lowers.append(v);quantiles+=1
        v=max(mean_tv,*event_lowers);check(f'full {i} joint and transfer TV',rec['full_TV_statistical_lower']==v and rec['shifted_target_TV_statistical_lower']==max(F(0),v-transfer));print('verified confirmation',i,flush=True)
    targets={s:read(ROOT/f'results/iteration37_local/target_{s:03d}.json')['law']['intervals'] for s in (0,54,134)};primary=np.load(ROOT/'results/iteration35_panel/inputs.npz')['shifted'];ep=read(external/'plan.json');cal=read(external/'case_00.audit.json');check('15 existing official adapter calibrations',cal['passed'] and len(cal['validation_cases'])==15 and all(r['off_support_samples']==0 for r in cal['validation_cases']));exe=ROOT/'results/iteration38_frugal_runtime/frugal_adapter.exe';replay=out/'official_replay';replay.mkdir()
    for i,job in enumerate(ep['jobs'][1:],1):
        rec=read(external/f'case_{i:02d}.audit.json');V=np.load(external/f'case_{i:02d}.npz')['covariance'];s=job['start'];ix=list(range(s,s+10))+list(range(s+144,s+154));check(f'external {i} identical input',np.array_equal(V,primary[np.ix_(ix,ix)]));target=targets[s]
        if job['panel']=='frugal':
            q=tuple(F(float(v)) for v in rec['distribution']);check(f'frugal {i} finite export',len(q)==1024 and all(v>=0 for v in q));lo=sum(max(F(0),a-v,v-b) for v,(a,b) in zip(q,target))/2;hi=sum(max(abs(v-a),abs(v-b)) for v,(a,b) in zip(q,target))/2;check(f'frugal {i} exported-vector distance',tuple(rec['exported_vector_half_L1_interval'])==(lo,hi) and rec['exported_vector_normalization_defect']==sum(q)-1);check(f'frugal {i} official metadata',rec['sample_count']==sum(rec['histogram_counts'])==65536 and rec['seed']==382000+i and rec['omp_threads']==1 and rec['cutoff']==job['cutoff'] and rec['modes']==10 and not rec['trace']['nonfinite_threshold_path_steps']);csv=replay/f'case_{i:02d}.csv';np.savetxt(csv,V,delimiter=',',fmt='%.17g');check(f'frugal {i} input roundtrip',np.array_equal(np.loadtxt(csv,delimiter=','),V));command=[str(exe),'--input',str(csv),'--modes','10','--cutoff',str(job['cutoff']),'--samples','65536','--seed',str(382000+i),'--threads','1','--enumerate'];process=subprocess.run(command,capture_output=True,text=True,check=True);again=json.loads(process.stdout);save(replay/f'case_{i:02d}.json',again);check(f'frugal {i} independent process replay',again['distribution']==rec['distribution'] and again['histogram_counts']==rec['histogram_counts']);external_replays+=1
        else:
            table={tuple(map(int,r['prefix'])):None if r['q_hex'] is None else F(float.fromhex(r['q_hex'])) for r in rec['oracle']};old=read(ROOT/f'results/iteration37_local/case_{rec["context_source"]:02d}.first.json');check(f'forest {i} all prefixes',len(table)==len(old)==len(rec['oracle'])==1023)
            for j,(a,b) in enumerate(zip(rec['oracle'],old)):
                same=a['prefix']==b['prefix'] and [v+s for v in a.get('retained',())]==b.get('retained',[]) and [[v+s for v in e] for e in a.get('edges',())]==b.get('edges',[]);difference=abs(float.fromhex(a['q_hex'])-float.fromhex(b['q_hex'])) if a['q_hex'] is not None and b['q_hex'] is not None else (0.0 if a['q_hex']==b['q_hex'] else float('inf'));c=rec['context_checks'][j];check(f'forest context {i}:{j}',same and difference<=1e-12 and c['same_structure']==same and c['probability_difference']==difference and c['q_hex_equal']==(a['q_hex']==b['q_hex']));context_checks+=1
            chosen=[tuple(j for j in range(10) if mask>>j&1) for mask in range(1024)]
            for a in rec['audits']:
                result,fail,used=reconstruct(10,chosen,target,table,a['uniform_bits'])
                for key,value in result.items():
                    if key in a:check(f'forest law {i}:{a["uniform_bits"]} {key}',a[key]==value)
    check('verifier sources unchanged',hashes=={p:digest(ROOT/p) for p in names});save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(r['passed'] for r in checks),full_conditionals_replayed=paths,exact_mean_root_checks=root_checks,exact_binomial_quantile_checks=quantiles,small_full_context_checks=context_checks,official_process_replays=external_replays,source_hashes=hashes,dependencies=deps,scope='99% simultaneous iid mean/tail confidence; Frugal exported-vector distances are numerical trace diagnostics, not strict finite-RNG sampler-TV certificates'));print('verified',len(checks),flush=True)
if __name__=='__main__':main()
