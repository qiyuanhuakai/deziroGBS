"""Independent exact witness, event-confidence and full-trajectory verification."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import mpmath as mp
from iteration22_external import digest,save
from verify_iteration34 import decode,reconstruct
from iteration37_reference import weights,retained,kruskal
import itertools
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def integers(A):
    a=np.asarray(A);values=[F(float(x)) for x in a.flat];d=math.lcm(*(v.denominator for v in values));return np.array([v.numerator*(d//v.denominator) for v in values],dtype=object).reshape(a.shape),d


def outward(a,b):
    scale=1<<256;return F(a.numerator*scale//a.denominator,scale),F(-((-b.numerator*scale)//b.denominator),scale)


def main():
    out=ROOT/'results/iteration37_validation';out.mkdir(exist_ok=False);names=['experiments/verify_iteration37.py','experiments/iteration37_reference.py','experiments/verify_iteration34.py','experiments/iteration22_external.py'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:shutil.copy2(ROOT/n,out/Path(n).name)
    read=lambda p:decode(json.loads(Path(p).read_text(encoding='utf-8')));checks=[];cert_count=hp_count=0;high=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(name)
    def real_certificate(c,A,den,label):
        nonlocal cert_count
        cert_count+=1;Z=np.array(c['real'],dtype=object);d=c['denominator'];R=np.array(c['R_real'],dtype=object);s=c['scale'];n=len(Z)
        check(label+' exact matrix',np.array_equal(Z*den,A*d) and np.array_equal(Z,Z.T));check(label+' real triangular witness',not any(int(x) for row in c['imag'] for x in row) and not any(int(x) for row in c['R_imag'] for x in row) and all(int(R[i,i])>0 for i in range(n)) and not any(int(R[i,j]) for i in range(n) for j in range(i+1,n)))
        E=(R@Z)@R.T-np.eye(n,dtype=object)*d*s*s;delta=F(max(sum(abs(int(x)) for x in row) for row in E),d*s*s);trace=F(sum(int(x)**2 for x in R.flat),s*s);lam=(1-delta)/trace;check(label+' exact residual',0<=delta<1 and delta==c['diagnostics']['exact_residual_upper']);check(label+' eigenvalue',lam==c['diagnostics']['minimum_eigenvalue_lower']>0);detR=math.prod((F(int(R[j,j]),s) for j in range(n)),start=F(1));bounds=outward((1-delta)**n/detR**2,(1+delta)**n/detR**2);check(label+' determinant',tuple(c['determinant_interval'])==bounds);return delta,trace,lam
    def model_check(rec,arrays,label):
        B=arrays['B'];noise=arrays['F'];m=len(B);br,dr=integers(B.real);bi,di=integers(B.imag);d=max(dr,di);S=np.block([[br*(d//dr),bi*(d//di)],[bi*(d//di),-br*(d//dr)]]);K=np.eye(2*m,dtype=object)*d-S;check(label+' symmetric B',np.array_equal(B,B.T));delta,trace,_=real_certificate(rec['core'],K,d,label+' core');Z,dz=integers(noise);td=2*dz*dz*d;T=Z.T@(K@Z)+np.eye(noise.shape[1],dtype=object)*td;real_certificate(rec['noise'],T,td,label+' noise');R=np.array(rec['core']['R_real'],dtype=object);dy=rec['core']['scale']**2;dc=max(dy,dz*dz);C=(2*(R.T@R)-np.eye(2*m,dtype=object)*dy)*(dc//dy)+(Z@Z.T)*(dc//(dz*dz));radius=2*delta*trace/(1-delta);check(label+' center and radius',dc==rec['center_denominator'] and np.array_equal(C,np.array(rec['center_numerators'],dtype=object)) and radius==rec['operator_radius']);return C,dc,radius
    def enclosure_check(r,C,dc,radius,label):
        modes=tuple(r['modes']);m=len(C)//2;ix=modes+tuple(j+m for j in modes);A=C[np.ix_(ix,ix)]+np.eye(len(ix),dtype=object)*dc;_,_,lam=real_certificate(r['certificate'],A,2*dc,label);eta=radius/(2*lam);check(label+' relative perturbation',eta==r['relative_perturbation'] and 0<=eta<1);lo,hi=r['certificate']['determinant_interval'];bounds=outward(lo*(1-eta)**len(ix),hi*(1+eta)**len(ix));check(label+' perturbed determinant',tuple(r['perturbed_determinant_interval'])==bounds);a,b=r['raw_vacuum_interval'];check(label+' exact root inequalities',a>=0 and a*a<=1/bounds[1]<=1/bounds[0]<=b*b);check(label+' physical probability intersection',tuple(r['vacuum_interval'])==(max(F(0),a),min(F(1),b)));return tuple(r['vacuum_interval'])
    def law_check(law,vac,label):
        modes=law['modes'];n=len(modes)
        for clicks in range(1<<n):
            zeros=((1<<n)-1)^clicks;lo=hi=F(0)
            for subset in range(1<<n):
                if subset&~clicks:continue
                mask=zeros|subset;key=tuple(modes[j] for j in range(n) if mask>>j&1);a,b=vac[key]
                if subset.bit_count()%2:lo-=b;hi-=a
                else:lo+=a;hi+=b
            check(label+f' atom {clicks}',tuple(law['raw_intervals'][clicks])==(lo,hi) and tuple(law['intervals'][clicks])==(max(F(0),lo),min(F(1),hi)))
        check(label+' mass',sum(a for a,b in law['intervals'])<=1<=sum(b for a,b in law['intervals']))
    def moment_check(r,vac,m,label):
        means=(sum(1-vac[(j,)][1] for j in range(m)),sum(1-vac[(j,)][0] for j in range(m)));elo,ehi=means;check(label+' mean',tuple(r['mean_interval'])==means)
        for p in r['pairs']:
            i,j=p['modes'];a,b=vac[(i,j)];lo=max(F(0),1-vac[(i,)][1]-vac[(j,)][1]+a);hi=min(F(1),1-vac[(i,)][0]-vac[(j,)][0]+b);check(label+f' pair {i}:{j}',tuple(p['click_pair_interval'])==(lo,hi) and lo<=hi);elo+=2*lo;ehi+=2*hi
        check(label+' all pairs',len(r['pairs'])==m*(m-1)//2 and len({tuple(p['modes']) for p in r['pairs']})==m*(m-1)//2);check(label+' second moment variance',tuple(r['second_moment_interval'])==(elo,ehi) and tuple(r['variance_interval'])==(max(F(0),elo-means[1]**2),max(F(0),ehi-means[0]**2)))
    local=ROOT/'results/iteration37_local';full=ROOT/'results/iteration37_full';pilot=ROOT/'results/iteration37_pilot';deps={}
    for folder,n in ((pilot,8),(local,24),(full,4)):
        p=read(folder/'plan.json');s=read(folder/'summary.json');rows=read(folder/'rows.json');check(folder.name+' complete frozen panel',s['planned']==s['finished']==s['completed']==n and s['unchanged_sources'] and len(rows)==n and all(r['status']=='completed' for r in rows));deps.update(p['dependencies'])
        for name,sha in p['source_hashes'].items():check(folder.name+' source '+name,digest(ROOT/name)==sha);check(folder.name+' snapshot '+name,digest(folder/'source_snapshot'/name)==sha)
    for name,sha in deps.items():check('dependency '+name,digest(ROOT/name)==sha)
    inputs=np.load(ROOT/'results/iteration35_panel/inputs.npz');base=read(ROOT/'results/iteration35_panel/case_07.audit.json');C,dc,radius=model_check(base,inputs,'exact physical model');V=inputs['shifted'];W=weights(V);selection_checks=graph_checks=0
    def structure(r,order,job,label,independent):
        nonlocal selection_checks,graph_checks
        if r['q_hex'] is None:return
        h=r['prefix'];clicks=[order[j] for j,b in enumerate(h) if b=='1'];kept=r['retained'];omitted=[j for j in clicks if j not in kept];edges=[tuple(e) for e in r['edges']];check(label+' selection structure',len(set(kept))==len(kept)==min(4,len(clicks)) and set(kept).issubset(clicks));parents={j:j for j in omitted};acyclic=True
        def root(j):
            while parents[j]!=j:j=parents[j]
            return j
        for a,b in edges:
            if a not in parents or b not in parents or root(a)==root(b):acyclic=False;break
            parents[root(a)]=root(b)
        check(label+' spanning forest',acyclic and len(edges)==max(0,len(omitted)-1))
        if independent:
            check(label+' dense independent selector',tuple(kept)==retained(V,W,order,h,job['selection']));selection_checks+=1
            expected=set(zip(omitted[:-1],omitted[1:])) if job['forest']=='chain' else kruskal(W,omitted);check(label+' independent graph',set(edges)==expected);graph_checks+=1
    laws={};enclosures=0
    for start in (0,54,134):
        r=read(local/f'target_{start:03d}.json');vac={(): (F(1),F(1))};check(f'block {start} certificate count',len(r['enclosures'])==1023)
        for j,e in enumerate(r['enclosures']):vac[tuple(e['modes'])]=enclosure_check(e,C,dc,radius,f'new enclosure {start}:{j}');enclosures+=1
        law_check(r['law'],vac,f'block {start}');laws[start]=r['law']
    local_calls=0
    for i,job in enumerate(read(local/'plan.json')['jobs']):
        rec=read(local/f'case_{i:02d}.audit.json');first=read(local/f'case_{i:02d}.first.json');reverse=read(local/f'case_{i:02d}.reverse.json');expected=[''.join(map(str,h)) for depth in range(10) for h in itertools.product((0,1),repeat=depth)];check(f'local {i} all prefixes',len(first)==len(reverse)==1023 and [r['prefix'] for r in first]==expected and [r['prefix'] for r in reverse]==list(reversed(expected)));check(f'local {i} reverse equality',not rec['reverse_mismatches'] and all(all(a.get(k)==b.get(k) for k in ('prefix','q_hex','retained','edges')) for a,b in zip(first,reversed(reverse))))
        for j,r in enumerate(first):structure(r,rec['order'],job,f'local {i}:{j}',True)
        table={tuple(map(int,r['prefix'])):None if r['q_hex'] is None else F(float.fromhex(r['q_hex'])) for r in first};chosen=[tuple(j for j in range(10) if mask>>j&1) for mask in range(1024)]
        for audit in rec['audits']:
            result,fail,used=reconstruct(10,chosen,laws[job['start']]['intervals'],table,audit['uniform_bits'])
            for key,value in result.items():
                if key in audit:check(f'local {i} audit {audit["uniform_bits"]} {key}',audit[key]==value)
        check(f'local {i} native count',rec['native_diagnostics']['probability_calls']<=2046);local_calls+=rec['native_diagnostics']['probability_calls']
        print('verified local',i,flush=True)
    U=np.load(full/'uniforms.npz')['uniforms'];check('independent confirmation stream',np.array_equal(U,np.random.default_rng(372440).random((64,144))) and not np.array_equal(U[0],np.load(pilot/'uniforms.npz')['uniforms'][0]));plan=read(full/'plan.json');alpha=F(1,3200);check('simultaneous statistical allocation',plan['simultaneous_events']==32 and plan['per_event_one_sided_alpha']==alpha and 32*alpha==plan['familywise_alpha']==F(1,100));mom=read(ROOT/'results/iteration36_recovery/case_11.audit.json')['moments'];mu=mom['mean_interval'][1];var=mom['variance_interval'][1];events_checked=0;native_calls=0;conditional_attempts=0
    for i,job in enumerate(plan['jobs']):
        rec=read(full/f'case_{i:02d}.audit.json');outputs=np.load(full/f'case_{i:02d}.npz')['outputs'];attempts=rec['attempts'];check(f'full {i} fixed attempts',len(attempts)==64);calls=0
        for j,a in enumerate(attempts):
            r=read(full/f'attempts/case_{i:02d}/attempt_{j:03d}.json');path='';count=0;failed=False;check(f'attempt {i}:{j} summary',all(r[k]==v for k,v in a.items()))
            for q in r['conditionals']:
                check(f'attempt {i}:{j}:{count} path',q['prefix']==path);structure(q,tuple(range(144)),job,f'full {i}:{j}:{count}',j%8==0 and count in (16,32,64,96,128,143))
                if q['q_hex'] is None:failed=True;break
                probability=float.fromhex(q['q_hex']);check(f'attempt {i}:{j}:{count} guarded probability',0<=probability<=1);path+=str(int(float(U[j,count])<probability));count+=1
            calls+=len(r['conditionals']);check(f'attempt {i}:{j} outcome',count==r['uniforms_consumed'] and len(r['conditionals'])==r['conditional_attempts'] and ((failed and r['outcome']=='numerical_failure' and r['failed_prefix']==path and np.all(outputs[j]==-1)) or (not failed and r['outcome']=='completed' and count==144 and r['clicks']==path.count('1') and r['output']==[int(x) for x in path] and np.array_equal(outputs[j],r['output']))))
        check(f'full {i} count',rec['native_diagnostics']['probability_calls']<=calls==rec['conditional_attempts']);native_calls+=rec['native_diagnostics']['probability_calls'];conditional_attempts+=calls;check(f'full {i} fixed events',[e['threshold'] for e in rec['events']]==[40,50,60,70,80,90,100,None])
        for e in rec['events']:
            threshold=e['threshold'];k=sum(a['outcome']=='numerical_failure' if threshold is None else a['outcome']=='completed' and a['clicks']>=threshold for a in attempts);qlo=e['sampler_probability_lower'];check(f'event {i}:{threshold} successes',k==e['successes'] and e['attempts']==64)
            def tail(p):return sum(F(math.comb(64,j))*p**j*(1-p)**(64-j) for j in range(k,65))
            check(f'event {i}:{threshold} exact binomial inverse',qlo==0 if k==0 else qlo>=0 and tail(qlo)<=alpha<tail(qlo+F(1,1<<40)))
            upper=F(0) if threshold is None else min(F(1),mu/threshold,var/(var+(threshold-mu)**2) if threshold>mu else F(1));check(f'event {i}:{threshold} target bound and TV',e['target_probability_upper']==upper and e['full_extended_TV_statistical_lower']==max(F(0),qlo-upper));events_checked+=1
        print('verified full',i,flush=True)
    print('independent 90-digit exact-model solves',flush=True);mp.mp.dps=90;mf=lambda x:mp.mpf(F(x).numerator)/F(x).denominator;k=base['core'];K=mp.matrix([[mp.mpf(x)/k['denominator'] for x in row] for row in k['real']]);L=mp.cholesky(K);wanted=sorted({j for start in (0,54,134) for j in (*range(start,start+10),*range(start+144,start+154))});columns={};n=288
    for column in wanted:
        y=[mp.mpf(0)]*n;x=[mp.mpf(0)]*n
        for j in range(column,n):y[j]=(int(j==column)-mp.fsum(L[j,t]*y[t] for t in range(column,j)))/L[j,j]
        for j in range(n-1,-1,-1):x[j]=(y[j]-mp.fsum(L[t,j]*x[t] for t in range(j+1,n)))/L[j,j]
        columns[column]=x
    noise=inputs['F'];noise_mp={j:[mp.mpf(float(x)) for x in noise[j]] for j in wanted}
    for start in (0,54,134):
        ix=list(range(start,start+10))+list(range(start+144,start+154));Ctrue=mp.matrix([[2*columns[j][i]-int(i==j)+mp.fsum(a*b for a,b in zip(noise_mp[i],noise_mp[j])) for j in ix] for i in ix]);vs=[mp.mpf(1)]
        for mask in range(1,1024):
            modes=[j for j in range(10) if mask>>j&1];positions=modes+[j+10 for j in modes];Q=mp.matrix([[(Ctrue[i,j]+int(i==j))/2 for j in positions] for i in positions]);vs.append(1/mp.sqrt(mp.det(Q)))
        values=[]
        for clicks in range(1024):
            value=mp.fsum((-1)**s.bit_count()*vs[(1023^clicks)|s] for s in range(1024) if not s&~clicks);a,b=laws[start]['intervals'][clicks];check(f'90 digit exact-model atom {start}:{clicks}',mf(a)<=value<=mf(b));hp_count+=1;values.append(value)
        high.append(dict(start=start,all_vacuum=mp.nstr(vs[-1],85),maximum_atom=mp.nstr(max(values),85),digits=90));print('verified high precision block',start,flush=True)
    check('verifier sources unchanged',hashes=={n:digest(ROOT/n) for n in names});save(out/'checks.json',checks);save(out/'high_precision.json',high);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),matrix_certificates=cert_count,new_vacuum_certificates=enclosures,high_precision_atom_checks=hp_count,predeclared_statistical_tests=events_checked,native_full_calls_replayed=native_calls,full_conditional_attempts=conditional_attempts,local_native_calls=local_calls,independent_selector_checks=selection_checks,independent_graph_checks=graph_checks,source_hashes=hashes,dependencies=deps,statistical_scope='99% simultaneous lower confidence bounds conditional on iid uniform-grid attempts; local stopped-law TV intervals are not full-law upper bounds'));print('verified',len(checks),'matrices',cert_count,'high precision',hp_count,flush=True)
if __name__=='__main__':main()
