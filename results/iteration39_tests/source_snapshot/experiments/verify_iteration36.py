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
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def integers(A):
    a=np.asarray(A);values=[F(float(x)) for x in a.flat];d=math.lcm(*(v.denominator for v in values));return np.array([v.numerator*(d//v.denominator) for v in values],dtype=object).reshape(a.shape),d


def outward(a,b):
    scale=1<<256;return F(a.numerator*scale//a.denominator,scale),F(-((-b.numerator*scale)//b.denominator),scale)


def main():
    out=ROOT/'results/iteration36_validation';out.mkdir(exist_ok=False);names=['experiments/verify_iteration36.py','experiments/verify_iteration34.py','experiments/iteration22_external.py'];hashes={n:digest(ROOT/n) for n in names}
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
    panel=ROOT/'results/iteration36_panel';recovery=ROOT/'results/iteration36_recovery';plan=read(panel/'plan.json');original=read(panel/'rows.json');rs=read(recovery/'summary.json');summary=read(panel/'summary.json');rows=original[:6]+read(recovery/'rows.json');check('preserved original type comparison failure',summary['completed']==6 and all(r['status']=='completed' for r in original[:6]) and all(r['status']=='failed' and 'AssertionError' in r['traceback'] for r in original[6:]) and 'certificate(model.core)==base' in original[6]['traceback']);check('effective 16 jobs',len(rows)==16 and rs['planned']==rs['finished']==rs['completed']==10 and rs['original_failure_preserved'] and all(r['status']=='completed' for r in rows));deps={}
    for folder in (panel,recovery):
        p=read(folder/'plan.json');s=read(folder/'summary.json');check(folder.name+' frozen jobs',p['jobs']==plan['jobs'] and s['unchanged_sources']);deps.update(p['dependencies'])
        for n,sha in p['source_hashes'].items():check(folder.name+' source '+n,digest(ROOT/n)==sha);check(folder.name+' snapshot '+n,digest(folder/'source_snapshot'/n)==sha)
    for n,sha in deps.items():check('dependency '+n,digest(ROOT/n)==sha)
    inputs=np.load(ROOT/'results/iteration35_panel/inputs.npz');base=read(ROOT/'results/iteration35_panel/case_07.audit.json');main=read(recovery/'case_06.audit.json');arrays=np.load(recovery/'case_06.npz');check('same exact physical target',np.array_equal(inputs['B'],arrays['B']) and np.array_equal(inputs['F'],arrays['F']) and np.array_equal(inputs['shifted'],arrays['shifted']) and all(main[k]==base[k] for k in ('core','noise','center_numerators','center_denominator','operator_radius','vacuum_interval')));C,dc,radius=model_check(main,arrays,'main');records=[]
    index=read(recovery/'certificate_index.json')
    for k in range(index['shards']):records.extend(read(recovery/f'certificates_{k:02d}.json'))
    check('enclosure count',len(records)==index['count']==11316);vac={(): (F(1),F(1))}
    for j,r in enumerate(records):vac[tuple(r['modes'])]=enclosure_check(r,C,dc,radius,f'large enclosure {j}')
    check('unique vacuum subsets',len(vac)==11317);mom=read(recovery/'case_11.audit.json')['moments'];moment_check(mom,vac,144,'main moments');mp.mp.dps=90;small_refs={};mf=lambda x:mp.mpf(F(x).numerator)/F(x).denominator
    def reference_law(V):
        n=V.rows//2;vs=[mp.mpf(1)]
        for mask in range(1,1<<n):
            modes=[j for j in range(n) if mask>>j&1];ix=modes+[j+n for j in modes];Q=mp.matrix([[(V[i,j]+int(i==j))/2 for j in ix] for i in ix]);vs.append(1/mp.sqrt(mp.det(Q)))
        law=[]
        for clicks in range(1<<n):law.append(mp.fsum((-1)**s.bit_count()*vs[(((1<<n)-1)^clicks)|s] for s in range(1<<n) if not s&~clicks))
        return vs,law
    for i in range(6):
        rec=read(panel/f'case_{i:02d}.audit.json');a=np.load(panel/f'case_{i:02d}.npz');c,dd,eps=model_check(rec,a,f'small {i}');vv={(): (F(1),F(1))}
        for j,r in enumerate(rec['enclosures']):vv[tuple(r['modes'])]=enclosure_check(r,c,dd,eps,f'small enclosure {i}:{j}')
        law_check(rec['law'],vv,f'small {i}');moment_check(rec['moments'],vv,rows[i]['modes'],f'small {i}')
        m=rows[i]['modes']
        if m not in small_refs:
            k=rec['core'];K=mp.matrix([[mp.mpf(x)/k['denominator'] for x in row] for row in k['real']]);N=mp.matrix([[mp.mpf(float(x)) for x in row] for row in a['F']]);V=2*K**-1-mp.eye(2*m)+N*N.T;small_refs[m]=reference_law(V)
        for mask,(lo,hi) in enumerate(rec['law']['intervals']):check(f'90 digit small atom {i}:{mask}',mf(lo)<=small_refs[m][1][mask]<=mf(hi));hp_count+=1
    for i in range(7,11):
        rec=read(recovery/f'case_{i:02d}.audit.json');law=rec['law'];law_check(law,vac,f'block {i}');first=rows[i]['start'];old=read(ROOT/f'results/iteration35_panel/target_{first:03d}.json')['target_intervals'];lo=sum(max(F(0),a-d,c-b) for (a,b),(c,d) in zip(law['intervals'],old))/2;hi=sum(max(abs(a-d),abs(b-c)) for (a,b),(c,d) in zip(law['intervals'],old))/2;check(f'block {i} physical target TV',tuple(rec['shifted_target_TV_interval'])==(lo,hi));chosen=[tuple(j for j in range(8) if mask>>j&1) for mask in range(256)]
        for audit in rec['audits']:
            previous=read(ROOT/f"results/iteration35_panel/case_{audit['source_case']:02d}.audit.json");table={tuple(r['prefix']):None if r['q_hex'] is None else F(float.fromhex(r['q_hex'])) for r in previous['oracle']};result,fail,used=reconstruct(8,chosen,law['intervals'],table,audit['uniform_bits'])
            for key,value in result.items():
                if key in audit:check(f'block {i} audit {audit["source_case"]}:{audit["uniform_bits"]} {key}',audit[key]==value)
    U=np.load(recovery/'uniforms.npz')['uniforms'];check('new fixed shared uniforms',np.array_equal(U,np.random.default_rng(361440).random((64,144))) and np.array_equal(U,np.load(panel/'uniforms.npz')['uniforms']));mu=mom['mean_interval'][1];var=mom['variance_interval'][1];alpha=F(1,3200);check('simultaneous statistical allocation',plan['simultaneous_events']==32 and plan['per_event_one_sided_alpha']==alpha and 32*alpha==plan['familywise_alpha']==F(1,100));events_checked=0;native_calls=0
    for i in range(12,16):
        rec=read(recovery/f'case_{i:02d}.audit.json');outputs=np.load(recovery/f'case_{i:02d}.npz')['outputs'];attempts=rec['attempts'];check(f'sampling {i} fixed attempts',len(attempts)==64);calls=0
        for j,a in enumerate(attempts):
            r=read(recovery/f'attempts/case_{i:02d}/attempt_{j:03d}.json');path='';count=0;failed=False;check(f'attempt {i}:{j} summary',all(r[k]==v for k,v in a.items()))
            for q in r['conditionals']:
                check(f'attempt {i}:{j}:{count} path',q['prefix']==path)
                if q['q_hex'] is None:failed=True;break
                p=float.fromhex(q['q_hex']);check(f'attempt {i}:{j}:{count} guarded probability',0<=p<=1);path+=str(int(float(U[j,count])<p));count+=1
            calls+=len(r['conditionals']);check(f'attempt {i}:{j} outcome',count==r['uniforms_consumed'] and len(r['conditionals'])==r['native_calls'] and ((failed and r['outcome']=='numerical_failure' and r['failed_prefix']==path and np.all(outputs[j]==-1)) or (not failed and r['outcome']=='completed' and count==144 and r['clicks']==path.count('1') and r['output']==[int(x) for x in path] and np.array_equal(outputs[j],r['output']))))
        check(f'sampling {i} actual native count',calls==rows[i]['native_calls']==rec['native_diagnostics']['probability_calls']);native_calls+=calls
        check(f'sampling {i} predeclared events',[e['threshold'] for e in rec['events']]==[40,50,60,70,80,90,100,None])
        for e in rec['events']:
            threshold=e['threshold'];k=sum(a['outcome']=='numerical_failure' if threshold is None else a['outcome']=='completed' and a['clicks']>=threshold for a in attempts);qlo=e['sampler_probability_lower'];check(f'event {i}:{threshold} successes',k==e['successes'] and e['attempts']==64)
            def tail(p):return sum(F(math.comb(64,j))*p**j*(1-p)**(64-j) for j in range(k,65))
            check(f'event {i}:{threshold} exact binomial inverse',qlo==0 if k==0 else qlo>=0 and tail(qlo)<=alpha<tail(qlo+F(1,1<<40)))
            upper=F(0) if threshold is None else min(F(1),mu/threshold,var/(var+(threshold-mu)**2) if threshold>mu else F(1));check(f'event {i}:{threshold} physical target tail and TV',e['target_probability_upper']==upper and e['full_extended_TV_statistical_lower']==max(F(0),qlo-upper));events_checked+=1
    print('exact checks complete; independent 90-digit main solves',flush=True);k=main['core'];K=mp.matrix([[mp.mpf(x)/k['denominator'] for x in row] for row in k['real']]);L=mp.cholesky(K);wanted=sorted({j for start in (0,18,54,136) for j in (*range(start,start+8),*range(start+144,start+152))});columns={};n=288
    for column in wanted:
        y=[mp.mpf(0)]*n;x=[mp.mpf(0)]*n
        for j in range(column,n):y[j]=(int(j==column)-mp.fsum(L[j,t]*y[t] for t in range(column,j)))/L[j,j]
        for j in range(n-1,-1,-1):x[j]=(y[j]-mp.fsum(L[t,j]*x[t] for t in range(j+1,n)))/L[j,j]
        columns[column]=x
    noise=arrays['F'];noise_mp={j:[mp.mpf(float(x)) for x in noise[j]] for j in wanted}
    def true_cov(ix):return mp.matrix([[2*columns[j][i]-int(i==j)+mp.fsum(a*b for a,b in zip(noise_mp[i],noise_mp[j])) for j in ix] for i in ix])
    for i,start in enumerate((0,18,54,136),7):
        ix=list(range(start,start+8))+list(range(start+144,start+152));V=true_cov(ix);vs,law=reference_law(V);rec=read(recovery/f'case_{i:02d}.audit.json')
        for mask,value in enumerate(law):
            a,b=rec['law']['intervals'][mask];check(f'90 digit main atom {start}:{mask}',mf(a)<=value<=mf(b));hp_count+=1
        high.append(dict(start=start,all_vacuum=mp.nstr(vs[-1],85),maximum_probability=mp.nstr(max(law),85),digits=90))
    for modes in ((0,143),(18,54),(7,140),(25,61),(1,19)):
        ix=list(modes)+[j+144 for j in modes];V=true_cov(ix);value=1/mp.sqrt(mp.det((V+mp.eye(4))/2));a,b=vac[modes];check('90 digit distant pair '+str(modes),mf(a)<=value<=mf(b));hp_count+=1
    check('verifier sources unchanged',hashes=={n:digest(ROOT/n) for n in names});save(out/'checks.json',checks);save(out/'high_precision.json',high);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),matrix_certificates=cert_count,high_precision_atom_or_pair_checks=hp_count,predeclared_statistical_tests=events_checked,native_calls_replayed=native_calls,source_hashes=hashes,dependencies=deps,statistical_scope='99% simultaneous lower confidence bounds, conditional on iid uniform-grid attempts; no full-law TV upper bound'));print('verified',len(checks),'matrices',cert_count,'high precision',hp_count,flush=True)
if __name__=='__main__':main()
