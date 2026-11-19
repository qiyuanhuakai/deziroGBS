"""Independent integer witness reconstruction; no new certificate code imported."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import mpmath as mp
from iteration22_external import save,digest
from verify_iteration34 import decode,reconstruct,exact_determinant
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def integers(A):
    values=[F(x) for x in np.asarray(A).flat];d=math.lcm(*(x.denominator for x in values));return np.array([x.numerator*(d//x.denominator) for x in values],dtype=object).reshape(np.asarray(A).shape),d


def fraction_array(A):return np.array([[F(float(x)) for x in row] for row in A],dtype=object)


def rounded_interval(a,b):
    scale=1<<256;return F(a.numerator*scale//a.denominator,scale),F(-((-b.numerator*scale)//b.denominator),scale)


def main():
    out=ROOT/'results/iteration35_validation';out.mkdir(exist_ok=False);sources=['experiments/verify_iteration35.py','experiments/verify_iteration34.py','experiments/iteration22_external.py'];source_hashes={n:digest(ROOT/n) for n in sources}
    for n in sources:shutil.copy2(ROOT/n,out/Path(n).name)
    checks=[];cert_count=mp_count=small_exact=0
    def read(p):return decode(json.loads(Path(p).read_text(encoding='utf-8')))
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(name)
    def check_certificate(c,A,B=None,label='certificate'):
        nonlocal cert_count,small_exact
        cert_count+=1;ar=np.array(c['real'],dtype=object);ai=np.array(c['imag'],dtype=object);den=c['denominator'];n=len(ar);r=np.array(c['R_real'],dtype=object);s=np.array(c['R_imag'],dtype=object);scale=c['scale'];x,dx=integers(A);y,dy=integers(np.zeros(x.shape,dtype=object) if B is None else B)
        check(label+' exact input',np.array_equal(ar*dx,x*den) and np.array_equal(ai*dy,y*den));check(label+' Hermitian',np.array_equal(ar,ar.T) and np.array_equal(ai,-ai.T));check(label+' triangular invertible witness',all(int(r[i,i])>0 and not int(s[i,i]) for i in range(n)) and not any(int(r[i,j]) or int(s[i,j]) for i in range(n) for j in range(i+1,n)))
        if any(int(v) for v in ai.flat):
            er=(r@ar)@r.T+(s@ar)@s.T-(s@ai)@r.T+(r@ai)@s.T;ei=(s@ar)@r.T-(r@ar)@s.T+(r@ai)@r.T+(s@ai)@s.T
        else:er=(r@ar)@r.T;ei=np.zeros(ar.shape,dtype=object);check(label+' real witness',not any(int(v) for v in s.flat))
        er-=np.eye(n,dtype=object)*(den*scale**2);delta=F(max(sum(abs(int(er[i,j]))+abs(int(ei[i,j])) for j in range(n)) for i in range(n)),den*scale**2);diag=c['diagnostics'];trace=F(sum(int(v)**2 for v in r.flat)+sum(int(v)**2 for v in s.flat),scale**2)
        check(label+' exact residual',delta==diag['exact_residual_upper'] and 0<=delta<1);check(label+' eigenvalue bound',(1-delta)/trace==diag['minimum_eigenvalue_lower']>0)
        if 'determinant_interval' in c:
            detR=math.prod((F(int(r[j,j]),scale) for j in range(n)),start=F(1));bounds=rounded_interval((1-delta)**n/detR**2,(1+delta)**n/detR**2);check(label+' determinant outward',tuple(c['determinant_interval'])==bounds)
            if n<=4:
                truth=exact_determinant(A);check(label+' exact elimination determinant',bounds[0]<=truth<=bounds[1]);small_exact+=1
        return delta,trace
    panel=ROOT/'results/iteration35_panel';plan=read(panel/'plan.json');rows=read(panel/'rows.json');summary=read(panel/'summary.json');deps=dict(plan['dependencies']);check('formal 49 completed',summary['planned']==summary['finished']==summary['completed']==len(rows)==49 and summary['unchanged_sources'] and summary['primary_valid'])
    for folder in (panel,ROOT/'results/iteration35_pilot'):
        p=read(folder/'plan.json');s=read(folder/'summary.json');check(str(folder.name)+' unchanged',s['unchanged_sources'])
        for n,sha in p['source_hashes'].items():check(str(folder.name)+' source '+n,digest(ROOT/n)==sha);check(str(folder.name)+' snapshot '+n,digest(folder/'source_snapshot'/n)==sha)
        deps.update(p['dependencies'])
    for n,sha in deps.items():check('reused '+n,digest(ROOT/n)==sha)
    inputs=np.load(panel/'inputs.npz');old=inputs['original'];primary=inputs['shifted'];check('original preserved',np.array_equal(old,np.load(ROOT/'results/iteration31_panel/large_input.npz')['covariance']));check('explicit primary rounding',np.array_equal(primary,old+2.**-40*np.eye(len(old))));main_model=None;mp.mp.dps=90;high=[]
    for i,row in enumerate(rows):
        rec=read(panel/f'case_{i:02d}.audit.json');arrays=np.load(panel/f'case_{i:02d}.npz');check(f'job {i} frozen identity',rec['job']==plan['jobs'][i]);kind=row['panel']
        if kind=='model':
            if row['modes']==144:
                original_model=np.load(ROOT/'results/iteration28_panel/case_14.npz');check(f'model {i} exact reused B,F',np.array_equal(arrays['B'],original_model['full_pairing']) and np.array_equal(arrays['F'],original_model['full_noise_factor']))
            else:
                m=row['modes'];rng=np.random.default_rng(350000+m);a=rng.integers(-2,3,(m,m));b=rng.integers(-2,3,(m,m));expected=((a+a.T)+1j*(b+b.T))/(1<<(m.bit_length()+6));noise=rng.integers(-2,3,(2*m,m+1))/32.;check(f'model {i} fixed small input',np.array_equal(arrays['B'],expected) and np.array_equal(arrays['F'],noise))
            B=arrays['B'];noise=arrays['F'];m=len(B);n=2*m;S=np.block([[fraction_array(B.real),fraction_array(B.imag)],[fraction_array(B.imag),-fraction_array(B.real)]]);O=np.block([[np.zeros((m,m),dtype=object),np.eye(m,dtype=object)],[-np.eye(m,dtype=object),np.zeros((m,m),dtype=object)]]);check(f'model {i} symplectic algebra',np.array_equal(S,S.T) and np.array_equal(S@O,-O@S));K=np.eye(n,dtype=object)-S;d,tr=check_certificate(rec['core'],K,label=f'core {i}');N=fraction_array(noise);T=np.eye(noise.shape[1],dtype=object)+(N.T@(K@N))/2;check_certificate(rec['noise'],T,label=f'noise {i}');kl,ku=rec['core']['determinant_interval'];tl,tu=rec['noise']['determinant_interval'];vl,vu=rec['vacuum_interval'];check(f'model {i} vacuum sqrt',vl>=0 and vl*vl<=kl/tu<=ku/tl<=vu*vu and vu-vl>0)
            c=rec['core'];R=np.array(c['R_real'],dtype=object);Y=R.T@R;dy=c['scale']**2;Z,df=integers(noise);dc=rec['center_denominator'];C=(2*Y-np.eye(n,dtype=object)*dy)*(dc//dy)+(Z@Z.T)*(dc//df**2);check(f'model {i} exact center',np.array_equal(C,np.array(rec['center_numerators'],dtype=object)));radius=2*d/(1-d)*tr;trace=(2*tr/(1+d)-n,2*tr/(1-d)-n);check(f'model {i} radius trace',radius==rec['operator_radius'] and tuple(rec['core_trace_interval'])==trace)
            def comparison(V):
                Z,dv=integers(V);den=math.lcm(dc,dv);D=C*(den//dc)-Z*(den//dv);entry=F(max(abs(int(x)) for x in D.flat),den);norm=F(max(sum(abs(int(x)) for x in r) for r in D),den);return norm+radius,max(F(0),entry-radius)
            if m==144:
                err,lower=comparison(old);check(f'model {i} error upper',err==rec['original_comparison']['operator_difference_upper'] and lower==rec['original_comparison']['operator_difference_lower'])
                for e,t in rec['transfers'].items():
                    V=old+2.**int(e)*np.eye(n);shifts=[F(float(V[j,j]))-F(float(old[j,j])) for j in range(n)];margin=min(shifts)-err;newerr,_=comparison(V);check(f'model {i} shift {e}',t['minimum_actual_diagonal_shift']==min(shifts) and t['maximum_actual_diagonal_shift']==max(shifts) and t['added_noise_PSD_margin_lower']==margin and t['added_noise_operator_upper']==newerr and t['noise_order_proved']==(margin>0))
                    b=t['all_measurement_TV_transfer_upper'];check(f'model {i} transfer bound {e}',b is None if margin<=0 else b>=0 and b*b>=newerr*trace[1]/4)
                if row['bits']==128:main_model=rec
            if row['bits']==128:
                print('high precision independent model',m,flush=True)
                def mp_matrix(a):return mp.matrix([[mp.mpf(F(v).numerator)/F(v).denominator for v in rr] for rr in a])
                km=mp_matrix(K);tm=mp_matrix(T);lk=mp.cholesky(km);lt=mp.cholesky(tm);value=mp.fprod(lk[j,j] for j in range(n))/mp.fprod(lt[j,j] for j in range(len(T)));a=mp.mpf(vl.numerator)/vl.denominator;b=mp.mpf(vu.numerator)/vu.denominator;check(f'model {i} 90 digit vacuum',a<=value<=b);mp_count+=1;high.append(dict(modes=m,vacuum=mp.nstr(value,85),digits=90))
        elif kind=='uncertainty':
            V=arrays['covariance'];m=len(V)//2;O=np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]])
            if rec['status']=='strictly_physical':check_certificate(rec['certificate'],V,O,label=f'physical {i}')
            else:
                w=arrays['witness'];a,da=integers(w.real);b,db=integers(w.imag);d=math.lcm(da,db);a*=d//da;b*=d//db;Z,dv=integers(V);value=F(int(a@(Z@a)+b@(Z@b)-2*dv*(a@np.r_[b[m:],-b[:m]])),dv*int(a@a+b@b));check(f'negative or unresolved {i}',value==rec['exact_rayleigh'] and (value<0)==(rec['status']=='strict_violation'))
        elif kind=='prefix':
            first=row['start'];target=read(panel/f'target_{first:03d}.json');modes=list(range(first,first+8));ix=modes+[j+144 for j in modes];V=primary[np.ix_(ix,ix)];check(f'prefix {i} correct physical marginal',np.array_equal(V,arrays['covariance']));order=modes+[j for j in range(144) if j not in modes];check(f'prefix {i} measurement order',list(arrays['full_order'])==order);table={tuple(r['prefix']):None if r['q_hex'] is None else F(float.fromhex(r['q_hex'])) for r in rec['oracle']};chosen=[tuple(j for j in range(8) if mask>>j&1) for mask in range(256)]
            for audit in rec['audits']:
                result,fail,used=reconstruct(8,chosen,target['target_intervals'],table,audit['uniform_bits'])
                for key,value in result.items():
                    if key in audit:check(f'prefix {i} {audit["uniform_bits"]} '+key,audit[key]==value)
                check(f'prefix {i} full stopped coverage',result['unselected_proxy_mass']==0)
            check(f'prefix {i} observations',sum(r['observations'] for r in rec['oracle'])==row['native_calls']==rec['native_diagnostics']['probability_calls'] and row['duplicate_checks']==row['native_calls']-len(table))
        else:
            U=arrays['uniforms'];outputs=arrays['outputs'];check(f'sampling {i} fixed uniforms',np.array_equal(U,np.random.default_rng(row['seed']).random((8,144))));records=rec['conditionals']
            for k,a in enumerate(rec['attempts']):
                path=();observed=[r for r in records if r['attempt']==k];failure=False;consumed=0
                for r in observed:
                    check(f'sampling {i}:{k}:{len(path)} prefix',tuple(r['prefix'])==path)
                    if r['q_hex'] is None:failure=True;break
                    q=float.fromhex(r['q_hex']);check(f'sampling {i}:{k}:{len(path)} guarded q',0<=q<=1);path+=(int(float(U[k,consumed])<q),);consumed+=1
                check(f'sampling {i}:{k} replay',consumed==a['uniforms_consumed'] and ((failure and a['outcome']=='numerical_failure' and tuple(a['failed_prefix'])==path and np.all(outputs[k]==-1)) or (not failure and a['outcome']=='completed' and len(path)==144 and np.array_equal(outputs[k],path))))
            check(f'sampling {i} no hidden attempts',len(rec['attempts'])==8 and len(records)==row['native_calls']==rec['native_diagnostics']['probability_calls'])
    for first in (0,18,54,136):
        target=read(panel/f'target_{first:03d}.json');V=np.load(panel/f'target_{first:03d}.npz')['covariance'];vac=[(F(1),F(1))]
        for c in target['certificates']:
            mask=c['mask'];modes=[j for j in range(8) if mask>>j&1];ix=modes+[j+8 for j in modes];A=V[np.ix_(ix,ix)];Q=(fraction_array(A)+np.eye(len(ix),dtype=object))/2;check_certificate(c,Q,label=f'target {first}:{mask}');lo,hi=c['determinant_interval'];a,b=target['vacua'][mask];check(f'vacuum root {first}:{mask}',0<=a<=b and a*a<=1/hi<=1/lo<=b*b);vac.append((a,b))
            if mask in (1,3,15,255):
                mm=mp.matrix([[mp.mpf(F(x).numerator)/F(x).denominator for x in rr] for rr in Q]);value=1/mp.sqrt(mp.det(mm));check(f'target 90 digit {first}:{mask}',mp.mpf(a.numerator)/a.denominator<=value<=mp.mpf(b.numerator)/b.denominator);mp_count+=1
        for clicks in range(256):
            zeros=255^clicks;lo=hi=F(0)
            for subset in range(256):
                if subset&~clicks:continue
                a,b=vac[zeros|subset]
                if subset.bit_count()%2:lo-=b;hi-=a
                else:lo+=a;hi+=b
            check(f'target atom {first}:{clicks}',tuple(target['raw_atom_intervals'][clicks])==(lo,hi) and tuple(target['target_intervals'][clicks])==(max(F(0),lo),min(F(1),hi)))
        check(f'target interval mass {first}',sum(a for a,b in target['target_intervals'])<=1<=sum(b for a,b in target['target_intervals']))
    check('verification sources unchanged',source_hashes=={n:digest(ROOT/n) for n in sources});deps['experiments/verify_iteration34.py']=digest(ROOT/'experiments/verify_iteration34.py');save(out/'checks.json',checks);save(out/'high_precision.json',high);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),matrix_certificates=cert_count,high_precision_crosschecks=mp_count,exact_small_determinants=small_exact,source_hashes=source_hashes,dependencies=deps));print('verified',len(checks),'certificates',cert_count,'high precision',mp_count,flush=True)
if __name__=='__main__':main()
