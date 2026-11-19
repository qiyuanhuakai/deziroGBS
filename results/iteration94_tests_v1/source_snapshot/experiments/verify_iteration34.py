"""Independent atom-by-atom reconstruction and exact matrix-witness checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,math,shutil,sys
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import numpy as np
import mpmath as mp
from iteration22_external import digest,save
ROOT=Path(__file__).resolve().parents[1]

def decode(v):
    if isinstance(v,dict):return F(v['rational']) if set(v)=={'rational'} else {k:decode(x) for k,x in v.items()}
    if isinstance(v,list):return [decode(x) for x in v]
    return v

def exact_determinant(a):
    a=[[F(x) for x in row] for row in a];n=len(a);answer=F(1)
    for j in range(n):
        pivot=next((i for i in range(j,n) if a[i][j]),None)
        if pivot is None:return F(0)
        if pivot!=j:a[j],a[pivot]=a[pivot],a[j];answer=-answer
        d=a[j][j];answer*=d
        for i in range(j+1,n):
            scale=a[i][j]/d
            for k in range(j+1,n):a[i][k]-=scale*a[j][k]
    return answer

def dyadic_integers(a):
    ratios=[float(x).as_integer_ratio() for x in np.asarray(a).flat];den=max(d for n,d in ratios);return np.array([n*(den//d) for n,d in ratios],dtype=object).reshape(a.shape),den

def reconstruct(m,chosen,intervals,table,bits):
    masses=[];failures={};used=set()
    for pattern in chosen:
        weight=F(1);prefix=();clicks=set(pattern)
        for j in range(m):
            used.add(prefix);q=table[prefix]
            if q is None:
                if prefix in failures:assert failures[prefix]==weight
                failures[prefix]=weight;weight=F(0);break
            if bits is not None:q=F(-((-q.numerator*(1<<bits))//q.denominator),1<<bits)
            bit=int(j in clicks);weight*=q if bit else 1-q
            if not weight:break
            prefix+=(bit,)
        masses.append(weight)
    failure=sum(failures.values(),F(0));selected=sum(masses,F(0));overlap=sum((min(bounds[0],q) for bounds,q in zip(intervals,masses)),F(0));lower=max(sum(max(F(0),bounds[0]-q) for bounds,q in zip(intervals,masses)),failure+sum(max(F(0),q-bounds[1]) for bounds,q in zip(intervals,masses)))
    return dict(proxy_masses=masses,selected_proxy_mass=selected,target_selected_lower=sum(lo for lo,hi in intervals),target_selected_upper=min(F(1),sum(hi for lo,hi in intervals)),confirmed_overlap=overlap,tv_lower=lower,tv_upper=1-overlap,seen_failure_mass=failure,unselected_proxy_mass=1-selected-failure,queries=len(used)),failures,used

def main():
    sys.set_int_max_str_digits(0)
    out=ROOT/'results/iteration34_validation';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);checks=[]
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
    def check(name,passed,**values):
        checks.append(dict(name=name,passed=bool(passed),**values))
        if not passed:save(out/'checks.failed.json',checks);raise AssertionError(name)
    panel=ROOT/'results/iteration34_panel';rows=read(panel/'rows.json');plan=read(panel/'plan.json');summary=read(panel/'summary.json');dependencies=dict(plan['dependencies']);recovered=False
    if summary['completed']!=42:
        check('preserved decimal serialization failure',summary['completed']==41 and [r['sequence'] for r in rows if r['status']!='completed']==[37] and 'integer string conversion' in rows[37]['traceback']);recovery=ROOT/'results/iteration34_recovery';rp=read(recovery/'plan.json');rs=read(recovery/'summary.json');check('recovery outcome',rs['planned']==rs['finished']==rs['completed']==1 and rs['unchanged_sources'] and rs['original_failure_preserved'] and rp['integer_string_digit_limit']==0)
        for n,v in rp['source_hashes'].items():check('recovery source '+n,digest(ROOT/n)==v);check('recovery snapshot '+n,digest(recovery/'source_snapshot'/n)==v)
        for n,v in rp['dependencies'].items():check('recovery dependency '+n,digest(ROOT/n)==v);dependencies[n]=v
        rows[37]=read(recovery/'case_37.json');recovered=True
    check('effective formal jobs',len(rows)==summary['planned']==summary['finished']==42 and all(r['status']=='completed' for r in rows) and summary['unchanged_sources'])
    for n,v in dependencies.items():check('dependency '+n,digest(ROOT/n)==v)
    for n,v in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==v);check('source snapshot '+n,digest(panel/'source_snapshot'/n)==v)
    pilot=ROOT/'results/iteration34_pilot';p=read(pilot/'summary.json');check('pilot unchanged',p['unchanged_sources'])
    for n,v in p['source_hashes'].items():check('pilot snapshot '+n,digest(pilot/'source_snapshot'/n)==v)
    for n,v in p['dependencies'].items():check('pilot dependency '+n,digest(ROOT/n)==v);dependencies[n]=v
    build=read(ROOT/'results/iteration31_build/build.json')
    for n,v in build['source_hashes'].items():check('native binding '+n,digest(ROOT/n)==v);dependencies[n]=v
    prior=read(ROOT/'results/iteration33_panel/rows.json');integrals={};integral_rows=[];exact_complements=0;matrix_certificates=0;large_direct=[];total_duplicate_checks=0;maximum_bound_width=F(0)
    with mp.workdps(110):
        convert=lambda x:mp.mpf(x.numerator)/x.denominator
        for i,row in enumerate(rows):
            print('verifying',i,flush=True);case_panel=ROOT/'results/iteration34_recovery' if recovered and i==37 else panel;data=decode(read(case_panel/f'case_{i:02d}.audit.json'));chosen=[tuple(c) for c in data['patterns']];intervals=data['target_intervals'];table={tuple(map(int,r['prefix'])):None if r['q_hex'] is None else F(float.fromhex(r['q_hex'])) for r in data['oracle']};m=row['modes'];check(f'{i} target interval domain',all(0<=lo<=hi<=1 for lo,hi in intervals) and sum(lo for lo,hi in intervals)<=1)
            check(f'{i} conditional domain',all(q is None or 0<=q<=1 for q in table.values()))
            if row['panel']=='reused':
                old=np.load(ROOT/f"results/iteration32_panel/case_{row['source_case']:02d}.npz");p=[F(float(v)) for v in old['target']];total=sum(p);p=[v/total for v in p];check(f'{i} target normalization',all(lo==hi==p[sum(1<<j for j in c)] for c,(lo,hi) in zip(chosen,intervals)))
                for h,q in table.items():check(f'{i} reused conditional '+str(h),q==F(float(old['conditionals'][(1<<len(h))-1+sum(b<<j for j,b in enumerate(h))])))
            else:
                saved=np.load(case_panel/f'case_{i:02d}.npz');V=saved['covariance'];R=saved['R'];n=2*m;check(f'{i} triangular witness',np.array_equal(R,np.tril(R)) and np.all(np.diag(R)>0));A,dv=dyadic_integers(V);T,dr=dyadic_integers(R);Qint=A+np.eye(n,dtype=object)*dv;scale=2*dv*dr*dr
                # Different multiplication association from the primary run.
                E=T@(Qint@T.T)-np.eye(n,dtype=object)*scale;delta=F(max(sum(abs(int(x)) for x in line) for line in E),scale);cert=data['gaussian_certificate'];check(f'{i} exact whitening witness',delta==cert['diagnostics']['whitening_residual_inf'] and delta<1);matrix_certificates+=1
                detR=math.prod((F(float(v)) for v in np.diag(R)),start=F(1));cert_atoms={tuple(r['pattern']):r['interval'] for r in cert['atoms']};lo,hi=cert_atoms[()];rawlo=detR/(1+delta)**m;rawhi=detR/(1-delta)**m;check(f'{i} exact vacuum enclosure',lo<=rawlo<=rawhi<=hi and rawlo-lo<F(1,1<<128) and hi-rawhi<F(1,1<<128))
                if row['panel']!='original':
                    e=row['exponent']
                    if row['family']=='thermal':
                        v=1+F(1,1<<e);check(f'{i} exact thermal input',all(F(float(V[a,b]))==(v if a==b else 0) for a in range(n) for b in range(n)))
                    else:
                        a=[F(-1 if j%3==1 else 1,1<<e) for j in range(n)];check(f'{i} exact physical classical displacement',all(F(float(V[j,k]))==int(j==k)+a[j]*a[k] for j in range(n) for k in range(n)))
                    by_count={len(c):bounds for c,bounds in zip(chosen,intervals)}
                    for k,(lo,hi) in sorted(by_count.items()):
                        if row['family']=='thermal':
                            q=F(1,(1<<(e+1))+1);expected=q**k*(1-q)**(m-k);check(f'{i} analytic thermal atom {k}',lo==hi==expected)
                        else:
                            key=(m,e,k)
                            if key not in integrals:
                                intensity=mp.mpf(2)**(-2*e)/2
                                f=lambda z:mp.exp(-z*z/2-(m-k)*intensity*z*z)*(-mp.expm1(-intensity*z*z))**k
                                integrals[key]=mp.sqrt(2/mp.pi)*mp.quad(f,[0,1,3,6,mp.inf]);integral_rows.append(dict(modes=m,exponent=e,clicks=k,value=mp.nstr(integrals[key],100)))
                            check(f'{i} independent positive Gaussian integral {k}',convert(lo)<=integrals[key]<=convert(hi))
                    for c,(lo,hi) in cert_atoms.items():
                        a,b=by_count[len(c)];check(f'{i} Gaussian certificate contains analytic target '+str(c),lo<=a<=b<=hi)
                if m<=8:
                    Q=[[(F(float(V[a,b]))+int(a==b))/2 for b in range(n)] for a in range(n)]
                    for removed in cert_atoms:
                        keep=[j for j in range(n) if j%m not in removed];D=exact_determinant([[Q[a][b] for b in keep] for a in keep]);terms=[cert_atoms[c] for k in range(len(removed)+1) for c in combinations(removed,k)];lo=sum(a for a,b in terms);hi=sum(b for a,b in terms);check(f'{i} independent exact complement determinant '+str(removed),lo>=0 and lo*lo*D<=1<=hi*hi*D);exact_complements+=1
                if row['panel']=='original':
                    reference={}
                    for removed in ((),(0,),(54,),(135,)):
                        keep=[j for j in range(n) if j%m not in removed];Q=mp.matrix([[(mp.mpf(float(V[a,b]))+int(a==b))/2 for b in keep] for a in keep]);L=mp.cholesky(Q);reference[removed]=1/mp.fprod(L[j,j] for j in range(len(keep)));value=reference[removed] if not removed else reference[removed]-reference[()];lo,hi=cert_atoms[removed];check(f'{i} direct 110-digit complement '+str(removed),convert(lo)<=value<=convert(hi));large_direct.append(dict(pattern=removed,value=mp.nstr(value,100)))
                observations=sum(r['observations'] for r in data['oracle']);check(f'{i} native calls',observations==row['native_calls']==row['native_diagnostics']['probability_calls']);check(f'{i} duplicate checks',observations-len(table)==row['duplicate_prefix_checks']);total_duplicate_checks+=row['duplicate_prefix_checks']
                check(f'{i} one reset per attempted selected trajectory',row['native_diagnostics']['reset_requests']==len(chosen))
            for bits in (None,53):
                previous_upper=F(1);previous_lower=F(0)
                for stored in (a for a in data['audits'] if a['uniform_bits']==bits):
                    indices=[j for j,c in enumerate(chosen) if len(c)<=stored['max_clicks']];subset=[chosen[j] for j in indices];bounds=[intervals[j] for j in indices];expected,failures,used=reconstruct(m,subset,bounds,table,bits)
                    for key,value in expected.items():check(f'{i} bits {bits} layer {stored["max_clicks"]} '+key,stored[key]==value)
                    check(f'{i} bits {bits} failure atoms {stored["max_clicks"]}',{tuple(r['prefix']):r['mass'] for r in stored['failures']}==failures)
                    check(f'{i} bits {bits} monotonic {stored["max_clicks"]}',previous_lower<=stored['tv_lower']<=stored['tv_upper']<=previous_upper);previous_lower=stored['tv_lower'];previous_upper=stored['tv_upper']
                    check(f'{i} bits {bits} conservation {stored["max_clicks"]}',stored['selected_proxy_mass']+stored['seen_failure_mass']+stored['unselected_proxy_mass']==1)
                    if stored['max_clicks']==m:
                        width=stored['tv_upper']-stored['tv_lower'];maximum_bound_width=max(maximum_bound_width,width);check(f'{i} bits {bits} full interval closes',width<=sum(hi-lo for lo,hi in bounds))
                        if row['panel']=='reused' and bits is None:check(f'{i} prior complete TV',abs(float(stored['tv_upper'])-prior[row['source_case']-24]['metrics']['extended_tv'])<1e-20)
            if row['panel']!='reused':
                observed={tuple(map(int,r['prefix'])):0 for r in data['oracle']}
                for c in chosen:
                    h=();clicks=set(c)
                    for j in range(m):
                        observed[h]+=1;q=table[h]
                        if q is None:break
                        bit=int(j in clicks)
                        if (q if bit else 1-q)==0:break
                        h+=(bit,)
                check(f'{i} all observation counts',all(observed[tuple(map(int,r['prefix']))]==r['observations'] for r in data['oracle']))
    physical=ROOT/'results/iteration34_physicality';ph=read(physical/'plan.json');ps=read(physical/'summary.json');check('physicality supplemental jobs',ps['completed']==ps['planned']==5 and ps['unchanged_sources']);physical_rows=decode(read(physical/'rows.json'))
    for n,v in ph['source_hashes'].items():check('physicality source '+n,digest(ROOT/n)==v);check('physicality snapshot '+n,digest(physical/'source_snapshot'/n)==v)
    for n,v in ph['dependencies'].items():check('physicality dependency '+n,digest(ROOT/n)==v);dependencies[n]=v
    for i,r in enumerate(physical_rows):
        d=np.load(physical/f'case_{i:02d}.npz');V=d['covariance'];w=d['witness'];a=[F(float(x)) for x in w.real];b=[F(float(x)) for x in w.imag];n=len(a);m=n//2
        diagonal=sum((F(float(V[j,j]))*(a[j]**2+b[j]**2) for j in range(n)),F(0));off_diagonal=2*sum((F(float(V[j,k]))*(a[j]*a[k]+b[j]*b[k]) for j in range(n) for k in range(j+1,n)),F(0));symplectic=2*sum((a[j]*b[j+m]-a[j+m]*b[j] for j in range(m)),F(0));form=diagonal+off_diagonal-symplectic;norm=sum(x*x+y*y for x,y in zip(a,b));quotient=form/norm
        check(f'physicality exact witness {i}',norm>0 and form==r['exact_quadratic_form'] and norm==r['exact_norm_squared'] and quotient==r['exact_rayleigh']);check(f'physicality sign {i}',(quotient<0)==r['strict_violation_proved']==(i==0))
        if i==0:check('physicality original input identity',np.array_equal(V,np.load(ROOT/'results/iteration31_panel/large_input.npz')['covariance']))
        else:
            exponent=int(r['job'].split('_')[-1])
            if r['job'].startswith('thermal'):check(f'physicality thermal exact control {i}',quotient==F(1,1<<exponent))
            else:
                vector=[F(-1 if j%3==1 else 1,1<<exponent) for j in range(n)];check(f'physicality exact PSD construction {i}',all(F(float(V[j,k]))==int(j==k)+vector[j]*vector[k] for j in range(n) for k in range(n)))
    check('verifier unchanged',digest(source)==sha);save(out/'checks.json',checks);save(out/'positive_integrals.json',integral_rows);save(out/'large_direct_complements.json',large_direct);result=dict(count=len(checks),passed=sum(c['passed'] for c in checks),dependencies=dependencies,verifier_sha256=sha,matrix_certificates=matrix_certificates,independent_exact_small_complements=exact_complements,independent_positive_integrals=len(integral_rows),independent_large_direct_complements=len(large_direct),runtime_duplicate_prefix_checks=total_duplicate_checks,maximum_full_target_interval_width=float(maximum_bound_width),physicality_witnesses=5,strict_original_uncertainty_violation=True,original_exact_rayleigh_float=float(physical_rows[0]['exact_rayleigh']));save(out/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='dependencies'},indent=2))
if __name__=='__main__':main()
