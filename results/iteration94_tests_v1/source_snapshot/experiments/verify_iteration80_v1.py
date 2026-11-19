"""Independent frozen-evidence checks; no sampler DLL is loaded.
Numerical Fock/linear-algebra references and ideal-iid confidence assumptions
remain explicit. Exact rational checks apply only to integer alias table laws.
"""
from pathlib import Path
import os,json,hashlib,argparse,math,statistics
from fractions import Fraction
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
checks=[]
def check(ok,label):
    checks.append(label)
    assert ok,label
for base in sorted((ROOT/'results').glob('iteration80*')):
    if not base.is_dir():continue
    if (base/'manifest.json').exists():
        for name,digest in read(base/'manifest.json')['artifacts'].items():check(H(base/name)==digest,'artifact '+base.name+'/'+name)
    if (base/'plan.json').exists():
        plan=read(base/'plan.json')
        for name,digest in plan.get('hashes',{}).items():check(H(ROOT/name)==digest,'executed source/binary '+name)
        for key in ('source_sha256','driver_sha256'):
            if key in plan:check(H(ROOT/'experiments'/(base.name+'.py'))==plan[key],'driver '+base.name)
tests=read(ROOT/'results/iteration80_tests_v1/validation.json');check(tests['passed'] and tests['transport80_passed']==13,'full suite and 13 new tests')
for name,digest in read(ROOT/'results/iteration80_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
partial={}
for name in ('iteration80_transport_pilot_v2','iteration80_alias_audit_v1'):
    base=ROOT/'results'/name;check(not (base/'summary.json').exists(),'retained incomplete run '+name);partial[name]={p.name:H(p) for p in base.iterdir() if p.is_file()}
# Independently check finite-design primal/dual constraints.
base=ROOT/'results/iteration80_pair_transport_preflight_v1';rows=read(base/'summary.json')['rows']
for row in rows:
    d=np.load(base/(row['label']+'.npz'));v,w,p,Q,R=d['v'],d['weights'],d['marginals'],d['Q'],d['R'];m,n=v.shape;ell=(abs(v)**2).sum(axis=1)
    check(np.max(abs(abs(v@v.T)**2/n-Q))<1e-14 and np.max(abs((np.outer(ell,ell)+abs(v@v.conj().T)**2+abs(v@v.T)**2)/(n*(n+2))-R))<1e-14,'Q R independently reconstructed')
    check(np.max(abs(p.T@(p*w[:,None])-R))<1e-12,'real design fourth moment')
    if row['success']:
        P=d['couplings'];check(P.min()>=0 and np.max(abs(P.sum(axis=2)-p))<1e-8 and np.max(abs(np.einsum('l,lij->ij',w,P)-Q))<1e-8,'LP primal constraints')
    else:
        dual=d['dual'];nr=len(w)*m;original=[]
        for l in range(len(w)):
            for k,(i,j) in enumerate(d['pairs']):original.append(dual[l*m+i]+(dual[l*m+j] if i!=j else 0)+w[l]*dual[nr+k])
        objective=float(d['rhs']@dual);check(max(original)<=1e-12 and abs(dual).max()<=1+1e-12 and objective>.12,'finite design numerical infeasibility dual');check(abs(objective-row['minimum_l1_constraint_slack'])<1e-10,'dual primal objective equality')
base=ROOT/'results/iteration80_ipf_pilot_v1'
for row in read(base/'summary.json')['rows']:
    if not row['success']:continue
    d=np.load(base/(row['label']+'.npz'));P,w,p,Q=d['couplings'],d['weights'],d['marginals'],d['Q'];check(P.min()>=0 and np.max(abs(P-P.transpose(0,2,1)))<1e-14 and np.max(abs(P.sum(axis=2)-p))<1e-8 and np.max(abs(np.einsum('l,lij->ij',w,P)-Q))<1e-8,'IPF constraints')
# Full small law via vector-valued polynomial recurrence (all latents together).
for version,calibration in ((1,'iteration80_pair_transport_preflight_v1'),(3,'iteration80_ipf_pilot_v1')):
    base=ROOT/f'results/iteration80_transport_pilot_v{version}';d=np.load(ROOT/'results'/calibration/'strong4.npz');P,w=d['couplings'],d['weights'];m=P.shape[1];law={(0,)*m:np.ones(len(w))};pairs=[]
    for i in range(m):
        for j in range(i,m):x=np.zeros(m,int);x[i]+=1;x[j]+=1;pairs.append((tuple(x),P[:,i,j]*(1 if i==j else 2)))
    for N in range(2,15,2):
        new={}
        for x,values in law.items():
            for y,masses in pairs:
                key=tuple(a+b for a,b in zip(x,y));new[key]=new.get(key,0)+values*masses
        law=new
        if N not in (2,4,8,14):continue
        saved=np.load(base/f'law_{N}.npz');prob=np.array([w@law[tuple(x)] for x in saved['states']]);check(np.max(abs(prob-saved['law']))<1e-14,'independent complete polynomial law');x=saved['states'].astype(float);pair=(x.T@(x*prob[:,None])-np.diag(prob@x))/(N*(N-1));target=(d['Q']+(N-2)*d['R'])/(N-1);check(np.max(abs(pair-target))<1e-9,'full law ordered-pair moment')
        lookup={tuple(x):i for i,x in enumerate(saved['states'])};hist=np.bincount([lookup[tuple(x)] for x in saved['counts']],minlength=len(prob));check(np.array_equal(hist,saved['histogram']) and np.all(saved['counts'].sum(axis=1)==N),'fixed total raw histogram')
# Reconstruct all fresh 1M and 5M bounds without reusing saved histograms.
reference=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');radix=int(reference['cutoff'])+1;codes=reference['codes'];target=reference['target'];powers=np.array([radix**j for j in range(4)],np.uint64);lookup=np.full(radix**4,-1,np.int32);lookup[codes]=np.arange(len(codes));
for version in (1,2):
    base=ROOT/f'results/iteration80_pnr_quality_v{version}';summary=read(base/'summary.json');plan=read(base/'plan.json');N=plan['samples_per_law'];radius=math.sqrt(math.log(800)/(2*N))
    for row in summary['rows']:
        panel=row['panel'];pilot=np.load(base/f'pilot_{panel}.npz');hist=np.zeros(len(target),np.int64);hc=np.zeros(16,np.int64);photons=odd=maximum=0
        for path in sorted(base.glob(f'counts_{panel}_*.npz')):
            x=np.load(path)['counts'];total=x.sum(axis=1);inside=total<int(reference['cutoff'])+1;labels=np.full(len(x),len(codes),np.int32);labels[inside]=lookup[(x[inside].astype(np.uint64)@powers).astype(np.int64)];check(labels.min()>=0,'complete Fock outcome lookup');hist+=np.bincount(labels,minlength=len(target));hc+=np.bincount(((x>0).astype(np.int64)@2**np.arange(4)),minlength=16);photons+=int(total.sum());odd+=int(np.count_nonzero(total%2));maximum=max(maximum,int(total.max()))
        check(hist.sum()==N and odd==0 and photons/N==row['mean_photons'] and maximum==row['maximum_photons'],'fresh record counts parity and moments')
        for key,h,t,event,tail in [('pnr',hist,target,pilot['pnr_event'],target[-1]),('click',hc,pilot['click_target'],pilot['click_event'],0.)]:
            empirical=h/N;tv=float(abs(empirical-t).sum()/2);lo=max(0.,float((empirical[event]-t[event]).sum())-radius);hi=min(1.,tv+radius+tail);check(max(abs(tv-row[key]['empirical_tv']),abs(lo-row[key]['lower']),abs(hi-row[key]['upper']))<1e-14,'independent nominal 99 percent TV bounds')
# Exact rational alias table audits, including tiny positive events.
for case in ('strong4','identity2'):
    result=read(ROOT/f'results/iteration80_alias_audit_v2/{case}.json');d=np.load(ROOT/f'results/iteration80_ipf_pilot_v1/{case}.npz');P=d['couplings'];m=P.shape[1];inputs=[d['weights']]+[np.array([p[i,j]*(1 if i==j else 2) for i in range(m) for j in range(i,m)]) for p in P];deltas=[]
    for row,weights in zip(result['audits'],inputs):
        levels=1<<(64-row['bits']);counts=[0]*len(row['thresholds'])
        for i,(threshold,alias) in enumerate(zip(row['thresholds'],row['aliases'])):counts[i]+=threshold;counts[alias]+=levels-threshold
        check(sum(counts)==2**64 and counts==list(map(int,row['word_counts'])),'integer alias exhaustive word counts');ratios=[float(w).as_integer_ratio() for w in weights];den=max(b for a,b in ratios);ints=[a*(den//b) for a,b in ratios];s=sum(ints);delta=Fraction(sum(abs(c*s-v*2**64) for c,v in zip(counts,ints)),2*2**64*s);check(delta==Fraction(int(row['tv_numerator']),int(row['tv_denominator'])),'exact rational TV');deltas.append(delta)
    bound=deltas[0]+7*max(deltas[1:]);check(bound==Fraction(int(result['N14_bound_numerator']),int(result['N14_bound_denominator'])),'exact rational coupling bound')
# Wide constraints, fixed-total samples, cached replay, and independent vacuum coefficients.
def coeff(eigen,exponent,degree):
    h=[np.longdouble(1)];traces=[exponent*np.sum(eigen.astype(np.longdouble)**k) for k in range(1,degree+1)]
    for j in range(1,degree+1):h.append(sum(traces[k-1]*h[j-k] for k in range(1,j+1))/j)
    return h[-1]
def norm(a,j):
    value=np.longdouble(1)
    for k in range(1,j+1):value*=np.longdouble(a+k-1)/k
    return value
def mobius(vac):
    law=vac[np.arange(len(vac))^(len(vac)-1)].copy()
    for bit in range((len(vac)-1).bit_length()):
        for mask in range(len(vac)):
            if mask>>bit&1:law[mask]-=law[mask^(1<<bit)]
    return law
for case in range(4):
    base=ROOT/'results/iteration80_wide_pilot_v1';d=np.load(base/f'input_{case}.npz');v,w,p=d['v'],d['weights'],d['marginals'];m,n=v.shape;P=np.concatenate([np.load(base/f'table_{case}_{first}.npz')['couplings'] for first in range(0,len(w),64)])
    check(P.min()>=0 and np.max(abs(P.sum(axis=2)-p))<1e-8 and np.max(abs(np.einsum('l,lij->ij',w,P)-d['Q']))<1e-8,'wide calibrated constraints');check(np.max(abs(p.T@(p*w[:,None])-d['R']))<1e-12,'wide design moment')
    counts=np.load(base/f'counts_{case}.npz')['counts'];check(np.array_equal(counts,np.load(ROOT/f'results/iteration80_cached_pilot_v1/counts_{case}.npz')['counts']) and np.all(counts.sum(axis=1)==34),'100k cached bitwise replay')
    P/=P.sum(axis=(1,2))[:,None,None];w=w/w.sum();marginal=P.sum(axis=2);vac=np.ones(256,np.longdouble);transport=vac.copy()
    for mask in range(1,256):
        ids=[i for i in range(8) if mask>>i&1];remaining=v[[i for i in range(m) if i not in ids]];_,s,Vh=np.linalg.svd(remaining,full_matrices=False);f=s[:,None]*Vh;eigen=np.linalg.svd(f@f.T,compute_uv=False)**2;vac[mask]=coeff(eigen,.5,17)/norm(n/2,17);nohit=1-2*marginal[:,ids].sum(axis=1)+P[:,ids][:,:,ids].sum(axis=(1,2));transport[mask]=np.dot(w.astype(np.longdouble),nohit.astype(np.longdouble)**17)
    saved=np.load(ROOT/f'results/iteration80_marginal_checks_v1/conditional_{case}.npz');purelaw,law=mobius(vac),mobius(transport);check(np.max(abs(purelaw-saved['pure']))<1e-11 and np.max(abs(law-saved['transport']))<1e-12,'wide pure and transport marginal reconstructed');check(law.min()>=0 and purelaw.min()>=0,'no negative probability clipping')
resource=read(ROOT/'results/iteration80_wide_pilot_v1/case_4.json');check(not resource['success'] and resource['full_table_bytes']==77460406272,'explicit 77.46 GB construction refusal')
# All predeclared direct order laws, including worse alternatives.
base=ROOT/'results/iteration80_order_pilot_v1'
for row in read(base/'summary.json')['rows']:
    check(row['success'],'all ordering panels completed')
    if row['case']>=2:continue
    d=np.load(base/f"row_{row['case']}_{row['method']}_{row['budget']}.npz");order=np.load(base/f"input_{row['case']}.npz")['orders'][['original','click_ascending','click_descending','spectral','neighbor','degree_ascending'].index(row['method'])];prob=d['conditional_probabilities'];law=np.prod(np.where(d['patterns'][:,order],prob,1-prob),axis=1);check(np.max(abs(law-d['cpu_law']))<1e-9 and abs(abs(law-d['target']).sum()/2-row['numerical_full_tv'])<1e-14,'physical-port complete direct click law')
for kind in ('rolling','guarded'):
    summary=read(ROOT/f'results/iteration80_{kind}_pilot_v1/summary.json');check('Invalid' in json.dumps(summary) or 'invalid' in json.dumps(summary),'retained explicit invalid probability failures '+kind)
# Complete matched timing panels, no partial means or cross-run splicing.
base=ROOT/'results/iteration80_timing_v1';timing=read(base/'summary.json');check(len(timing['rows'])==54 and all(r['success'] for r in timing['rows']),'54 fresh complete timing panels')
for row in timing['rows']:
    c,r,k,b=(row[key] for key in ('case','rep','kind','budget'));d=np.load(base/f'counts_{c}_{r}_{k}_{b}.npz');check(d['clicks'].shape==(row['records'],row['outputs']) and np.isin(d['clicks'],[0,1]).all(),'timing complete records')
    if 'counts' in d:check(np.array_equal(d['counts']>0,d['clicks']),'PNR timing threshold conversion')
for key,median in timing['medians'].items():
    c,k,b=key.split('_');values=[r['wall_seconds'] for r in timing['rows'] if (r['case'],r['kind'],r['budget'])==(int(c),k,int(b))];check(len(values)==3 and statistics.median(values)==median,'three repetition median')
for rep in range(3):check(np.array_equal(np.load(base/f'batch_cached_{rep}.npz')['counts'],np.load(base/f'batch_uncached_{rep}.npz')['counts']),'100k fresh cached and uncached identical')
report=dict(passed=True,checks=len(checks),full_suite=tests,partial_failure_artifacts=partial,scope=__doc__)
(OUT/'validation.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());(OUT/'checks.json').write_bytes((json.dumps(checks,indent=2)+'\n').encode());print(json.dumps(dict(passed=True,checks=len(checks))),flush=True)
