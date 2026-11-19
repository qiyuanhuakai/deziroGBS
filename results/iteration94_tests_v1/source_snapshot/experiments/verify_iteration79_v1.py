"""Reconstruct frozen full laws, photon-pair matrices, raw statistics and timings.
This verifier uses no sampling DLL; numerical reference and iid assumptions remain.
"""
from pathlib import Path
import os,json,hashlib,argparse,math,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=100
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
checks=[]
def check(ok,label):
    checks.append(label)
    assert ok,label
for base in sorted((ROOT/'results').glob('iteration79*')):
    if base.is_dir() and (base/'manifest.json').exists():
        for name,digest in read(base/'manifest.json')['artifacts'].items():check(H(base/name)==digest,'frozen '+base.name+'/'+name)
    if base.is_dir() and (base/'plan.json').exists():
        plan=read(base/'plan.json')
        for name,digest in plan.get('hashes',plan.get('source_hashes',{})).items():check(H(ROOT/name)==digest,'executed source/binary '+name)
tests=read(ROOT/'results/iteration79_tests_v1/validation.json');check(tests['passed'] and tests['joint_pnr_passed']==6 and tests['block_click_passed']==5 and tests['block_philox_passed']==4,'full suite with 15 new interface tests')
for name,digest in read(ROOT/'results/iteration79_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
base=ROOT/'results/iteration79_pilot_v1';summary=read(base/'summary.json');check(summary['completed'] and all(r['success'] for group in ('click','large') for r in summary[group]),'CPU pilot completed')
d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');radix=int(d['cutoff'])+1;states=np.array([(d['codes']//radix**j)%radix for j in range(4)]).T;totals=states.sum(axis=1);v=d['v'];n=v.shape[1];ell=(abs(v)**2).sum(axis=1);A=abs(v@v.T)**2;C=abs(v@v.conj().T)**2;M=np.outer(ell,ell)/n**2;Q=A/n;R=(np.outer(ell,ell)+C+A)/(n*(n+2))
for row in summary['joint']:
    N,b=row['total'],row['block'];law={(0,0,0,0):1.};left=N;sizes=[]
    while left:
        size=min(left,b);sizes.append(size);mask=totals==size;factor=d['target'][:-1][mask];factor=factor/factor.sum();new={}
        for first,p0 in law.items():
            for second,q in zip(states[mask],factor):key=tuple(np.array(first)+second);new[key]=new.get(key,0.)+p0*q
        law=new;left-=size
    saved=np.load(base/f"joint_{row['case']}.npz");ground=np.array([law[tuple(x)] for x in saved['states']]);check(np.max(abs(ground-saved['target']))<1e-14,'independent block convolution')
    x=saved['states'].astype(float);pair=(x.T@(x*ground[:,None])-np.diag(ground@x))/(N*(N-1));k=sum(s*(s-2) for s in sizes)/(N*(N-1));formula=Q/(N-1)+k*R+((N-2)/(N-1)-k)*M;check(np.max(abs(pair-formula))<2e-13,'block pair formula from full Fock law')
    lookup={tuple(x):i for i,x in enumerate(saved['states'])};hist=np.bincount([lookup[tuple(x)] for x in saved['counts']],minlength=len(x));check(np.array_equal(hist,saved['histogram']) and abs(abs(hist/100000-ground).sum()/2-row['empirical_tv_to_block_law'])<1e-14,'fresh fixed-total histogram');check(np.all(saved['counts'].sum(axis=1)==N) and abs(abs(ground-saved['pure_target']).sum()/2-row['numerical_block_law_tv_to_pure'])<1e-14,'fixed total and full approximation error')
check(np.array_equal(np.load(base/'strict_replay.npz')['counts'],np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz')['counts']),'strict branch unchanged')
for row in summary['click']:
    d=np.load(base/f"click_{row['case']}_{row['width']}_{row['budget']}.npz");check(abs(d['law'].sum()-1)<1e-9 and np.all(d['law']>=0) and abs(abs(d['law']-d['target']).sum()/2-row['numerical_full_tv'])<1e-14,'CPU complete click law')
base=ROOT/'results/iteration79_gpu_pilot_v1';gpu=read(base/'summary.json');check(gpu['completed'] and all(r['success'] for group in ('laws','displaced','large') for r in gpu[group]),'GPU pilot completed without discarded panels')
for row in gpu['laws']:
    d=np.load(base/f"law_{row['case']}_{row['width']}_{row['budget']}.npz");law=np.prod(np.where(d['patterns'],d['probabilities'],1-d['probabilities']),axis=1);check(np.array_equal(law,d['law']) and np.max(abs(law-d['cpu_law']))<1e-9 and abs(law.sum()-1)<1e-9,'GPU complete law and independent CPU Mobius oracle');check(abs(abs(law-d['target']).sum()/2-row['numerical_tv'])<1e-14,'GPU full numerical TV');check(np.isin(d['clicks'],[0,1]).all() and np.all((d['draws']>=0)&(d['draws']<1)),'GPU records and Philox range')
    if row['width']==1:check(row['width1_bitwise_replay'],'old GPU bitwise replay')
for row in gpu['displaced']:
    d=np.load(base/f"displaced_{row['width']}_{row['budget']}.npz");law=np.prod(np.where(d['clicks'],d['conditional_click_probabilities'],1-d['conditional_click_probabilities']),axis=1);check(np.array_equal(law,d['law']) and np.max(abs(law-d['oracle']))<1e-9,'displaced trajectory mass oracle')
base=ROOT/'results/iteration79_joint_preflight_v1'
for case in (0,1):
    d=np.load(base/f'input_{case}.npz');v=d['v'];N=int(d['total']);m,n=v.shape;ell=(abs(v)**2).sum(axis=1);A=abs(v@v.T)**2;C=abs(v@v.conj().T)**2;M=np.outer(ell,ell)/n**2;Q=A/n;R=(np.outer(ell,ell)+C+A)/(n*(n+2));S=(np.outer(ell,ell)+C)/(n*(n+1));T=(Q+(N-2)*R)/(N-1)
    check(np.array_equal(T,d['target']),'fixed-total analytic target')
    for b in (2,4,8,12):
        sizes=[b]*(N//b)+([N%b] if N%b else []);k=sum(s*(s-2) for s in sizes)/(N*(N-1));J=Q/(N-1)+k*R+((N-2)/(N-1)-k)*M;stored=np.load(base/f'matrices_{case}_{b}.npz');check(np.max(abs(J-stored['joint']))<1e-15,'independent block matrix');check(np.max(abs(T-J-((N-2)/(N-1)-k)*(R-M)))<1e-15,'missing interblock coherence identity')
base=ROOT/'results/iteration79_strong_v1';strong=read(base/'summary.json');target=read(base/'target.json');d=np.load(base/'input.npz');v,t,W=d['v'],float(d['t']),d['weights'];m,n=v.shape;a=mp.mpf(n)/2;z=mp.mpf(t)**2;term=(1-z)**a;ca=cl=la=ll=ja=tail=mp.mpf(0)
for j in range(1,257):
    term*=z*(a+j-1)/j;da=(a+j)/(2*a*(a+1)*(2*j-1));dl=(j-1)/(2*a*(a+1)*(2*j-1));ca+=term*da;cl+=term*dl
    if 2*j<=12:la+=term*da;ll+=term*dl
    else:tail+=term;ja+=term/(2*j-1)
ratio=z*(a+256)/257;remainder=term*ratio/(1-ratio);ell=(abs(v)**2).sum(axis=1);A=abs(v@v.T)**2;L=np.outer(ell,ell)+abs(v@v.conj().T)**2;pair=float(ca)*A+float(cl)*L;check(np.array_equal(pair,d['target_pair']),'100-digit negative-binomial averaged pair target');means=[float(np.sum(w*pair)) for w in W]
for size in (4,16):
    V=v[:size];o=np.block([[V.real,-V.imag],[V.imag,V.real]]);x=np.diag(np.r_[np.full(n,-t/(1+t)),np.full(n,t/(1-t))]);q=np.eye(2*size)+o@x@o.T;means.append(float(1/mp.sqrt(mp.det(mp.matrix(q.tolist())))))
check(np.array_equal(means,target['target_means']) and abs(remainder-mp.mpf(target['omitted_mass_upper']))<mp.mpf('1e-190'),'vacuum targets and positive truncation remainder')
predictions={'joint':float(la)*A+float(ll)*L+float(ja)*A/n+float(tail-ja)*np.outer(ell,ell)/n**2,'sphere':float(la)*A+float(ll)*L+float(tail)*L/(n*(n+1))}
for kind,P in predictions.items():check(np.allclose([float(np.sum(w*P)) for w in W],target['predicted_hybrid_pair_means'][kind],atol=1e-15,rtol=0),'analytic hybrid pair means')
for row in strong['rows']:
    values=[];seconds=0;tail_records=0
    for batch in range(10):
        d=np.load(base/f"counts_{row['kind']}_{batch}.npz");counts=d['counts'];x=counts.astype(float);N=x.sum(axis=1);den=N*np.maximum(N-1,0);obs=[]
        for w in W:
            numerator=(x*(x@w)).sum(axis=1)-x@np.diag(w);obs.append(np.divide(numerator,den,out=np.zeros(len(x)),where=den>0))
        obs.extend([(x[:,:size].sum(axis=1)==0).astype(float) for size in (4,16)]);obs=np.array(obs).T;check(np.array_equal(obs,d['observables']) and np.array_equal(N,d['totals']) and not (N%2).any(),'strong raw observables and parity');values.append(obs);r=read(base/f"row_{row['kind']}_{batch}.json");seconds+=r['seconds'];tail_records+=r['statistics']['tail_records']
    values=np.concatenate(values);mu=values.mean(axis=0);var=values.var(axis=0,ddof=1);radius=np.sqrt(2*var*math.log(4000)/len(values))+7*math.log(4000)/(3*(len(values)-1));lower=np.maximum(0,abs(mu-means)-radius-float(remainder));check(np.array_equal(mu,row['mean']) and np.array_equal(radius,row['nominal_joint99_radii']) and np.array_equal(lower,row['nominal_full_tv_lowers']),'joint 99 percent nominal observable lower bounds');check(seconds==row['seconds'] and tail_records==row['tail_records'],'complete strong sampling records')
base=ROOT/'results/iteration79_timing_v1';timing=read(base/'summary.json');check(timing['completed'] and len(timing['rows'])==87 and all(r['success'] for r in timing['rows']),'87 fresh matched timing panels')
for row in timing['rows']:
    case,rep,kind,width,budget=(row[k] for k in ('case','rep','kind','width','budget'));d=np.load(base/f'counts_{case}_{rep}_{kind}_{width}_{budget}.npz');check(d['clicks'].shape==(row['records'],row['outputs']) and np.isin(d['clicks'],[0,1]).all(),'timing returned full batch')
    if 'counts' in d:check(np.array_equal(d['counts']>0,d['clicks']) and np.array_equal(d['counts'].sum(axis=1),d['totals']),'timing PNR to clicks conversion')
    if kind in ('old','gpu'):
        inputs=row['inputs'];dd=(2*inputs)**2;nd=row['records']*2*inputs;nm=row['records']*row['outputs'];slots=row['records']*budget*2**(width-1);expected=(slots*dd+slots*2*inputs*4+dd+4*row['outputs']*inputs+nd+2*nm+3*row['records']+4*inputs)*8+nm+8*row['records'];check(expected==row['cuda_timings']['workspace_bytes'],'actual allocated GPU workspace')
med={}
for case,kind,width,budget in set((r['case'],r['kind'],r['width'],r['budget']) for r in timing['rows']):med[f'{case}_{kind}_{width}_{budget}']=statistics.median(r['wall_seconds'] for r in timing['rows'] if (r['case'],r['kind'],r['width'],r['budget'])==(case,kind,width,budget))
check(med==timing['medians'],'medians use only complete fresh panel');check(read(ROOT/'results/iteration79_upstream_v1.json')['sha']=='fa8b796f2afebd4845a2e335e457735952617ef8','Frugal upstream rechecked')
report=dict(passed=True,checks=len(checks),full_suite=tests,timing_medians=med,strong_rows=strong['rows'],limitations='No universal full-law certificate. Pair distances are diagnostics. Nominal confidence bounds assume independent samples and numerical reference accuracy. Frugal and Jiuzhang timing comparisons do not imply equivalent quality or tasks.')
(OUT/'checks.json').write_bytes((json.dumps(checks,indent=2)+'\n').encode());(OUT/'validation.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(dict(passed=True,checks=len(checks),tests=tests['tests_run'],successes=tests['successes'])),flush=True)
