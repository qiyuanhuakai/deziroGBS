"""Independent raw-record reconstruction of iteration76 evidence and bounds."""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,math,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=[]
def check(ok,label):checks.append(label);assert ok,label
for base in sorted((ROOT/'results').glob('iteration76*')):
    if base.is_dir() and (base/'manifest.json').exists():
        for name,digest in read(base/'manifest.json')['artifacts'].items():check(H(base/name)==digest,'frozen '+base.name+'/'+name)
tests=read(ROOT/'results/iteration76_tests_v1/validation.json');check(tests['passed'] and tests['projected_pnr_passed']==6,'full suite and all six new projected tests')
for name,digest in read(ROOT/'results/iteration76_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
for version in (1,3):
    base=ROOT/f'results/iteration76_projected_pilot_v{version}';summary=read(base/'summary.json');data=read(base/'checks.json');check(summary['passed'] and data['passed'] and [len(data[k]) for k in ('moments','compressions','rates')]==[80,15,20],'complete projected independent checks')
    for row in data['moments']:
        n,k,tau=row['inputs'],row['k'],row['tau'];d=np.load(base/f'probe_{n}_{k}_{tau}.npz');scale=max(1,abs(d['reference']),abs(d['reference_prime']));check(abs(d['mass'][0]-d['reference'])<1e-10*scale and abs(d['prime'][0]-d['reference_prime'])<1e-10*scale,'independent Wick moment and derivative');check(d['diagnostics'][0]<=min(n,2*k) and d['diagnostics'][2]<1e-12,'bounded tangent subspace')
    for row in data['compressions']:
        d=np.load(base/f"compression_{row['inputs']}_{row['k']}.npz");delta=max(float(abs(d['projected_cov']-d['reference_cov']).max()),float(abs(d['projected_mean']-d['reference_mean']).max()));check(delta==row['maximum_difference'] and delta<1e-9,'independent complete compression moments')
    for row in data['rates']:
        d=np.load(base/f"rates_{row['inputs']}_{row['k']}.npz");delta=float(abs(d['rates'][d['selected']]-d['references']).max());check(delta==row['maximum_difference'] and delta<1e-9 and d['rates'].min()>=0,'positive detector rate versus independent Wick')
    for case in (0,1):
        for k in (3,7,15,31):
            a=np.load(base/f'counts_{case}_{k}_old.npz')['counts'];b=np.load(base/f'counts_{case}_{k}_projected.npz')['counts'];check(a.shape==(4096,1 if case==0 else 4) and np.array_equal(a,b),'complete old/new pilot records')
base=ROOT/'results/iteration76_pnr_scale_v1';pnr=read(base/'summary.json');check(pnr['passed'] and len(pnr['rows'])==27,'all 27 PNR timing panels');pnrmed={};paired=0
for case in range(4):
    data=np.load(base/f'input_{case}.npz');n=len(data['t']);k=31 if case==0 else 7;z=mp.mpf(float(data['t'][0]))**2;shape=mp.mpf(n)/2;term=(1-z)**shape;mass=term
    for pairs in range(1,k//2+1):term*=z*(shape+pairs-1)/pairs;mass+=term
    target=read(base/f'target_{case}.json');check(abs(1-mass-mp.mpf(target['ideal_prefix_tv_upper']))<mp.mpf('1e-70'),'analytic all-input negative-binomial tail');check(np.max(abs(data['v'].conj().T@data['v']-np.eye(n)))<1e-12,'physical input isometry')
    kinds=sorted(set(r['kind'] for r in pnr['rows'] if r['case']==case));pnrmed[str(case)]={}
    for rep in range(3):
        reference=np.load(base/f'counts_{case}_{rep}_projected3.npz')['counts'];paired+=len(reference)
        for kind in kinds:
            counts=np.load(base/f'counts_{case}_{rep}_{kind}.npz')['counts'];row=read(base/f'row_{case}_{rep}_{kind}.json');check(np.array_equal(counts,reference) and counts.shape==(row['records'],row['outputs']),'paired PNR output records');check(int(counts.sum())==row['statistics']['photons'] and int((counts.sum(axis=1)%2!=0).sum())==row['odd_records'],'raw PNR timing statistics')
    for kind in kinds:
        group=[r for r in pnr['rows'] if r['case']==case and r['kind']==kind];check(sorted(r['rep'] for r in group)==[0,1,2],'three uncontended PNR repetitions');median=statistics.median(r['wall_seconds'] for r in group);check(median==pnr['medians'][str(case)][kind],'PNR median reconstructed');pnrmed[str(case)][kind]=median
    if case:check(read(base/f'old_resource_{case}.json')['rejected'],'old capacity rejection separate from successful timing')
base=ROOT/'results/iteration76_scale_click_v1';click=read(base/'summary.json');check(click['passed'] and len(click['rows'])==45,'45 complete direct/Frugal timings');clickmed={}
for row in click['rows']:
    label=f"{row['case']}_{row['rep']}_{row['method']}_{row['budget']}";bits=np.load(base/f'records_{label}.npz')['clicks'];check(row['returncode']==0 and bits.shape==(128,row['outputs']) and np.isin(bits,[0,1]).all(),'complete returned click records');check(float(bits.sum(axis=1).mean())==row['observed_mean_clicks'],'observed clicks recomputed')
for case in range(3):
    clickmed[str(case)]={}
    for kind,k in sorted(set((r['method'],r['budget']) for r in click['rows'] if r['case']==case)):
        group=[r for r in click['rows'] if (r['case'],r['method'],r['budget'])==(case,kind,k)];check(sorted(r['rep'] for r in group)==[0,1,2],'three matched direct/Frugal repetitions');value=statistics.median(r['wall_seconds'] for r in group);check(value==click['medians'][str(case)][f'{kind}_{k}'],'direct/Frugal median reconstructed');clickmed[str(case)][f'{kind}_{k}']=value
    for k in (1,4,8):
        d=np.load(base/f'replay_{case}_{k}.npz');check(np.array_equal(d['old_clicks'],d['new_clicks']) and abs(d['old_probabilities']-d['new_probabilities']).max()<1e-8,'expanded direct GPU arithmetic replay')
base=ROOT/'results/iteration76_strong_pnr_v1';stress=read(base/'summary.json');check(stress['completed'],'strong pressure results present');stressrows=[]
for row in stress['rows']:
    if not row['success']:check(bool(row['error']),'strong failure preserved');stressrows.append(row);continue
    counts=np.load(base/f"counts_{row['case']}.npz")['counts'];total=counts.sum(axis=1);odd=int((total%2!=0).sum());radius=math.sqrt(math.log(300)/(2*len(counts)));lower=max(0,odd/len(counts)-radius);check(counts.shape==(row['records'],row['outputs']) and odd==row['odd_records'] and lower==row['nominal_joint99_full_tv_lower'],'fixed parity event full-TV lower bound');check(float(total.mean())==row['mean_photons'] and int(total.max())==row['maximum_photons'],'uncensored strong photon counts');stressrows.append(row)
base=ROOT/'results/iteration76_pnr_quality_v1';quality=read(base/'summary.json');check(quality['passed'] and len(quality['rows'])==2,'two fresh independent quality laws');radius=math.sqrt(math.log(800)/(2*1000000));check(radius==quality['radius'],'eight one-sided bounds share .01');data=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');cutoff=int(data['cutoff']);codes=data['codes'];target=data['target'];b=cutoff+1;powers=np.array([b**j for j in range(4)],np.uint64);lookup=np.full(b**4,-1,np.int32);lookup[codes]=np.arange(len(codes),dtype=np.int32);clicktarget=np.array(list(map(float,read(ROOT/'results/iteration75_task_comparison_v1/target.json')['probabilities'])));records=0;seeds=[]
def hist(counts):
    total=counts.sum(axis=1);inside=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];check(labels.min()>=0,'all fresh PNR records mapped');bits=sum((counts[:,j]>0).astype(np.int64)*2**j for j in range(4));return np.bincount(labels,minlength=len(target)),np.bincount(bits,minlength=16)
for stored in quality['rows']:
    panel=stored['panel'];pilot=np.load(base/f'pilot_{panel}.npz');hp,hc=hist(pilot['counts']);ct=pilot['click_target'];check(abs(ct-clicktarget).max()<1e-13,'independent 80-digit click target');check(np.array_equal(hp/20000>target,pilot['pnr_event']) and np.array_equal(hc/20000>ct,pilot['click_event']),'independent pilot-fixed events');h=np.zeros(len(target),np.int64);c=np.zeros(16,np.int64);odd=photons=maximum=0;seconds=0.
    for batch in range(20):
        counts=np.load(base/f'counts_{panel}_{batch}.npz')['counts'];row=read(base/f'status_{panel}_{batch}.json');seed=0x86000000000000+(panel*20+batch)*2**40;seeds.append(seed);check(counts.shape==(50000,4) and counts.dtype==np.uint32 and row['seed']==seed,'new independent batch and disjoint seed range');a,z=hist(counts);h+=a;c+=z;total=counts.sum(axis=1);photons+=int(total.sum());odd+=int((total%2!=0).sum());maximum=max(maximum,int(total.max()));seconds+=row['seconds'];records+=len(counts);check(row['statistics']['photons']==int(total.sum()),'native/fresh raw count total')
    saved=np.load(base/f'histogram_{panel}.npz');check(np.array_equal(h,saved['pnr']) and np.array_equal(c,saved['click']),'both fresh histograms reconstructed')
    for kind,histogram,tar,event,tail in [('pnr',h,target,pilot['pnr_event'],target[-1]),('click',c,ct,pilot['click_event'],0.)]:
        empirical=histogram/1000000;tv=float(abs(empirical-tar).sum()/2);lower=max(0.,float((empirical[event]-tar[event]).sum())-radius);upper=min(1.,tv+radius+tail);check(np.allclose([tv,lower,upper],[stored[kind][key] for key in ('empirical_tv','lower','upper')],atol=1e-15,rtol=0),'fresh complete '+kind+' TV bounds')
    check(odd==stored['odd_records'] and photons/1000000==stored['mean_photons'] and maximum==stored['maximum_photons'] and seconds==stored['seconds'],'fresh parity mean maximum and time')
check(records==2000000 and len(set(seeds))==40,'two million fresh independent records')
shutil.copy2(__file__,OUT/Path(__file__).name);save('checks.json',checks);result=dict(passed=True,checks=len(checks),full_suite=tests,pnr_medians=pnrmed,click_medians=clickmed,paired_pnr_records=paired,fresh_quality_records=records,quality_rows=quality['rows'],strong_stress=stressrows,source_sha256=H(Path(__file__)),arbitrary_input_dimension_superiority=False);save('validation.json',result);print(json.dumps({k:result[k] for k in ('passed','checks','paired_pnr_records','fresh_quality_records','pnr_medians','click_medians')}),flush=True)
