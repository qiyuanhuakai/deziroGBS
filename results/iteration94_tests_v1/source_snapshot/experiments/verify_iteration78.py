"""Independent reconstruction of iteration78 laws, fresh records and timing panels."""
from pathlib import Path
import os,json,hashlib,shutil,argparse,math,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=100
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
checks=[]
def check(ok,label):checks.append(label);assert ok,label
for base in sorted((ROOT/'results').glob('iteration78*')):
    if base.is_dir() and (base/'manifest.json').exists():
        for name,digest in read(base/'manifest.json')['artifacts'].items():check(H(base/name)==digest,'frozen '+base.name+'/'+name)
tests=read(ROOT/'results/iteration78_tests_v1/validation.json');check(tests['passed'] and tests['sphere_pnr_passed']==6 and tests['alias_click_passed']==6,'full suite with twelve new native tests')
for name,digest in read(ROOT/'results/iteration78_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
failed=ROOT/'results/iteration78_pilot_v1';check(read(failed/'build_sphere.json')['returncode']==0 and read(failed/'build_table.json')['returncode']!=0,'first table compilation failure retained')
for folder in ('iteration78_pilot_v1','iteration78_pilot_v2','iteration78_alias_pilot_v1'):
    for name,digest in read(ROOT/f'results/{folder}/plan.json')['hashes'].items():check(H(ROOT/name)==digest,'first executed source unchanged '+name)
base=ROOT/'results/iteration78_pilot_v2';pilot=read(base/'summary.json');check(pilot['passed'],'sphere and table pilots completed')
for row in pilot['sphere']:
    d=np.load(base/f"sphere_{row['case']}.npz");states=d['states'];lookup={tuple(state):i for i,state in enumerate(states)};histogram=np.bincount([lookup[tuple(x)] for x in d['counts']],minlength=len(states));tv=float(abs(histogram/len(d['counts'])-d['target']).sum()/2);check(np.array_equal(histogram,d['histogram']) and tv==row['empirical_tv'] and tv<.025,'complete fixed-total thermal histogram');check(np.all(d['counts'].sum(axis=1)==row['total']) and abs(d['target'].sum()-1)<1e-11,'thermal conditional support and mass')
for folder in ('iteration78_pilot_v2','iteration78_alias_pilot_v1'):
    base=ROOT/f'results/{folder}';summary=read(base/'summary.json');check(summary['passed'],'complete numerical-law pilot')
    for row in summary['table']:
        d=np.load(base/f"table_{row['case']}.npz");target=np.array(list(map(float,read(base/f"table_target_{row['case']}.json")['probabilities'])));delta=float(abs(d['law']-target).sum()/2);check(delta==row['numerical_full_tv'] and delta<1e-12,'complete 80-digit Gaussian click law');check(np.array_equal(target,d['target']),'target export')
    a=np.load(base/'strict_replay.npz');b=np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz');check(np.array_equal(a['counts'],b['counts']) and np.array_equal(a['totals'],b['totals']),'unchanged strict number branch')
timings={}
for folder,expected in [('iteration78_click_timing_v1',30),('iteration78_click_timing_v2',42),('iteration78_pnr_timing_v1',36)]:
    base=ROOT/f'results/{folder}';summary=read(base/'summary.json');check(summary['passed'] and len(summary['rows'])==expected,'all '+str(expected)+' matched timing panels');med={}
    for row in summary['rows']:
        case,rep,kind,k=row['case'],row['rep'],row['kind'],row['budget'];d=np.load(base/f'counts_{case}_{rep}_{kind}_{k}.npz');check(d['clicks'].shape==(row['records'],row['outputs']) and np.isin(d['clicks'],[0,1]).all(),'complete returned timing records')
        if 'counts' in d.files:check(np.array_equal(d['counts']>0,d['clicks']) and np.array_equal(d['counts'].sum(axis=1),d['totals']) and not (d['totals']%2).any(),'exact total and PNR click conversion')
    for case,kind,k in set((r['case'],r['kind'],r['budget']) for r in summary['rows']):
        group=[r for r in summary['rows'] if (r['case'],r['kind'],r['budget'])==(case,kind,k)];check(sorted(r['rep'] for r in group)==[0,1,2],'three complete repetitions');key=f'{case}_{kind}_{k}';med[key]=statistics.median(r['wall_seconds'] for r in group);check(med[key]==summary['medians'][key],'median reconstructed')
    if 'click_timing' in folder:
        for case in (0,1):
            d=np.load(base/f'fused_law_{case}.npz');law=np.prod(np.where(d['patterns'],d['probabilities'],1-d['probabilities']),axis=1);check(np.array_equal(law,d['law']) and abs(law-d['target']).sum()/2<1e-10,'complete direct CUDA law')
            if folder.endswith('v2'):
                d=np.load(base/f'alias_law_{case}.npz');check(abs(d['law']-d['target']).sum()/2<1e-12,'integer alias law on timing input')
            for rep in range(3):
                for kind in (('table','alias') if folder.endswith('v2') else ('table',)):
                    a=np.load(base/f'counts_{case}_{rep}_{kind}_1.npz')['clicks'];b=np.load(base/f'counts_{case}_{rep}_{kind}_16.npz')['clicks'];check(np.array_equal(a,b),'thread-count-independent direct records')
    else:
        for case in range(3):
            for rep in range(3):
                k=8 if case==0 else 12;a=np.load(base/f'counts_{case}_{rep}_iid_{k}.npz');b=np.load(base/f'counts_{case}_{rep}_sphere_{k}.npz');mask=a['totals']<=k;check(np.array_equal(a['totals'],b['totals']) and np.array_equal(a['counts'][mask],b['counts'][mask]),'strict branch coupling across tail policies')
    timings[folder]=med
base=ROOT/'results/iteration78_pnr_quality_v1';quality=read(base/'summary.json');check(quality['passed'] and len(quality['rows'])==2,'two fresh million-record laws');radius=math.sqrt(math.log(800)/2000000);check(radius==quality['radius'],'eight one-sided bounds share .01');data=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');cutoff=int(data['cutoff']);codes=data['codes'];target=data['target'];powers=np.array([(cutoff+1)**j for j in range(4)],np.uint64);lookup=np.full((cutoff+1)**4,-1,np.int32);lookup[codes]=np.arange(len(codes),dtype=np.int32);clicktarget=np.array(list(map(float,read(ROOT/'results/iteration75_task_comparison_v1/target.json')['probabilities'])));records=0;seeds=[]
def hist(counts):
    total=counts.sum(axis=1);mask=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[mask]=lookup[(counts[mask].astype(np.uint64)@powers).astype(np.int64)];check(labels.min()>=0,'fresh records mapped to full PNR target');bits=sum((counts[:,j]>0).astype(np.int64)*2**j for j in range(4));return np.bincount(labels,minlength=len(target)),np.bincount(bits,minlength=16)
for stored in quality['rows']:
    panel=stored['panel'];pilot=np.load(base/f'pilot_{panel}.npz');hp,hc=hist(pilot['counts']);ct=pilot['click_target'];check(abs(ct-clicktarget).max()<1e-13,'independent 80-digit click target');check(np.array_equal(hp/20000>target,pilot['pnr_event']) and np.array_equal(hc/20000>ct,pilot['click_event']),'pilot fixed events');h=np.zeros(len(target),np.int64);c=np.zeros(16,np.int64);photons=odd=maximum=tails=0;seconds=0.
    for batch in range(20):
        counts=np.load(base/f'counts_{panel}_{batch}.npz')['counts'];row=read(base/f'status_{panel}_{batch}.json');seed=0x93000000000000+(panel*20+batch)*2**40;seeds.append(seed);check(counts.shape==(50000,4) and counts.dtype==np.uint32 and row['seed']==seed,'disjoint new quality batch');a,b=hist(counts);h+=a;c+=b;total=counts.sum(axis=1);photons+=int(total.sum());odd+=int((total%2!=0).sum());maximum=max(maximum,int(total.max()));tails+=int(row['statistics'].get('tail_records',0));seconds+=row['seconds'];records+=len(counts);check(row['statistics']['photons']==int(total.sum()),'quality native photon totals')
    saved=np.load(base/f'histogram_{panel}.npz');check(np.array_equal(h,saved['pnr']) and np.array_equal(c,saved['click']),'complete raw histograms reconstructed')
    for kind,histogram,tar,event,extra in [('pnr',h,target,pilot['pnr_event'],target[-1]),('click',c,ct,pilot['click_event'],0.)]:
        empirical=histogram/1000000;tv=float(abs(empirical-tar).sum()/2);lower=max(0.,float((empirical[event]-tar[event]).sum())-radius);upper=min(1.,tv+radius+extra);check(np.allclose([tv,lower,upper],[stored[kind][key] for key in ('empirical_tv','lower','upper')],atol=1e-15,rtol=0),'fresh full '+kind+' nominal TV bounds')
    check(odd==stored['odd_records'] and photons/1000000==stored['mean_photons'] and maximum==stored['maximum_photons'] and seconds==stored['seconds'] and tails==stored['tail_records'],'quality parity mean max time tail')
check(records==2000000 and len(set(seeds))==40,'two million independent quality records')

base=ROOT/'results/iteration78_pair_checks_v1';pairs=read(base/'validation.json');check(pairs['passed'] and pairs['checks']==70,'70 complete Fock pair matrices')
for row in pairs['rows']:
    d=np.load(base/(row['label']+'.npz'));x=d['states'].astype(float);q=d['probabilities'];N=int(d['total']);reference=(x.T@(x*q[:,None])-np.diag(q@x))/(N*(N-1));check(np.array_equal(reference,d['reference']) and abs(reference-d['formula']).max()<2e-11,'full pair matrix reconstructed from independent Fock states')
base=ROOT/'results/iteration78_strong_v1';strong=read(base/'summary.json');target=read(base/'target.json');d=np.load(base/'input.npz');v,t,W=d['v'],float(d['t']),d['weights'];m,n=v.shape;a=mp.mpf(n)/2;z=mp.mpf(t)**2;term=(1-z)**a;ca=cl=mp.mpf(0)
for j in range(1,257):
    term*=z*(a+j-1)/j;ca+=term*(a+j)/(2*a*(a+1)*(2*j-1));cl+=term*(j-1)/(2*a*(a+1)*(2*j-1))
ratio=z*(a+256)/257;remainder=term*ratio/(1-ratio);ell=(abs(v)**2).sum(axis=1);A=abs(v@v.T)**2;L=np.outer(ell,ell)+abs(v@v.conj().T)**2;reference_pair=float(ca)*A+float(cl)*L;check(np.array_equal(reference_pair,d['target_pair']) and abs(remainder-mp.mpf(target['omitted_mass_upper']))<mp.mpf('1e-190'),'target pair coefficients and positive analytic remainder');target_means=[float(np.sum(w*reference_pair)) for w in W]
for size in (4,16):
    V=v[:size];o=np.block([[V.real,-V.imag],[V.imag,V.real]]);x=np.diag(np.r_[np.full(n,-t/(1+t)),np.full(n,t/(1-t))]);q=np.eye(2*size)+o@x@o.T;target_means.append(float(1/mp.sqrt(mp.det(mp.matrix(q.tolist())))))
check(target_means==target['target_means'],'independent pair and vacuum targets')
for row in strong['rows']:
    kind=row['kind'];all_values=[];odd=tails=0;seconds=0
    for batch in range(10):
        d=np.load(base/f'counts_{kind}_{batch}.npz');x=d['counts'].astype(float);totals=x.sum(axis=1);den=totals*np.maximum(totals-1,0);values=[]
        for w in W:
            numerator=(x*(x@w)).sum(axis=1)-x@np.diag(w);values.append(np.divide(numerator,den,out=np.zeros(len(x)),where=den>0))
        values.extend([(x[:,:size].sum(axis=1)==0).astype(float) for size in (4,16)]);values=np.array(values).T;check(np.array_equal(values,d['observables']) and np.array_equal(totals,d['totals']),'strong raw observable reconstruction');all_values.append(values);odd+=int((totals%2).sum());tails+=int((totals>12).sum());seconds+=read(base/f'row_{kind}_{batch}.json')['seconds']
    values=np.concatenate(all_values);mean=values.mean(axis=0);variance=values.var(axis=0,ddof=1);radii=np.sqrt(2*variance*math.log(4000)/len(values))+7*math.log(4000)/(3*(len(values)-1));lowers=np.maximum(0,abs(mean-target_means)-radii-float(remainder));check(np.array_equal(mean,row['mean']) and np.array_equal(variance,row['variance']) and np.array_equal(radii,row['nominal_joint99_radii']) and np.array_equal(lowers,row['nominal_full_tv_lowers']),'joint twenty-sided nominal confidence bounds');check(odd==row['odd_records'] and tails==row['tail_records'] and seconds==row['seconds'],'strong total statistics')
c=timings['iteration78_click_timing_v2'];pnr=timings['iteration78_pnr_timing_v1'];ratios=dict(small_frugal_cpu_over_alias1=c['0_cpu_4']/c['0_alias_1'],small_frugal_gpu_over_alias1=c['0_gpu_4']/c['0_alias_1'],small_cuda_over_alias1=c['0_fused_8']/c['0_alias_1'],small_alias16_over_frugal_cpu=c['0_alias_16']/c['0_cpu_4'],eight_frugal_cpu5_over_alias1=c['1_cpu_5']/c['1_alias_1'],eight_cuda128_over_alias1=c['1_fused_128']/c['1_alias_1'],large_frugal_cpu2_over_sphere=pnr['1_cpu_2']/pnr['1_sphere_12'],large_frugal_cpu3_over_sphere=pnr['1_cpu_3']/pnr['1_sphere_12'],strong128_sphere_microseconds=pnr['2_sphere_12']/512*1e6,strong128_over_jiuzhang3_different_task=pnr['2_sphere_12']/512/1.27e-6)
check(quality['rows'][1]['pnr']['upper']<quality['rows'][0]['pnr']['lower'] and quality['rows'][1]['click']['upper']<quality['rows'][0]['click']['lower'],'simultaneous full-law quality improvement at equal total budget')
result=dict(passed=True,checks=len(checks),full_suite=tests,timing_medians=timings,ratios=ratios,quality=quality,strong=strong,pair_checks=70,arbitrary_input_energy_superiority=False,source_sha256=H(Path(__file__)))
shutil.copy2(__file__,OUT/Path(__file__).name)
for name,obj in [('checks.json',checks),('validation.json',result)]:
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
print(json.dumps({k:result[k] for k in ('passed','checks','ratios')}),flush=True)
