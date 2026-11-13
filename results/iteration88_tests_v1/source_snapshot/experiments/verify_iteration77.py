"""Reconstruct frozen iteration77 raw-record evidence, tails and matched timings."""
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
def tail(n,t,k):
    a=mp.mpf(n)/2;z=mp.mpf(float(t))**2;term=(1-z)**a;s=term
    for j in range(1,k//2+1):term*=z*(a+j-1)/j;s+=term
    return 1-s
for base in sorted((ROOT/'results').glob('iteration77*')):
    if base.is_dir() and (base/'manifest.json').exists():
        for name,digest in read(base/'manifest.json')['artifacts'].items():check(H(base/name)==digest,'frozen '+base.name+'/'+name)
tests=read(ROOT/'results/iteration77_tests_v1/validation.json');check(tests['passed'] and tests['number_pnr_passed']==8,'full suite and eight number interface tests')
for name,digest in read(ROOT/'results/iteration77_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
for version in (1,2):
    base=ROOT/f'results/iteration77_number_pilot_v{version}';summary=read(base/'summary.json');data=read(base/'checks.json');check(summary['checks_passed'] and summary['all_samples_succeeded'] and len(data['fock'])==266 and len(data['large'])==45,'independent Fock and Wick panels')
    check(summary['capacity_fixed10']['success']==(version==2),'first resource failure retained and pruning succeeds')
    for row in data['fock']:
        d=np.load(base/f"fock_{row['case']}_{row['total']}_{row['k']}.npz");delta=max(float(abs(d['mass']-d['reference'])),float(abs(d['rates']-d['reference_rates']).max()));check(delta<2e-10*row['reference_scale'],'Fock detector coefficients');check(abs(float(d['rates'].sum())-(row['total']-row['k'])*float(d['mass']))<2e-10*max(1,(row['total']-row['k'])*float(d['mass'])),'photon conservation detector sum')
    for row in data['large']:
        d=np.load(base/f"large_{row['inputs']}_{row['total']}_{row['k']}.npz");check(max(float(abs(d['mass']-d['reference'])),float(abs(d['rates'][d['selected']]-d['reference_rates']).max()))<2e-10,'independent large input Wick coefficient')
base=ROOT/'results/iteration77_budget_pilot_v1';budget=read(base/'summary.json');check(budget['passed'],'all expanded budget panels succeeded')
for row in budget['strict_replays']:
    d=np.load(base/f"old_replay_{row['case']}.npz");old=np.load(ROOT/f"results/iteration77_number_pilot_v2/counts_{row['case']}.npz");check(np.array_equal(d['counts'],old['counts']) and np.array_equal(d['totals'],old['totals']),'strict budget archived replay')
reference=np.load(base/'coupling_64.npz')
for row in budget['couplings']:
    k=row['budget'];d=np.load(base/f'coupling_{k}.npz');mask=d['totals']<=k;check(np.array_equal(d['totals'],reference['totals']) and np.array_equal(d['counts'][mask],reference['counts'][mask]),'exact total branch coupling');check(int((~mask).sum())==row['statistics']['tail_records'],'explicit tail statistics')
for row in budget['expanded']:
    case=row['case'];d=np.load(base/f'input_{case}.npz');counts=np.load(base/f'counts_{case}.npz');total=counts['counts'].sum(axis=1);check(np.array_equal(total,counts['totals']) and not (total%2).any(),'expanded exact total even parity');check(abs(tail(row['inputs'],float(d['t']),row['budget'])-mp.mpf(row['ideal_tv_upper']))<mp.mpf('1e-70'),'analytic ideal total-tail TV upper')
base=ROOT/'results/iteration77_task_comparison_v1';timing=read(base/'summary.json');check(timing['passed'] and len(timing['rows'])==36,'36 complete matched timing panels');medians={}
for row in timing['rows']:
    case,rep,kind,k=row['case'],row['rep'],row['kind'],row['budget'];d=np.load(base/f'records_{case}_{rep}_{kind}_{k}.npz');check(d['clicks'].shape==(row['records'],4 if case==0 else 144) and np.isin(d['clicks'],[0,1]).all(),'returned click records')
    if 'counts' in d.files:
        check(np.array_equal(d['counts']>0,d['clicks']) and np.array_equal(d['counts'].sum(axis=1),d['totals']),'PNR threshold conversion')
        check(row['statistics']['photons']==int(d['totals'].sum()),'PNR raw total statistics')
        if kind!='projected':check(not (d['totals']%2).any(),'number sampler parity')
for case,kind,k in sorted(set((r['case'],r['kind'],r['budget']) for r in timing['rows'])):
    group=[r for r in timing['rows'] if (r['case'],r['kind'],r['budget'])==(case,kind,k)];check(sorted(r['rep'] for r in group)==[0,1,2],'three rotated repetitions');label=f'{case}_{kind}_{k}';medians[label]=statistics.median(r['wall_seconds'] for r in group);check(medians[label]==timing['medians'][label],'matched median reconstructed')
for case in (0,1):
    for rep in range(3):
        a=np.load(base/f'records_{case}_{rep}_strict_0.npz');b=np.load(base/f'records_{case}_{rep}_number_64.npz');mask=a['totals']<=64;check(np.array_equal(a['totals'],b['totals']) and np.array_equal(a['counts'][mask],b['counts'][mask]),'timing strict budget exact coupling')
base=ROOT/'results/iteration77_pnr_quality_v1';quality=read(base/'summary.json');check(quality['passed'] and len(quality['rows'])==2,'two fresh million-record laws');radius=math.sqrt(math.log(800)/2000000);check(radius==quality['radius'],'eight one-sided bounds share .01');data=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');cutoff=int(data['cutoff']);codes=data['codes'];target=data['target'];powers=np.array([(cutoff+1)**j for j in range(4)],np.uint64);lookup=np.full((cutoff+1)**4,-1,np.int32);lookup[codes]=np.arange(len(codes),dtype=np.int32);clicktarget=np.array(list(map(float,read(ROOT/'results/iteration75_task_comparison_v1/target.json')['probabilities'])));records=0;seeds=[]
def hist(counts):
    total=counts.sum(axis=1);mask=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[mask]=lookup[(counts[mask].astype(np.uint64)@powers).astype(np.int64)];check(labels.min()>=0,'fresh records mapped to full PNR target');bits=sum((counts[:,j]>0).astype(np.int64)*2**j for j in range(4));return np.bincount(labels,minlength=len(target)),np.bincount(bits,minlength=16)
for stored in quality['rows']:
    panel=stored['panel'];pilot=np.load(base/f'pilot_{panel}.npz');hp,hc=hist(pilot['counts']);ct=pilot['click_target'];check(abs(ct-clicktarget).max()<1e-13,'independent 80-digit click target');check(np.array_equal(hp/20000>target,pilot['pnr_event']) and np.array_equal(hc/20000>ct,pilot['click_event']),'pilot fixed events');h=np.zeros(len(target),np.int64);c=np.zeros(16,np.int64);photons=odd=maximum=tails=0;seconds=0.
    for batch in range(20):
        counts=np.load(base/f'counts_{panel}_{batch}.npz')['counts'];row=read(base/f'status_{panel}_{batch}.json');seed=0x8b000000000000+(panel*20+batch)*2**40;seeds.append(seed);check(counts.shape==(50000,4) and counts.dtype==np.uint32 and row['seed']==seed,'disjoint new quality batch');a,b=hist(counts);h+=a;c+=b;total=counts.sum(axis=1);photons+=int(total.sum());odd+=int((total%2!=0).sum());maximum=max(maximum,int(total.max()));tails+=int(row['statistics'].get('tail_records',0));seconds+=row['seconds'];records+=len(counts);check(row['statistics']['photons']==int(total.sum()),'quality native photon totals')
    saved=np.load(base/f'histogram_{panel}.npz');check(np.array_equal(h,saved['pnr']) and np.array_equal(c,saved['click']),'complete raw histograms reconstructed')
    for kind,histogram,tar,event,extra in [('pnr',h,target,pilot['pnr_event'],target[-1]),('click',c,ct,pilot['click_event'],0.)]:
        empirical=histogram/1000000;tv=float(abs(empirical-tar).sum()/2);lower=max(0.,float((empirical[event]-tar[event]).sum())-radius);upper=min(1.,tv+radius+extra);check(np.allclose([tv,lower,upper],[stored[kind][key] for key in ('empirical_tv','lower','upper')],atol=1e-15,rtol=0),'fresh full '+kind+' nominal TV bounds')
    check(odd==stored['odd_records'] and photons/1000000==stored['mean_photons'] and maximum==stored['maximum_photons'] and seconds==stored['seconds'] and tails==stored['tail_records'],'quality parity mean max time tail')
check(records==2000000 and len(set(seeds))==40,'two million independent quality records')
base=ROOT/'results/iteration77_terminal_collision_v1';terminal=read(base/'checks.json');check(terminal['passed'] and len(terminal['terminal'])==192 and len(terminal['collision_formulas'])==64,'terminal coefficient and collision formula coverage')
for row in terminal['terminal']:
    d=np.load(base/f"terminal_{row['case']}_{row['total']}_{row['k']}.npz");scale=max(1e-290,float(d['reference']),float(d['reference_rates'].max()));check(max(float(abs(d['mass'][0]-d['reference'])),float(abs(d['rates']-d['reference_rates']).max()))<1e-8*scale,'terminal relative Fock coefficient check')
for row in terminal['collision_formulas']:check(abs(row['reference']-row['formula'])<1e-11,'Fock collision formula validation')
collision=read(ROOT/'results/iteration77_collision_bound_v1/validation.json');values=[];odd=tails=0
for batch in range(10):
    d=np.load(base/f'counts_{batch}.npz');x=d['counts'].astype(float);totals=x.sum(axis=1);f=np.divide((x*(x-1)).sum(axis=1),totals*np.maximum(totals-1,0),out=np.zeros(len(x)),where=totals>=2);check(np.array_equal(f,d['collision']) and np.array_equal(totals,d['totals']),'raw collision observable and exact total');values.extend(f);odd+=int((totals%2).sum());tails+=int((totals>12).sum())
mean=float(np.mean(values));variance=float(np.var(values,ddof=1));radius=math.sqrt(2*variance*math.log(400)/len(values))+7*math.log(400)/(3*(len(values)-1));lower=max(0,abs(mean-float(collision['target_partial_expectation']))-radius-float(collision['target_remainder_upper']));check(mean==collision['empirical_mean'] and variance==collision['sample_variance'] and radius==collision['nominal_two_sided99_radius'] and lower==collision['nominal_full_tv_lower'] and lower>0,'positive nominal collision full-TV lower');check(mp.mpf(collision['target_remainder_upper'])>0 and odd==0 and tails==99611,'positive corrected tail bound and even records')
ratios=dict(large_projected_over_number=medians['1_projected_31']/medians['1_number_64'],large_fused_over_number=medians['1_fused_4']/medians['1_number_64'],large_frugal_cpu2_over_number=medians['1_cpu_2']/medians['1_number_64'],large_frugal_cpu3_over_number=medians['1_cpu_3']/medians['1_number_64'],small_number_over_frugal_cpu4=medians['0_number_64']/medians['0_cpu_4'],small_fused_over_frugal_cpu4=medians['0_fused_8']/medians['0_cpu_4'],large_number_microseconds=medians['1_number_64']/1024*1e6,large_number_over_jiuzhang3_different_task=medians['1_number_64']/1024/1.27e-6)
result=dict(passed=True,checks=len(checks),full_suite=tests,timing_medians=medians,ratios=ratios,quality_rows=quality['rows'],fresh_quality_records=records,collision=collision,budget_expanded=budget['expanded'],arbitrary_input_energy_superiority=False,source_sha256=H(Path(__file__)))
shutil.copy2(__file__,OUT/Path(__file__).name)
for name,obj in [('checks.json',checks),('validation.json',result)]:
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
print(json.dumps({k:result[k] for k in ('passed','checks','ratios')}),flush=True)
