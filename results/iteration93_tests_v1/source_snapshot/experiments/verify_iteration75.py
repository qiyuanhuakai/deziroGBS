"""Recompute 12M count records, event laws, complete small laws and timing groups.
No new throughput measurements are taken by this verifier.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,math,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.frugal_law import full_law,zero_probability
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=[]
def check(ok,label):
    checks.append(label)
    assert ok,label
for base in sorted((ROOT/'results').glob('iteration75*')):
    if base.is_dir() and (base/'manifest.json').exists():
        for name,digest in read(base/'manifest.json')['artifacts'].items():check(H(base/name)==digest,'frozen '+base.name+'/'+name)
tests=read(ROOT/'results/iteration75_tests_v1/validation.json');check(tests['passed'] and (tests['deep_pnr_passed'],tests['resplit_pnr_passed'],tests['prefix_pnr_passed'])==(5,4,5),'full suite and fourteen new tests')
for name,digest in read(ROOT/'results/iteration75_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
for name,file,expected in [('deep','moment_checks.json',40),('resplit','decomposition_checks.json',25)]:
    base=ROOT/f'results/iteration75_{name}_pilot_v1';r=read(base/file);check(r['passed'] and len(r['rows'])==expected,name+' independent checks');check(read(base/'summary.json')['all_samples_succeeded'],name+' all pilot samples')
probes=read(ROOT/'results/iteration75_prefix_pilot_v1/probe_checks.json');check(probes['passed'] and len(probes['moments'])==80,'eighty prefix moment and tangent checks')
check(len(probes['compressions'])==20 and all(r['returncode']==0 and r['maximum_difference']<1e-10 for r in probes['compressions']),'twenty independent Gaussian compression moments')
check(len(probes['high_degree'])==21 and all(max(r['relative_mass_error'],r['relative_prime_error'])<1e-12 for r in probes['high_degree']),'twenty-one high-degree analytic factorial moments')
check(read(ROOT/'results/iteration75_prefix_pilot_v1/summary.json')['all_samples_succeeded'],'all prefix pilots succeeded')
for p in (ROOT/'results/iteration75_prefix_pilot_v1').glob('norm_*.npz'):
    d=np.load(p);scale=max(1,abs(d['reference']),abs(d['reference_prime']));check(abs(d['mass']-d['reference'])<1e-10*scale and abs(d['prime']-d['reference_prime'])<1e-10*scale,'independent prefix Wick values '+p.name)
base=ROOT/'results/iteration75_pnr_quality_v2';quality=read(base/'summary.json');radius=math.sqrt(math.log(4800)/(2*1000000));check(quality['passed'] and len(quality['rows'])==12 and radius==quality['radius'],'twelve simultaneous independent laws');records=0;seeds=[];bounds=[]
for stored in quality['rows']:
    panel=stored['panel'];ci=stored['case'];data=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{ci}.npz');v,t,cutoff,codes,target=data['v'],data['t'],int(data['cutoff']),data['codes'],data['target'];m=len(v);b=cutoff+1;powers=np.array([b**j for j in range(m)],np.uint64);lookup=np.full(b**m,-1,np.int32);lookup[codes]=np.arange(len(codes),dtype=np.int32)
    def hist(counts):
        total=counts.sum(axis=1);inside=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];check(labels.min()>=0,'raw PNR mapping');clicklabels=sum((counts[:,j]>0).astype(np.int64)*2**j for j in range(m));return np.bincount(labels,minlength=len(target)),np.bincount(clicklabels,minlength=2**m)
    pilot=np.load(base/f'pilot_{panel}.npz');hp,hc=hist(pilot['counts']);ct=pilot['click_target'];check(np.array_equal(hp/20000>target,pilot['pnr_event']) and np.array_equal(hc/20000>ct,pilot['click_event']),'predeclared pilot events');h=np.zeros(len(target),np.int64);c=np.zeros(2**m,np.int64);odd=photons=maximum=0;seconds=0.
    for batch in range(20):
        counts=np.load(base/f'counts_{panel}_{batch}.npz')['counts'];row=read(base/f'status_{panel}_{batch}.json');seed=0x7c000000000000+(panel*20+batch)*2**40;seeds.append(seed);check(row['seed']==seed and counts.shape==(50000,m) and counts.dtype==np.uint32,'cohort count and nonoverlap seed');a,z=hist(counts);h+=a;c+=z;total=counts.sum(axis=1);odd+=int((total%2!=0).sum());photons+=int(total.sum());maximum=max(maximum,int(total.max()));seconds+=row['seconds'];records+=len(counts);check(row['statistics']['photons']==int(total.sum()),'native/raw photon total')
    if ci==1:independent_click=np.array(list(map(float,read(ROOT/'results/iteration75_task_comparison_v1/target.json')['probabilities'])))
    else:independent_click=np.array([math.sqrt(1-t[0]**2),1-math.sqrt(1-t[0]**2)])
    check(abs(ct-independent_click).max()<1e-13,'independent click target')
    saved=np.load(base/f'histogram_{panel}.npz');check(np.array_equal(h,saved['pnr']) and np.array_equal(c,saved['click']),'both histograms reconstructed')
    for kind,histogram,tar,event,tail in [('pnr',h,target,pilot['pnr_event'],target[-1]),('click',c,ct,pilot['click_event'],0.)]:
        empirical=histogram/1000000;tv=float(abs(empirical-tar).sum()/2);lower=max(0.,float((empirical[event]-tar[event]).sum())-radius);upper=min(1.,tv+radius+tail);check(np.allclose([tv,lower,upper],[stored[kind][k] for k in ('empirical_tv','lower','upper')],rtol=0,atol=1e-15),'complete '+kind+' TV bounds');bounds.append(dict(panel=panel,kind=kind,lower=lower,upper=upper))
    check(odd==stored['odd_records'] and photons/1000000==stored['mean_photons'] and maximum==stored['maximum_photons'] and seconds==stored['seconds'],'parity mean maximum and sampling cost')
check(records==12000000 and len(set(seeds))==240,'12M records and 240 disjoint seed ranges')
base=ROOT/'results/iteration75_task_comparison_v1';target=np.array(list(map(float,read(base/'target.json')['probabilities'])));lawrows=read(base/'law_summary.json')['rows']
for k in (1,2,4,8):
    d=np.load(base/f'fused_law_{k}.npz');law=np.prod(np.where(d['patterns'],d['probabilities'],1-d['probabilities']),axis=1);check(np.array_equal(law,d['law']) and abs(law.sum()-1)<1e-10,'complete direct click law');ref=next(r for r in lawrows if r['kind']=='fused' and r['budget']==k);check(abs(float(abs(law-target).sum()/2)-ref['tv'])<1e-15,'fused target TV')
for k in (1,2,3,4):
    export=read(base/f'frugal_export_{k}.json');check(export['audit_identical_outputs'],'Frugal original executable audit')
    for model in ('continuous','uint64'):
        law=full_law(4,export['prefix_tree'],model=model);tv=float(abs(np.array(list(map(float,law['distribution'])))-target).sum()/2);ref=next(r for r in lawrows if r['kind']=='frugal_'+model and r['budget']==k);check(abs(tv-ref['tv'])<1e-15,'Frugal complete '+model+' law')
def verify_timing(base,expected,shape):
    summary=read(base/'summary.json');check(summary['passed'] and len(summary['rows'])==expected,'complete timing group '+base.name);medians={}
    for row in summary['rows']:
        label=f"{row['rep']}_{row['kind']}_{row['budget']}";d=np.load(base/f'records_{label}.npz');bits=d['clicks'];check(bits.shape==shape and np.isin(bits,[0,1]).all(),'all timing records '+label)
        if 'counts' in d:check(np.array_equal(bits,d['counts']>0),'PNR to click mapping '+label)
    for kind,k in sorted({(r['kind'],r['budget']) for r in summary['rows']}):
        group=[r for r in summary['rows'] if (r['kind'],r['budget'])==(kind,k)];check(sorted(r['rep'] for r in group)==[0,1,2],'three repetitions');value=statistics.median(r['wall_seconds'] for r in group);key=f'{kind}_{k}';check(value==summary['medians'][key],'recomputed timing median');medians[key]=value
    return summary,medians
small,smallmed=verify_timing(base,33,(65536,4))
for rep in range(3):
    reference=np.load(base/f'records_{rep}_prefix_31.npz')['counts'];total=reference.sum(axis=1)
    for k in (7,15,23):
        counts=np.load(base/f'records_{rep}_prefix_{k}.npz')['counts'];different=np.any(counts!=reference,axis=1);check(not different[total<=k].any() and (counts[different].sum(axis=1)>=k+1).all(),'common random prefix coupling')
base=ROOT/'results/iteration75_lowrank_v2';lowrank,lowmed=verify_timing(base,12,(4096,144));data=np.load(base/'input.npz');v,t=data['v'],data['t'];O=mp.matrix(np.block([[v.real,-v.imag],[v.imag,v.real]]).tolist());vac=1/mp.sqrt(mp.det(mp.eye(4)+mp.matrix(data['x'].tolist())*O.T*O));tail=mp.mpf(float(t[0]))**32;target=read(base/'target.json');check(abs(vac-mp.mpf(target['vacuum']))<mp.mpf('1e-70') and abs(tail-mp.mpf(target['ideal_prefix31_tv_bound']))<mp.mpf('1e-70'),'80 digit vacuum and analytic geometric tail')
for k in (1,2,3):
    d=np.load(base/f'zero_{k}.npz');row=read(base/f'event_{k}.json');check(int((~np.isfinite(d['threshold'])).sum())==row['nonfinite'] and int(((d['threshold']<0)|(d['threshold']>1)).sum())==row['saturated'],'explicit nonfinite and saturated thresholds')
    for model in ('continuous','uint64'):
        prob=math.prod(zero_probability(float(q),model) for q in d['threshold']);value=mp.mpf(prob.numerator)/prob.denominator;check(abs(value-mp.mpf(row['models'][model]['vacuum_probability']))<mp.mpf('1e-70') and abs(abs(value-vac)-mp.mpf(row['models'][model]['full_tv_lower_bound']))<mp.mpf('1e-70'),'zero event law and TV lower bound')
    for rep in range(3):
        if k==1:continue
        row=read(base/f'row_{rep}_frugal_{k}.json');observed=row['zero_records']/4096;expected=float(read(base/f'event_{k}.json')['models']['uint64']['vacuum_probability']);check(abs(observed-expected)<.05,'actual Frugal zero-event frequency consistent with exported law')
interrupted=ROOT/'results/iteration75_scale_click_v2';receipt=read(ROOT/'results/iteration75_scale_supervision_v1/receipt.json');check(receipt['status']=='operator_budget_interrupted' and receipt['completed_rows']==23 and not receipt['summary_present'],'GPU budget interruption is not a successful timing')
for name,digest in receipt['artifact_hashes'].items():check(H(interrupted/name)==digest,'preserved partial artifact '+name)
rows=[read(p) for p in interrupted.glob('row_0_*.json')];check(len(rows)==18 and all(r['returncode']==0 for r in rows),'complete 192-output subset within interrupted run');scalemed={}
for kind,k in sorted({(r['method'],r['budget']) for r in rows}):
    group=[r for r in rows if (r['method'],r['budget'])==(kind,k)];check(sorted(r['rep'] for r in group)==[0,1,2],'192-output matched repetitions');scalemed[f'{kind}_{k}']=statistics.median(r['wall_seconds'] for r in group)
for p in (ROOT/'results/iteration75_scale_click_v1').glob('replay_*.npz'):
    d=np.load(p);check(np.array_equal(d['old_clicks'],d['new_clicks']) and abs(d['old_probabilities']-d['new_probabilities']).max()<1e-8,'expanded direct-click replay '+p.name)
index=read(ROOT/'results/iteration75_frugal_gpu_build_v3/index_equivalence.json');check(index['passed'] and sum(r['entries'] for r in index['rows'])==268995 and all(r['all_bytes_equal'] for r in index['rows']),'268995 official GPU index entries byte equal')
check(not read(ROOT/'results/iteration75_scale_click_v1/summary.json')['passed'],'initial adapter rejections preserved')
for version in (1,2):check(not read(ROOT/f'results/iteration75_frugal_gpu_build_v{version}/summary.json')['passed'],'failed GPU build preserved')
shutil.copy2(__file__,OUT/Path(__file__).name);save('checks.json',checks);result=dict(passed=True,checks=len(checks),full_suite=tests,raw_quality_records_recomputed=records,quality_bounds=bounds,small_medians=smallmed,lowrank_medians=lowmed,completed_192_output_medians=scalemed,large_input_full_tv_claim=False,source_sha256=H(Path(__file__)));save('validation.json',result);print(json.dumps({k:result[k] for k in ('passed','checks','raw_quality_records_recomputed','small_medians','lowrank_medians','completed_192_output_medians')}),flush=True)
