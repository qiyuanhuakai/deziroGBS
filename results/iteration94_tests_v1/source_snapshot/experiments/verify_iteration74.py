"""Independently reconstruct all new pure PNR records and CUDA comparisons."""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,math,statistics,functools
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_structured_pnr import NativeStructuredPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=[]
def check(condition,label):checks.append(label);assert condition,label
for base in sorted((ROOT/'results').glob('iteration74*')):
    if base.is_dir() and (base/'manifest.json').exists():
        for name,digest in read(base/'manifest.json')['artifacts'].items():check(H(base/name)==digest,'frozen '+base.name+'/'+name)
interrupted=ROOT/'results/iteration74_click_benchmark_v1';receipt=read(interrupted/'interruption.json');check(receipt['status']=='user_interrupted' and receipt['completed_rows']==52 and not receipt['summary_present'],'user-interrupted partial benchmark retained')
for name,digest in receipt['artifact_hashes'].items():check(H(interrupted/name)==digest,'interrupted artifact '+name)
tests=read(ROOT/'results/iteration74_tests_v3/validation.json');check(tests['passed'] and tests['structured_pnr_passed']==5 and tests['classical_tail_passed']==3 and tests['fused_click_passed']==3,'all tests and eleven new interface tests')
for name,digest in read(ROOT/'results/iteration74_tests_v3/source_hashes.json').items():check(H(ROOT/name)==digest,'tested scientific source '+name)
for version in (1,2):
    failed=read(ROOT/f'results/iteration74_tests_v{version}/validation.json');check(not failed['passed'] and failed['errors']==4 and failed['failures']==0 and all('WinError 206' in e['traceback'] for e in failed['error_details']),'preserved DLL directory capacity failures')
diag=read(ROOT/'results/iteration74_runtime_directory_diagnostic_v1/summary.json');check(diag['registered']==337 and diag['failure']['winerror']==206 and diag['recovered'],'isolated DLL directory limit and recovery')
for name in ('structured','classical'):
    base=ROOT/f'results/iteration74_{name}_pilot_v1';summary=read(base/'summary.json');check(summary['all_samples_succeeded'] and len(summary['rows'])==28,'all '+name+' pilot batches completed');positive=read(base/'positive_validation.json');check(len(positive['norm_checks'])==60 and len(positive['endpoints'])==40,'independent moment and endpoint panels complete')
    for endpoint in positive['endpoints']:
        if endpoint['remaining']==0 and endpoint['k']%2:check(endpoint['native']==0,'exact odd pure endpoint')
    for p in base.glob('norm_*.npz'):
        data=np.load(p);check(abs(data['native'][0]-data['target'])<1e-10*max(1,abs(data['target'])),'stored positive norm matches independent Wick')

base=ROOT/'results/iteration74_pure_quality_v1';quality=read(base/'summary.json');radius=math.sqrt(math.log(1200)/(2*2000000));check(quality['passed'] and radius==quality['radius'] and len(quality['rows'])==6,'six predeclared independent pure laws');bounds=[];allseeds=[];records=0
for ci in (0,1):
    data=np.load(base/f'input_{ci}.npz');v,t,cutoff,codes,target=data['v'],data['t'],int(data['cutoff']),data['codes'],data['target'];m=len(v);b=cutoff+1;powers=np.array([b**j for j in range(m)],np.uint64);lookup=np.full(b**m,-1,np.int32);lookup[codes]=np.arange(len(codes),dtype=np.int32);matrix=-(v*t)@v.T;p0=float(np.sqrt(np.prod(1-t*t)))
    def histogram(counts):
        total=counts.sum(axis=1);inside=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];check(labels.min()>=0,'all raw pure patterns mapped');return np.bincount(labels,minlength=len(target))
    sector=np.zeros(cutoff+1)
    if ci==0:
        sector[0]=np.sqrt(1-t[0]**2)
        for j in range(1,cutoff//2+1):sector[2*j]=sector[2*j-2]*(2*j-1)/(2*j)*t[0]**2
    else:sector[::2]=(1-t[0]**2)*t[0]**np.arange(0,cutoff+1,2)
    totals=sum((codes//powers[j])%b for j in range(m));actual=np.bincount(totals.astype(int),weights=target[:-1],minlength=cutoff+1);check(np.max(abs(actual-sector))<1e-12 and abs(target.sum()-1)<1e-12,'Fock sectors and analytic squeezed total law')
    rng=np.random.default_rng(745901+ci)
    for photons in (0,2,4,6,8,10):
        for rep in range(10):
            counts=rng.multinomial(photons,np.full(m,1/m));indices=np.repeat(np.arange(m),counts)
            @functools.lru_cache(None)
            def haf(mask):
                if not mask:return 1+0j
                i=(mask&-mask).bit_length()-1;rest=mask&~(1<<i);return sum(matrix[indices[i],indices[j]]*haf(rest&~(1<<j)) for j in range(len(indices)) if rest>>j&1)
            expected=p0*abs(haf((1<<photons)-1))**2/math.prod(math.factorial(int(n)) for n in counts);code=int(counts.astype(np.uint64)@powers);check(abs(expected-target[lookup[code]])<1e-12,'independent hafnian probability versus Fock recurrence')
    for ki,k in enumerate((0,1,3)):
        stored=next(r for r in quality['rows'] if r['case']==ci and r['retained']==k);hist=np.zeros(len(target),np.int64);odd=photons=maximum=0;seconds=0.;pilot=np.load(base/f'pilot_{ci}_{k}.npz');event=histogram(pilot['counts'])/20000>target;check(np.array_equal(event,pilot['fixed_event']),'fixed event reconstructed from independent pilot')
        for batch in range(20):
            counts=np.load(base/f'counts_{ci}_{k}_{batch}.npz')['counts'];row=read(base/f'status_{ci}_{k}_{batch}.json');seed=0x76000000000000+(ci*60+ki*20+batch)*2**40;check(row['seed']==seed and counts.shape==(100000,m) and counts.dtype==np.uint32,'complete independent pure batch and seed');allseeds.append(seed);hist+=histogram(counts);total=counts.sum(axis=1);odd+=int(np.count_nonzero(total%2));photons+=int(total.sum());maximum=max(maximum,int(total.max()));seconds+=row['seconds'];records+=len(counts);check(row['statistics']['photons']==int(total.sum()),'pure raw counts native total')
        empirical=hist/2000000;tv=float(abs(empirical-target).sum()/2);lower=max(0,float((empirical[event]-target[event]).sum())-radius);upper=min(1,tv+radius+target[-1]);check(np.array_equal(hist,np.load(base/f'histogram_{ci}_{k}.npz')['histogram']) and abs(tv-stored['empirical_tv'])<1e-15 and np.allclose([lower,upper],stored['nominal_full_tv_interval'],atol=1e-15,rtol=0),'pure full TV bounds recomputed');check(odd==stored['odd_records'] and photons/2000000==stored['mean_photons'] and maximum==stored['maximum_photons'] and seconds==stored['seconds'],'pure parity, mean, uncensored maxima and cost');bounds.append(dict(case=ci,retained=k,tv=tv,lower=lower,upper=upper,odd_records=odd,mean_photons=photons/2000000,seconds=seconds))
check(records==12000000 and len(set(allseeds))==120,'12M new independent pure records')
for k in (1,3):check(not read(base/f'old_{k}.json')['success'],'old same pure seed failure retained')
noise=ROOT/'results/iteration74_noise_replay_v1';replay=read(noise/'summary.json');check(replay['passed'] and replay['records']==4000000,'previous noisy holds replay completely')
for row in replay['rows']:
    ci,batch=row['case'],row['batch'];a=np.load(noise/f'counts_{ci}_{batch}.npz')['counts'];old=np.load(ROOT/f'results/iteration73_pnr_holdout_v1/counts_{ci}_poly3_{batch}.npz')['counts'];check(np.array_equal(a,old),'noisy count-by-count historical replay')

laws=ROOT/'results/iteration74_small_laws_v1';law_summary=read(laws/'summary.json');check(law_summary['passed'] and len(law_summary['rows'])==64,'all 64 complete fused small laws')
for row in law_summary['rows']:
    data=np.load(laws/f"law_{row['case']}_{row['version']}_{row['budget']}.npz");law=np.prod(np.where(data['patterns'],data['probabilities'],1-data['probabilities']),axis=1);check(np.array_equal(law,data['law']) and np.array_equal(data['patterns'],data['clicks']),'complete law reconstructed from all conditional paths');check(abs(float(abs(law-data['target']).sum()/2)-row['tv_to_target'])<1e-15 and float(abs(law-data['reference']).sum()/2)<1e-8,'complete target and CPU law comparisons')
gpu_medians={};gpu_records=0
for version in (1,2):
    base=ROOT/f'results/iteration74_fused_comparison_v{version}';comparison=read(base/'summary.json');check(comparison['passed'] and len(comparison['pairs'])==72,'72 fused comparison pairs')
    for pair in comparison['pairs']:
        label=f"{pair['panel']}_{pair['rep']}";a=np.load(base/f'records_{label}_old.npz');bdata=np.load(base/f'records_{label}_fused.npz');check(np.array_equal(a['clicks'],bdata['clicks']),'all fused comparison click bits');delta=float(abs(a['probabilities']-bdata['probabilities']).max());check(delta==pair['maximum_probability_difference'] and delta<1e-8,'all fused conditional probabilities');gpu_records+=len(a['clicks'])
        if version==1:check(np.array_equal(a['probabilities'],bdata['probabilities']) and np.array_equal(a['diagnostics'],bdata['diagnostics']),'v1 bitwise arithmetic replay')
        rows=[r for r in comparison['rows'] if r['panel']==pair['panel'] and r['rep']==pair['rep']];d=100 if pair['case']=='large' else 4 if pair['case']==3 else 6;oldrow=next(r for r in rows if r['kind']=='old');newrow=next(r for r in rows if r['kind']=='fused');check(oldrow['cuda_timings']['workspace_bytes']-newrow['cuda_timings']['workspace_bytes']==pair['count']*pair['budget']*(d*d+d)*8,'matrix and mean workspace halved')
    gpu_medians[str(version)]=comparison['medians']
capacity=ROOT/'results/iteration74_capacity_v1';cap=read(capacity/'summary.json');check(cap['passed'] and len(cap['rows'])==9,'all capacity attempts retained')
for row in cap['rows']:
    if row['kind']=='old':check(not row['success'] and '75 percent' in row['error'],'old large workspace explicitly rejected')
    else:check(row['success'],'fused 16384 batch succeeded')
for rep in range(3):
    old=np.load(capacity/f'old_split_{rep}.npz')
    for kind in ('fused','reduced'):
        new=np.load(capacity/f'records_{rep}_{kind}.npz');check(new['clicks'].shape==(16384,144) and np.array_equal(new['clicks'],old['clicks']),'capacity records equal split baseline');check(float(abs(new['probabilities']-old['probabilities']).max())<1e-8,'capacity probabilities match');gpu_records+=16384
bench=ROOT/'results/iteration74_click_benchmark_v2';benchmark=read(bench/'summary.json');check(benchmark['passed'] and len(benchmark['rows'])==57,'57 fresh Frugal timing panels');medians={}
for row in benchmark['rows']:
    label=f"{row['panel']}_{row['rep']}_{row['method']}_{row['budget']}";clicks=np.load(bench/f'records_{label}.npz')['clicks'];m=4 if row['panel']=='small_3' else 8 if row['panel']=='small_4' else 144;check(row['returncode']==0 and clicks.shape==(row['samples'],m) and np.isin(clicks,[0,1]).all(),'all benchmark host bit arrays')
for panel in sorted(set(r['panel'] for r in benchmark['rows'])):
    medians[panel]={f'{kind}_{budget}':statistics.median(r['wall_seconds'] for r in benchmark['rows'] if (r['panel'],r['method'],r['budget'])==(panel,kind,budget)) for kind,budget in sorted(set((r['method'],r['budget']) for r in benchmark['rows'] if r['panel']==panel))}
runtime=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];public=0
for ci in (0,1):
    data=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{ci}.npz');seed=read(ROOT/f'results/iteration74_pure_quality_v1/status_{ci}_3_0.json')['seed']
    with NativeStructuredPNRSampler(data['v'],data['t'],ROOT/'results/iteration74_structured_pilot_v1/structured.dll',retained=3,workers=4,runtime_dirs=runtime) as model:r=model.sample(2048,seed)
    check(np.array_equal(r['counts'],np.load(ROOT/f'results/iteration74_pure_quality_v1/counts_{ci}_3_0.npz')['counts'][:2048]),'public pure cohort replay');public+=2048
data=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz')
with NativePhiloxClickSampler(data['v'],data['x0'],ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=ROOT/'results/iteration74_fused_pilot_v2/philox.dll',mean=data['mean'],noise_variance=data['variance'],branch_budget=8,runtime_dirs=runtime) as model:r=model.sample(64,744501)
check(np.array_equal(r['clicks'],np.load(capacity/'records_0_reduced.npz')['clicks'][:64]),'public large capacity replay');public+=64
shutil.copy2(__file__,OUT/Path(__file__).name);save('checks.json',checks);result=dict(passed=True,checks=len(checks),full_suite=tests,new_pure_records_recomputed=records,historical_noisy_records_recomputed=4000000,gpu_records_compared=gpu_records,public_records_replayed=public,pure_bounds=bounds,click_medians=medians,gpu_comparison_medians=gpu_medians,capacity_medians={kind:statistics.median(r['wall_seconds'] for r in cap['rows'] if r['kind']==kind) for kind in ('fused','reduced')},source_sha256=H(Path(__file__)));save('validation.json',result);print(json.dumps({k:result[k] for k in ('passed','checks','new_pure_records_recomputed','historical_noisy_records_recomputed','public_records_replayed','click_medians','capacity_medians')}),flush=True)
