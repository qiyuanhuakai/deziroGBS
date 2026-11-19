"""Verify every holdout count, frozen source, native replay and timing panel."""
from pathlib import Path
import os,json,hashlib,shutil,sys,math,statistics,argparse,ctypes as C
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=[]
def check(ok,label):checks.append(label);assert ok,label

for base in sorted((ROOT/'results').glob('iteration73*')):
    if base.is_dir() and (base/'manifest.json').exists():
        for name,digest in read(base/'manifest.json')['artifacts'].items():check(H(base/name)==digest,'frozen artifact '+base.name+'/'+name)
tests=read(ROOT/'results/iteration73_tests_v1/validation.json');check(tests['passed'] and tests['unchanged_sources'] and tests['polynomial_pnr_passed']==5 and tests['philox_click_passed']==5,'complete suite and ten public API tests')
for name,digest in read(ROOT/'results/iteration73_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested scientific source frozen '+name)
for row in read(ROOT/'results/iteration73_parity_diagnostic_v1/summary.json')['rows']:check(not row['success'] and 'p=-4.4906869' in row['error'],'pure parity failure disclosed and retained')
probe=read(ROOT/'results/iteration73_polynomial_validation_v1/summary.json');check(probe['passed'] and len(probe['moment_checks'])==30,'independent polynomial quadrature complete')
for p in (ROOT/'results/iteration73_polynomial_validation_v1').glob('probe_*.npz'):
    d=np.load(p);check(np.max(abs(d['native_mean']-d['quadrature_mean']))<1e-9 and np.max(abs(d['native_x']-d['quadrature_x']))<1e-9 and abs(d['native_mass'][0]-d['quadrature_mass'])/max(1,abs(d['quadrature_mass']))<1e-9,'stored native and independent quadrature moments agree')

hold=ROOT/'results/iteration73_pnr_holdout_v1';summary=read(hold/'summary.json');plan=read(hold/'plan.json');radius=math.sqrt(math.log(1600)/(2*2000000));check(summary['passed'] and len(summary['rows'])==8 and radius==summary['radius']==plan['radius'],'predeclared eight-law simultaneous confidence allocation');records=0;seeds=[];bounds=[]
for row in summary['rows']:
    case,method=row['case'],row['method'];data=np.load(ROOT/f'results/iteration73_polynomial_pilot_v2/input_{case}.npz');cutoff=int(data['cutoff']);codes=data['codes'];law=data['target'];base=cutoff+1;lookup=np.full(base**4,-1,np.int32);lookup[codes]=np.arange(len(codes));powers=np.array([base**j for j in range(4)],np.uint64);hist=np.zeros(len(law),np.int64);photons=maximum=extreme=0;seconds=0
    for batch in range(20):
        counts=np.load(hold/f'counts_{case}_{method}_{batch}.npz')['counts'];status=read(hold/f'status_{case}_{method}_{batch}.json');seeds.append(status['seed']);check(status['returncode']==0 and counts.dtype==np.uint32 and counts.shape==(100000,4),'complete uncut holdout batch')
        totals=counts.sum(axis=1);inside=totals<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];check(labels.min()>=0,'all PNR patterns mapped');hist+=np.bincount(labels,minlength=len(law));photons+=int(totals.sum());maximum=max(maximum,int(totals.max()));extreme+=int(np.count_nonzero(totals>=128));records+=len(counts);seconds+=status['seconds'];check(status['statistics']['photons']==int(totals.sum()),'native total photons verified')
    prior=np.load(ROOT/(f'results/iteration71_pnr_branch_pilot_v2/counts_{case}_2_1.npz' if method=='finite' else f'results/iteration73_polynomial_pilot_v2/hist_{case}_{method[-1]}.npz'))['histogram'];event=prior/prior.sum()>law;stored=np.load(hold/f'histogram_{case}_{method}.npz');check(np.array_equal(hist,stored['histogram']) and np.array_equal(event,stored['fixed_event']),'all raw records and fixed pilot event reconstructed')
    empirical=hist/hist.sum();tv=float(abs(empirical-law).sum()/2);lower=max(0,float(empirical[event].sum()-law[event].sum())-radius);upper=min(1,tv+radius+float(law[-1]));check(abs(tv-row['empirical_tv'])<1e-15 and np.allclose([lower,upper],row['nominal_full_tv_interval'],atol=1e-15,rtol=0),'full PNR bounds recomputed');check(row['maximum_photons']==maximum and row['events_ge128']==extreme and row['mean_photons']==photons/2000000 and abs(row['seconds']-seconds)<1e-12,'PNR uncensored tails and cost verified')
    target_mean=.5*(np.trace(data['x'])+data['variance'].sum()+data['mean']@data['mean']);bounds.append(dict(case=case,method=method,lower=lower,upper=upper,tv=tv,seconds=seconds,target_mean=float(target_mean),mean_photons=photons/2000000,maximum_photons=maximum,events_ge128=extreme))
check(len(set(seeds))==160 and min(b-a for a,b in zip(sorted(seeds),sorted(seeds)[1:]))>=2**40,'disjoint predeclared holdout seed ranges')
lookup_bounds={(r['case'],r['method']):r for r in bounds};check(lookup_bounds[3,'poly3']['upper']<.01 and lookup_bounds[4,'poly3']['upper']<.01,'both selected K3 input bounds below one percent');quality_factor=lookup_bounds[3,'poly0']['lower']/lookup_bounds[3,'poly3']['upper'];check(quality_factor>4.5,'strict case3 improvement versus Gaussian reset')

philox=ROOT/'results/iteration73_philox_pilot_v1';pg=read(philox/'summary.json');check(pg['passed'] and len(pg['replays'])==9,'Philox pilot complete');thermal=np.load(philox/'thermal.npz');clicks=thermal['clicks'];hist=np.bincount(clicks@np.array([1,2]),minlength=4)/len(clicks);tv=float(abs(hist-thermal['target']).sum()/2);check(abs(tv-pg['thermal']['thermal_tv'])<1e-15 and tv<.005,'analytic thermal law from raw clicks')
draws=thermal['draws'];check(np.isfinite(draws).all() and draws.min()>=0 and draws.max()<1 and np.count_nonzero(np.modf(draws*2**32)[0])==pg['thermal']['non_32bit_grid'],'full double precision uniform inputs')
for row in pg['replays']:
    d=np.load(philox/f"replay_{row['case']}_{row['budget']}.npz");check(np.array_equal(d['cpu'],d['gpu']) and float(abs(d['cpu_probabilities']-d['gpu_probabilities']).max())==row['maximum_conditional_error'],'explicit random-input CPU GPU replay')
build=ROOT/'results/iteration73_frugal_cpu_build_v5';check(read(build/'summary.json')['passed'],'AVX2 fixed build passed')
original=(ROOT/'third_party/FrugalGBS/include/fgbs/cpu/cpu_sampler.hpp').read_text();expected=original.replace('NaiveSum<T>& operator+=(T value);','__attribute__((always_inline)) inline NaiveSum<T>& operator+=(T value) { _sum += value; return *this; }').replace('KahanSum<T>& operator+=(T value);','__attribute__((always_inline)) inline KahanSum<T>& operator+=(T value) { const auto y=value-_compensation; const auto t=_sum+y; _compensation=(t-_sum)-y; _sum=t; return *this; }');check((build/'patched_include/fgbs/cpu/cpu_sampler.hpp').read_text()==expected,'Frugal exact disclosed compatibility transformation')

bench=ROOT/'results/iteration73_click_benchmark_v1';benchmark=read(bench/'summary.json');check(benchmark['passed'] and len(benchmark['rows'])==86,'all 86 click timing panels complete');medians={}
for row in benchmark['rows']:
    panel,rep,kind,budget=row['panel'],row['rep'],row['method'],row['budget']
    label=f'{panel}_{rep}_{kind}_{budget}' if not panel.startswith('diagnostic') else f'diagnostic_{kind}_{budget}'
    out=np.load(bench/f'records_{label}.npz')['clicks'];m=4 if panel=='small_3' else 8 if panel=='small_4' else 144;check(row['returncode']==0 and out.shape==(row['samples'],m) and np.isin(out,[0,1]).all(),'complete benchmark bit records')
for panel in sorted(set(r['panel'] for r in benchmark['rows'])):
    medians[panel]={f'{kind}_{budget}':statistics.median(r['wall_seconds'] for r in benchmark['rows'] if (r['panel'],r['method'],r['budget'])==(panel,kind,budget)) for kind,budget in sorted(set((r['method'],r['budget']) for r in benchmark['rows'] if r['panel']==panel))}
for rep in range(3):
    for budget in (1,4,8):
        a=np.load(bench/f'records_large_512_{rep}_philox_{budget}.npz')['clicks'];b=np.load(bench/f'records_large_4096_{rep}_philox_{budget}.npz')['clicks'];check(np.array_equal(a,b[:512]),'large Philox records independent of batch length')

large_pnr=ROOT/'results/iteration73_pnr_large_v1';lp=read(large_pnr/'summary.json');check(lp['completed'] and len(lp['rows'])==12,'twelve large PNR attempts retained');large_pnr_medians={}
for method in ('poly0','poly1','poly3','finite'):
    selected=[r for r in lp['rows'] if r['method']==method];large_pnr_medians[method]=dict(successes=sum(r['success'] for r in selected),wall_seconds=statistics.median(r['wall_seconds'] for r in selected) if all(r['success'] for r in selected) else None)
    for row in selected:
        if row['success']:
            counts=np.load(large_pnr/f"counts_{row['rep']}_{method}.npz")['counts'];check(counts.shape==(32,144) and counts.dtype==np.uint32 and counts.sum()==row['statistics']['photons'],'complete large PNR counts and statistics')

runtime=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];replayed=0
for case in (3,4):
    data=np.load(ROOT/f'results/iteration73_polynomial_pilot_v2/input_{case}.npz');seed=read(hold/f'status_{case}_poly3_0.json')['seed']
    with NativePolynomialPNRSampler(data['v'],data['x'],ROOT/'results/iteration73_polynomial_pilot_v2/polynomial.dll',noise_variance=data['variance'],retained=3,workers=4,runtime_dirs=runtime) as model:actual=model.sample(2048,seed)['counts']
    check(np.array_equal(actual,np.load(hold/f'counts_{case}_poly3_0.npz')['counts'][:2048]),'public PNR holdout replay');replayed+=2048
data=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz')
with NativePhiloxClickSampler(data['v'],data['x0'],ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=philox/'philox.dll',mean=data['mean'],noise_variance=data['variance'],runtime_dirs=runtime) as model:
    for rep in range(3):
        actual=model.sample(64,737501+rep)['clicks'];expected=np.load(bench/f'records_large_4096_{rep}_philox_4.npz')['clicks'];check(np.array_equal(actual,expected[:64]),'public large Philox replay');replayed+=64
with NativePhiloxClickSampler(np.eye(2,dtype=complex),np.zeros((4,4)),ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=philox/'philox.dll',noise_variance=np.full(4,.4),runtime_dirs=runtime) as model:
    a=model.sample(1999,733002,row_offset=17,export_random=True);b=model.sample(2097,733002,row_offset=2016,export_random=True);old=np.load(philox/'batch_invariance.npz')
    for key in ('clicks','realized_means','uniform_draws'):check(np.array_equal(np.concatenate([a[key],b[key]]),old[key]),'independent stream continuation replay '+key)
    replayed+=4096
shutil.copy2(__file__,OUT/Path(__file__).name);save('checks.json',checks);save('validation.json',dict(passed=True,checks=len(checks),full_suite=tests,raw_pnr_records_recomputed=records,public_records_replayed=replayed,pnr_bounds=bounds,case3_certified_reset_quality_factor=quality_factor,thermal_tv=tv,click_medians=medians,large_pnr=large_pnr_medians,source_sha256=H(Path(__file__))))
print(json.dumps(dict(passed=True,checks=len(checks),raw_pnr_records_recomputed=records,public_records_replayed=replayed,case3_quality_factor=quality_factor,click_medians=medians,large_pnr=large_pnr_medians)),flush=True)
