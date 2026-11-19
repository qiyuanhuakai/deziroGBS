"""Independent byte, complete-record, law, public API and timing verification."""
from pathlib import Path
import os,json,hashlib,shutil,sys,statistics,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_cuda_click_branches import NativeCudaBranchClickSampler
from gbslab.native_pnr_newton import NativeNewtonPNRSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=[]
def check(condition,label):
    checks.append(label);assert condition,label

for folder in ('iteration72_cuda_pilot_v1','iteration72_newton_pilot_v1','iteration72_frugal_gpu_comparison_v1','iteration72_small_gpu_comparison_v1'):
    base=ROOT/'results'/folder
    for name,digest in read(base/'manifest.json')['artifacts'].items():check(H(base/name)==digest,'archive bytes '+folder+'/'+name)
tests=read(ROOT/'results/iteration72_tests_v1/validation.json')
check(tests['passed'] and tests['unchanged_sources'] and tests['cuda_click_passed']==6 and tests['newton_pnr_passed']==4,'full suite and ten new tests')
for name,digest in read(ROOT/'results/iteration72_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source remains frozen '+name)
for version in range(1,6):
    build=ROOT/f'results/iteration72_frugal_gpu_build_v{version}'
    summary=read(build/'summary.json');check(summary['passed']==(version==5),'all failed builds retained '+str(version))
    for name,digest in read(build/'plan.json')['upstream_sha256'].items():check(H(ROOT/'third_party/FrugalGBS'/name)==digest,'official source unchanged '+str(version)+' '+name)
    if version==5:
        original=(ROOT/'third_party/FrugalGBS/include/fgbs/layout.hpp').read_text()
        expected=original.replace('BITSTRING&& bitstring','BITSTRING bitstring').replace('#pragma omp task if (fork) shared(bitstring_layout)','// Host compatibility: recurse serially; OpenMP correlator loop is retained.')
        check((build/'layout_compatibility.hpp').read_text()==expected,'only documented host layout workaround')
        for name,digest in summary['binaries'].items():check(H(build/name)==digest,'Frugal binary bytes '+name)

pnr=ROOT/'results/iteration72_newton_pilot_v1';summary=read(pnr/'summary.json')
check(summary['all_batches_succeeded'] and len(summary['replay_rows'])==40,'forty complete PNR replay batches')
pnr_summary=[];raw_records=0
for case in (3,4):
    photons=iterations=fallbacks=flow_calls=0;max_residual=0
    for batch in range(20):
        actual=np.load(pnr/f'replay_{case}_{batch}.npz');counts=actual['counts'];stats=actual['stats']
        expected=np.load(ROOT/f'results/iteration71_pnr_holdout/counts_{case}_branches_{batch}.npz')['counts']
        check(counts.dtype==np.uint32 and counts.shape==(100000,4),'complete PNR shape')
        check(np.array_equal(counts,expected),'every original PNR entry reproduced')
        check(stats[0]==counts.sum() and stats[1]==len(counts) and stats[2]==counts.sum(axis=1).max(),'PNR statistics independently reconstructed')
        raw_records+=len(counts);photons+=int(stats[0]);flow_calls+=int(stats[7]);iterations+=int(stats[8]);fallbacks+=int(stats[9]);max_residual=max(max_residual,float(stats[10]))
    medians={method:statistics.median(r['seconds'] for r in summary['timing_rows'] if r['case']==case and r['method']==method) for method in ('newton','bisection')}
    for rep in range(3):
        a=np.load(pnr/f'timing_{case}_{rep}_newton.npz')['counts'];b=np.load(pnr/f'timing_{case}_{rep}_bisection.npz')['counts'];check(np.array_equal(a,b),'matched PNR timing records reproduced')
    pnr_summary.append(dict(case=case,medians=medians,speedup=medians['bisection']/medians['newton'],photons=photons,flow_calls=flow_calls,iterations=iterations,iterations_per_photon=iterations/photons,fallbacks=fallbacks,maximum_survival_residual=max_residual))

pilot=ROOT/'results/iteration72_cuda_pilot_v1';gpu_summary=read(pilot/'summary.json');check(gpu_summary['passed'],'CUDA pilot passed');small=[]
for row in gpu_summary['small']:
    case,budget=row['case'],row['budget'];g=np.load(pilot/f'small_gpu_{case}_{budget}.npz');c=np.load(pilot/f'small_cpu_{case}_{budget}.npz');count,m=g['output'].shape
    patterns=((np.arange(count)[:,None]>>np.arange(m))&1).astype(np.uint8)
    check(np.array_equal(g['output'],patterns) and not g['errors'][:,0].any(),'all small forced trajectories completed')
    law=np.prod(np.where(patterns,g['probabilities'],1-g['probabilities']),axis=1)
    ref=np.load(ROOT/f'results/iteration71_branch_pilot_v1/law_{case}_{budget}.npz')['law'];target=np.load(ROOT/f'results/iteration71_branch_pilot_v1/input_{case}.npz')['target']
    tv=float(abs(law-target).sum()/2);delta=float(abs(law-ref).sum()/2)
    check(abs(law.sum()-1)<1e-10 and delta<1e-10,'full small induced law preserved numerically')
    check(abs(tv-row['law_tv_to_target'])<1e-15,'small target TV reconstructed')
    small.append(dict(case=case,budget=budget,target_tv=tv,cpu_gpu_tv=delta,maximum_conditional_delta=float(abs(g['probabilities']-c['probabilities']).max())))
for row in gpu_summary['large']:
    rep,budget=row['rep'],row['budget'];g=np.load(pilot/f'large_gpu_{rep}_{budget}.npz');c=np.load(pilot/f'large_cpu_{rep}_{budget}.npz');old=np.load(ROOT/f'results/iteration71_frugal_comparison/large_{rep}_branch_{budget}.npz')['clicks']
    check(not g['errors'][:,0].any() and np.array_equal(g['output'],c['output']) and np.array_equal(c['output'],old),'large same-input CPU GPU and archive records agree')

bench=ROOT/'results/iteration72_frugal_gpu_comparison_v1';bench_summary=read(bench/'summary.json');check(bench_summary['passed'] and len(bench_summary['rows'])==21,'twenty-one matched GPU timing batches')
for bits in (32,64):
    check(not np.load(bench/f'vacuum_{bits}.npz')['clicks'].any(),'Frugal GPU analytic vacuum')
    for case in (3,4):
        data=np.load(bench/f'validation_{case}_{bits}.npz');clicks=data['clicks'];m=clicks.shape[1];hist=np.bincount((clicks@np.left_shift(np.uint64(1),np.arange(m,dtype=np.uint64))).astype(np.int64),minlength=2**m)/len(clicks)
        tv=float(abs(hist-data['target']).sum()/2);stored=read(bench/f'validation_{case}_{bits}.json');check(abs(tv-stored['tv_to_reference'])<1e-15 and tv<.025,'Frugal validation histogram from every raw row')
medians={};sampling_medians={}
for kind,budget in sorted(set((r['method'],r['budget']) for r in bench_summary['rows'])):
    label=f'{kind}_{budget}';rows=[r for r in bench_summary['rows'] if r['method']==kind and r['budget']==budget]
    medians[label]=statistics.median(r['wall_seconds'] for r in rows)
    sampling_medians[label]=statistics.median((r['sampling_seconds'] if kind=='branch' else r['timings']['sampling']) for r in rows)
    for row in rows:
        actual=np.load(bench/f"records_{row['rep']}_{label}.npz")['clicks'];check(actual.shape==(512,144) and np.isin(actual,[0,1]).all(),'complete binary GPU benchmark records')
        if kind=='branch':check(np.array_equal(actual,np.load(ROOT/f"results/iteration71_frugal_comparison/large_{row['rep']}_branch_{budget}.npz")['clicks']),'public CUDA full benchmark replay')

runtime=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];public_records=0
for case in (3,4):
    data=np.load(ROOT/f'results/iteration71_pnr_branch_pilot_v2/input_{case}.npz');seed=read(pnr/f'replay_{case}_0.json')['seed']
    with NativeNewtonPNRSampler(data['v'],data['x'],pnr/'newton.dll',noise_variance=data['variance'],workers=4,runtime_dirs=runtime) as model:
        actual=model.sample(2048,seed)['counts'];expected=np.load(pnr/f'replay_{case}_0.npz')['counts'];check(np.array_equal(actual,expected[:len(actual)]),'independent public Newton replay');public_records+=len(actual)
data=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz')
with NativeCudaBranchClickSampler(data['v'],data['x0'],pilot/'cpu.dll',gpu_library=pilot/'gpu.dll',mean=data['mean'],noise_variance=data['variance'],runtime_dirs=runtime) as model:
    for rep in range(3):
        for budget in (1,4,8):
            actual=model.sample(32,715901+rep,branch_budget=budget)['clicks'];expected=np.load(bench/f'records_{rep}_branch_{budget}.npz')['clicks'];check(np.array_equal(actual,expected[:32]),'independent public CUDA replay');public_records+=len(actual)

old=read(ROOT/'results/iteration71_frugal_comparison/summary.json')
old_frugal_cpu={f'frugal_cpu_{k}':statistics.median(r['compute_seconds'] for r in old['rows'] if r['panel']=='large_144' and r['method']==f'frugal_{k}') for k in (2,3)}
small_bench=ROOT/'results/iteration72_small_gpu_comparison_v1';sb=read(small_bench/'summary.json');check(sb['passed'] and len(sb['rows'])==18,'eighteen small GPU benchmark batches')
small_medians={}
for case,kind,budget in sorted(set((r['case'],r['method'],r['budget']) for r in sb['rows'])):
    selected=[r for r in sb['rows'] if (r['case'],r['method'],r['budget'])==(case,kind,budget)]
    small_medians[f'{case}_{kind}_{budget}']=dict(wall_seconds=statistics.median(r['wall_seconds'] for r in selected),reference_induced_tv=selected[0]['reference_induced_tv'])
    for row in selected:
        record=np.load(small_bench/f"records_{case}_{row['rep']}_{kind}_{budget}.npz")
        check(record['clicks'].shape==(16384,4 if case==3 else 8) and np.isin(record['clicks'],[0,1]).all(),'complete small benchmark output')
        check(abs(float(abs(record['reference_law']-record['target']).sum()/2)-row['reference_induced_tv'])<1e-15,'small full-law quality anchor reconstructed')
shutil.copy2(__file__,OUT/Path(__file__).name)
save('validation.json',dict(passed=True,checks=len(checks),raw_pnr_records_recomputed=raw_records,public_records_replayed=public_records,full_suite=tests,pnr=pnr_summary,small_click=small,gpu_medians=medians,gpu_sampling_medians=sampling_medians,small_gpu_medians=small_medians,previous_frugal_cpu_medians=old_frugal_cpu,source_sha256=H(Path(__file__))))
save('checks.json',checks)
print(json.dumps(dict(passed=True,checks=len(checks),pnr=pnr_summary,gpu_medians=medians,raw_pnr_records_recomputed=raw_records,public_records_replayed=public_records)),flush=True)
