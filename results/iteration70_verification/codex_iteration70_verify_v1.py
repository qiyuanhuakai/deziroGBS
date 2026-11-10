from pathlib import Path
import hashlib,json,shutil,math,re,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iteration70_verification';OUT.mkdir(exist_ok=False)
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):
    with p.open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=0
def check(ok,label=''):
    global checks
    assert ok,label;checks+=1
mapping=read(ROOT/'results/iteration70_archive_mapping.json');arch=ROOT/'results/iteration70_archive'
for row in mapping['rows']:
    check(H(ROOT/row['source'])==row['sha256'],row['source']);check(H(ROOT/row['destination'])==row['sha256'],row['destination']);check((ROOT/row['destination']).stat().st_size==row['bytes'])
mapped={r['source'] for r in mapping['rows']}
for p in (ROOT/'.git').glob('codex_iteration70*.py'):
    if p.relative_to(ROOT).as_posix() not in mapped:shutil.copy2(p,OUT/p.name)
manifests=0
for folder in [p for p in arch.iterdir() if p.is_dir()]+[p for p in (ROOT/'results').glob('iteration70*') if p.is_dir() and p not in (arch,OUT)]:
    p=folder/'manifest.json'
    if not p.exists():continue
    manifests+=1
    for relative,sha in read(p).get('artifacts',{}).items():
        check(isinstance(sha,str),str(p));target=folder/Path(*('git_source_snapshot' if x=='.git' else x for x in Path(relative).parts));check(H(target)==sha,str(target))
for name in ('iteration70_build','iteration70_public_validation','iteration70_pgf_reference','iteration70_benchmark','iteration70_quality_extension','iteration70_quality_extension_v2'):
    check(read(ROOT/'results'/name/'summary.json')['passed'],name)
for name in ('iteration70_flow_pilot_v1_20261110','iteration70_flow_pilot_v2_20261110','iteration70_parallel_pilot_v1_20261110','iteration70_pgf_quad_v1_20261110','iteration70_noise_screen_v2_20261110','iteration70_noise_holdout_v1_20261110','iteration70_frugal_speed_v2_20261110','iteration70_frugal_large_v1_20261110','iteration70_direct_click_pilot_v1_20261110','iteration70_dual_benchmark_v1_20261110'):
    check(read(arch/name/'summary.json')['passed'],name)
tests=read(ROOT/'results/iteration70_tests/validation.json');check(tests['executed_gate_passed'] and tests['unchanged_sources']);check(tests['failures']==tests['errors']==0);check(tests['parallel_tests_passed']==9 and tests['direct_click_tests_passed']==7)
tested=read(ROOT/'results/iteration70_tests/source_hashes_at_start.json');doc_changes=[]
for name,sha in tested.items():
    if name=='native/README.md':
        if H(ROOT/name)!=sha:doc_changes.append(name)
    else:check(H(ROOT/name)==sha,name)
new_python=sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments').glob('iteration70*.py') if p.relative_to(ROOT).as_posix() not in tested)
check(set(new_python)=={'experiments/iteration70_benchmark.py','experiments/iteration70_quality_extension_v2.py'},str(new_python))
for name in new_python:compile((ROOT/name).read_text(encoding='utf-8'),name,'exec');check(True,name)
check(H(ROOT/'experiments/iteration70_quality_extension_v2.py')==read(ROOT/'results/iteration70_quality_extension_v2/plan.json')['source_sha256'])
check(H(ROOT/'experiments/iteration70_benchmark.py')==H(ROOT/'results/iteration70_benchmark/iteration70_benchmark.py'))
for case in range(5):
    a=np.load(arch/f'iteration70_pgf_quad_v1_20261110/target_{case}.npz');b=np.load(ROOT/f'results/iteration70_pgf_reference/target_{case}.npz')
    for key in ('codes','probabilities','binary128','covariance','v'):check(np.array_equal(a[key],b[key]),f'oracle {case} {key}')
direct=read(arch/'iteration70_direct_click_pilot_v1_20261110/summary.json');check(len(direct['checks'])==240);check(all(r['passed'] for r in direct['checks']))
for row in direct['rows']:
    data=np.load(arch/f"iteration70_direct_click_pilot_v1_20261110/counts_{row['case']}_{row['label']}.npz");clicks=data['clicks'];check(np.all((clicks==0)|(clicks==1)));hist=np.bincount(clicks@np.array([1,2,4,8]),minlength=16);check(np.array_equal(hist,data['histogram']));check(float(.5*abs(hist/len(clicks)-data['target']).sum())==row['observed_threshold_tv'])
# The generic public benchmark reproduces all earlier output records, not times.
benchmark=read(ROOT/'results/iteration70_benchmark/summary.json')
for rep in range(3):
    for method in ('direct','pnr'):
        a=np.load(ROOT/f'results/iteration70_benchmark/records_{rep}_{method}.npz');b=np.load(arch/f'iteration70_dual_benchmark_v1_20261110/records_{rep}_{method}.npz');check(np.array_equal(a['output'],b['output']),f'benchmark {rep} {method}')
    for cutoff in (2,3):
        a=read(ROOT/f'results/iteration70_benchmark/frugal_{rep}_{cutoff}.json');b=read(arch/f'iteration70_dual_benchmark_v1_20261110/frugal_{rep}_{cutoff}.json');check(a['outputs']==b['outputs'],f'Frugal {rep} {cutoff}')
times={m:statistics.median(r['compute_seconds'] for r in benchmark['rows'] if r['method']==m) for m in ('direct','pnr','frugal_K2','frugal_K3')}
# Independently reconstruct both complete extended histograms from every count.
quality=[];target=np.load(arch/'iteration70_pgf_quad_v1_20261110/target_4.npz');cutoff=int(target['cutoff']);base=cutoff+1;codes=target['codes'];lookup=np.full(base**4,-1,np.int32);lookup[codes]=np.arange(len(codes));powers=np.array([base**j for j in range(4)],np.uint64)
for name in ('iteration70_quality_extension','iteration70_quality_extension_v2'):
    folder=ROOT/'results'/name;plan=read(folder/'plan.json');summary=read(folder/'summary.json');saved=np.load(folder/'histogram.npz');law=saved['target'];event=saved['fixed_event'];hist=np.zeros(len(law),np.int64);photons=0;maximum=0;ge128=0
    for row in summary['rows']:
        counts=np.load(folder/f"batch_{row['batch']:02d}.npz")['counts'];total=counts.sum(axis=1);inside=total<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];check(min(labels)>=0);hist+=np.bincount(labels,minlength=len(law));check(len(counts)==row['samples']);check(int(total.sum())==row['photons']);check(int(total.max())==row['maximum_photons']);check(int(sum(total>=128))==row['events_ge128']);check(int(sum(~inside))==row['tail_events']);photons+=int(total.sum());maximum=max(maximum,int(total.max()));ge128+=int(sum(total>=128))
    check(np.array_equal(hist,saved['histogram']));count=summary['samples'];check(int(hist.sum())==count);tv=float(.5*abs(hist/count-law).sum());radius=math.sqrt(math.log(2/.01)/(2*count));lower=max(0.,float((hist/count)[event].sum()-law[event].sum())-radius);upper=min(1.,tv+radius+summary['target_tail']);check(tv==summary['empirical_joint_binned_tv']);check([lower,upper]==summary['nominal_full_pnr_tv_interval']);check(photons/count==summary['mean_photons']);check(maximum==summary['maximum_photons'] and ge128==summary['events_ge128']);quality.append(dict(experiment=name,samples=count,tv=tv,interval=[lower,upper],maximum=maximum,ge128=ge128))
check(quality[-1]['interval'][1]<.01)
# New Markdown local links must resolve. Web sources stay explicit URLs.
for name in ('RESEARCH_REPORT_ITERATION70.md','research/iteration70_methods.md','research/iteration70_frontier_comparison.md','research/iteration71_plan.md'):
    for dest in re.findall(r'\]\(([^)]+)\)',(ROOT/name).read_text(encoding='utf-8')):
        if '://' not in dest and not dest.startswith('#'):check(((ROOT/name).parent/dest.split('#')[0]).exists(),name+': '+dest)
save(OUT/'validation.json',dict(passed=True,checks=checks,archive_files=mapping['files'],archive_bytes=mapping['bytes'],verified_manifests=manifests,tests_run=tests['tests_run'],tests_passed=tests['passed'],tested_code_unchanged=True,documentation_changed_after_tests=doc_changes,added_experiment_drivers_executed_after_tests=new_python,public_benchmark_outputs_equal_prototype=True,times=times,extended_quality_recomputed=quality))
scope=['.gitattributes','README.md','native/README.md','RESEARCH_REPORT_ITERATION70.md','research/iteration70_methods.md','research/iteration70_frontier_comparison.md','research/iteration71_plan.md','native/gaussian_reset_parallel.cpp','native/gaussian_direct_click.cpp','native/gaussian_pgf_quad.cpp','src/gbslab/native_gaussian_reset_parallel.py','src/gbslab/native_direct_click.py','tests/test_native_gaussian_reset_parallel.py','tests/test_native_direct_click.py','results/iteration70_archive_mapping.json']
scope.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments').glob('iteration70*.py'))
for folder in (ROOT/'results').glob('iteration70*'):
    if folder.is_dir():scope.extend(p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file())
check(len(scope)==len(set(scope)));artifacts={name:dict(sha256=H(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in sorted(scope)};check(max(r['bytes'] for r in artifacts.values())<100_000_000)
save(ROOT/'results/iteration70_manifest.json',dict(artifacts=artifacts,scope='Both native sampler routes, small-system PNR oracle, all frozen pilots/failures, public builds/replay/benchmarks, 20M and 80M raw-count quality runs, full tests, and independent verification. Manifest excludes its own bytes.'))
scope.append('results/iteration70_manifest.json');(ROOT/'.git/iteration70_stage_paths.nul').write_bytes(b''.join(n.encode()+b'\0' for n in sorted(scope)));print(json.dumps(dict(passed=True,checks=checks,files=len(scope),bytes=sum(r['bytes'] for r in artifacts.values())),indent=2),flush=True)
