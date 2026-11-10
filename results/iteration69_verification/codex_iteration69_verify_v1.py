from pathlib import Path
import hashlib,json,shutil,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iteration69_verification';OUT.mkdir(exist_ok=False)
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):
    with p.open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=0
def check(ok):
    global checks
    assert ok;checks+=1
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
mapping=read(ROOT/'results/iteration69_archive_mapping.json')
for record in mapping['artifacts']:
    check(H(ROOT/record['archive'])==record['sha256']);check(H(ROOT/record['original'])==record['sha256'])
archive=ROOT/'results/iteration69_archive';historical_manifests=0
for folder in archive.iterdir():
    if not folder.is_dir():continue
    for name in ('manifest.json','failure_manifest.json'):
        p=folder/name
        if not p.exists():continue
        historical_manifests+=1
        for relative,value in read(p).get('artifacts',{}).items():
            check(isinstance(value,str));portable=Path(*('git_source_snapshot' if part=='.git' else part for part in Path(relative).parts));check(H(folder/portable)==value)
assembly=read(archive/'iteration69_public_assembly_v2.json');check(H(ROOT/'native/gaussian_reset_cpu.cpp')==assembly['output_sha256'])
for name,value in assembly['inputs'].items():check(H(archive/name)==value)
tests=read(ROOT/'results/iteration69_tests/validation.json');check(tests['executed_gate_passed'] and tests['unchanged_sources']);check(tests['failures']==tests['errors']==0);check(tests['new_native_tests_passed']==10 and tests['legacy_native_tests_passed']==18)
source_hashes=read(ROOT/'results/iteration69_tests/source_hashes_at_start.json')
current_sources=sorted({p for base in ('experiments','src','tests') for p in (ROOT/base).rglob('*.py')}|set((ROOT/'native').glob('*')))
check(set(source_hashes)=={p.relative_to(ROOT).as_posix() for p in current_sources if p.is_file()})
for name,value in source_hashes.items():check(H(ROOT/name)==value)
for name in ('iteration69_build','iteration69_public_validation','iteration69_benchmark'):
    folder=ROOT/'results'/name;check(read(folder/'summary.json')['passed'])
    for path,value in read(folder/'manifest.json')['artifacts'].items():check(H(folder/path)==value)
benchmark=read(ROOT/'results/iteration69_benchmark/summary.json');replayed_rows=[]
for row in benchmark['rows']:
    name=f"case_{row['case']}_rep_{row['repetition']}_method_{row['method']}.npz";actual=np.load(ROOT/'results/iteration69_benchmark'/name);expected=np.load(archive/'iteration69_displaced_benchmark_v2_20261110'/name);check(np.array_equal(actual['counts'],expected['counts']));check(np.array_equal(actual['stats'][[0,1,5,6]],expected['stats'][[0,1,5,6]]));replayed_rows.append(name)
for folder in ('iteration69_reset_precision_v3_20261110','iteration69_reset_diagonal_v4_20261110','iteration69_reset_blas_v5_20261110','iteration69_displaced_v6_20261110','iteration69_anchor_v8_20261110','iteration69_anchor_v9_20261110','iteration69_displaced_benchmark_v1_20261110','iteration69_displaced_benchmark_v2_20261110','iteration69_envelope_exact_v1_20261110'):
    check(read(archive/folder/'summary.json')['passed'])
quality=[]
for case in range(4):
    target=np.load(archive/'iteration69_quality_pilot_20261110'/f'input_{case}.npz')['target'];check(abs(target.sum()-1)<1e-14 and min(target)>=0)
    for folder in ('iteration69_quality_pilot_20261110','iteration69_displaced_v6_20261110'):
        counts=np.load(archive/folder/f'case_{case}_method_3.npz')['counts'];mask=np.sum((counts>0)*(1<<np.arange(4)),axis=1);hist=np.bincount(mask,minlength=16);tv=float(.5*np.sum(abs(hist/len(counts)-target)));stored=read(archive/folder/f'case_{case}_method_3.json');check(tv==stored['empirical_threshold_tv']);quality.append(dict(case=case,folder=folder,tv=tv))
save(OUT/'validation.json',dict(passed=True,checks=checks,archive_files=len(mapping['artifacts']),archive_bytes=mapping['total_bytes'],historical_manifests=historical_manifests,current_source_files=len(source_hashes),public_benchmark_rows_equal_private=len(replayed_rows),quality_recomputed=quality,tests_run=tests['tests_run'],tests_passed=tests['passed']))
scope=['.gitattributes','README.md','RESEARCH_REPORT_ITERATION69.md','native/README.md','native/gaussian_reset_cpu.cpp','src/gbslab/native_gaussian_reset.py','tests/test_native_gaussian_reset.py','experiments/iteration69_build.py','experiments/iteration69_tests.py','experiments/iteration69_benchmark.py','experiments/iteration69_validate_public.py','research/iteration69_methods.md','research/iteration69_frontier_comparison.md','research/iteration70_plan.md','results/iteration69_archive_mapping.json']
for name in ('iteration69_archive','iteration69_build','iteration69_tests','iteration69_benchmark','iteration69_public_validation','iteration69_verification'):
    scope.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'results'/name).rglob('*') if p.is_file())
check(len(scope)==len(set(scope)))
artifacts={name:dict(sha256=H(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in sorted(scope)}
dependencies={name:H(ROOT/name) for name in ['results/iteration38_frugal_runtime/frugal_adapter.exe','results/iteration38_frugal_runtime/build.json','results/iteration31_build/libwire_forest.dll','results/iteration68_build/native_cpu.dll','results/iteration68_build/native_cuda.dll']}
save(ROOT/'results/iteration69_manifest.json',dict(artifacts=artifacts,historical_dependencies=dependencies,scope='Publication bytes, including all frozen pilots, numerical failures, public source, final benchmark, full tests and independent verification. Manifest excludes its own bytes.'))
scope.append('results/iteration69_manifest.json');(ROOT/'.git/iteration69_stage_paths.nul').write_bytes(b''.join(name.encode('utf-8')+b'\0' for name in sorted(scope)))
print(json.dumps(dict(passed=True,checks=checks,publication_files=len(scope),total_bytes=sum(r['bytes'] for r in artifacts.values()),source_files=len(source_hashes))))
