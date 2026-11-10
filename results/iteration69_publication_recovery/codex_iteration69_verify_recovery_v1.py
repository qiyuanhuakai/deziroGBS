from pathlib import Path
import hashlib,json,subprocess,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iteration69_publication_recovery';OUT.mkdir(exist_ok=False);FIRST='77fef897263782605eb6766ec120f7d5163dd7a0'
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):
    with p.open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=0
def check(ok):
    global checks
    assert ok;checks+=1
for p in [Path(__file__),ROOT/'.git/codex_iteration69_publish_v1.py',ROOT/'.git/codex_iteration69_science_local.json',ROOT/'.git/codex_iteration69_stage_check.json']:shutil.copy2(p,OUT/p.name)
old=read(ROOT/'results/iteration69_manifest.json');changed=[]
for name,row in old['artifacts'].items():
    if H(ROOT/name)!=row['sha256']:
        check(name in ('native/README.md','RESEARCH_REPORT_ITERATION69.md'));blob=subprocess.check_output(['git','show',FIRST+':'+name],cwd=ROOT);check(hashlib.sha256(blob).hexdigest()==row['sha256']);changed.append(name)
    else:check(True)
for record in read(ROOT/'results/iteration69_archive_mapping.json')['artifacts']:check(H(ROOT/record['archive'])==record['sha256'])
tests=read(ROOT/'results/iteration69_tests_recovery/validation.json');check(tests['executed_gate_passed'] and tests['unchanged_sources']);check((tests['tests_run'],tests['passed'],tests['failures'],tests['errors'])==(812,783,0,0));check(tests['new_native_tests_passed']==10)
hashes=read(ROOT/'results/iteration69_tests_recovery/source_hashes_at_start.json');sources={p.relative_to(ROOT).as_posix() for base in ('experiments','src','tests') for p in (ROOT/base).rglob('*.py')};sources|={p.relative_to(ROOT).as_posix() for p in (ROOT/'native').glob('*') if p.is_file()};check(sources==set(hashes))
for name,value in hashes.items():check(H(ROOT/name)==value)
validation=ROOT/'results/iteration69_public_validation_v2';check(read(validation/'summary.json')['passed'])
for name,value in read(validation/'manifest.json')['artifacts'].items():check(H(validation/name)==value)
for case in range(4):
    for selector in ('anchor','diagonal'):
        name=f'case_{case}_{selector}';check(read(validation/(name+'.json'))['passed']);check(not (ROOT/(name+'.json')).exists());check(np.array_equal(np.load(validation/(name+'.npz'))['counts'],np.load(ROOT/'results/iteration69_public_validation'/(name+'.npz'))['counts']))
for row in read(ROOT/'results/iteration69_validation_recovery/recovery.json')['records']:check(H(ROOT/row['recovered_path'])==row['sha256'])
save(OUT/'validation.json',dict(passed=True,checks=checks,first_local_science_commit=FIRST,first_commit_not_pushed_before_recovery=True,superseded_document_paths=changed,current_source_files=len(hashes),tests_run=812,tests_passed=783,public_replay_samples=400000,scope='Computational core and measured data unchanged. V2 fixes per-case JSON destination, retains first output bytes, replays from portable archive and repeats all tests.'))
scope=set(old['artifacts']);scope.add('results/iteration69_manifest.json');scope.add('experiments/iteration69_validate_public_v2.py')
for name in ('iteration69_validation_recovery','iteration69_public_validation_v2','iteration69_tests_recovery','iteration69_publication_recovery'):
    scope.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'results'/name).rglob('*') if p.is_file())
artifacts={name:dict(sha256=H(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in sorted(scope)}
save(ROOT/'results/iteration69_manifest_recovery.json',dict(artifacts=artifacts,original_manifest_sha256=H(ROOT/'results/iteration69_manifest.json'),original_science_commit=FIRST,superseded_document_paths=changed,historical_dependencies=old['historical_dependencies'],scope='Final publication bytes after output-directory recovery. Original scientific artifacts, first manifest and local first commit retained. This manifest excludes its own bytes.'))
scope.add('results/iteration69_manifest_recovery.json');(ROOT/'.git/iteration69_recovery_stage_paths.nul').write_bytes(b''.join(name.encode()+b'\0' for name in sorted(scope)))
print(json.dumps(dict(passed=True,checks=checks,files=len(scope),bytes=sum(r['bytes'] for r in artifacts.values()),source_files=len(hashes))))
