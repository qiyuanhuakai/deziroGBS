from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
ROOT=Path(__file__).resolve().parents[1];BASE='4b30056e1c9f071cb7547cf065e8a28e1da61297';BRANCH='research/iteration73-analytic-pnr-philox'
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    with p.open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def run(args):
    r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if r.returncode:raise RuntimeError(json.dumps(dict(command=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
    return r.stdout.strip(),r.stderr.strip()
def names(args):return {p.decode() for p in subprocess.check_output(args,cwd=ROOT).split(b'\0') if p}
def api(sha):return json.loads(run(['gh','api',f'repos/qiyuanhuakai/deziroGBS/git/commits/{sha}'])[0])
def check_index(paths):
    index={}
    for row in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0'):
        if row:
            meta,name=row.split(b'\t',1);index[name.decode()]=meta.split()[1].decode()
    for name in paths:
        data=(ROOT/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==index[name],name
    assert names(['git','diff','--cached','--name-only','-z'])==set(paths)
    run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
assert run(['git','branch','--show-current'])[0]==BRANCH
assert run(['git','rev-parse','HEAD'])[0]==BASE
assert run(['git','ls-remote','origin','refs/heads/main'])[0].split()[0]==BASE
verification=json.loads((ROOT/'results/iteration73_verification/validation.json').read_text());assert verification['passed'] and verification['full_suite']['passed']
paths=names(['git','diff','--name-only','-z'])|names(['git','ls-files','--others','--exclude-standard','-z'])
for folder in (ROOT/'results').glob('iteration73*'):
    if folder.is_dir():paths|={p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()}
assert 'RESEARCH_REPORT_ITERATION73.md' in paths
artifacts={name:dict(bytes=(ROOT/name).stat().st_size,sha256=H(ROOT/name)) for name in sorted(paths)}
assert max(r['bytes'] for r in artifacts.values())<100000000
save(ROOT/'results/iteration73_manifest.json',dict(base_commit=BASE,artifacts=artifacts,files=len(paths),bytes=sum(r['bytes'] for r in artifacts.values()),verification_sha256=H(ROOT/'results/iteration73_verification/validation.json'),scope='Analytic polynomial photon-subtraction PNR with 16M new independent records; device Philox direct clicks; stronger Frugal AVX2 compatibility builds; 86 timing panels; pure-state zero-probability failure and all failed builds retained. Large-system speed is not equal-quality superiority. Manifest excludes itself and subsequent publication receipt.'))
paths.add('results/iteration73_manifest.json');stage=ROOT/'.git/iteration73_stage_paths.nul';stage.write_bytes(b'\0'.join(name.encode() for name in sorted(paths))+b'\0')
run(['git','add','-f','--pathspec-from-file=.git/iteration73_stage_paths.nul','--pathspec-file-nul']);check_index(paths)
save(ROOT/'.git/codex_iteration73_stage_check.json',dict(passed=True,files=len(paths),all_staged_blobs_byte_exact=True,manifest_sha256=H(ROOT/'results/iteration73_manifest.json')))
message=ROOT/'.git/iteration73_science_message.txt';message.write_bytes(b'Advance analytic PNR and device Philox clicks against AVX2 Frugal\n\nAdd analytic photon-subtraction polynomial trajectories and 16M new PNR records. Both selected K3 full-TV nominal joint-99% upper bounds are below one percent; preserve the pure squeezed parity boundary failure.\n\nMove click RNG to CUDA Philox and repair Frugal Windows AVX2 summation ABI with an identical-body inline patch. Record all 86 timing panels, including small-system Frugal advantages and 2.052431 seconds per 4096 large four-branch samples without equal-quality claims.\n\nValidation: 2630 independent checks; 861 tests run, 832 passed, zero failures/errors; ten new public API tests. All initial failed source versions and raw results retained.\n')
run(['git','-c','maintenance.auto=false','-c','gc.auto=0','commit','--file',str(message)]);science=run(['git','rev-parse','HEAD'])[0];tree=run(['git','rev-parse','HEAD^{tree}'])[0]
save(ROOT/'.git/codex_iteration73_science_local.json',dict(sha=science,tree=tree,parent=BASE));assert not run(['git','status','--porcelain'])[0];print('science committed',science,flush=True)
push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote=api(science);assert remote['sha']==science and remote['tree']['sha']==tree and [p['sha'] for p in remote['parents']]==[BASE]
save(ROOT/'.git/codex_iteration73_science_remote.json',dict(commit=remote,push=push));print('science remote verified',science,flush=True)
receipt=ROOT/'results/iteration73_publication_receipt';receipt.mkdir(exist_ok=False);shutil.copy2(Path(__file__),receipt/Path(__file__).name);shutil.copy2(ROOT/'.git/codex_iteration73_stage_check.json',receipt/'staged_bytes.json')
save(receipt/'receipt.json',dict(science_commit=science,science_tree=tree,parent=BASE,remote_commit_api=remote,atomic_push=push,manifest_sha256=H(ROOT/'results/iteration73_manifest.json'),verification_sha256=H(ROOT/'results/iteration73_verification/validation.json'),tests_sha256=H(ROOT/'results/iteration73_tests_v1/validation.json'),recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
files={p.relative_to(ROOT).as_posix() for p in receipt.iterdir() if p.is_file()};run(['git','add','-f','--',*sorted(files)]);check_index(files);run(['git','-c','maintenance.auto=false','-c','gc.auto=0','commit','-m','Record verified iteration 73 publication'])
final=run(['git','rev-parse','HEAD'])[0];finaltree=run(['git','rev-parse','HEAD^{tree}'])[0];save(ROOT/'.git/codex_iteration73_receipt_local.json',dict(sha=final,tree=finaltree,parent=science))
push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote=api(final);assert remote['sha']==final and remote['tree']['sha']==finaltree and [p['sha'] for p in remote['parents']]==[science]
refs=run(['git','ls-remote','origin','refs/heads/main',f'refs/heads/{BRANCH}'])[0].splitlines();assert len(refs)==2 and all(line.split()[0]==final for line in refs)
run(['git','update-ref','refs/heads/main',final,BASE]);run(['git','switch','main']);assert not run(['git','status','--porcelain'])[0]
result=dict(passed=True,science_commit=science,final_commit=final,tree=finaltree,remote_refs=refs,receipt_commit_api=remote,final_push=push,worktree_clean=True,report=f'https://github.com/qiyuanhuakai/deziroGBS/blob/{final}/RESEARCH_REPORT_ITERATION73.md');save(ROOT/'.git/codex_iteration73_final.json',result);print(json.dumps({k:result[k] for k in ('passed','science_commit','final_commit','worktree_clean','report')}),flush=True)
