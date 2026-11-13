from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
ROOT=Path(__file__).resolve().parents[1];BASE='13f613de6bca4983c47eef4d5908951721cf15ac';BRANCH='research/iteration80-calibrated-pairs-click-order'
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
verification=json.loads((ROOT/'results/iteration80_verification_v2/validation.json').read_text());assert verification['passed'] and verification['full_suite']['passed']
paths=names(['git','diff','--cached','--name-only','-z'])|names(['git','diff','--name-only','-z'])|names(['git','ls-files','--others','--exclude-standard','-z'])
for folder in (ROOT/'results').glob('iteration80*'):
    if folder.is_dir():paths|={p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()}
assert 'RESEARCH_REPORT_ITERATION80.md' in paths
artifacts={name:dict(bytes=(ROOT/name).stat().st_size,sha256=H(ROOT/name)) for name in sorted(paths)}
assert max(r['bytes'] for r in artifacts.values())<100000000
save(ROOT/'results/iteration80_manifest_v1.json',dict(base_commit=BASE,artifacts=artifacts,files=len(paths),bytes=sum(r['bytes'] for r in artifacts.values()),verification_sha256=H(ROOT/'results/iteration80_verification_v2/validation.json'),scope='Calibrated shared photon pairs with cached integer alias plans. Fresh 5M per method complete PNR and click bounds, 130 exact rational alias tables, independent wide marginal checks. Final isolated timing v3 includes all 66 CPU/GPU Frugal panels; retain v2 but exclude its overlapped timings. Preserve finite-design infeasibility, high-order counterexamples and explicit rolling-merge failures. No universal or equal-quality superiority. Manifest excludes itself and receipt.'))
paths.add('results/iteration80_manifest_v1.json');stage=ROOT/'.git/iteration80_stage_paths.nul';stage.write_bytes(b'\0'.join(name.encode() for name in sorted(paths))+b'\0')
run(['git','add','-f','--pathspec-from-file=.git/iteration80_stage_paths.nul','--pathspec-file-nul']);run(['git','add','--renormalize','-f','--pathspec-from-file=.git/iteration80_stage_paths.nul','--pathspec-file-nul']);check_index(paths)
save(ROOT/'.git/codex_iteration80_stage_check.json',dict(passed=True,files=len(paths),all_staged_blobs_byte_exact=True,manifest_sha256=H(ROOT/'results/iteration80_manifest_v1.json')))
message=ROOT/'.git/iteration80_science_message.txt';message.write_bytes((f"Add calibrated shared photon pairs and cached sampling plans\n\nRestore fixed-total pair statistics and compare complete PNR and click laws using fresh 5M records per method. Preserve finite-design infeasibility, higher-order counterexamples and explicit rolling-merge failures.\n\nPublish 66 isolated setup-inclusive timing panels with CPU and GPU Frugal. Include integer alias audits and input-only click ordering; no universal or equal-quality claim.\n\nValidation: {verification['checks']} independent checks; {verification['full_suite']['tests_run']} tests run, {verification['full_suite']['successes']} passed, zero failures/errors; thirteen new tests.\n").encode())
run(['git','-c','maintenance.auto=false','-c','gc.auto=0','commit','--file',str(message)]);science=run(['git','rev-parse','HEAD'])[0];tree=run(['git','rev-parse','HEAD^{tree}'])[0]
save(ROOT/'.git/codex_iteration80_science_local.json',dict(sha=science,tree=tree,parent=BASE));assert not run(['git','status','--porcelain'])[0];print('science committed',science,flush=True)
push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote=api(science);assert remote['sha']==science and remote['tree']['sha']==tree and [p['sha'] for p in remote['parents']]==[BASE]
save(ROOT/'.git/codex_iteration80_science_remote.json',dict(commit=remote,push=push));print('science remote verified',science,flush=True)
receipt=ROOT/'results/iteration80_publication_receipt';receipt.mkdir(exist_ok=False);shutil.copy2(Path(__file__),receipt/Path(__file__).name);shutil.copy2(ROOT/'.git/codex_iteration80_stage_check.json',receipt/'staged_bytes.json')
save(receipt/'receipt.json',dict(science_commit=science,science_tree=tree,parent=BASE,remote_commit_api=remote,atomic_push=push,manifest_sha256=H(ROOT/'results/iteration80_manifest_v1.json'),verification_sha256=H(ROOT/'results/iteration80_verification_v2/validation.json'),tests_sha256=H(ROOT/'results/iteration80_tests_v1/validation.json'),recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
files={p.relative_to(ROOT).as_posix() for p in receipt.iterdir() if p.is_file()};run(['git','add','-f','--',*sorted(files)]);check_index(files);run(['git','-c','maintenance.auto=false','-c','gc.auto=0','commit','-m','Record verified iteration 80 publication'])
final=run(['git','rev-parse','HEAD'])[0];finaltree=run(['git','rev-parse','HEAD^{tree}'])[0];save(ROOT/'.git/codex_iteration80_receipt_local.json',dict(sha=final,tree=finaltree,parent=science))
push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote=api(final);assert remote['sha']==final and remote['tree']['sha']==finaltree and [p['sha'] for p in remote['parents']]==[science]
refs=run(['git','ls-remote','origin','refs/heads/main',f'refs/heads/{BRANCH}'])[0].splitlines();assert len(refs)==2 and all(line.split()[0]==final for line in refs)
run(['git','update-ref','refs/heads/main',final,BASE]);run(['git','switch','main']);assert not run(['git','status','--porcelain'])[0]
result=dict(passed=True,science_commit=science,final_commit=final,tree=finaltree,remote_refs=refs,receipt_commit_api=remote,final_push=push,worktree_clean=True,report=f'https://github.com/qiyuanhuakai/deziroGBS/blob/{final}/RESEARCH_REPORT_ITERATION80.md');save(ROOT/'.git/codex_iteration80_final.json',result);print(json.dumps({k:result[k] for k in ('passed','science_commit','final_commit','worktree_clean','report')}),flush=True)
