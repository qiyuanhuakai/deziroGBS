from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
ROOT=Path(__file__).resolve().parents[1];FIRST='77fef897263782605eb6766ec120f7d5163dd7a0';BASE='47774d5af47159e1426e63979571bea0254e1233';BRANCH='research/iteration69-gaussian-reset'
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):
    with p.open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def run(args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if p.returncode:raise RuntimeError(json.dumps(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
    return p.stdout.strip(),p.stderr.strip()
def api(sha):return json.loads(run(['gh','api',f'repos/qiyuanhuakai/deziroGBS/git/commits/{sha}'])[0])
assert run(['git','rev-parse','HEAD'])[0]==FIRST;assert run(['git','branch','--show-current'])[0]==BRANCH
assert run(['git','ls-remote','origin','refs/heads/main'])[0].split()[0]==BASE
run(['git','add','-f','--pathspec-from-file=.git/iteration69_recovery_stage_paths.nul','--pathspec-file-nul'])
oid=run(['git','hash-object','-w','--no-filters','.gitattributes'])[0];run(['git','update-index','--cacheinfo',f'100644,{oid},.gitattributes'])
paths=set(p.decode() for p in (ROOT/'.git/iteration69_recovery_stage_paths.nul').read_bytes().split(b'\0') if p);index={}
for row in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0'):
    if row:
        meta,name=row.split(b'\t',1);index[name.decode()]=meta.split()[1].decode()
for name in paths:
    data=(ROOT/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==index[name],name
manifest=json.loads((ROOT/'results/iteration69_manifest_recovery.json').read_text(encoding='utf-8'))
for name,row in manifest['artifacts'].items():assert H(ROOT/name)==row['sha256'],name
status=run(['git','status','--porcelain'])[0];assert not any(line.startswith('??') or line[:2] not in ('A ','M ') for line in status.splitlines()),status
save(ROOT/'.git/codex_iteration69_recovery_stage_check.json',dict(passed=True,files=len(paths),manifest_sha256=H(ROOT/'results/iteration69_manifest_recovery.json')))
run(['git','commit','-m','Recover replay summary paths and verify final iteration 69 package']);science=run(['git','rev-parse','HEAD'])[0];tree=run(['git','rev-parse','HEAD^{tree}'])[0];save(ROOT/'.git/codex_iteration69_recovered_science_local.json',dict(sha=science,tree=tree,parent=FIRST));assert not run(['git','status','--porcelain'])[0];print('recovery committed',science,flush=True)
push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote_science=api(science);remote_first=api(FIRST);assert remote_science['tree']['sha']==tree and [p['sha'] for p in remote_science['parents']]==[FIRST];assert [p['sha'] for p in remote_first['parents']]==[BASE]
save(ROOT/'.git/codex_iteration69_recovered_science_remote.json',dict(commit=remote_science,first_commit=remote_first,push=push));print('science remote verified',science,flush=True)
receipt=ROOT/'results/iteration69_publication_receipt';receipt.mkdir(exist_ok=False)
shutil.copy2(ROOT/'.git/codex_iteration69_recovery_stage_check.json',receipt/'staged_bytes.json');shutil.copy2(Path(__file__),receipt/Path(__file__).name)
save(receipt/'receipt.json',dict(science_commit=science,science_tree=tree,parent=FIRST,first_science_commit=FIRST,previous_main=BASE,branch=BRANCH,remote_commit_api=remote_science,remote_first_commit_api=remote_first,atomic_push=push,manifest_sha256=H(ROOT/'results/iteration69_manifest_recovery.json'),tests_sha256=H(ROOT/'results/iteration69_tests_recovery/validation.json'),recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
files=sorted(p.relative_to(ROOT).as_posix() for p in receipt.iterdir() if p.is_file());run(['git','add','-f','--',*files]);assert set(run(['git','diff','--cached','--name-only'])[0].splitlines())==set(files);run(['git','commit','-m','Record verified iteration 69 publication'])
final=run(['git','rev-parse','HEAD'])[0];finaltree=run(['git','rev-parse','HEAD^{tree}'])[0];save(ROOT/'.git/codex_iteration69_receipt_local.json',dict(sha=final,tree=finaltree,parent=science))
final_push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote_final=api(final);assert remote_final['tree']['sha']==finaltree and [p['sha'] for p in remote_final['parents']]==[science]
refs=run(['git','ls-remote','origin','refs/heads/main',f'refs/heads/{BRANCH}'])[0].splitlines();assert len(refs)==2 and all(line.split()[0]==final for line in refs)
run(['git','update-ref','refs/heads/main',final,BASE]);run(['git','switch','main']);assert not run(['git','status','--porcelain'])[0]
result=dict(passed=True,first_science_commit=FIRST,science_commit=science,final_commit=final,tree=finaltree,remote_refs=refs,receipt_commit_api=remote_final,final_push=final_push,worktree_clean=True,report=f'https://github.com/qiyuanhuakai/deziroGBS/blob/{final}/RESEARCH_REPORT_ITERATION69.md');save(ROOT/'.git/codex_iteration69_final.json',result);print(json.dumps({k:result[k] for k in ('passed','science_commit','final_commit','tree','worktree_clean','report')}),flush=True)
