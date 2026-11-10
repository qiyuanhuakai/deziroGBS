from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
ROOT=Path(__file__).resolve().parents[1];BASE='0031f7ba0bffd1d6c38d67b0d11c856bbd471314';BRANCH='research/iteration70-dual-sampling'
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):
    with p.open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def run(args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if p.returncode:raise RuntimeError(json.dumps(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
    return p.stdout.strip(),p.stderr.strip()
def api(sha):return json.loads(run(['gh','api',f'repos/qiyuanhuakai/deziroGBS/git/commits/{sha}'])[0])
assert run(['git','branch','--show-current'])[0]==BRANCH
assert run(['git','rev-parse','HEAD'])[0]==BASE
assert run(['git','ls-remote','origin','refs/heads/main'])[0].split()[0]==BASE
manifest=json.loads((ROOT/'results/iteration70_manifest.json').read_text(encoding='utf-8'))
verification=json.loads((ROOT/'results/iteration70_verification/validation.json').read_text(encoding='utf-8'));assert verification['passed']
for name,row in manifest['artifacts'].items():assert H(ROOT/name)==row['sha256'],name
paths=set(p.decode() for p in (ROOT/'.git/iteration70_stage_paths.nul').read_bytes().split(b'\0') if p)
assert paths==set(manifest['artifacts'])|{'results/iteration70_manifest.json'}
run(['git','add','-f','--pathspec-from-file=.git/iteration70_stage_paths.nul','--pathspec-file-nul'])
# Verify every staged blob against the actual publication bytes, including attributes.
index={}
for row in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0'):
    if row:
        meta,name=row.split(b'\t',1);index[name.decode()]=meta.split()[1].decode()
for name in paths:
    data=(ROOT/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==index[name],name
staged=set(run(['git','diff','--cached','--name-only'])[0].splitlines());assert staged==paths
status=run(['git','status','--porcelain'])[0];assert not any(line[:2] not in ('A ','M ') for line in status.splitlines()),status
run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
save(ROOT/'.git/codex_iteration70_stage_check.json',dict(passed=True,base=BASE,files=len(paths),manifest_sha256=H(ROOT/'results/iteration70_manifest.json'),verification_sha256=H(ROOT/'results/iteration70_verification/validation.json'),all_staged_blobs_byte_exact=True))
message=ROOT/'.git/iteration70_science_message.txt';message.write_bytes(b'Add independent native PNR and direct-click sampling routes\n\nImplement safeguarded Newton flow and deterministic CPU trajectory workers for PNR, plus a direct binary-click Gaussian moment-closure core. Retain generic-CPU same-input Frugal comparisons and all numerical failures.\n\nAdd a binary128 small-system PNR oracle and full-count 20M/80M quality experiments without truncating rare tails. Public interfaces replay 800576 records. Full suite: 828 run, 799 passed, no failures/errors; historical skips retained. No universal or equal-quality quantum-advantage claim.\n')
run(['git','commit','--file',str(message)]);science=run(['git','rev-parse','HEAD'])[0];tree=run(['git','rev-parse','HEAD^{tree}'])[0]
save(ROOT/'.git/codex_iteration70_science_local.json',dict(sha=science,tree=tree,parent=BASE));assert not run(['git','status','--porcelain'])[0];print('science committed',science,flush=True)
push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote_science=api(science);assert remote_science['sha']==science and remote_science['tree']['sha']==tree and [p['sha'] for p in remote_science['parents']]==[BASE]
save(ROOT/'.git/codex_iteration70_science_remote.json',dict(commit=remote_science,push=push));print('science remote verified',science,flush=True)
receipt=ROOT/'results/iteration70_publication_receipt';receipt.mkdir(exist_ok=False);shutil.copy2(ROOT/'.git/codex_iteration70_stage_check.json',receipt/'staged_bytes.json');shutil.copy2(Path(__file__),receipt/Path(__file__).name)
save(receipt/'receipt.json',dict(science_commit=science,science_tree=tree,parent=BASE,branch=BRANCH,remote_commit_api=remote_science,atomic_push=push,manifest_sha256=H(ROOT/'results/iteration70_manifest.json'),verification_sha256=H(ROOT/'results/iteration70_verification/validation.json'),tests_sha256=H(ROOT/'results/iteration70_tests/validation.json'),recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
files=sorted(p.relative_to(ROOT).as_posix() for p in receipt.iterdir() if p.is_file());run(['git','add','-f','--',*files]);assert set(run(['git','diff','--cached','--name-only'])[0].splitlines())==set(files);run(['git','commit','-m','Record verified iteration 70 publication'])
final=run(['git','rev-parse','HEAD'])[0];finaltree=run(['git','rev-parse','HEAD^{tree}'])[0];save(ROOT/'.git/codex_iteration70_receipt_local.json',dict(sha=final,tree=finaltree,parent=science))
final_push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote_final=api(final);assert remote_final['tree']['sha']==finaltree and [p['sha'] for p in remote_final['parents']]==[science]
refs=run(['git','ls-remote','origin','refs/heads/main',f'refs/heads/{BRANCH}'])[0].splitlines();assert len(refs)==2 and all(line.split()[0]==final for line in refs)
run(['git','update-ref','refs/heads/main',final,BASE]);run(['git','switch','main']);assert not run(['git','status','--porcelain'])[0]
result=dict(passed=True,science_commit=science,final_commit=final,tree=finaltree,remote_refs=refs,receipt_commit_api=remote_final,final_push=final_push,worktree_clean=True,report=f'https://github.com/qiyuanhuakai/deziroGBS/blob/{final}/RESEARCH_REPORT_ITERATION70.md');save(ROOT/'.git/codex_iteration70_final.json',result);print(json.dumps({k:result[k] for k in ('passed','science_commit','final_commit','tree','worktree_clean','report')}),flush=True)
