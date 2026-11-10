from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
ROOT=Path(__file__).resolve().parents[1]
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):
    with p.open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def run(args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if p.returncode:raise RuntimeError(json.dumps(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
    return p.stdout.strip(),p.stderr.strip()
def api(sha):return json.loads(run(['gh','api',f'repos/qiyuanhuakai/deziroGBS/git/commits/{sha}'])[0])
stage=json.loads((ROOT/'.git/codex_iteration69_stage_check.json').read_text(encoding='utf-8'));assert stage['passed'];base=stage['base'];branch='research/iteration69-gaussian-reset'
assert run(['git','branch','--show-current'])[0]==branch;assert run(['git','rev-parse','HEAD'])[0]==base
remote=run(['git','ls-remote','origin','refs/heads/main'])[0].split()[0];assert remote==base
run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
message=ROOT/'.git/iteration69_science_message.txt';message.write_bytes(b'Add native Gaussian reset sampling and detector proposals\n\nImplement centered and displaced PNR resets in C++, with diagonal and coherent-reference rejection. Record same-input Frugal quality comparisons, count-tail failures, and repeated OpenBLAS baselines up to 8176 outputs.\n\nPreserve all exploratory failures and verify the public build against 400000 archived samples. Full suite: 812 run, 783 passed, zero failures/errors; historical skips retained. No universal or quantum-advantage claim.\n')
commit_output=run(['git','commit','--file',str(message)]);science=run(['git','rev-parse','HEAD'])[0];tree=run(['git','rev-parse','HEAD^{tree}'])[0]
save(ROOT/'.git/codex_iteration69_science_local.json',dict(sha=science,tree=tree,base=base,commit_output=commit_output));print('science committed',science,flush=True)
assert not run(['git','status','--porcelain'])[0]
push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{branch}']);remote_science=api(science);assert remote_science['sha']==science and remote_science['tree']['sha']==tree and [p['sha'] for p in remote_science['parents']]==[base]
save(ROOT/'.git/codex_iteration69_science_remote.json',dict(commit=remote_science,push=push));print('science remote verified',science,flush=True)
receipt=ROOT/'results/iteration69_publication_receipt';receipt.mkdir(exist_ok=False);shutil.copy2(ROOT/'.git/codex_iteration69_stage_check.json',receipt/'staged_bytes.json');shutil.copy2(Path(__file__),receipt/Path(__file__).name)
save(receipt/'receipt.json',dict(science_commit=science,science_tree=tree,parent=base,branch=branch,remote_commit_api=remote_science,atomic_push=push,manifest_sha256=H(ROOT/'results/iteration69_manifest.json'),verification_sha256=H(ROOT/'results/iteration69_verification/validation.json'),tests_sha256=H(ROOT/'results/iteration69_tests/validation.json'),recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
files=sorted(p.relative_to(ROOT).as_posix() for p in receipt.iterdir() if p.is_file());run(['git','add','-f','--',*files]);assert set(run(['git','diff','--cached','--name-only'])[0].splitlines())==set(files)
run(['git','commit','-m','Record verified iteration 69 publication']);final=run(['git','rev-parse','HEAD'])[0];finaltree=run(['git','rev-parse','HEAD^{tree}'])[0];save(ROOT/'.git/codex_iteration69_receipt_local.json',dict(sha=final,tree=finaltree,parent=science))
final_push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{branch}']);remote_final=api(final);assert remote_final['tree']['sha']==finaltree and [p['sha'] for p in remote_final['parents']]==[science]
refs=run(['git','ls-remote','origin','refs/heads/main',f'refs/heads/{branch}'])[0].splitlines();assert len(refs)==2 and all(line.split()[0]==final for line in refs)
run(['git','update-ref','refs/heads/main',final,base]);run(['git','switch','main']);assert not run(['git','status','--porcelain'])[0]
result=dict(passed=True,science_commit=science,final_commit=final,tree=finaltree,remote_refs=refs,receipt_commit_api=remote_final,final_push=final_push,worktree_clean=True,report=f'https://github.com/qiyuanhuakai/deziroGBS/blob/{final}/RESEARCH_REPORT_ITERATION69.md')
save(ROOT/'.git/codex_iteration69_final.json',result);print(json.dumps({k:result[k] for k in ('passed','science_commit','final_commit','tree','worktree_clean','report')}),flush=True)
