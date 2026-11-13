from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
ROOT=Path(__file__).resolve().parents[1];BASE='e8ea50570cbdd5b968f92518dcf566f480087c06';BRANCH='research/iteration88-window-click-pnr-depth'
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
verification=json.loads((ROOT/'results/iteration88_verification_v1/validation.json').read_text());assert verification['passed'] and verification['full_suite']['passed']
paths=names(['git','diff','--cached','--name-only','-z'])|names(['git','diff','--name-only','-z'])|names(['git','ls-files','--others','--exclude-standard','-z'])
changed_paths=set(paths)
for folder in (ROOT/'results').glob('iteration88*'):
    if folder.is_dir():paths|={p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()}
assert 'RESEARCH_REPORT_ITERATION88.md' in paths
artifacts={name:dict(bytes=(ROOT/name).stat().st_size,sha256=H(ROOT/name)) for name in sorted(paths)}
assert max(r['bytes'] for r in artifacts.values())<100000000
save(ROOT/'results/iteration88_manifest_v1.json',dict(base_commit=BASE,artifacts=artifacts,files=len(paths),bytes=sum(r['bytes'] for r in artifacts.values()),verification_sha256=H(ROOT/'results/iteration88_verification_v1/validation.json'),scope='Complete two-mode PNR quality for12 preregistered65536-record tasks, with analytic tails and all207 failure outcomes. Four new full1152 direct-click jobs under one47-event nominal99 family reveal no certified projection advantage and one nonphysical projection failure, diagnosed by independent100-digit sums. The separate small-PNR24-event99 family gives at least98 percent across both. Serial timings, exact replay tests, Frugal scope and Jiuzhang throughput remain distinct. No arbitrary-input superiority or whole1152-pattern TV upper bound.'))
paths=changed_paths|{'results/iteration88_manifest_v1.json'};stage=ROOT/'.git/iteration88_stage_paths.nul';stage.write_bytes(b'\0'.join(name.encode() for name in sorted(paths))+b'\0')
run(['git','add','-f','--pathspec-from-file=.git/iteration88_stage_paths.nul','--pathspec-file-nul']);run(['git','add','--renormalize','-f','--pathspec-from-file=.git/iteration88_stage_paths.nul','--pathspec-file-nul']);check_index(paths)
save(ROOT/'.git/codex_iteration88_stage_check.json',dict(passed=True,files=len(paths),all_staged_blobs_byte_exact=True,manifest_sha256=H(ROOT/'results/iteration88_manifest_v1.json')))
message=ROOT/'.git/iteration88_science_message.txt';message.write_bytes((f"Measure full PNR accuracy and diagnose nonphysical click moment projection\n\nGenerate12 preregistered65536-record two-port PNR tasks and four full1152 direct-click quality jobs. Preserve every failure, the rejected oldest-factor closure, and the first compile attempt. Match full conditional second moments while retaining one click, then diagnose its negative probability by independent100-digit sums. Report scoped confidence intervals, serial performance and hardware gaps without claiming general superiority.\n\nValidation: {verification['checks']} independent final checks; {verification['full_suite']['tests_run']} tests run, {verification['full_suite']['successes']} successful, zero failures/errors; five new native tests. Separate99-percent families combine to at least98 percent.\n").encode())
run(['git','-c','maintenance.auto=false','-c','gc.auto=0','commit','--file',str(message)]);science=run(['git','rev-parse','HEAD'])[0];tree=run(['git','rev-parse','HEAD^{tree}'])[0]
save(ROOT/'.git/codex_iteration88_science_local.json',dict(sha=science,tree=tree,parent=BASE));assert not run(['git','status','--porcelain'])[0];print('science committed',science,flush=True)
push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote=api(science);assert remote['sha']==science and remote['tree']['sha']==tree and [p['sha'] for p in remote['parents']]==[BASE]
save(ROOT/'.git/codex_iteration88_science_remote.json',dict(commit=remote,push=push));print('science remote verified',science,flush=True)
receipt=ROOT/'results/iteration88_publication_receipt';receipt.mkdir(exist_ok=False);shutil.copy2(Path(__file__),receipt/Path(__file__).name);shutil.copy2(ROOT/'.git/codex_iteration88_stage_check.json',receipt/'staged_bytes.json')
save(receipt/'receipt.json',dict(science_commit=science,science_tree=tree,parent=BASE,remote_commit_api=remote,atomic_push=push,manifest_sha256=H(ROOT/'results/iteration88_manifest_v1.json'),verification_sha256=H(ROOT/'results/iteration88_verification_v1/validation.json'),tests_sha256=H(ROOT/'results/iteration88_tests_v1/validation.json'),recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
files={p.relative_to(ROOT).as_posix() for p in receipt.iterdir() if p.is_file()};run(['git','add','-f','--',*sorted(files)]);check_index(files);run(['git','-c','maintenance.auto=false','-c','gc.auto=0','commit','-m','Record verified iteration 88 publication'])
final=run(['git','rev-parse','HEAD'])[0];finaltree=run(['git','rev-parse','HEAD^{tree}'])[0];save(ROOT/'.git/codex_iteration88_receipt_local.json',dict(sha=final,tree=finaltree,parent=science))
push=run(['git','push','--atomic','origin','HEAD:refs/heads/main',f'HEAD:refs/heads/{BRANCH}']);remote=api(final);assert remote['sha']==final and remote['tree']['sha']==finaltree and [p['sha'] for p in remote['parents']]==[science]
refs=run(['git','ls-remote','origin','refs/heads/main',f'refs/heads/{BRANCH}'])[0].splitlines();assert len(refs)==2 and all(line.split()[0]==final for line in refs)
run(['git','update-ref','refs/heads/main',final,BASE]);run(['git','switch','main']);assert not run(['git','status','--porcelain'])[0]
result=dict(passed=True,science_commit=science,final_commit=final,tree=finaltree,remote_refs=refs,receipt_commit_api=remote,final_push=push,worktree_clean=True,report=f'https://github.com/qiyuanhuakai/deziroGBS/blob/{final}/RESEARCH_REPORT_ITERATION88.md');save(ROOT/'.git/codex_iteration88_final.json',result);print(json.dumps({k:result[k] for k in ('passed','science_commit','final_commit','worktree_clean','report')}),flush=True)
