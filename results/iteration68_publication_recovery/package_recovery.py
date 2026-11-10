from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iteration68_publication_recovery';OUT.mkdir(exist_ok=False)
def H(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(p,v):
 with p.open('xb') as stream:stream.write((json.dumps(v,indent=2)+'\n').encode())
shutil.copy2(__file__,OUT/'package_recovery.py');shutil.copy2(ROOT/'.git/codex_iteration68_publish_check.py',OUT/'first_publish_check.py')
first=ROOT/'results/iteration68_manifest.json';old=json.loads(first.read_text());artifacts=dict(old['artifacts']);aliases={}
for name,sha in old['artifacts'].items():
 if '/.git/' not in name:continue
 target=name.replace('/.git/','/git_source_snapshot/');dest=ROOT/target;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copy2(ROOT/name,dest);assert H(dest)==sha;aliases[name]=target;del artifacts[name];artifacts[target]=sha
save(OUT/'first_check_failure.json',dict(status='failed_before_publication',reason='Git refuses nested .git path components even when explicitly forced. Original source bytes and original manifests remain unchanged; only archive paths are mapped.',first_missing_path='results/iteration68_pilots/iteration68_cuda_setup_recovery_20261110/source_snapshot/.git/codex_iteration68_cuda_setup_recovery.py',source_sha256=H(OUT/'first_publish_check.py'),original_manifest_sha256=H(first),alias_count=len(aliases)))
artifacts[first.relative_to(ROOT).as_posix()]=H(first)
for p in OUT.iterdir():
 if p.is_file():artifacts[p.relative_to(ROOT).as_posix()]=H(p)
manifest=ROOT/'results/iteration68_manifest_recovery.json';save(manifest,dict(artifacts=artifacts,dependencies=old['dependencies'],aliases=aliases,count=len(artifacts),bytes=sum((ROOT/n).stat().st_size for n in artifacts),dependency_count=old['dependency_count'],original_manifest='results/iteration68_manifest.json'))
paths=set(artifacts)|{'results/iteration68_manifest_recovery.json'}
stage=ROOT/'.git/codex_iteration68_recovery_stage_paths';stage.write_bytes(b'\0'.join(n.encode() for n in sorted(paths))+b'\0');subprocess.run(['git','add','-f','--pathspec-from-file='+str(stage),'--pathspec-file-nul'],cwd=ROOT,check=True)
print(len(aliases),'source paths mapped; original bytes preserved',flush=True)
