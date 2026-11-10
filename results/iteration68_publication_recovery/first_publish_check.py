from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.git/codex_iteration68_prepublication.json';assert not OUT.exists()
BASE='e263bff2ce056f88334c85edde7a6a3740bbdfc3'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def H(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert git('rev-parse','HEAD').decode().strip()==BASE
assert git('diff','--name-only')==b''
assert git('ls-files','--others','--exclude-standard')==b''
index={}
for record in git('ls-files','--stage','-z').split(b'\0'):
 if record:
  header,path=record.split(b'\t',1);mode,oid,stage=header.split();assert stage==b'0';index[path.decode()]=oid
process=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(name):
 process.stdin.write(index[name]+b'\n');process.stdin.flush();header=process.stdout.readline();fields=header.split();assert len(fields)==3 and fields[1]==b'blob',(name,header);size=int(fields[2]);data=process.stdout.read(size);assert len(data)==size;assert process.stdout.read(1)==b'\n';return data
manifest=json.loads((ROOT/'results/iteration68_manifest.json').read_text());checked=0
for name,sha in manifest['artifacts'].items():
 data=blob(name);assert hashlib.sha256(data).hexdigest()==sha,name;assert H(ROOT/name)==sha,name;checked+=1
packed={}
for meta in ROOT.glob('research_archive/publication/iteration*_large_artifacts/manifest.json'):
 key=meta.relative_to(ROOT).as_posix()
 if key not in index:continue
 records=json.loads(blob(key))['artifacts']
 for name in manifest['dependencies']:
  if name in records:packed[name]=records[name]
for name,sha in manifest['dependencies'].items():
 assert H(ROOT/name)==sha,name
 if name in index:assert hashlib.sha256(blob(name)).hexdigest()==sha,name
 else:
  whole=hashlib.sha256();size=0
  for part in packed[name]['parts']:
   data=blob(part['path']);assert len(data)==part['bytes'] and hashlib.sha256(data).hexdigest()==part['sha256'];whole.update(data);size+=len(data)
  assert whole.hexdigest()==sha and size==packed[name]['bytes'],name
archive=json.loads(blob('results/iteration68_pilots/archive_manifest.json'))
for record in archive['artifacts'].values():
 assert H(ROOT/record['original_path'])==record['sha256']
 if 'path' in record:assert hashlib.sha256(blob(record['path'])).hexdigest()==record['sha256']
 else:
  whole=hashlib.sha256();size=0
  for part in record['parts']:
   data=blob(part['path']);assert hashlib.sha256(data).hexdigest()==part['sha256'] and len(data)==part['bytes'];whole.update(data);size+=len(data)
  assert whole.hexdigest()==record['sha256'] and size==record['bytes']
tests=json.loads(blob('results/iteration68_tests_with_legacy/validation.json'));assert (tests['tests_run'],tests['passed'],tests['failures'],tests['errors'],tests['new_native_tests_passed'])==(802,773,0,0,18)
sources=json.loads(blob('results/iteration68_tests_with_legacy/source_hashes_at_start.json'))
for name,sha in sources.items():assert H(ROOT/name)==sha,name
added=git('diff','--cached','--name-only','--diff-filter=A','-z').decode().split('\0')
new_code=[n for n in added if n.endswith(('.py','.cpp','.cu')) and n.startswith(('src/','tests/','experiments/','native/'))]
assert all(n in sources for n in new_code),new_code
for name in (ROOT/'.git/codex_iteration68_stage_paths').read_bytes().decode().split('\0'):
 if not name:continue
 data=blob(name);expected=(ROOT/name).read_bytes()
 if name in ('.gitattributes','README.md','RESEARCH_REPORT_ITERATION68.md','research/iteration68_frontier_comparison.md','research/iteration69_plan.md'):expected=expected.replace(b'\r\n',b'\n')
 assert data==expected,name
process.stdin.close();assert process.wait()==0
payload=dict(passed=True,base_commit=BASE,staged_tree=git('write-tree').decode().strip(),artifact_count=checked,historical_dependencies_verified=len(manifest['dependencies']),pilot_originals_verified=archive['count'],scientific_sources_verified=len(sources),new_scientific_sources=new_code,tests=tests,manifest_sha256=H(ROOT/'results/iteration68_manifest.json'))
OUT.write_bytes((json.dumps(payload,indent=2)+'\n').encode());print({k:v for k,v in payload.items() if not isinstance(v,(dict,list))},flush=True)
