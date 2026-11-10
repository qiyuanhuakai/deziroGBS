from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iteration68_pilots';OUT.mkdir(exist_ok=False)
def H(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
records={};files=[]
for folder in sorted((ROOT/'.git').glob('iteration68_*')):
 if folder.is_dir():files.extend(p for p in sorted(folder.rglob('*')) if p.is_file())
for pattern in ('codex_iteration68*.py','codex_iteration68_pilots.json','iteration68*.cpp','iteration68*.cu','iteration68_native_adapter*.py'):
 files.extend((ROOT/'.git').glob(pattern))
for p in sorted(set(files)):
 name=p.relative_to(ROOT/'.git').as_posix();record=dict(original_path=p.relative_to(ROOT).as_posix(),sha256=H(p),bytes=p.stat().st_size)
 if p.stat().st_size<=64*1024*1024:
  dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert H(dest)==record['sha256'];record['path']=dest.relative_to(ROOT).as_posix()
 else:
  parts=[];whole=hashlib.sha256();folder=OUT/(name+'.parts');folder.mkdir(parents=True)
  with p.open('rb') as stream:
   i=0
   while True:
    data=stream.read(64*1024*1024)
    if not data:break
    target=folder/f'{i:04d}.bin';target.write_bytes(data);whole.update(data);parts.append(dict(path=target.relative_to(ROOT).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));i+=1
  assert whole.hexdigest()==record['sha256'];record['parts']=parts
 records[name]=record
(OUT/'archive_manifest.json').write_bytes((json.dumps(dict(artifacts=records,count=len(records),original_bytes=sum(r['bytes'] for r in records.values()),scope='All frozen private iteration68 pilot sources and outputs, including failed setup/parser attempts. Large derived input is split without changing bytes.'),indent=2)+'\n').encode())
print(len(records),'pilot artifacts archived',flush=True)
