"""Losslessly split the retained oversized, incomplete v1 input for publication.
Original remains untouched locally. Restore with --restore OUTPUT_FILE; never
overwrite an existing destination. Publication uses byte-identical chunk stream.
"""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--restore',type=Path);a=p.parse_args();folder=ROOT/'results/iteration81_large_input_archive_v1'
def H(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
if a.restore:
 manifest=json.loads((folder/'manifest.json').read_text());h=hashlib.sha256();size=0
 with a.restore.open('xb') as f:
  for row in manifest['chunks']:
   b=(folder/row['name']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256'];f.write(b);h.update(b);size+=len(b)
 assert h.hexdigest()==manifest['original_sha256'] and size==manifest['original_bytes']
else:
 folder.mkdir(exist_ok=False);original=ROOT/'results/iteration81_factor_pilot_v1/input_large128.npz';rows=[];h=hashlib.sha256();size=0
 with original.open('rb') as f:
  for index,b in enumerate(iter(lambda:f.read(16<<20),b'')):
   name=f'part_{index:03d}.bin';(folder/name).write_bytes(b);h.update(b);size+=len(b);rows.append(dict(name=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 assert h.hexdigest()==H(original) and size==original.stat().st_size
 (folder/'manifest.json').write_bytes((json.dumps(dict(original=original.relative_to(ROOT).as_posix(),original_sha256=h.hexdigest(),original_bytes=size,chunks=rows,artifacts={r['name']:r['sha256'] for r in rows},source_sha256=H(Path(__file__)),scope=__doc__),indent=2)+'\n').encode())
print('Verified byte-exact chunk archive',flush=True)
