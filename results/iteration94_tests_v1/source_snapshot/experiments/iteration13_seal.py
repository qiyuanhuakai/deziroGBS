"""Write a frozen-artifact manifest, or verify its exact staged Git blob bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'results/iteration13_manifest.json'


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['create','verify-staged']); args=p.parse_args()
    if args.action=='create':
        if MANIFEST.exists(): raise FileExistsError(MANIFEST)
        files=sorted(f for directory in ROOT.glob('results/iteration13_*') if directory.is_dir() for f in directory.rglob('*') if f.is_file())
        data={'artifacts':{f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'count':len(files),'bytes':sum(f.stat().st_size for f in files)}
        MANIFEST.write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps({'count':data['count'],'bytes':data['bytes']})); return
    manifest=json.loads(MANIFEST.read_text())
    index={}
    for record in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0'):
        if not record: continue
        header,path=record.split(b'\t',1); mode,oid,stage=header.split()
        if stage!=b'0': raise RuntimeError('Unmerged index')
        index[path.decode('utf-8')]=oid
    entries=list(manifest['artifacts'].items()); checked=0
    for start in range(0,len(entries),32):
        batch=entries[start:start+32]
        data=subprocess.check_output(['git','cat-file','--batch'],input=b'\n'.join(index[path] for path,_ in batch)+b'\n',cwd=ROOT)
        pos=0
        for path,digest in batch:
            end=data.index(b'\n',pos); header=data[pos:end].split(); size=int(header[2]); pos=end+1
            blob=data[pos:pos+size]; pos+=size+1
            if hashlib.sha256(blob).hexdigest()!=digest: raise RuntimeError('Staged hash mismatch: '+path)
            checked+=1
        assert pos==len(data)
    print(json.dumps({'staged_artifacts_verified':checked,'expected':manifest['count']}))


if __name__=='__main__': main()
