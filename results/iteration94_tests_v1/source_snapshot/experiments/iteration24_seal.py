"""Freeze round 24 bytes and verify new plus reused Git blobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]; MANIFEST=ROOT/'results/iteration24_manifest.json'


def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['create','verify-staged']); args=p.parse_args()
    if args.action=='create':
        if MANIFEST.exists(): raise FileExistsError(MANIFEST)
        chain=json.loads((ROOT/'results/iteration24_chain/summary.json').read_text()); assert chain['planned']==chain['finished']==chain['passed']==57 and chain['unchanged_sources']
        audit=json.loads((ROOT/'results/iteration24_validation/validation.json').read_text()); assert audit['count']==audit['passed']==2073
        geo=json.loads((ROOT/'results/iteration24_geometry/summary.json').read_text()); assert geo['count']==301 and geo['passed']==156 and geo['conditional_pure_trace_upper_estimate'] is None
        files=sorted(f for d in ROOT.glob('results/iteration24_*') if d.is_dir() for f in d.rglob('*') if f.is_file())
        artifacts={f.relative_to(ROOT).as_posix():digest(f) for f in files}; dependencies={name:sha for name,sha in audit['dependencies'].items() if name not in artifacts}
        assert all(digest(ROOT/name)==sha for name,sha in dependencies.items())
        assert all((ROOT/name).stat().st_size<90*2**20 for name in [*artifacts,*dependencies])
        data=dict(artifacts=artifacts,dependencies=dependencies,count=len(files),bytes=sum(f.stat().st_size for f in files),dependency_count=len(dependencies),dependency_bytes=sum((ROOT/name).stat().st_size for name in dependencies))
        MANIFEST.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:v for k,v in data.items() if k not in ('artifacts','dependencies')})); return
    manifest=json.loads(MANIFEST.read_text()); combined=dict(manifest['artifacts']); combined.update(manifest['dependencies']); index={}
    for record in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0'):
        if not record: continue
        header,path=record.split(b'\t',1); mode,oid,stage=header.split(); assert stage==b'0'; index[path.decode()]=oid
    entries=list(combined.items()); checked=0
    for start in range(0,len(entries),16):
        batch=entries[start:start+16]; data=subprocess.check_output(['git','cat-file','--batch'],input=b'\n'.join(index[name] for name,_ in batch)+b'\n',cwd=ROOT); pos=0
        for name,sha in batch:
            end=data.index(b'\n',pos); size=int(data[pos:end].split()[2]); pos=end+1; blob=data[pos:pos+size]; pos+=size+1
            if hashlib.sha256(blob).hexdigest()!=sha: raise RuntimeError('Staged bytes differ: '+name)
            checked+=1
        assert pos==len(data)
    print(json.dumps(dict(verified=checked,new=manifest['count'],reused=manifest['dependency_count'])))


if __name__=='__main__': main()
