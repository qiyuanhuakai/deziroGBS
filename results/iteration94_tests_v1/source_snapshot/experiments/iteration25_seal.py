"""Freeze new artifacts and verify exact Git bytes, including reused states."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'results/iteration25_manifest.json'


def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['create','verify-staged']); args=p.parse_args()
    if args.action=='create':
        if MANIFEST.exists(): raise FileExistsError(MANIFEST)
        panel=json.loads((ROOT/'results/iteration25_panel_retry/summary.json').read_text(encoding='utf-8'))
        assert panel['finished']==panel['planned']==20 and panel['other']==0 and panel['unchanged_sources']
        compare=json.loads((ROOT/'results/iteration25_compare/summary.json').read_text(encoding='utf-8'))
        assert compare['finished']==compare['passed']==18 and compare['unchanged_sources']
        audit=json.loads((ROOT/'results/iteration25_validation/validation.json').read_text(encoding='utf-8')); assert audit['count']==audit['passed']
        analysis=json.loads((ROOT/'results/iteration25_analysis/summary.json').read_text(encoding='utf-8'))
        assert analysis['validation_checks']==audit['count'] and analysis['validation_passed']==audit['passed']
        assert digest(ROOT/'experiments/iteration25_analyze.py')==analysis['analyzer_sha256']==digest(ROOT/'results/iteration25_analysis/iteration25_analyze.py')
        files=sorted(f for directory in ROOT.glob('results/iteration25_*') if directory.is_dir() for f in directory.rglob('*') if f.is_file())
        artifacts={f.relative_to(ROOT).as_posix():digest(f) for f in files}
        dependencies={name:sha for name,sha in audit['dependencies'].items() if name not in artifacts}
        assert all(digest(ROOT/name)==sha for name,sha in dependencies.items())
        data=dict(artifacts=artifacts,dependencies=dependencies,count=len(files),bytes=sum(f.stat().st_size for f in files),dependency_count=len(dependencies),dependency_bytes=sum((ROOT/name).stat().st_size for name in dependencies))
        MANIFEST.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in data.items() if k not in ('artifacts','dependencies')})); return
    manifest=json.loads(MANIFEST.read_text()); combined=dict(manifest['artifacts']); combined.update(manifest['dependencies']); index={}
    for record in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0'):
        if not record: continue
        header,path=record.split(b'\t',1); mode,oid,stage=header.split()
        if stage!=b'0': raise RuntimeError('Unmerged index')
        index[path.decode('utf-8')]=oid
    packaged={}
    for round_number in (21,23,25):
        path=ROOT/f'research_archive/publication/iteration{round_number}_large_artifacts/manifest.json'
        if not path.exists(): continue
        records=json.loads(path.read_text())['artifacts']; relevant={k:v for k,v in records.items() if k in combined}
        if not relevant: continue
        staged=subprocess.check_output(['git','cat-file','blob',index[path.relative_to(ROOT).as_posix()].decode()],cwd=ROOT)
        if staged!=path.read_bytes(): raise RuntimeError('Staged package metadata mismatch')
        packaged.update(relevant)
    checked=0
    for path,record in packaged.items():
        whole=hashlib.sha256(); size=0
        for part in record['parts']:
            data=subprocess.check_output(['git','cat-file','blob',index[part['path']].decode()],cwd=ROOT)
            assert hashlib.sha256(data).hexdigest()==part['sha256'] and len(data)==part['bytes']
            whole.update(data); size+=len(data)
        assert size==record['bytes'] and whole.hexdigest()==record['sha256']==combined[path]
        checked+=1
    entries=[(name,sha) for name,sha in combined.items() if name not in packaged]
    for start in range(0,len(entries),16):
        batch=entries[start:start+16]
        data=subprocess.check_output(['git','cat-file','--batch'],input=b'\n'.join(index[name] for name,_ in batch)+b'\n',cwd=ROOT)
        pos=0
        for name,sha in batch:
            end=data.index(b'\n',pos); size=int(data[pos:end].split()[2]); pos=end+1
            blob=data[pos:pos+size]; pos+=size+1
            if hashlib.sha256(blob).hexdigest()!=sha: raise RuntimeError('Staged bytes differ: '+name)
            checked+=1
        assert pos==len(data)
    print(json.dumps(dict(staged_new_and_reused_artifacts_verified=checked,expected=manifest['count']+manifest['dependency_count'],new_artifacts=manifest['count'],reused_states=manifest['dependency_count'])))


if __name__=='__main__': main()
