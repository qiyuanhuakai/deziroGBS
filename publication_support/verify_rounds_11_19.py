"""Verify frozen round 11--19 evidence against committed Git blob bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]

def git(*args,input=None): return subprocess.check_output(['git',*args],input=input,cwd=ROOT)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',required=True); p.add_argument('--head',required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    base=git('rev-parse',args.base).decode().strip(); head=git('rev-parse',args.head).decode().strip()
    subprocess.run(['git','merge-base','--is-ancestor',base,head],cwd=ROOT,check=True)
    index={}
    for record in git('ls-tree','-r','-z',head).split(b'\0'):
        if not record: continue
        header,path=record.split(b'\t',1); mode,kind,oid=header.split()
        if kind==b'blob': index[path.decode()]=oid
    manifests=['results/iteration11_analysis_release/artifact_manifest.json','results/iteration12_analysis/artifact_manifest.json']+[f'results/iteration{k}_manifest.json' for k in range(13,20)]
    expected={}; records=[]
    for path in manifests:
        blob=git('cat-file','blob',index[path].decode()); data=json.loads(blob); entries=data.get('artifacts',data)
        for name,digest in entries.items():
            normalized=name.replace('\\','/')
            if normalized in expected and expected[normalized]!=digest: raise ValueError('Conflicting artifact hashes: '+normalized)
            expected[normalized]=digest
        records.append(dict(path=path,sha256=hashlib.sha256(blob).hexdigest(),entries=len(entries)))
    entries=list(expected.items()); checked=0; total_bytes=0
    for start in range(0,len(entries),32):
        batch=entries[start:start+32]; output=git('cat-file','--batch',input=b'\n'.join(index[path] for path,_ in batch)+b'\n'); pos=0
        for path,digest in batch:
            end=output.index(b'\n',pos); header=output[pos:end].split(); size=int(header[2]); pos=end+1
            blob=output[pos:pos+size]; pos+=size+1
            if hashlib.sha256(blob).hexdigest()!=digest: raise ValueError('Committed artifact mismatch: '+path)
            checked+=1; total_bytes+=size
        assert pos==len(output)
    commits=git('log','--reverse','--format=%H %s',base+'..'+head).decode().splitlines()
    changed=[x.decode() for x in git('diff','--name-only','-z',base,head).split(b'\0') if x]
    report=dict(format='deziroGBS-rounds11-19-publication-preflight-v1',verified_at_utc=datetime.now(timezone.utc).isoformat(),repository='qiyuanhuakai/deziroGBS',
        base_commit=base,science_commit=head,science_tree=git('rev-parse',head+'^{tree}').decode().strip(),commits=commits,changed_paths=len(changed),
        frozen_manifests=records,unique_artifacts_verified=checked,artifact_bytes_verified=total_bytes,
        scope='Exact committed artifact bytes verified. Publication preserves successful and failed evidence; this check is not a claim that the full scientific test gate passes.')
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('commits','frozen_manifests')},indent=2))

if __name__=='__main__': main()
