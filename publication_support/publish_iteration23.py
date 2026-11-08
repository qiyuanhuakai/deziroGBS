"""Publish verified frozen blobs in bounded ordinary fast-forward batches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.git/codex_iteration23_publish.json'
BRANCH='research/iteration23-20261109'


def git(*args,input=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,input=input)


def save(data):
    STATE.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','next','finish']); args=p.parse_args()
    assert git('branch','--show-current').decode().strip()==BRANCH
    if args.action=='prepare':
        if STATE.exists(): raise FileExistsError(STATE)
        subprocess.run([sys.executable,'experiments/iteration23_seal.py','verify-staged'],cwd=ROOT,check=True)
        parent=git('rev-parse','HEAD').decode().strip()
        remote=git('ls-remote','origin','refs/heads/main').decode().split()[0]
        assert remote==parent and not git('ls-remote','origin','refs/heads/'+BRANCH).strip()
        paths=git('diff','--cached','--name-only','-z').split(b'\0'); changed={p.decode() for p in paths if p}
        records=[]
        for raw in git('ls-files','--stage','-z').split(b'\0'):
            if not raw: continue
            header,path=raw.split(b'\t',1); mode,oid,stage=header.decode().split(); name=path.decode()
            if name in changed:
                assert stage=='0'; size=int(git('cat-file','-s',oid))
                assert size<100*2**20
                records.append(dict(path=name,mode=mode,oid=oid,bytes=size))
        assert len(records)==len(changed)
        # Large data first; the final batch makes the complete research tree visible.
        large=sorted([r for r in records if r['bytes']>=2**20],key=lambda r:r['path'])
        small=sorted([r for r in records if r['bytes']<2**20],key=lambda r:r['path'])
        batches=[]; batch=[]; size=0
        for r in large:
            if size+r['bytes']>600*2**20 and batch: batches.append(batch); batch=[]; size=0
            batch.append(r); size+=r['bytes']
        if batch: batches.append(batch)
        batches.append(small)
        data=dict(parent=parent,branch=BRANCH,expected_tree=git('write-tree').decode().strip(),batches=batches,commits=[],pushed=0,finished=False)
        save(data)
        git('read-tree','HEAD')
        print(json.dumps(dict(batches=len(batches),files=len(records),expected_tree=data['expected_tree']))); return
    data=json.loads(STATE.read_text()); assert data['branch']==BRANCH
    if args.action=='next':
        i=data['pushed']
        if i==len(data['batches']): print('All batches already published'); return
        if len(data['commits'])==i:
            expected=data['parent'] if i==0 else data['commits'][-1]
            assert git('rev-parse','HEAD').decode().strip()==expected and not git('diff','--cached','--name-only').strip()
            entries=data['batches'][i]
            index=''.join(f"{r['mode']} {r['oid']}\t{r['path']}\n" for r in entries).encode()
            git('update-index','--index-info',input=index)
            git('commit','-m',f'Research iteration 23: verified publication batch {i+1}/{len(data["batches"])}')
            data['commits'].append(git('rev-parse','HEAD').decode().strip()); save(data)
        assert len(data['commits'])==i+1 and git('rev-parse','HEAD').decode().strip()==data['commits'][i]
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()
        allowed={data['commits'][i]}
        if i: allowed.add(data['commits'][i-1])
        assert (not remote and i==0) or (remote and remote[0] in allowed)
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True)
        data['pushed']+=1; save(data); print(json.dumps(dict(pushed=data['pushed'],total=len(data['batches'])))); return
    assert data['pushed']==len(data['batches'])
    head=git('rev-parse','HEAD').decode().strip()
    assert head==data['commits'][-1] and git('rev-parse','HEAD^{tree}').decode().strip()==data['expected_tree']
    assert not git('status','--porcelain').strip()
    remote=git('ls-remote','origin','refs/heads/main').decode().split()[0]
    assert remote in (data['parent'],head)
    subprocess.run(['git','merge-base','--is-ancestor',data['parent'],head],cwd=ROOT,check=True)
    subprocess.run(['git','push','--atomic','origin','HEAD:refs/heads/main','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True)
    local_main=git('rev-parse','refs/heads/main').decode().strip()
    subprocess.run(['git','merge-base','--is-ancestor',local_main,head],cwd=ROOT,check=True)
    git('update-ref','refs/heads/main',head,local_main)
    data['finished']=True; save(data); print(json.dumps(dict(science_commit=head,tree=data['expected_tree'])))


if __name__=='__main__': main()
