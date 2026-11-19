"""Seal scientific outputs and exact reused input bytes without rewriting history."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];MANIFEST=ROOT/'results/iteration61_manifest.json'

def digest(p):
    with open(p,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('create','verify-staged'));action=parser.parse_args().action
    if action=='create':
        if MANIFEST.exists():raise FileExistsError(MANIFEST)
        r=json.loads((ROOT/'results/iteration61_order/summary.json').read_text());assert r['planned']==r['finished']==r['completed']==15 and r['unchanged_sources'] and r['unchanged_dependencies']
        total=0
        for folder in ('order_validation',):
            directory=ROOT/f'results/iteration61_{folder}';r=json.loads((directory/'validation.json').read_text());check_count=0
            for name in r.get('check_shards',['checks.json']):
                checks=json.loads((directory/name).read_text());assert all(c['passed'] for c in checks);check_count+=len(checks)
            assert check_count==r['count']==r['passed'];total+=check_count
            if folder=='order_validation':assert r['new_runs']==15 and r['main_new_runs']==8 and r['control_new_runs']==7 and r['new_prefix_nodes']==595953 and r['all_prefix_nodes']==858093 and r['replayed_steps']==32844
        tests=json.loads((ROOT/'results/iteration61_tests/validation.json').read_text());assert tests['executed_gate_passed'] and tests['unchanged_sources'] and tests['new_tests_passed']==0
        visual=json.loads((ROOT/'results/iteration61_analysis/visual_review.json').read_text());assert visual['accepted'] and digest(ROOT/'results/iteration61_analysis/order_effects.png')==visual['sha256']
        dependencies={}
        for directory in ROOT.glob('results/iteration61_*'):
            if not directory.is_dir():continue
            for filename in ('plan.json','summary.json','validation.json','build.json'):
                p=directory/filename
                if not p.exists():continue
                r=json.loads(p.read_text())
                for n,s in r.get('source_hashes',{}).items():assert digest(ROOT/n)==s==digest(directory/'source_snapshot'/n)
                for n,s in r.get('dependencies',{}).items():assert digest(ROOT/n)==s;dependencies[n]=s
        for n in ('results/iteration25_tests/validation.json','results/iteration31_build/libwire_forest.dll','results/iteration55_manifest.json','results/iteration60_manifest.json'):dependencies[n]=digest(ROOT/n)
        archives={};archive_dir=ROOT/'results/iteration61_dependency_archive';archive_dir.mkdir(exist_ok=False)
        for n,s in list(dependencies.items()):
            if n.startswith('results/iteration61_'):continue
            original=subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)
            if hashlib.sha256(original).hexdigest()==s:continue
            payload=(ROOT/n).read_bytes();assert n.endswith('.json') and original.replace(b'\r\n',b'\n')==payload.replace(b'\r\n',b'\n');p=archive_dir/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p);archives[n]=dict(path=p.relative_to(ROOT).as_posix(),sha256=s,inherited_git_sha256=hashlib.sha256(original).hexdigest())
        (archive_dir/'archive_map.json').write_text(json.dumps(archives,indent=2)+'\n',encoding='utf-8');files=sorted(p for d in ROOT.glob('results/iteration61_*') if d.is_dir() for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts);artifacts={p.relative_to(ROOT).as_posix():digest(p) for p in files};dependencies={n:s for n,s in dependencies.items() if n not in artifacts};r=dict(artifacts=artifacts,dependencies=dependencies,dependency_archives=archives,count=len(artifacts),bytes=sum(p.stat().st_size for p in files),dependency_count=len(dependencies),dependency_bytes=sum((ROOT/n).stat().st_size for n in dependencies),scientific_checks=total);MANIFEST.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in r.items() if not isinstance(v,dict)}));return
    manifest=json.loads(MANIFEST.read_text());index={}
    for row in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0'):
        if not row:continue
        header,path=row.split(b'\t',1);mode,oid,stage=header.split();assert stage==b'0';index[path.decode()]=oid
    def blob(n):return subprocess.check_output(['git','cat-file','blob',index[n].decode()],cwd=ROOT)
    assert blob('results/iteration61_manifest.json')==MANIFEST.read_bytes();combined={**manifest['artifacts'],**manifest['dependencies']};aliases=manifest['dependency_archives']
    for n,r in aliases.items():
        original=blob(n);assert hashlib.sha256(original).hexdigest()==r['inherited_git_sha256'];assert original.replace(b'\r\n',b'\n')==(ROOT/r['path']).read_bytes().replace(b'\r\n',b'\n');assert digest(ROOT/n)==combined[n]==r['sha256']
    entries=[(aliases[n]['path'] if n in aliases else n,s) for n,s in combined.items()];checked=0
    for start in range(0,len(entries),16):
        batch=entries[start:start+16];data=subprocess.check_output(['git','cat-file','--batch'],input=b'\n'.join(index[n] for n,s in batch)+b'\n',cwd=ROOT);pos=0
        for n,s in batch:
            end=data.index(b'\n',pos);size=int(data[pos:end].split()[2]);pos=end+1;payload=data[pos:pos+size];pos+=size+1;assert hashlib.sha256(payload).hexdigest()==s==digest(ROOT/n),n;checked+=1
        assert pos==len(data)
    print(json.dumps(dict(exact_working_and_git_payload_checks=checked,expected=manifest['count']+manifest['dependency_count'],archived_reused_inputs=len(aliases))))
if __name__=='__main__':main()
