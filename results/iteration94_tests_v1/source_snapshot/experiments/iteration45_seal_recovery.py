"""Verify exact archived dependency bytes when old Git JSON has different EOL.

Packaging-only post-suite recovery. Original scientific manifest is immutable.
The annex archives the two original working-byte inputs without altering old
Git manifests. All other dependencies and new artifacts retain direct checks.
"""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    manifest=json.loads((ROOT/'results/iteration45_manifest.json').read_text(encoding='utf-8'));annex_name='results/iteration45_packaging/annex.json';annex=json.loads((ROOT/annex_name).read_text(encoding='utf-8'));index={}
    for row in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0'):
        if not row:continue
        header,path=row.split(b'\t',1);mode,oid,stage=header.split();assert stage==b'0';index[path.decode()]=oid
    def blob(name):return subprocess.check_output(['git','cat-file','blob',index[name].decode()],cwd=ROOT)
    assert blob(annex_name)==(ROOT/annex_name).read_bytes()
    combined=dict(manifest['artifacts']);combined.update(manifest['dependencies']);aliases=annex['dependency_archives']
    for name,record in aliases.items():
        assert name in manifest['dependencies'];original=blob(name);archived=(ROOT/record['path']).read_bytes();assert sha(original)==record['inherited_git_sha256'];assert original.replace(b'\r\n',b'\n')==archived.replace(b'\r\n',b'\n');assert sha(archived)==combined[name]==record['sha256'];assert sha((ROOT/name).read_bytes())==combined[name]
    entries=[(aliases[n]['path'] if n in aliases else n,s) for n,s in combined.items()]+list(annex['artifacts'].items());checked=0
    for start in range(0,len(entries),16):
        batch=entries[start:start+16];data=subprocess.check_output(['git','cat-file','--batch'],input=b'\n'.join(index[n] for n,s in batch)+b'\n',cwd=ROOT);pos=0
        for n,s in batch:
            end=data.index(b'\n',pos);size=int(data[pos:end].split()[2]);pos=end+1;payload=data[pos:pos+size];pos+=size+1;assert sha(payload)==s==sha((ROOT/n).read_bytes()),n;checked+=1
        assert pos==len(data)
    source='experiments/iteration45_seal_recovery.py';assert sha(blob(source))==sha((ROOT/source).read_bytes())==annex['post_suite_packaging_source_sha256'];print(json.dumps(dict(base_manifest_checks=len(combined),archived_dependency_inputs=len(aliases),annex_artifact_checks=len(annex['artifacts']),total_payload_checks=checked,post_suite_packaging_source_verified=True)))
if __name__=='__main__':main()
