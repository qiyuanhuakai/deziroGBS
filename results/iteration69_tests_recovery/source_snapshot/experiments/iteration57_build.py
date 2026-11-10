"""Build bounded prefix adapter with unchanged official sampler and round39 hooks."""
import json,shutil,subprocess,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'results/iteration57_build';out.mkdir(exist_ok=False);oldpath=ROOT/'results/iteration39_build/build.json';old=json.loads(oldpath.read_text());manifest=json.loads((ROOT/'results/iteration39_manifest.json').read_text());assert digest(oldpath)==manifest['artifacts'][oldpath.relative_to(ROOT).as_posix()];names=[*old['source_hashes'],'experiments/iteration57_paths.cpp','experiments/iteration57_build.py','experiments/iteration57_path_hooks.hpp','research/iteration57_overlap_plan.md'];hashes={n:digest(ROOT/n) for n in names}
    for n,sha in old['source_hashes'].items():assert hashes[n]==sha
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    deps={oldpath.relative_to(ROOT).as_posix():digest(oldpath)}
    for p in (ROOT/'results/iteration39_build/generated').rglob('*'):
        if p.is_file():n=p.relative_to(ROOT).as_posix();assert digest(p)==manifest['artifacts'][n];deps[n]=digest(p)
    cmd=old['command'].copy();cmd[cmd.index(str(ROOT/'experiments/frugal_extended.cpp'))]=str(ROOT/'experiments/iteration57_paths.cpp');exe=out/'frugal_paths.exe';cmd[-1]=str(exe);assert digest(cmd[0])==old['compiler_sha256'];meta=dict(source_hashes=hashes,dependencies=deps,command=cmd,compiler_sha256=old['compiler_sha256'],same_target_implementation=True,scope='Complete specified output paths; unchanged public sampler and old generated hooks');(out/'plan.json').write_text(json.dumps(meta,indent=2)+'\n');begin=time.perf_counter()
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:r=subprocess.run(cmd,stdout=stdout,stderr=stderr)
    meta.update(returncode=r.returncode,seconds=time.perf_counter()-begin,unchanged_sources=all(digest(ROOT/n)==sha for n,sha in hashes.items()))
    if r.returncode==0:
        meta['executable_sha256']=digest(exe);meta['runtime_files']={}
        for n,sha in old['runtime_files'].items():p=oldpath.parent/n;assert digest(p)==sha;shutil.copy2(p,out/n);meta['runtime_files'][n]=sha
    (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:meta[k] for k in ('returncode','seconds','unchanged_sources')}));assert r.returncode==0
if __name__=='__main__':main()
