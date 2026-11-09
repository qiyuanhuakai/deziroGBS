"""Actual144 sampler prefix enumeration, depth16, unchanged round42 binary."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,subprocess,time,traceback,gzip,hashlib
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from gbslab.frugal_law import full_law,distance_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
def main():
    out=ROOT/'results/iteration56_prefix';out.mkdir(exist_ok=False);names=['research/iteration56_prefix_plan.md',*[f'experiments/{n}.py' for n in ('iteration56_prefix_run','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent')],'src/gbslab/frugal_law.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    def reuse(n,it=None):
        sha=digest(ROOT/n)
        if it:assert sha==json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text())['artifacts'][n]
        deps[n]=sha;return ROOT/n
    exe=reuse('results/iteration42_build/frugal_prefix.exe',42);build=read('results/iteration42_build/build.json');assert digest(exe)==build['executable_sha256']
    for p in exe.parent.glob('*.dll'):reuse(p.relative_to(ROOT).as_posix(),42)
    jobs=[dict(label=f'global_{k}',modes=144,cutoff=k,seed=391440,csv=f'results/iteration39_global/case_{k-1:02d}.csv',reference=f'results/iteration53_prefix/global_{k}.stdout.json.gz') for k in range(1,5)]
    for job in jobs:reuse(job['csv'],39);reuse(job['reference'],53)
    # Enumeration does not depend on the target job; the two may run concurrently.
    save(out/'plan.json',dict(jobs=jobs,depth=16,samples=17,source_hashes=hashes,dependencies=deps));rows=[]
    for job in jobs:
        start=time.perf_counter();label=job['label'];row=dict(label=label,cutoff=job['cutoff']);print('start',label,flush=True)
        try:
            cmd=[str(exe),'--input',str(ROOT/job['csv']),'--modes','144','--cutoff',str(job['cutoff']),'--samples','17','--seed','391440','--threads','1','--audit','--prefix-tree','--prefix-depth','16']
            p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800);stdout=p.stdout;data=gzip.compress(stdout,compresslevel=6,mtime=0);(out/(label+'.stdout.json.gz')).write_bytes(data);(out/(label+'.stderr.log')).write_bytes(p.stderr);storage={'stdout':dict(path=label+'.stdout.json.gz',raw_bytes=len(stdout),compressed_bytes=len(data),raw_sha256=hashlib.sha256(stdout).hexdigest())};assert p.returncode==0;raw=decode(json.loads(stdout));old=read(job['reference']);assert raw['prefix_reverse_verified'] and raw['prefix_depth']==16 and raw['outputs']==old['outputs'] and raw['audit_paths']==old['audit_paths']
            table={r['prefix']:r for r in raw['prefix_tree']};assert len(table)==65535 and all(table[r['prefix']]==r for r in old['prefix_tree']);steps=replay({**raw,'prefix_tree':[]})
            for s in range(17):
                for j in range(16):assert table[raw['outputs'][s][:j]]['threshold_hex']==raw['audit_paths'][s][j][1]
            laws={m:full_law(16,raw['prefix_tree'],model=m) for m in ('continuous','uint64')};rawaudit=(json.dumps(encode(dict(job=job,command=cmd,laws=laws,replayed_steps=steps)),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8');data=gzip.compress(rawaudit,compresslevel=6,mtime=0);(out/(label+'.audit.json.gz')).write_bytes(data);storage['audit']=dict(path=label+'.audit.json.gz',raw_bytes=len(rawaudit),compressed_bytes=len(data),raw_sha256=hashlib.sha256(rawaudit).hexdigest());save(out/(label+'.storage.json'),storage);row.update(status='completed',timings=raw['timings_seconds'],replayed_steps=steps)
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('end',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=4,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
