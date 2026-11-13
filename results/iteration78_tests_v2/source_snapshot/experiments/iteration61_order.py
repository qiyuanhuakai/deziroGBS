"""Frozen-order actual144 prefix experiments and complete ten-mode controls."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,gzip,hashlib,shutil,subprocess,time,traceback
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from gbslab.frugal_law import full_law,distance_interval,uniform_model_transfer_bound
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def order(n,depth,kind):
    first=list(range(depth))
    if kind=='reverse':first=first[::-1]
    elif kind=='even_odd':first=first[::2]+first[1::2]
    else:assert kind=='identity'
    return first+list(range(depth,n))

def mask_map(permutation):
    return [sum(((mask>>j)&1)<<p for j,p in enumerate(permutation)) for mask in range(1<<len(permutation))]

def main():
    out=ROOT/'results/iteration61_order';out.mkdir(exist_ok=False)
    names=['research/iteration61_order_plan.md',*[f'experiments/{n}.py' for n in ('iteration61_order','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    manifests={}
    def reuse(n,it=None):
        sha=digest(ROOT/n);deps[n]=sha
        if it is not None:
            if it not in manifests:
                name=f'results/iteration{it}_manifest.json';deps[name]=digest(ROOT/name);manifests[it]=json.loads((ROOT/name).read_text())
            assert sha==manifests[it]['artifacts'][n],n
        return ROOT/n
    def read(n,it=None):
        p=reuse(n,it);b=p.read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    exe=reuse('results/iteration42_build/frugal_prefix.exe',42);build=read('results/iteration42_build/build.json',42);assert digest(exe)==build['executable_sha256']
    for p in exe.parent.glob('*.dll'):reuse(p.relative_to(ROOT).as_posix(),42)
    V=np.load(reuse('results/iteration35_panel/inputs.npz',35))['shifted'];assert V.shape==(288,288)
    physical=read('results/iteration56_target/target.json',56);targets={144:physical['law']['intervals'],10:read('results/iteration37_local/target_000.json',37)['law']['intervals']}
    read('results/iteration56_prefix_validation/validation.json',56);read('results/iteration56_prefix_validation/bounds.json',56)
    for k in range(1,5):
        read(f'results/iteration56_prefix/global_{k}.stdout.json.gz',56);reuse(f'results/iteration56_prefix/global_{k}.audit.json.gz',56)
    for k in (1,4):read(f'results/iteration39_local/case_{k-1:02d}.stdout.json',39);reuse(f'results/iteration39_local/case_{k-1:02d}.audit.json',39)
    jobs=[]
    for kind in ('identity','reverse','even_odd'):
        for k in (1,4):jobs.append(dict(label=f'local_{kind}_k{k}',scope='control10',modes=10,depth=10,cutoff=k,seed=391000+k,kind=kind,permutation=order(10,10,kind)))
    jobs.append(dict(label='global_identity_k1',scope='control144',modes=144,depth=16,cutoff=1,seed=391440,kind='identity',permutation=order(144,16,'identity')))
    for kind in ('reverse','even_odd'):
        for k in range(1,5):jobs.append(dict(label=f'global_{kind}_k{k}',scope='main144',modes=144,depth=16,cutoff=k,seed=391440,kind=kind,permutation=order(144,16,kind)))
    save(out/'plan.json',dict(jobs=jobs,samples=17,source_hashes=hashes,dependencies=deps,transfer_upper=encode(physical['transfer_upper'])));rows=[]
    for job in jobs:
        label=job['label'];start=time.perf_counter();row=dict(label=label,scope=job['scope'],kind=job['kind'],cutoff=job['cutoff']);print('start',label,flush=True)
        try:
            n=job['modes'];depth=job['depth'];perm=job['permutation'];ix=perm+[144+j for j in perm];cov=V[np.ix_(ix,ix)];csv=out/(label+'.csv');np.savetxt(csv,cov,delimiter=',',fmt='%.17g');assert np.array_equal(np.loadtxt(csv,delimiter=','),cov)
            cmd=[str(exe),'--input',str(csv),'--modes',str(n),'--cutoff',str(job['cutoff']),'--samples','17','--seed',str(job['seed']),'--threads','1','--audit','--prefix-tree','--prefix-depth',str(depth)]
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800);(out/(label+'.stderr.log')).write_bytes(proc.stderr);payload=proc.stdout;data=gzip.compress(payload,compresslevel=6,mtime=0);(out/(label+'.stdout.json.gz')).write_bytes(data);storage={'stdout':dict(path=label+'.stdout.json.gz',raw_bytes=len(payload),compressed_bytes=len(data),raw_sha256=hashlib.sha256(payload).hexdigest())};save(out/(label+'.storage.json'),storage);assert proc.returncode==0
            raw=decode(json.loads(payload));assert raw['prefix_depth']==depth and raw['prefix_reverse_verified'] and len(raw['prefix_tree'])==(1<<depth)-1;steps=replay({**raw,'prefix_tree':[]});table={r['prefix']:r for r in raw['prefix_tree']}
            for s in range(17):
                for j in range(depth):assert table[raw['outputs'][s][:j]]['threshold_hex']==raw['audit_paths'][s][j][1]
            if job['kind']=='identity':
                old=read(f'results/iteration56_prefix/global_1.stdout.json.gz',56) if n==144 else read(f'results/iteration39_local/case_{job["cutoff"]-1:02d}.stdout.json',39)
                assert raw['outputs']==old['outputs'][:17] and raw['audit_paths']==old['audit_paths'][:17] and table=={r['prefix']:r for r in old['prefix_tree']}
            mapping=mask_map(perm[:depth]);target=[targets[n][j] for j in mapping];laws={m:full_law(depth,raw['prefix_tree'],model=m) for m in ('continuous','uint64')};tv={m:distance_interval(v['distribution'],target) for m,v in laws.items()};coupling=sum(abs(a-b) for a,b in zip(laws['uint64']['distribution'],laws['continuous']['distribution']))/2;assert coupling<=uniform_model_transfer_bound(depth)
            mapped=[]
            for s in raw['outputs']:
                result=['']*n
                for j,p in enumerate(perm):result[p]=s[j]
                mapped.append(''.join(result))
            record=dict(job=job,command=cmd,laws=laws,TV_intervals=tv,uniform_model_distance=coupling,mapped_outputs=mapped,replayed_steps=steps)
            payload=(json.dumps(encode(record),indent=2)+'\n').encode();data=gzip.compress(payload,compresslevel=6,mtime=0);(out/(label+'.audit.json.gz')).write_bytes(data);storage['audit']=dict(path=label+'.audit.json.gz',raw_bytes=len(payload),compressed_bytes=len(data),raw_sha256=hashlib.sha256(payload).hexdigest());save(out/(label+'.storage.json'),storage);row.update(status='completed',TV_lower=float(tv['uint64'][0]),TV_upper=float(tv['uint64'][1]),timings=raw['timings_seconds'],replayed_steps=steps)
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('end',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=15,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,unchanged_dependencies=True,dependencies=deps))
if __name__=='__main__':main()
