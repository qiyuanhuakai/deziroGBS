"""Finite RNG calibration and bounded, explicitly exploratory scaling pilots."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,hashlib,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    out=ROOT/'results/iteration39_pilot';out.mkdir(exist_ok=False);exe=ROOT/'results/iteration39_build/frugal_extended.exe';build=json.loads((exe.parent/'build.json').read_text());assert digest(exe)==build['executable_sha256'];source_hashes={n:digest(ROOT/n) for n in ('experiments/iteration39_pilot.py','experiments/build_frugal_extended.py','experiments/frugal_extended.cpp')}
    for n in source_hashes:dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    primary=ROOT/'results/iteration35_panel/inputs.npz';assert digest(primary)==json.loads((ROOT/'results/iteration35_manifest.json').read_text())['artifacts'][primary.relative_to(ROOT).as_posix()];V=np.load(primary)['shifted'];jobs=[dict(modes=2,cutoff=k,samples=129,kind=kind) for kind in ('vacuum','thermal','tmsv') for k in (1,2)]+[dict(modes=10,cutoff=k,samples=129,kind='primary') for k in (1,2,5)]+[dict(modes=m,cutoff=k,samples=2,kind='primary') for m in (32,64,144) for k in (1,2,3)]+[dict(modes=144,cutoff=4,samples=2,kind='primary')];save(out/'plan.json',dict(jobs=jobs,exploratory_only=True,timeout_per_case_seconds=600,seed_base=390100,source_hashes=source_hashes,dependencies={primary.relative_to(ROOT).as_posix():digest(primary),'results/iteration39_build/frugal_extended.exe':digest(exe),'results/iteration39_build/build.json':digest(exe.parent/'build.json')}))
    rng=np.random.default_rng(390064);words={0,1,2,(1<<64)-1,(1<<64)-1024,(1<<64)-1025};words.update(int(w) for w in rng.bit_generator.random_raw(4096))
    for e in range(53,65):
        for base in (1<<e,(1<<e)-(1<<max(0,e-52)),(1<<e)-(1<<max(0,e-53))):
            for delta in range(-4,5):
                if 0<=base+delta<1<<64:words.add(base+delta)
    (out/'rng_words.txt').write_text('\n'.join(map(str,sorted(words)))+'\n');cmd=[str(exe),'--rng-probe',str(out/'rng_words.txt')];p=subprocess.run(cmd,capture_output=True,text=True,check=True);(out/'rng_probe.json').write_text(p.stdout);headers={};compiler=Path(build['command'][0]);stdlib=compiler.parent.parent/'lib/gcc/x86_64-w64-mingw32/15.1.0/include/c++/bits';(out/'rng_source').mkdir()
    for name in ('random.h','random.tcc'):
        source=stdlib/name;shutil.copy2(source,out/'rng_source'/name);headers[name]=dict(source=str(source),sha256=digest(source))
    save(out/'rng_source/provenance.json',dict(headers=headers,compiler_sha256=build['compiler_sha256'],assumption='Independent uniform uint64 words; not uniform seeds or proof of mt19937 independence'))
    rows=[]
    for i,j in enumerate(jobs):
        print('starting',i,j,flush=True);begin=time.perf_counter();r=dict(sequence=i,**j);m=j['modes']
        if j['kind']=='vacuum':local=np.eye(4)
        elif j['kind']=='thermal':local=np.diag([3.,5.,3.,5.])
        elif j['kind']=='tmsv':local=np.array([[1.25,.75,0,0],[.75,1.25,0,0],[0,0,1.25,-.75],[0,0,-.75,1.25]])
        else:ix=list(range(m))+list(range(144,144+m));local=V[np.ix_(ix,ix)]
        csv=out/f'case_{i:02d}.csv';np.savetxt(csv,local,delimiter=',',fmt='%.17g');np.savez_compressed(out/f'case_{i:02d}.npz',covariance=local);cmd=[str(exe),'--input',str(csv),'--modes',str(m),'--cutoff',str(j['cutoff']),'--samples',str(j['samples']),'--seed',str(390100+i),'--threads','1','--audit']
        if m<=10:cmd.append('--prefix-tree')
        r['command']=cmd
        try:
            with (out/f'case_{i:02d}.stdout.json').open('w') as stdout,(out/f'case_{i:02d}.stderr.log').open('w') as stderr:p=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=600)
            r['returncode']=p.returncode
            if p.returncode:raise RuntimeError('Native process failed; raw logs retained')
            rec=json.loads((out/f'case_{i:02d}.stdout.json').read_text());assert rec['audit_identical_outputs'] and len(rec['outputs'])==j['samples'];assert all(len(s)==m and set(s)<={'0','1'} for s in rec['outputs']);replayed=0
            for s,path in zip(rec['outputs'],rec['audit_paths']):
                assert len(path)==m
                for bit,(u,q,prev,z) in zip(s,path):assert int(float.fromhex(u)>float.fromhex(q))==int(bit);replayed+=1
            if m<=10:
                table={x['prefix']:x['threshold_hex'] for x in rec['prefix_tree']}
                for s,path in zip(rec['outputs'],rec['audit_paths']):
                    for pos,a in enumerate(path):
                        q=float.fromhex(table[s[:pos]]);actual=float.fromhex(a[1]);assert q==actual or np.isnan(q) and np.isnan(actual)
            r.update(status='completed',timings=rec['timings_seconds'],layout_entries=rec['layout_entries'],replayed_steps=replayed,saturated=rec['audit_saturated_steps'],nonfinite=rec['audit_nonfinite_steps'])
        except Exception:r.update(status='failed',traceback=traceback.format_exc())
        r['seconds']=time.perf_counter()-begin;rows.append(r);save(out/f'case_{i:02d}.json',r);save(out/'rows.json',rows);print('finished',i,r,flush=True)
    assert all(digest(ROOT/n)==sha for n,sha in source_hashes.items());save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,exploratory_only=True,rng_probe_words=len(words)))
if __name__=='__main__':main()
