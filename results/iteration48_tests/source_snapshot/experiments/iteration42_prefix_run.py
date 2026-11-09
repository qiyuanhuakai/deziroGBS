"""Complete initial-prefix laws of the actual144-mode official sampler."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,subprocess,time,traceback
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from gbslab.frugal_law import full_law,distance_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration42_prefix';out.mkdir(exist_ok=False)
    names=['research/iteration42_prefix_plan.md',*[f'experiments/{n}.py' for n in ('iteration42_prefix_run','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent')],'src/gbslab/frugal_law.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    read=lambda p:decode(json.loads(Path(p).read_text()))
    def reuse(n,it=None):
        sha=digest(ROOT/n)
        if it is not None:assert sha==json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text())['artifacts'][n]
        deps[n]=sha;return ROOT/n
    exe=reuse('results/iteration42_build/frugal_prefix.exe');build=read(reuse('results/iteration42_build/build.json'));assert digest(exe)==build['executable_sha256']
    for p in exe.parent.glob('*.dll'):reuse(p.relative_to(ROOT).as_posix())
    target=read(reuse('results/iteration37_local/target_000.json',37))['law']['intervals'];base=read(reuse('results/iteration35_panel/case_07.audit.json',35));transfer=base['transfers']['-40']['all_measurement_TV_transfer_upper'];reuse('results/iteration35_panel/case_12.audit.json',35);reuse('results/iteration35_panel/inputs.npz',35);reuse('results/iteration37_validation/validation.json',37)
    jobs=[]
    for k in range(1,6):
        jobs.append(dict(label=f'calibration_{k}',modes=10,cutoff=k,seed=391000+k,csv=str(reuse(f'results/iteration39_local/case_{k-1:02d}.csv',39)),reference=str(reuse(f'results/iteration39_local/case_{k-1:02d}.stdout.json',39))))
    for k in range(1,5):jobs.append(dict(label=f'global_{k}',modes=144,cutoff=k,seed=391440,csv=str(reuse(f'results/iteration39_global/case_{k-1:02d}.csv',39)),reference=str(reuse(f'results/iteration39_global/case_{k-1:02d}.stdout.json',39))))
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps,depth=10,samples=17,scope='Exact prefix laws; public samples only regression calibration'))
    invalid=[]
    for args in (['--prefix-tree','--prefix-depth','11'],['--prefix-tree','--prefix-depth','17'],['--prefix-depth','5']):
        p=subprocess.run([str(exe),'--input',jobs[0]['csv'],'--modes','10',*args],capture_output=True,text=True);invalid.append(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode!=0
    save(out/'invalid_depth.json',invalid);rows=[]
    for job in jobs:
        label=job['label'];row=dict(label=label,modes=job['modes'],cutoff=job['cutoff']);start=time.perf_counter();print('starting',label,flush=True)
        try:
            cmd=[str(exe),'--input',job['csv'],'--modes',str(job['modes']),'--cutoff',str(job['cutoff']),'--samples','17','--seed',str(job['seed']),'--threads','1','--audit','--prefix-tree','--prefix-depth','10']
            with (out/(label+'.stdout.json')).open('w') as stdout,(out/(label+'.stderr.log')).open('w') as stderr:p=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=1800)
            assert p.returncode==0;raw=read(out/(label+'.stdout.json'));old=read(job['reference']);assert raw['prefix_reverse_verified'] and raw['prefix_depth']==10
            assert raw['outputs']==old['outputs'][:17] and raw['audit_paths']==old['audit_paths'][:17]
            steps=replay({**raw,'prefix_tree':[]});table={r['prefix']:r['threshold_hex'] for r in raw['prefix_tree']}
            for s in range(17):
                for j in range(10):assert table[raw['outputs'][s][:j]]==raw['audit_paths'][s][j][1]
            if job['modes']==10:assert raw['prefix_tree']==old['prefix_tree']
            laws={model:full_law(10,raw['prefix_tree'],model=model) for model in ('continuous','uint64')};tv={model:distance_interval(law['distribution'],target) for model,law in laws.items()};q=laws['uint64']['distribution'];event=[i for i,(x,(a,b)) in enumerate(zip(q,target)) if x>b];Q=sum(q[i] for i in event);P=(sum(target[i][0] for i in event),sum(target[i][1] for i in event));gap=max(F(0),Q-P[1])
            result=dict(job=job,command=cmd,laws=laws,prefix_TV_intervals=tv,transfer_upper=transfer,full144_TV_lower=tv['uint64'][0] if job['modes']==144 else None,full144_shifted_TV_lower=max(F(0),tv['uint64'][0]-transfer) if job['modes']==144 else None,event=dict(masks=event,sampler_probability=Q,target_probability_interval=P,TV_lower=gap),replayed_steps=steps)
            save(out/(label+'.audit.json'),encode(result));row.update(status='completed',prefix_TV_lower=float(tv['uint64'][0]),prefix_TV_upper=float(tv['uint64'][1]),shifted_lower=float(max(F(0),tv['uint64'][0]-transfer)),event_lower=float(gap),timings=raw['timings_seconds'],replayed_steps=steps)
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('finished',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
