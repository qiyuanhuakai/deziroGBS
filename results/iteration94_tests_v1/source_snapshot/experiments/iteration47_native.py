"""Frozen official ten-mode laws on exactly represented nonclassical inputs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,subprocess,time,traceback
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from gbslab.frugal_law import full_law
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
def main():
    out=ROOT/'results/iteration47_native';out.mkdir(exist_ok=False);names=['research/iteration47_mixture_plan.md',*[f'experiments/{n}.py' for n in ('iteration47_native','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent')],*[f'src/gbslab/{n}.py' for n in ('frugal_law','interval_completion','pairwise_completion')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    summary=read('results/iteration47_inputs/summary.json');assert summary['completed']==8;exe=ROOT/'results/iteration39_build/frugal_extended.exe';manifest=json.loads((ROOT/'results/iteration39_manifest.json').read_text())
    for p in [exe,exe.parent/'build.json',*exe.parent.glob('*.dll')]:n=p.relative_to(ROOT).as_posix();deps[n]=digest(p);assert deps[n]==manifest['artifacts'][n]
    jobs=[dict(case=i,cutoff=k,seed=472000+i) for i in range(8) for k in (2,3,4)];sources=[read(f'results/iteration47_inputs/case_{i:02d}.json') for i in range(8)];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps));rows=[]
    for job in jobs:
        start=time.perf_counter();i=job['case'];k=job['cutoff'];label=f'case_{i:02d}_k{k}';row=dict(**job);print('start native',label,flush=True)
        try:
            source=sources[i];n=source['job']['modes'];V=source['physical']['covariance'];padded=[[F(a==b) for b in range(20)] for a in range(20)];ix=list(range(n))+list(range(10,10+n))
            for a in range(2*n):
                for b in range(2*n):padded[ix[a]][ix[b]]=V[a][b]
            csv=out/(label+'.csv');csv.write_text('\n'.join(','.join(format(float(x),'.17g') for x in line) for line in padded)+'\n');assert all(F(float(x))==x for line in padded for x in line)
            cmd=[str(exe),'--input',str(csv),'--modes','10','--cutoff',str(k),'--samples','3','--seed',str(job['seed']),'--threads','1','--audit','--prefix-tree']
            for name,command in (('stdout',cmd),('reverse',[str(exe),'--input',str(csv),'--modes','10','--cutoff',str(k),'--samples','0','--seed',str(job['seed']),'--threads','1','--prefix-tree','--reverse-prefixes'])):
                with (out/(label+'.'+name+'.json')).open('w') as stdout,(out/(label+'.'+name+'.stderr.log')).open('w') as stderr:p=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=300)
                assert p.returncode==0
            raw=json.loads((out/(label+'.stdout.json')).read_text());reverse=json.loads((out/(label+'.reverse.json')).read_text());assert {r['prefix']:r for r in raw['prefix_tree']}=={r['prefix']:r for r in reverse['prefix_tree']};steps=replay(raw);laws={m:full_law(10,raw['prefix_tree'],model=m) for m in ('continuous','uint64')};target=source['target']['target_intervals']+[(F(0),F(0))]*(1024-(1<<n));tvs={m:physical_tv_interval(law['distribution'],target) for m,law in laws.items()};leak={m:sum(law['distribution'][1<<n:]) for m,law in laws.items()};save(out/(label+'.audit.json'),encode(dict(job=job,command=cmd,laws=laws,physical_TV_intervals=tvs,vacuum_padding_leak=leak,replayed_steps=steps)));row.update(status='completed',TV_uint64=[float(x) for x in tvs['uint64']],vacuum_leak=float(leak['uint64']),replayed_steps=steps,timings=raw['timings_seconds'])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('end native',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=24,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
