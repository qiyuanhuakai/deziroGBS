"""Full prescribed-output traces without enumerating the full144 tree."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,gzip,hashlib,shutil,subprocess,time,traceback
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from gbslab.complete_path_overlap import path_probability
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration57_paths';out.mkdir(exist_ok=False)
    names=['research/iteration57_overlap_plan.md','research/iteration57_interval_note.md','src/gbslab/complete_path_overlap.py','src/gbslab/frugal_law.py',*[f'experiments/{n}.py' for n in ('iteration57_paths_run','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent')]]
    hashes={n:digest(ROOT/n) for n in names};deps={};rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    def bind(n,it):
        m=read(f'results/iteration{it}_manifest.json');deps[n]=digest(ROOT/n);assert deps[n]==m['artifacts'][n]
    build=read('results/iteration57_build/build.json');exe=ROOT/'results/iteration57_build/frugal_paths.exe';assert digest(exe)==build['executable_sha256'];deps[exe.relative_to(ROOT).as_posix()]=digest(exe)
    for n,s in build['runtime_files'].items():p=exe.parent/n;assert digest(p)==s;deps[p.relative_to(ROOT).as_posix()]=s
    jobs=[dict(label='local_4',modes=10,cutoff=4,samples=257,seed=391004,csv='results/iteration39_local/case_03.csv',reference='results/iteration39_local/case_03.stdout.json',reference_iteration=39,depth=10)]
    jobs += [dict(label=f'global_{k}',modes=144,cutoff=k,samples=17,seed=391440,csv=f'results/iteration39_global/case_{k-1:02d}.csv',reference=f'results/iteration56_prefix/global_{k}.stdout.json.gz',reference_iteration=56,depth=8) for k in range(1,5)]
    for job in jobs:bind(job['csv'],39);bind(job['reference'],job['reference_iteration'])
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps))
    for job in jobs:
        tick=time.perf_counter();label=job['label'];row=dict(job=job);print('start',label,flush=True)
        try:
            n=job['modes'];outputs=[''.join(str(mask>>j&1) for j in range(n)) for mask in range(1024)] if n==10 else ['0'*n]+[''.join('1' if j==pos else '0' for j in range(n)) for pos in (0,47,95,143)]
            path=out/(label+'.paths.txt');path.write_text('\n'.join(outputs)+'\n')
            cmd=[str(exe),'--input',str(ROOT/job['csv']),'--modes',str(n),'--cutoff',str(job['cutoff']),'--samples',str(job['samples']),'--seed',str(job['seed']),'--threads','1','--audit','--prefix-tree','--prefix-depth',str(job['depth']),'--probe-paths',str(path)]
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800);rawbytes=proc.stdout;data=gzip.compress(rawbytes,compresslevel=6,mtime=0);(out/(label+'.stdout.json.gz')).write_bytes(data);(out/(label+'.stderr.log')).write_bytes(proc.stderr)
            assert proc.returncode==0;raw=decode(json.loads(rawbytes));old=read(job['reference']);assert raw['outputs']==old['outputs'] and raw['audit_paths']==old['audit_paths'];table={r['prefix']:r for r in old['prefix_tree']};assert all(table[r['prefix']]==r for r in raw['prefix_tree']);steps=replay({**raw,'prefix_tree':[]});assert [r['output'] for r in raw['path_probes']]==outputs
            laws=[]
            for r in raw['path_probes']:
                for j,step in enumerate(r['steps'][:16 if n==144 else 10]):assert step==[table[r['output'][:j]][k] for k in ('threshold_hex','previous_hex','proposed_hex')]
                laws.append(dict(output=r['output'],models={model:path_probability(r['output'],r['steps'],model=model) for model in ('uint64','continuous')}))
            if n==10:
                oldaudit='results/iteration39_local/case_03.audit.json';bind(oldaudit,39);oldlaw=read(oldaudit)['laws']
                for model in ('uint64','continuous'):assert [r['models'][model]['probability'] for r in laws]==oldlaw[model]['distribution']
            save(out/(label+'.audit.json'),encode(dict(job=job,command=cmd,laws=laws,replayed_steps=steps)))
            storage=dict(path=label+'.stdout.json.gz',raw_bytes=len(rawbytes),compressed_bytes=len(data),raw_sha256=hashlib.sha256(rawbytes).hexdigest());save(out/(label+'.storage.json'),storage)
            row.update(status='completed',timings=raw['timings_seconds'],probe_seconds=raw['probe_seconds'],probe_control_checks=raw['probe_control_checks'],public_path_checks=raw['public_path_checks'],replayed_steps=steps,paths=len(laws),steps=sum(len(r['steps']) for r in raw['path_probes']))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-tick;rows.append(row);save(out/'rows.json',rows);print('end',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items())
    save(out/'summary.json',dict(planned=5,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))


if __name__=='__main__':main()
